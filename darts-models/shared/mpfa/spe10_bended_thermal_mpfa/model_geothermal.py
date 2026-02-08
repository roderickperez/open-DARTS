from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params
from darts.reservoirs.struct_reservoir import StructReservoir
import numpy as np

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.iapws.iapws_property import (Density_iapws_water, Density_iapws_steam,
                                                           Viscosity_iapws_water, Viscosity_iapws_steam,
                                                           Saturation_iapws_water, Saturation_iapws_steam,
                                                           Relperm_iapws_water, Relperm_iapws_steam,
                                                           Enthalpy_iapws_water, Enthalpy_iapws_steam,
                                                           Conductivity_iapws_water, Conductivity_iapws_steam)
from darts.physics.properties.enthalpy import EnthalpyBasic

from reservoir import UnstructReservoir


class Model_geothermal(DartsModel):
    def __init__(self, discr_type, model_folder, T=2000, report_step=100.0, physics_type="", customize_new_operator=False, zero: float = 1e-13):
        # call base class constructor
        super().__init__()

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.discr_type = discr_type
        self.model_folder = model_folder
        self.physics_type = physics_type

        self.T = T
        self.report_step = report_step
        self.customize_new_operator = customize_new_operator

        self.set_physics(zero)
        self.set_reservoir(model_folder)
        self.set_wells()

        self.set_sim_params(first_ts=0.001, mult_ts=2, max_ts=50, runtime=1000, tol_newton=1e-3, tol_linear=1e-6)

        self.timer.node["initialization"].stop()

        self.p_init = 200.0
        self.init_temp = 350.0
        self.inj = value_vector([300])
        self.water_rate = 2400 / 2  # m3/day
        self.initial_values = {
                               self.physics.vars[0]: self.p_init,
                               self.physics.vars[1]: self.init_temp
                              }

    def set_reservoir(self, model_folder):
        self.reservoir = UnstructReservoir(self.discr_type, model_folder, n_vars=self.physics.n_vars)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap.fill(2200)     # kJ/(m3K), specific capacity [J/kg/K] * density [kg/m3] / 1000
        rcond.fill(181.44)  # kJ/m/day/K, heat conductivity [W/m/K] / 1000 * 24 * 3600

        if self.discr_type == 'mpfa':
            self.reservoir.mesh.pz_bounds.resize(self.physics.n_vars * self.reservoir.n_bounds)
            pz_bounds = np.array(self.reservoir.mesh.pz_bounds, copy=False)
            pz_bounds[::self.reservoir.n_vars] = self.p_init
            for i in range(1, self.physics.nc):  # in geothermal model [pres, T], this loop will be skipped
                pz_bounds[i::self.reservoir.n_vars] = self.ini[i-1]

            pz_bounds[self.physics.n_vars-1::self.reservoir.n_vars] = self.init_temp

            self.reservoir.P_VAR = self.engine.P_VAR

    def set_wells(self):
        if self.discr_type == 'mpfa':
            # centroids = np.array(self.reservoir.discr_mesh.centroids, copy=False)[:self.reservoir.n_matrix]
            centroids = np.array([np.array([c.values[0], c.values[1]]) for
                                  c in self.reservoir.discr_mesh.centroids])[:self.reservoir.n_matrix]

            l_max = np.max(self.reservoir.mesh_data.points, axis=0)

            mult1 = 0.21
            mult2 = 1 - mult1
            well_coords = np.array([[mult1 * l_max[0], mult1 * l_max[1]],
                                    [mult1 * l_max[0], mult2 * l_max[1]],
                                    [mult2 * l_max[0], mult1 * l_max[1]],
                                    [mult2 * l_max[0], mult2 * l_max[1]]])
            well_names = ['PRD1', 'INJ1', 'INJ2', 'PRD2']
            self.well_cell_ids = []
            well_init_depth = 0.0
            nodes = np.array(self.reservoir.discr_mesh.nodes, copy=False)
            elems = np.array(self.reservoir.discr_mesh.elems, copy=False)
            for i, coord in enumerate(well_coords):
                ids = ((centroids[:,0] - coord[0]) ** 2 + (centroids[:,1] - coord[1]) ** 2).argsort()
                self.well_cell_ids.append(ids[:self.reservoir.nz])
                # adding well
                self.reservoir.add_well(well_names[i], depth=well_init_depth)
                # adding perforations
                for cell_id in ids[:self.reservoir.nz]:
                    cell = elems[cell_id]
                    pt_ids = self.reservoir.discr_mesh.elem_nodes[cell.pts_offset:cell.pts_offset+cell.n_pts]
                    pts = np.array([nodes[id].values for id in pt_ids])
                    # Calculate well_index (very primitive way....):
                    rw = 0.1
                    dx = np.max(pts, axis=0)[0] - np.min(pts, axis=0)[0]
                    dy = np.max(pts, axis=0)[1] - np.min(pts, axis=0)[1]
                    dz = self.reservoir.volume_all_cells[cell_id] / dx / dy
                    mean_perm_xx = self.reservoir.permeability[3 * cell_id]
                    mean_perm_yy = self.reservoir.permeability[3 * cell_id + 1]
                    mean_perm_zz = self.reservoir.permeability[3 * cell_id + 2]
                    rp_z = 0.28 * np.sqrt((mean_perm_yy / mean_perm_xx) ** 0.5 * dx ** 2 +
                                          (mean_perm_xx / mean_perm_yy) ** 0.5 * dy ** 2) / \
                           ((mean_perm_xx / mean_perm_yy) ** 0.25 + (mean_perm_yy / mean_perm_xx) ** 0.25)
                    wi_x = 0.0
                    wi_y = 0.0
                    wi_z = 2 * np.pi * np.sqrt(mean_perm_xx * mean_perm_yy) * dz / np.log(rp_z / rw)
                    well_index = np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)
                    # add perforation
                    self.reservoir.add_perforation(self.reservoir.wells[-1], cell_id, well_index=well_index)
        else:
            centroids = np.array([np.array([c[0], c[1]]) for
                                  c in self.reservoir.unstr_discr.centroid_all_cells])[:np.size(self.reservoir.poro)]

            l_max = np.max(self.reservoir.mesh_data.points, axis=0)

            mult1 = 0.21
            mult2 = 1 - mult1
            well_coords = np.array([[mult1 * l_max[0], mult1 * l_max[1]],
                                    [mult1 * l_max[0], mult2 * l_max[1]],
                                    [mult2 * l_max[0], mult1 * l_max[1]],
                                    [mult2 * l_max[0], mult2 * l_max[1]]])
            well_names = ['PRD1', 'INJ1', 'INJ2', 'PRD2']
            self.well_cell_ids = []
            well_init_depth = 0.0
            # nodes = np.array(self.reservoir.discr_mesh.nodes, copy=False)
            # elems = np.array(self.reservoir.discr_mesh.elems, copy=False)
            nodes = np.array(self.reservoir.mesh_data.points, copy=False)
            elems = np.array(self.reservoir.mesh_data.cells_dict["hexahedron"], copy=False)
            for i, coord in enumerate(well_coords):
                ids = ((centroids[:, 0] - coord[0]) ** 2 + (centroids[:, 1] - coord[1]) ** 2).argsort()
                self.well_cell_ids.append(ids[:self.reservoir.nz])
                # adding well
                self.reservoir.add_well(well_names[i], depth=well_init_depth)
                # adding perforations
                for cell_id in ids[:self.reservoir.nz]:
                    cell = elems[cell_id]
                    pts = nodes[cell]
                    # Calculate well_index (very primitive way....):
                    rw = 0.1
                    dx = np.max(pts, axis=0)[0] - np.min(pts, axis=0)[0]
                    dy = np.max(pts, axis=0)[1] - np.min(pts, axis=0)[1]
                    dz = self.reservoir.volume[cell_id] / dx / dy
                    mean_perm_xx = self.reservoir.permeability[3 * cell_id]
                    mean_perm_yy = self.reservoir.permeability[3 * cell_id + 1]
                    mean_perm_zz = self.reservoir.permeability[3 * cell_id + 2]
                    rp_z = 0.28 * np.sqrt((mean_perm_yy / mean_perm_xx) ** 0.5 * dx ** 2 +
                                          (mean_perm_xx / mean_perm_yy) ** 0.5 * dy ** 2) / \
                           ((mean_perm_xx / mean_perm_yy) ** 0.25 + (mean_perm_yy / mean_perm_xx) ** 0.25)
                    wi_x = 0.0
                    wi_y = 0.0
                    wi_z = 2 * np.pi * np.sqrt(mean_perm_xx * mean_perm_yy) * dz / np.log(rp_z / rw)
                    well_index = np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)
                    # add perforation
                    self.reservoir.add_perforation(self.reservoir.wells[-1], cell_id, well_index=well_index)

    def set_physics(self, zero: float = 1e-13):
        """Physical properties"""
        components = ['w']
        phases = ['water', 'steam']

        # cell_property corresponds to the variables in engine.X
        self.cell_property = ['pressure']
        self.cell_property += ['temperature']

        property_container = ModelProperties(phases_name=phases, components_name=components, min_z=zero / 10)

        # # Define property evaluators based on custom properties
        # property_container.density_ev = dict([('water', DensityBasic(compr=1e-5, dens0=1014)),
        #                                       ('steam', DensityBasic(compr=5e-3, dens0=50))])
        # property_container.viscosity_ev = dict([('water', ConstFunc(0.3)),
        #                                         ('steam', ConstFunc(0.03))])
        # property_container.rel_perm_ev = dict([('water', PhaseRelPerm("water", 0.1, 0.1)),
        #                                        ('steam', PhaseRelPerm("gas", 0.1, 0.1))])
        property_container.enthalpy_ev = dict([('water', EnthalpyBasic(hcap=4.18)),
                                               ('steam', EnthalpyBasic(hcap=0.035))])
        # property_container.conductivity_ev = dict([('water', ConstFunc(1.)),
        #                                            ('steam', ConstFunc(1.))])
        # property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        property_container.density_ev = dict([('water', Density_iapws_water()),
                                              ('steam', Density_iapws_steam())])
        property_container.viscosity_ev = dict([('water', Viscosity_iapws_water()),
                                                ('steam', Viscosity_iapws_steam())])
        property_container.saturation_ev = dict([('water', Saturation_iapws_water()),
                                               ('steam', Saturation_iapws_steam())])
        property_container.rel_perm_ev = dict([('water', Relperm_iapws_water()),
                                               ('steam', Relperm_iapws_steam())])
        # property_container.enthalpy_ev = dict([('water', Enthalpy_iapws_water()),
        #                                        ('steam', Enthalpy_iapws_steam())])
        property_container.conductivity_ev = dict([('water', Conductivity_iapws_water()),
                                                   ('steam', Conductivity_iapws_steam())])
        property_container.rock_energy_ev = EnthalpyBasic(hcap=1.0)

        # create physics
        thermal = True
        self.physics = Compositional(components, phases, self.timer,
                                     n_points=400, min_p=0, max_p=1000, min_z=zero, max_z=1 - zero,
                                     min_t=273.15 + 20, max_t=273.15 + 200, thermal=thermal, cache=True)
        self.physics.add_property_region(property_container)

    def set_well_controls(self):
        for i, w in enumerate(self.reservoir.wells):
            if w.name[:3] == 'PRD':
                w.control = self.physics.new_bhp_prod(self.p_init - 10)
                # w.control = self.physics.new_rate_prod(self.water_rate, 0)
            elif w.name[:3] == 'INJ':
                # w.control = self.physics.new_rate_inj(200, self.inj, 1)
                w.control = self.physics.new_bhp_inj(self.p_init + 10, self.inj)
                # w.control = self.physics.new_rate_inj(self.water_rate, self.inj, 0)
                # w.control = self.physics.new_bhp_inj(450, self.inj)

    def run(self, days=0, restart_dt=0, log_3d_body_path=0):
        if days:
            runtime = days
        else:
            runtime = self.runtime

        mult_dt = self.params.mult_ts
        max_dt = self.params.max_ts
        self.e = self.physics.engine

        # get current engine time
        t = self.e.t

        # same logic as in engine.run
        if np.fabs(t) < 1e-15:
            dt = self.params.first_ts
        elif restart_dt > 0:
            dt = restart_dt
        else:
            dt = self.params.max_ts

        # evaluate end time
        runtime += t
        ts = 0
        #
        if log_3d_body_path and self.physics.n_vars == 3:
            self.body_path_start()

        self.i_th_step = 0

        good_ts_counter = 0
        while t < runtime:
            # if t == 0:
            #     self.full_arr_operator = np.array(self.e.op_vals_arr)
            # else:
            #     self.full_arr_operator = np.vstack((self.full_arr_operator, np.array(self.e.op_vals_arr)))
            # np.savetxt("Operator_analytical.csv", self.full_arr_operator.T, delimiter=",")

            converged = self.run_timestep_python(dt, t)
            if converged:
                t += dt
                ts = ts + 1
                print("# %d \tT = %f\tDT = %f\tNI = %d\tLI=%d"
                      % (ts, t, dt, self.e.n_newton_last_dt, self.e.n_linear_last_dt))

                if self.e.n_newton_last_dt < 4:
                    dt *= mult_dt
                if self.e.n_newton_last_dt < 3:
                    good_ts_counter += 1
                else:
                    good_ts_counter = 0
                if good_ts_counter > 5:
                    good_ts_counter = 0
                    max_dt *= mult_dt
                    # if max_dt > 25: max_dt = 25
                    # self.params.max_ts = max_dt
                if dt > max_dt:
                    dt = max_dt

                if t + dt > runtime:
                    dt = runtime - t

                if log_3d_body_path and self.physics.n_vars == 3:
                    self.body_path_add_bodys(t)
                    nb_begin = self.reservoir.nx * self.reservoir.ny * (self.body_path_map_layer - 1) * 3
                    nb_end = self.reservoir.nx * self.reservoir.ny * (self.body_path_map_layer) * 3

                    self.save_matlab_map(self.body_path_axes[0] + '_ts_' + str(ts), self.e.X[nb_begin:nb_end:3])
                    self.save_matlab_map(self.body_path_axes[1] + '_ts_' + str(ts), self.e.X[nb_begin + 1:nb_end:3])
                    self.save_matlab_map(self.body_path_axes[2] + '_ts_' + str(ts), self.e.X[nb_begin + 2:nb_end:3])
            else:
                dt /= 2 * mult_dt
                # max_dt /= mult_dt
                print("Cut timestep to %f" % (dt))
                # if dt < 1e-8:
                #     break
            # if self.discr_type == 'mpfa':
            #     self.reservoir.write_to_vtk(self.output_directory, self.cell_property, self.i_th_step, self.engine)
            # else:
            #     self.reservoir.write_to_vtk_old_discretizer(self.output_directory, self.cell_property, self.i_th_step, self.engine)
            # self.i_th_step += 1

        # update current engine time
        self.e.t = runtime

        # print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.e.stat.n_timesteps_total, self.e.stat.n_timesteps_wasted,
        #                                                  self.e.stat.n_newton_total, self.e.stat.n_newton_wasted,
        #                                                  self.e.stat.n_linear_total, self.e.stat.n_linear_wasted))

    def run_timestep(self, dt, t):
        max_newt = self.params.max_i_newton
        self.e.n_linear_last_dt = 0
        well_tolerance_coefficient = 1e2
        res_history = []
        self.timer.node['simulation'].start()
        for i in range(max_newt + 1):
            self.e.run_single_newton_iteration(dt)
            self.e.newton_residual_last_dt = self.e.calc_newton_residual()
            # if i == 0 and self.e.newton_residual_last_dt > 5.E-5:
            #    self.e.newton_residual_last_dt = 1.0
            #    break
            self.e.well_residual_last_dt = self.e.calc_well_residual()
            self.e.n_newton_last_dt = i
            # print('matrix_res = ' + str(self.e.newton_residual_last_dt) + '\t' + 'well_res = ' + str(
            #     self.e.well_residual_last_dt))
            res_history.append((self.e.newton_residual_last_dt, self.e.well_residual_last_dt))
            #  check tolerance if it converges
            if ((
                    self.e.newton_residual_last_dt < self.params.tolerance_newton and self.e.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton)
                    or self.e.n_newton_last_dt == self.params.max_i_newton):
                if i > 0:  # min_i_newton
                    break
            # elif i > 0 and \
            #     res_history[-1][0] / res_history[0][0] < 1.e-8 and \
            #     res_history[-1][1] / res_history[0][1] < 1.e-8:
            #         break

            r_code = self.e.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.e.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.e.post_newtonloop(dt, t)
        self.timer.node['simulation'].stop()
        return converged


class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.array([18.015, 18.015])  # Molar quantities of water and steam
        super().__init__(phases_name, components_name, Mw, min_z, temperature=None)
        self.x = np.ones((self.nph, self.nc))

    def compute_saturation(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1] or [pres, comp_0, ..., comp_N-1, T]
        :return: updated value for saturation array
        """
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        temperature = vec_state_as_np[-1]

        sat = np.zeros(self.nph)
        for j in self.ph:
            sat[j] = self.saturation_ev[self.phases_name[j]].evaluate(pressure, temperature)

        return sat

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1] or [pres, comp_0, ..., comp_N-1, T]
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]
        temperature = vec_state_as_np[-1]

        self.ph = [0]  # phase number or index???

        for j in self.ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            # self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, 0)  # output in [kg/m3]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            # self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature)  # output in [cp]

        self.nu[0] = 1
        # self.compute_saturation(self.ph)
        self.sat = self.compute_saturation(state)

        for j in self.ph:
            # self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(pressure, temperature)
            self.pc[j] = 0

        mass_source = np.zeros(1)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source

    def evaluate_thermal(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :return: updated value for thermal properties
        """

        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        for j in self.ph:
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # kJ/kmol
            # self.kappa[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])
            #
            # self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(pressure, temperature)  # kJ/kmol
            self.kappa[j] = self.conductivity_ev[self.phases_name[j]].evaluate(pressure, temperature) # kJ/m/day/K, heat conductivity [W/m/K] / 1000 * 24 * 3600

        # Heat source and Reaction enthalpy, will be skipped in geothermal model
        energy_source = 0.
        if self.energy_source_ev:
            energy_source += self.energy_source_ev.evaluate(state)

        for j, reaction in self.kinetic_rate_ev.items():
            energy_source += reaction.evaluate_enthalpy(pressure, temperature, self.x, self.sat[-1])

        return self.enthalpy, self.kappa, energy_source



