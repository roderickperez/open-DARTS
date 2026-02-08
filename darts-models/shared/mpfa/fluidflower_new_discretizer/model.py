from reservoir import UnstructReservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh
import numpy as np
from dataclasses import dataclass, field
import meshio

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from dartsflash.libflash import CubicEoS, AQEoS
from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, Ziabakhsh2012, FlashParams, InitialGuess
from dartsflash.components import CompData


@dataclass
class Corey:
    nw: float
    ng: float
    swc: float
    sgc: float
    krwe: float
    krge: float
    labda: float
    p_entry: float
    def modify(self, std, mult):
        i = 0
        for attr, value in self.__dict__.items():
            if attr != 'type':
                setattr(self, attr, value * (1 + mult[i] * float(getattr(std, attr))))
            i += 1

    def random(self, std):
        for attr, value in self.__dict__.items():
            if attr != 'type':
                std_in = value * float(getattr(std, attr))
                param = np.random.normal(value, std_in)
                if param < 0:
                    param = 0
                setattr(self, attr, param)


@dataclass
class PorPerm:
    type: str
    poro: float
    perm: float
    anisotropy: list


class Model(DartsModel):
    def __init__(self, platform='cpu', pres_in=1.01325, temp_in=293., discr_type='mpfa',
                 problem_type='reservoir', mpfa_type='mpfa', thermal=False, init_filename=None):
        # call base class constructor
        super().__init__()

        self.init_filename = init_filename
        self.pres_in = pres_in
        self.discr_type = discr_type

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.reservoir = UnstructReservoir(discr_type=discr_type, problem_type=problem_type, mpfa_type=mpfa_type)

        # some numbers to check: volume = 2.8 x 1.3 x 0.025 = 0.091 m3, pore volume = volume * 0.44 = 0.04 m3
        volume = np.array(self.reservoir.mesh.volume, copy=False)
        porosity = np.array(self.reservoir.mesh.poro, copy=False)
        print("---------------------------------")
        print("Total volume = %8.5f" % (np.sum(volume)))
        print("Total pore volume = %8.5f" % (np.sum(volume*porosity)))
        print("---------------------------------")

        if thermal:
            temp_init = None
        else:
            temp_init = temp_in
        self.set_physics(temperature=temp_init, temp_in=temp_in, n_points=2001, platform=platform)

        """Configuration"""
        self.bound_cond = 'wells'
        self.physics_type = 'compositional'

        if discr_type == 'mpfa':
            self.reservoir.P_VAR = 0

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-6
        self.params.max_ts = 0.00694  # 10 minutes reporting time
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4
        self.params.max_i_newton = 6
        self.params.max_i_linear = 100
        self.params.newton_type = sim_params.newton_local_chop
        # self.params.newton_params[0] = 0.2
        self.params.linear_type = sim_params.cpu_gmres_cpr_amg#cpu_gmres_cpr_amg#cpu_superlu

        self.timer.node["initialization"].stop()

    def init(self, platform='cpu'):
        DartsModel.init(self, discr_type=self.discr_type, platform=platform)

    def set_physics(self, temperature=None, temp_in=None, n_points=2001, platform='cpu'):
        self.zero = 1e-8

        """Physical properties"""
        # Fluid components, ions and solid
        components = ["CO2", "H2O"]
        phases = ["V", "Aq"]
        nc = len(components)
        comp_data = CompData(components, setprops=True)

        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                               AQEoS.CompType.solute: AQEoS.Ziabakhsh2012
                               })

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        eos_used = ["PR", "AQ"]
        split_initial_guess = [InitialGuess.Ki.Henry_VA]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        self.ini_stream = [self.zero]  # z1
        self.inj_stream = [1 - self.zero]
        self.inj_rate = 10 * 24 * 60 / 1E6  # ml/min to m3/day

        if temperature is None:  # if None, then thermal=True
            thermal = True
            self.T_init = temp_in
        else:
            thermal = False
            self.T_init = None

        self.physics = CustomPhysics(components, phases, timer=self.timer,
                                     n_points=n_points, min_p=0.8 * self.pres_in,
                                     max_p=1.2 * self.pres_in, min_z=self.zero / 10, max_z=1 - self.zero / 10,
                                     thermal=thermal, min_t=0.8 * temperature, max_t=1.2 * temperature, cache=False)

        corey = {
            0: Corey(nw=2.0, ng=1.5, swc=0.11, sgc=0.06, krwe=0.80, krge=0.85, labda=2., p_entry=3e-3),
            1: Corey(nw=2.0, ng=1.5, swc=0.12, sgc=0.08, krwe=0.93, krge=0.95, labda=2., p_entry=3e-3),
            2: Corey(nw=2.5, ng=2.0, swc=0.14, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=3e-3),
            3: Corey(nw=2.5, ng=2.0, swc=0.32, sgc=0.14, krwe=0.71, krge=0.75, labda=2., p_entry=3e-3)
        }
        self.property_regions = [key for key in corey.keys()]
        for region in self.property_regions:
            property_container = PropertyContainer(components_name=components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=self.zero / 10, temperature=temperature)
            diff_coef = 8.64e-6

            property_container.flash_ev = NegativeFlash(flash_params, eos_used, split_initial_guess)
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(len(components)) * diff_coef)),
                                                    ('Aq', ConstFunc(np.ones(len(components)) * diff_coef))])

            if thermal:
                # property_container.enthalpy_ev = dict([('V', Enthalpy(hcap=0.035)),
                #                                        ('L', Enthalpy(hcap=0.035)),
                #                                        ('Aq', Enthalpy(hcap=0.00418*18.015)),]
                property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                       ('Aq', EoSEnthalpy(eos=aq)), ])

                property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                           ('Aq', ConstFunc(0.)), ])

            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey[region], 'V')),
                                                   ('Aq', ModBrooksCorey(corey[region], 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey[region])

            self.physics.add_property_region(property_container, region)

            property_container.output_props = {"satV": lambda ii=region: self.physics.property_containers[ii].sat[0],
                                               "xCO2": lambda ii=region: self.physics.property_containers[ii].x[1, 1],
                                               "rhoV": lambda ii=region: self.physics.property_containers[ii].dens[0],
                                               "rho_mA": lambda ii=region: self.physics.property_containers[ii].dens_m[1]}

        return

    def set_wells(self):
        well_index = 1e2

        if not self.init_filename:
            # id1 = self.reservoir.well_cells[0]
            # well_depth0 = self.reservoir.water_column_height - self.reservoir.well_centers[0,2]
            well_cells, well_depth = self.reservoir.find_well_cells(self.reservoir.well_centers[0])
            self.reservoir.add_well("INJ001", depth=well_depth)
            self.reservoir.add_perforation(self.reservoir.wells[-1], well_cells[0], well_index=well_index)

            # id2 = self.reservoir.well_cells[1]
            # well_depth1 = self.reservoir.water_column_height - self.reservoir.well_centers[1,2]
            # well_cells, well_depth = self.reservoir.find_well_cells(self.reservoir.well_centers[1])
            # self.reservoir.add_well("PROD001", depth=well_depth)
            # self.reservoir.add_perforation(self.reservoir.wells[-1], well_cells[0], well_index=well_index)

        # Add top layer production well for boundary condition
        if self.reservoir.discr_type == 'tpfa':
            max_z = np.max(self.reservoir.unstr_discr.mesh_data.points[:, 2])
            ids = np.where(self.reservoir.unstr_discr.mesh_data.points[self.reservoir.unstr_discr.mesh_data.cells_dict['wedge']][:, :, 2] == max_z)
            unique, counts = np.unique(ids[0], return_counts=True)
            self.reservoir.top_cells = unique[counts > 2]

        self.reservoir.add_well("P1", self.reservoir.min_depth)
        for nth_cell in self.reservoir.top_cells:
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], res_block=nth_cell, well_index=well_index)

        self.p_prod = self.pres_in + self.reservoir.min_depth * 0.1

    def add_wells_unstructured_cpp(self):
        layers_num = 1
        is_center = False
        Lx = max([pt.values[0] for pt in self.reservoir.discr_mesh.nodes])
        Ly = max([pt.values[1] for pt in self.reservoir.discr_mesh.nodes])
        Lz = max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])
        dx = np.sqrt(np.mean(self.reservoir.volume_all_cells) / \
                ( max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes]) - min([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])))

        setback = 0

        # find closest cells
        # dist = 1.E+10
        # pt1 = np.array([(setback + 0.5) * dx, (setback + 0.5) * dx, 0])
        # id1 = -1
        # for cell_id in range(self.reservoir.discr_mesh.region_ranges[elem_loc.MATRIX][0], self.reservoir.discr_mesh.region_ranges[elem_loc.MATRIX][1]):
        #     c = self.reservoir.discr_mesh.centroids[cell_id]
        #     cur_dist = (c.values[0] - pt1[0]) ** 2 + (c.values[1] - pt1[1]) ** 2 + c.values[2] ** 2
        #     if dist > cur_dist:
        #         dist = cur_dist
        #         id1 = cell_id
        # assert (id1 >= 0)
        #
        # dist = 1.E+10
        # pt2 = np.array([L - (setback + 0.5) * dx, L - (setback + 0.5) * dx, 0])
        # id2 = -1
        # for cell_id in range(self.reservoir.discr_mesh.region_ranges[elem_loc.MATRIX][0], self.reservoir.discr_mesh.region_ranges[elem_loc.MATRIX][1]):
        #     c = self.reservoir.discr_mesh.centroids[cell_id]
        #     cur_dist = (c.values[0] - pt2[0]) ** 2 + (c.values[1] - pt2[1]) ** 2 + c.values[2] ** 2
        #     if dist > cur_dist:
        #         dist = cur_dist
        #         id2 = cell_id
        # assert(id2 >= 0)

        n_cells = self.reservoir.discr_mesh.n_cells
        pt_x = np.array([c.values[0] for c in self.reservoir.discr_mesh.centroids])[:n_cells]
        pt_y = np.array([c.values[1] for c in self.reservoir.discr_mesh.centroids])[:n_cells]
        pt_z = np.array([c.values[2] for c in self.reservoir.discr_mesh.centroids])[:n_cells]

        id1 = ((pt_x - Lx/2) ** 2 + (pt_y - Ly/3) ** 2 + pt_z ** 2).argmin()
        id2 = ((pt_x - Lx/2) ** 2 + (pt_y - 2*Ly/3) ** 2 + pt_z ** 2).argmax()

        self.reservoir.add_well("INJ001", depth=0)
        for kk in range(layers_num):
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(id1 + kk), well_index=self.reservoir.well_index)

        self.reservoir.add_well("PROD001", depth=0)
        for kk in range(layers_num):
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(id2 + kk), well_index=self.reservoir.well_index)

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        if self.init_filename:
            mesh = meshio.read(filename=self.init_filename)
            props = ['pressure', 'CO2']
            cell_data = {}
            for geom_name, geom in mesh.cells_dict.items():
                #centers = np.append(centers, np.average(mesh.points[geom], axis=1), axis=0)
                for prop in props:
                    if prop in mesh.cell_data_dict:
                        if prop not in cell_data: cell_data[prop] = []
                        cell_data[prop].append(mesh.cell_data_dict[prop][geom_name])

            if self.physics.thermal:
                self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=cell_data['pressure'][0],
                                                              uniform_composition=cell_data['CO2'][0], uniform_temp=self.ini_stream[-1])
            else:
                self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=cell_data['pressure'][0],
                                                            uniform_composition=cell_data['CO2'][0])
        else:
            if self.physics.thermal:
                self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.pres_in,
                                                              uniform_composition=self.ini_stream[:-1], uniform_temp=self.ini_stream[-1])
            else:
                c = np.array([cent.values[2] for cent in self.reservoir.discr_mesh.centroids])
                nb = self.reservoir.mesh.n_res_blocks
                init_composition = np.zeros(nb)
                init_composition[c[:nb] > 0.4] = 0.008 * (1.0 - (0.5 - c[:nb][c[:nb] > 0.4]) / (0.5 - 0.4))
                init_composition[init_composition < 0] = self.zero
                #print(init_composition.min(), init_composition.max())
                #print(init_composition[c[:nb] > 0.4])
                self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=self.pres_in,
                                                            uniform_composition=init_composition)#self.ini_stream)

    def set_boundary_conditions(self):
        if not self.init_filename:
            self.reservoir.wells[0].control = self.physics.new_rate_inj(0, self.inj_stream, 0)
            #self.reservoir.wells[1].control = self.physics.new_rate_inj(0, self.inj_stream, 0)

        self.reservoir.wells[-1].control = self.physics.new_bhp_prod(self.p_prod)

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks

        if 1:
            layers_to_regions = {"ESF": 3, "C": 2, "D": 1, "E": 0, "F": 0, "G": 0, "W": 0}

            if self.reservoir.discr_type == 'mpfa':
                layers = self.reservoir.layers
            else:
                layers = self.reservoir.unstr_discr.layers

            for ith_layer, cells in enumerate(layers):
                region = layers_to_regions[self.reservoir.porperm[ith_layer].type]
                self.op_num[cells] = region
            self.op_num[n_res:] = len(self.property_regions)

            # C, D, E, ESF, F, Fault-1, Fault-2, G, W
            self.op_list = [*self.physics.acc_flux_itor.values(), self.physics.acc_flux_w_itor]

        else:
            self.op_num[n_res:] = 1
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]

    def output_properties(self):
        n_vars = self.physics.property_operators.n_vars
        n_props = self.physics.property_operators.n_props
        tot_props = n_vars + n_props

        property_array = np.zeros((self.reservoir.nb, tot_props))
        for j in range(n_vars):
            property_array[:, j] = self.physics.engine.X[j:self.reservoir.nb * n_vars:n_vars]

        values = value_vector(np.zeros(self.physics.n_ops))

        for i in range(self.reservoir.nb):
            state = []
            for j in range(n_vars):
                state.append(property_array[i, j])
            state = value_vector(np.asarray(state))
            self.physics.property_itor.evaluate(state, values)

            for j in range(n_props):
                property_array[i, j + n_vars] = values[j]

        return property_array

    # def run_python_my(self, days=0, restart_dt=0, verbose=0, max_res=1e20):
    #     runtime = days
    #
    #     mult_dt = self.params.mult_ts
    #     max_dt = self.max_dt
    #     self.e = self.physics.engine
    #
    #     # get current engine time
    #     t = self.e.t
    #
    #     # same logic as in engine.run
    #     if np.fabs(t) < 1e-15:
    #         dt = self.params.first_ts
    #     elif restart_dt > 0:
    #         dt = restart_dt
    #     else:
    #         dt = self.max_dt
    #
    #     # evaluate end time
    #     runtime += t
    #     ts = 0
    #
    #     counter_cut = 0
    #     counter_add = 0
    #     while t < runtime:
    #         converged = self.run_timestep_python(dt, t, verbose=0, max_res=max_res)
    #
    #         if converged:
    #             t += dt
    #             ts = ts + 1
    #             if verbose:
    #                 print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
    #                       % (ts, t, dt, self.e.n_newton_last_dt, self.e.n_linear_last_dt))
    #
    #             dt *= mult_dt
    #             if dt > max_dt:
    #                 dt = max_dt
    #
    #             if t + dt > runtime:
    #                 dt = runtime - t
    #
    #             if self.e.n_newton_last_dt < 5:
    #                 counter_add += 1
    #
    #             if counter_add > 20 and max_dt < self.params.max_ts:
    #                 max_dt *= 2
    #                 counter_add = 0
    #                 counter_cut = 0
    #                 print("Increase max timestep to %g" % max_dt)
    #
    #         else:
    #             dt /= mult_dt
    #             if verbose:
    #                 print("Cut timestep to %g" % dt)
    #             counter_cut += 1
    #             counter_add = 0
    #             if counter_cut > 5:
    #                 max_dt /= 4
    #                 counter_cut = 0
    #                 print("Cut max timestep to %g" % max_dt)
    #
    #             if dt < 1e-10:
    #                 break
    #     # update current engine time
    #     self.e.t = runtime
    #
    #     print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.e.stat.n_timesteps_total, self.e.stat.n_timesteps_wasted,
    #                                                      self.e.stat.n_newton_total, self.e.stat.n_newton_wasted,
    #                                                      self.e.stat.n_linear_total, self.e.stat.n_linear_wasted))
    #     self.max_dt = max_dt

class CustomPhysics(Compositional):
    def __init__(self, components, phases, timer,
                 n_points, min_p, max_p, min_z, max_z, min_t=None, max_t=None, thermal=False, cache=False):
        super().__init__(components=components, phases=phases, timer=timer, n_points=n_points,
                         min_p=min_p, max_p=max_p, min_z=min_z, max_z=max_z, min_t=min_t, max_t=max_t,
                         thermal=thermal, cache=cache)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list, uniform_temp: float = None):
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks
        nb_res = mesh.n_res_blocks
        """ Uniform Initial conditions """
        # set initial pressure
        pz_bounds = np.array(mesh.pz_bounds, copy=False)

        pressure_grad = 0.09995
        pressure = np.array(mesh.pressure, copy=False)
        if isinstance( uniform_pressure, (np.ndarray, np.generic) ):
            pressure[:nb_res] = uniform_pressure
            pz_bounds[::self.nc] = np.mean(uniform_pressure)
        else:
            depth = np.array(mesh.depth, copy=True)
            nonuniform_pressure = depth[:nb] * pressure_grad + uniform_pressure
            pressure[:] = nonuniform_pressure
            # pressure = np.array(mesh.pressure, copy=False)
            # pressure.fill(uniform_pressure)
            pz_bounds[::self.nc] = np.mean(nonuniform_pressure)

        # if thermal, set initial temperature
        if uniform_temp is not None:
            temperature = np.array(mesh.temperature, copy=False)
            temperature.fill(uniform_temp)

        # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        # composition[:] = np.array(uniform_composition)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c:(self.nc - 1) * nb_res:(self.nc - 1)] = uniform_composition[:]
                pz_bounds[1+c::self.nc] = np.mean(uniform_composition[:])
        else:
            for c in range(self.nc - 1):  # Denis
                composition[c::(self.nc - 1)] = uniform_composition[c]
                pz_bounds[1+c::self.nc] = np.mean(uniform_composition[c])

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.nc - 1):
            composition[c::(self.nc - 1)] = uniform_composition[c]


class ModBrooksCorey:
    def __init__(self, corey, phase):
        self.k_rw_e = corey.krwe
        self.k_rg_e = corey.krge

        self.swc = corey.swc
        self.sgc = corey.sgc

        self.nw = corey.nw
        self.ng = corey.ng

        self.phase = phase

    def evaluate(self, sat):
        if self.phase == "Aq":
            Se = (sat - self.swc)/(1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rw_e * Se ** self.nw
        else:
            Se = (sat - self.sgc) / (1 - self.swc - self.sgc)
            if Se > 1:
                Se = 1
            elif Se < 0:
                Se = 0
            k_r = self.k_rg_e * Se ** self.ng

        return k_r


class ModCapillaryPressure:
    def __init__(self, corey):
        self.swc = corey.swc
        self.p_entry = corey.p_entry
        self.labda = corey.labda
        # self.labda = 3
        self.eps = 1e-3

    def evaluate(self, sat):
        sat_w = sat[1]
        Se = (sat_w - self.swc)/(1 - self.swc)
        if Se < self.eps:
            Se = self.eps
        pc = self.p_entry * self.eps ** (1/self.labda) * Se ** (-1/self.labda)  # for p_entry to non-wetting phase
        # if Se > 1 - self.eps:
        #     pc = 0

        #pc = self.p_entry
        pc = 0.0#self.p_entry
        Pc = np.array([0, pc], dtype=object)  # V, Aq
        return Pc


class BrooksCorey:
    def __init__(self, wetting: bool):
        self.sat_wr = 0.15
        # self.sat_nwr = 0.1

        self.lambda_w = 4.2
        self.lambda_nw = 3.7

        self.wetting = wetting

    def evaluate(self, sat_w):
        # From Brooks-Corey (1964)
        Se = (sat_w - self.sat_wr)/(1-self.sat_wr)
        if Se > 1:
            Se = 1
        elif Se < 0:
            Se = 0

        if self.wetting:
            k_r = Se**((2+3*self.lambda_w)/self.lambda_w)
        else:
            k_r = (1-Se)**2 * (1-Se**((2+self.lambda_nw)/self.lambda_nw))

        if k_r > 1:
            k_r = 1
        elif k_r < 0:
            k_r = 0

        return k_r