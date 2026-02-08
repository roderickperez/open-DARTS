from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params, mech_operators, rsf_props, friction, contact_state, state_law, contact_solver, critical_stress, normal_condition
from reservoir import UnstructReservoir
import numpy as np
from darts.reservoirs.mesh.transcalc import TransCalculations as TC
from physics.physics_comp_sup import Poroelasticity
from physics.property_container import *
from physics.properties_basic import *
from darts.engines import vector_linear_solver_params, linear_solver_params

class Model(DartsModel):
    def __init__(self, n_points=64):
        super().__init__()
        self.n_points = n_points
        self.timer.node["initialization"].start()
        self.physics_type = 'poromechanics'

        self.reservoir = UnstructReservoir(timer=self.timer)
        self.set_physics()

        self.params.tolerance_newton = 1e-5 # Tolerance of newton residual norm ||residual||<tol_newt
        self.params.newton_type = sim_params.newton_global_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        self.params.max_i_newton = 10

        ls1 = linear_solver_params()
        ls1.linear_type = sim_params.cpu_superlu
        self.physics.engine.ls_params.append(ls1)

        # self.find_reference_cell()

        self.timer.node["initialization"].stop()

    def set_physics(self):
        self.zero = 1e-13
        """Physical properties"""
        self.property_container = model_properties(phases_name=['wat'], components_name=['w'],
                                                   min_z=self.zero/10)

        self.reservoir.fluid_density0 = 1014.0
        self.property_container.density_ev = dict([('wat', Density(compr=self.reservoir.fluid_compressibility,
                                                                   dens0=self.reservoir.fluid_density0))])
        self.property_container.viscosity_ev = dict([('wat', ViscosityConst(self.reservoir.fluid_viscosity))])
        self.property_container.rel_perm_ev = dict([('wat', PhaseRelPerm("single", 0.0, 0.0))])

        # create physics
        # self.physics = Poromechanics(timer=self.timer, physics_filename='input/physics.in',
        #             n_points=self.n_points, min_p=-1000, max_p=1000, max_u=1.E+20)
        self.physics = Poroelasticity(self.property_container, self.timer, n_points=400,
                                     min_p=-10, max_p=1000)
        self.engine = self.physics.engine

        self.reservoir.P_VAR = self.engine.P_VAR

    def init(self):
        self.set_boundary_conditions()
        self.reservoir.init_wells()
        self.physics.init_wells(self.reservoir.wells)
        self.set_initial_conditions()
        self.set_well_controls()
        self.set_op_list()
        self.reset()

        self.reservoir.mech_operators = mech_operators()
        self.reservoir.mech_operators.init(self.reservoir.mesh, self.reservoir.pm,
                                 self.physics.engine.P_VAR, self.physics.engine.Z_VAR,self.physics.engine.U_VAR,
                                 self.physics.engine.N_VARS, self.physics.engine.N_OPS, self.physics.engine.NC,
                                 self.physics.engine.ACC_OP, self.physics.engine.FLUX_OP, self.physics.engine.GRAV_OP)
        self.reservoir.mech_operators.prepare()
        self.init_contacts()
    def reinit_reference(self, output_directory):
        self.reservoir.turn_off_equilibrium()
        self.reservoir.write_to_vtk(output_directory, 0, self.physics)
        self.reservoir.eps_vol_ref = np.array(self.reservoir.mesh.ref_eps_vol, copy=False)
        self.reservoir.eps_vol_ref[:] = self.reservoir.mech_operators.eps_vol[:]
    def reinit(self, output_directory):
        self.reservoir.turn_off_equilibrium()
        self.reservoir.write_to_vtk(output_directory, 0, self.physics)
    def init_contacts(self):
        if hasattr(self.reservoir, 'contacts'):
            for contact in self.reservoir.contacts:
                contact.N_VARS = self.physics.engine.N_VARS
                contact.U_VAR = self.physics.engine.U_VAR
                contact.P_VAR = self.physics.engine.P_VAR
                contact.NT = self.physics.engine.N_VARS
                contact.U_VAR_T = self.physics.engine.U_VAR
                contact.P_VAR_T = self.physics.engine.P_VAR
                contact.init_friction(self.reservoir.pm, self.reservoir.mesh)
                contact.init_fault()
            self.physics.engine.contacts = self.reservoir.contacts
    def setup_contact_friction(self, contact_algorithm):
        if hasattr(self.reservoir, 'contacts'):
            for contact in self.physics.engine.contacts:
                friction_model = friction.STATIC#friction.SLIP_DEPENDENT#friction.RSF

                # allow to slip
                contact.set_state(contact_state.SLIP)
                # static friction coefficients
                mu0 = 0.5 * np.ones(len(contact.cell_ids))
                contact.mu0 = value_vector(mu0)
                contact.mu = contact.mu0

                # setup friction model
                contact.friction_model = friction_model
                # setup friction criterion
                contact.friction_criterion = critical_stress.TERZAGHI
                # setup normal condition
                contact.normal_condition = normal_condition.ZERO_GAP_CHANGE

                # RSF model
                if (friction_model == friction.RSF):
                    prop = rsf_props()
                    prop.min_vel = 1.E-11 * 86400
                    #theta = 1.0 / 86400.0 * np.ones(len(contact.cell_ids))
                    #prop.theta_n = value_vector(theta)
                    #prop.theta = value_vector(theta)
                    prop.a = 0.01#0.0008#0.0078
                    prop.b = 0.003
                    prop.crit_distance = 0.01 * 1.E-3
                    prop.ref_velocity = 0.01 * 1.E-6 * 86400
                    prop.law = state_law.AGEING_LAW
                    # set friction model
                    contact.rsf = prop

                for i in range(len(contact.eta)):
                    contact.eta[i] *= 0.0

                if contact_algorithm == contact_solver.LOCAL_ITERATIONS:
                    contact.init_local_iterations()

    def find_reference_cell(self):
        dims_min = np.array([self.reservoir.unstr_discr.mesh_data.points[:,0].min(), self.reservoir.unstr_discr.mesh_data.points[:,1].min(), self.reservoir.unstr_discr.mesh_data.points[:,2].min()])
        dims_max = np.array([self.reservoir.unstr_discr.mesh_data.points[:,0].max(), self.reservoir.unstr_discr.mesh_data.points[:,1].max(), self.reservoir.unstr_discr.mesh_data.points[:,2].max()])
        self.dims = dims_max - dims_min
        pt = [(dims_min[0] + dims_max[0]) / 2, dims_min[1] + 0.6 * (dims_max[1] - dims_min[1]), (dims_min[2] + dims_max[2]) / 2]
        #pt = [(dims_min[0] + dims_max[0]) / 2, dims_min[1] + 0.98 * (dims_max[1] - dims_min[1]), (dims_min[2] + dims_max[2]) / 2]
        dist = 1.E+10
        id = -1
        for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
            cur_dist = (cell.centroid[0] - pt[0]) ** 2 + (cell.centroid[1] - pt[1]) ** 2 + (cell.centroid[2] - pt[2]) ** 2
            if dist > cur_dist:
                dist = cur_dist
                id = cell_id
        if id != -1:    self.reservoir.ref_cell_id = id

    def add_wells(self):
        layers_num = 1
        is_at_corner = False
        is_wedge = False
        setback = 1
        if is_at_corner:
            self.reservoir.add_well("INJ001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(0 + kk),
                                               well_index=self.reservoir.well_index)

            end_id = self.reservoir.unstr_discr.mat_cells_tot - layers_num
            self.reservoir.add_well("PROD001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(end_id + kk),
                                               well_index=self.reservoir.well_index)
        else:
            if is_wedge:
                row_num = np.sqrt(self.reservoir.unstr_discr.mat_cells_tot / layers_num / 2)
                id1 = 2 * layers_num * (row_num + 1)
                id2 = self.reservoir.unstr_discr.mat_cells_tot - 2 * layers_num * (row_num + 2)
            else:
                row_num = np.sqrt(self.reservoir.unstr_discr.mat_cells_tot / layers_num)
                #id2 = int(layers_num * row_num * row_num / 2)
                #id2 = self.reservoir.unstr_discr.mat_cells_tot - layers_num * (row_num + 2)
                id1 = layers_num * ((setback + 1) * row_num - (setback + 1))
                id2 = layers_num * row_num * (row_num - (setback + 1)) + layers_num * setback
            self.reservoir.add_well("INJ001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(id1 + kk),
                                               well_index=self.reservoir.well_index)
            self.reservoir.add_well("PROD001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(id2 + kk),
                                               well_index=self.reservoir.well_index)

        # unstructured
        dist = 1.E+10
        mid = (np.min(self.reservoir.unstr_discr.mesh_data.points, axis=0) +
               np.max(self.reservoir.unstr_discr.mesh_data.points, axis=0)) / 2
        id = -1
        for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
            cur_dist = (cell.centroid[0] - mid[0]) ** 2 + (cell.centroid[1] - mid[1]) ** 2 + cell.centroid[2] ** 2
            if dist > cur_dist:
                dist = cur_dist
                id = cell_id

        # self.reservoir.add_well("PROD001", depth=0)
        # for kk in range(layers_num):
        #     self.reservoir.add_perforation(self.reservoir.wells[-1], int(id + kk),
        #                                    well_index=self.reservoir.well_index)
    def add_wells_frac(self):
        layers_num = 1
        is_center = False

        if is_center:
            # unstructured
            dist = 1.E+10
            mid = np.array([0, 0])#(np.min(self.reservoir.unstr_discr.mesh_data.points, axis=0) +
                   #np.max(self.reservoir.unstr_discr.mesh_data.points, axis=0)) / 2
            id = -1
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                cur_dist = (cell.centroid[0] - mid[0]) ** 2 + (cell.centroid[1] - mid[1]) ** 2 + cell.centroid[2] ** 2
                if dist > cur_dist:
                    dist = cur_dist
                    id = cell_id

            self.reservoir.add_well("PROD001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(id + kk),
                                               well_index=self.reservoir.well_index)
        else:
            setback = 0

            a = np.max(self.reservoir.unstr_discr.mesh_data.points[:, 0])
            coords = self.reservoir.unstr_discr.mat_cell_info_dict[0].coord_nodes_to_cell
            dx = np.max(coords[:,0]) - np.min(coords[:,0])
            dy = np.max(coords[:,1]) - np.min(coords[:,1])

            # find closest cells
            dist = 1.E+10
            pt1 = np.array([(setback + 0.5) * dx, a - (setback + 0.5) * dy, 0])
            id1 = -1
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                cur_dist = (cell.centroid[0] - pt1[0]) ** 2 + (cell.centroid[1] - pt1[1]) ** 2 + cell.centroid[2] ** 2
                if dist > cur_dist:
                    dist = cur_dist
                    id1 = cell_id
            assert (id1 >= 0)

            dist = 1.E+10
            pt2 = np.array([a - (setback + 0.5) * dx, (setback + 0.5) * dy, 0])
            id2 = -1
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                cur_dist = (cell.centroid[0] - pt2[0]) ** 2 + (cell.centroid[1] - pt2[1]) ** 2 + cell.centroid[2] ** 2
                if dist > cur_dist:
                    dist = cur_dist
                    id2 = cell_id
            assert(id2 >= 0)

            self.reservoir.add_well("PROD001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(id1 + kk),
                                               well_index=self.reservoir.well_index)

            self.reservoir.add_well("INJ001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(id2 + kk),
                                               well_index=self.reservoir.well_index)
    def add_wells_groningen(self):
        layers_num = 1
        is_center = False

        if is_center:
            # unstructured
            dist = 1.E+10
            mid = np.array([0, 0])#(np.min(self.reservoir.unstr_discr.mesh_data.points, axis=0) +
                   #np.max(self.reservoir.unstr_discr.mesh_data.points, axis=0)) / 2
            id = -1
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                cur_dist = (cell.centroid[0] - mid[0]) ** 2 + (cell.centroid[1] - mid[1]) ** 2 + cell.centroid[2] ** 2
                if dist > cur_dist:
                    dist = cur_dist
                    id = cell_id

            self.reservoir.add_well("PROD001", depth=0)
            for kk in range(layers_num):
                self.reservoir.add_perforation(self.reservoir.wells[-1], int(id + kk),
                                               well_index=self.reservoir.well_index)
        else:
            setback = 0

            # Reservoir
            throw = 0
            TopReservoir = -2830
            BottomReservoir = -3045
            CenterReservoir = (TopReservoir + BottomReservoir) / 2.
            hRes = 215
            GWC = -2995
            BottomBoundary = -3500
            h0 = BottomReservoir - BottomBoundary
            h1 = h0 - throw
            z1 = BottomBoundary + h0
            z3 = BottomBoundary + h0 + hRes

            # Calculate well_index (very primitive way....):
            rw = 0.1
            # WIx
            wi_x = 0.0
            # WIy
            wi_y = 0.0

            # find closest cells
            dist = 1.E+10
            pt1 = np.array([400, 500, (z1 + z3) / 2])
            id1 = -1
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                cur_dist = (cell.centroid[0] - pt1[0]) ** 2 + (cell.centroid[1] - pt1[1]) ** 2 + (cell.centroid[2] - pt1[2]) ** 2
                if dist > cur_dist:
                    dist = cur_dist
                    id1 = cell_id
            assert (id1 >= 0)

            dist = 1.E+10
            pt2 = np.array([1400, 500, (z1 + z3) / 2])
            id2 = -1
            for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items():
                cur_dist = (cell.centroid[0] - pt2[0]) ** 2 + (cell.centroid[1] - pt2[1]) ** 2 + (cell.centroid[2] - pt2[2]) ** 2
                if dist > cur_dist:
                    dist = cur_dist
                    id2 = cell_id
            assert(id2 >= 0)

            coords = self.reservoir.unstr_discr.mat_cell_info_dict[id1].coord_nodes_to_cell
            dx = np.max(coords[:, 0]) - np.min(coords[:, 0])
            dy = np.max(coords[:, 1]) - np.min(coords[:, 1])
            dz = np.max(coords[:, 2]) - np.min(coords[:, 2])
            hz = dz
            rp_z = 0.28 * np.sqrt((self.reservoir.permy[id1] / self.reservoir.permx[id1]) ** 0.5 * dx ** 2 +
                                  (self.reservoir.permx[id1] / self.reservoir.permy[id1]) ** 0.5 * dy ** 2) / \
                   ((self.reservoir.permx[id1] / self.reservoir.permy[id1]) ** 0.25 + (self.reservoir.permy[id1] / self.reservoir.permx[id1]) ** 0.25)
            wi_z = 2 * np.pi * np.sqrt(self.reservoir.permx[id1] * self.reservoir.permy[id1]) * hz / np.log(rp_z / rw)
            well_index1 = TC.darcy_constant * np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)

            self.reservoir.add_well("PROD001", depth=0)
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(id1),
                                               well_index=well_index1)

            coords = self.reservoir.unstr_discr.mat_cell_info_dict[id2].coord_nodes_to_cell
            dx = np.max(coords[:, 0]) - np.min(coords[:, 0])
            dy = np.max(coords[:, 1]) - np.min(coords[:, 1])
            dz = np.max(coords[:, 2]) - np.min(coords[:, 2])
            hz = dz
            rp_z = 0.28 * np.sqrt((self.reservoir.permy[id1] / self.reservoir.permx[id1]) ** 0.5 * dx ** 2 +
                                  (self.reservoir.permx[id1] / self.reservoir.permy[id1]) ** 0.5 * dy ** 2) / \
                   ((self.reservoir.permx[id1] / self.reservoir.permy[id1]) ** 0.25 + (self.reservoir.permy[id1] / self.reservoir.permx[id1]) ** 0.25)
            wi_z = 2 * np.pi * np.sqrt(self.reservoir.permx[id1] * self.reservoir.permy[id1]) * hz / np.log(rp_z / rw)
            well_index2 = TC.darcy_constant * np.sqrt(wi_x ** 2 + wi_y ** 2 + wi_z ** 2)

            self.reservoir.add_well("INJ001", depth=0)
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(id2),
                                               well_index=well_index2)

    def set_initial_conditions(self):
        #self.physics.set_uniform_initial_conditions(self.reservoir.mesh,
        #                                            uniform_pressure=self.reservoir.p_init,
        #                                            uniform_displacement=self.reservoir.u_init)
        self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh,
                                                    initial_pressure=self.reservoir.p_init,
                                                    initial_displacement=self.reservoir.u_init)
        return 0
    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # Add controls for production well:
                # Specify bhp for particular production well:
                self.physics.set_well_controls(well=w, is_control=True, is_inj=False, control_type=well_control_iface.BHP,
                                               target=self.p_init-20.)
            else:
                # For BHP control in injection well
                self.physics.set_well_controls(well=w, is_control=True, is_inj=True, control_type=well_control_iface.BHP,
                                               target=self.p_init+20., inj_composition=self.inj_comp, inj_temp=self.inj_temp)

        return 0

class model_properties(property_container):
    def __init__(self, phases_name, components_name, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, min_z=min_z)
        self.x = np.zeros((self.nph, self.nc))

    def evaluate(self, state):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        vec_state_as_np = np.asarray(state)
        pressure = vec_state_as_np[0]

        zc = np.append(vec_state_as_np[1:self.nc], 1 - np.sum(vec_state_as_np[1:self.nc]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        ph = list(range(0, self.nph))

        for j in ph:
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, 0)  # output in [kg/m3]
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate()  # output in [cp]

        return self.dens, self.mu

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        ph = list(range(0, self.nph))
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(1, 0)

        self.dens_m = [1025, 0.77]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)


        return self.sat, self.dens_m