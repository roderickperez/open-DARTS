from darts.models.darts_model import DartsModel
from darts.engines import value_vector, sim_params, mech_operators, rsf_props, sd_props, friction, contact_state, state_law, contact_solver, critical_stress, normal_condition
from reservoir import UnstructReservoir
import numpy as np

from physics.physics_comp_sup import Poroelasticity
from physics.property_container import *
from physics.properties_basic import *

class Model(DartsModel):
    def __init__(self, n_points=64):
        super().__init__()
        self.timer.node["initialization"].start()
        self.physics_type = 'poromechanics'

        self.cell_property = ['u_x', 'u_y', 'u_z', 'p']
        self.reservoir = UnstructReservoir(timer=self.timer)
        self.set_physics()

        self.params.first_ts = 1e-6  # Size of the first time-step [days]
        self.params.mult_ts = 1.5  # Time-step multiplier if newton is converged (i.e. dt_new = dt_old * mult_ts)
        self.params.max_ts = 0.1  # Max size of the time-step [days]
        self.params.tolerance_newton = 1e-7 # Tolerance of newton residual norm ||residual||<tol_newt
        self.params.tolerance_linear = 1e-10 # Tolerance for linear solver ||Ax - b||<tol_linslv
        self.params.newton_type = sim_params.newton_local_chop  # Type of newton method (related to chopping strategy?)
        self.params.newton_params = value_vector([0.2])  # Probably chop-criteria(?)
        self.params.linear_type = sim_params.cpu_superlu#sim_params.cpu_gmres_ilu0#sim_params.cpu_gmres_fs_cpr###sim_params.cpu_superlu
        self.runtime = 2  # Total simulations time [days], this parameters is overwritten in main.py!
        self.params.max_i_newton = 25
        self.params.max_i_linear = 500

        self.timer.node["initialization"].stop()
    def set_physics(self):
        self.zero = 1e-13
        """Physical properties"""
        self.property_container = model_properties(phases_name=['wat'], components_name=['w'],
                                                   min_z=self.zero / 10)

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

        self.reservoir.P_VAR = self.physics.engine.P_VAR
    def init(self):
        DartsModel.init(self)
        self.reservoir.mech_operators = mech_operators()
        self.reservoir.mech_operators.init(self.reservoir.mesh, self.reservoir.pm,
                                 self.physics.engine.P_VAR, self.physics.engine.Z_VAR,self.physics.engine.U_VAR,
                                 self.physics.engine.N_VARS, self.physics.engine.N_OPS, self.physics.engine.NC,
                                 self.physics.engine.ACC_OP, self.physics.engine.FLUX_OP, self.physics.engine.GRAV_OP)
        self.reservoir.mech_operators.prepare()
        self.init_contacts()
    def reinit(self, output_directory):
        self.reservoir.turn_off_equilibrium()
        self.reservoir.write_to_vtk(output_directory, 0, self.physics)
        self.reservoir.eps_vol_ref = np.array(self.reservoir.mesh.ref_eps_vol, copy=False)
        self.reservoir.eps_vol_ref[:] = self.reservoir.mech_operators.eps_vol[:]
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
    def set_initial_conditions(self):
        if 0:
            self.physics.set_initial_conditions_from_array(self.reservoir.mesh,
                                                           input_distribution={'pressure': self.reservoir.p_init},
                                                           uniform_displacement=self.reservoir.u_init)
        else:
            input_depth = [0., 1.]
            gradient = {'pressure': 100.}
            input_distribution = {'pressure': [self.reservoir.p_init,
                                               self.reservoir.p_init + input_depth[1] * gradient['pressure'] / 1000.]}
            self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                                 input_depth=input_depth,
                                                                 input_distribution=input_distribution,
                                                                 displacement_input=self.reservoir.u_init)

    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        return 0
    def setup_contact_friction(self, contact_algorithm):
        if hasattr(self.reservoir, 'contacts'):
            for contact in self.physics.engine.contacts:
                friction_model = friction.STATIC#friction.STATIC#friction.SLIP_DEPENDENT#friction.RSF

                # allow to slip
                contact.set_state(contact_state.SLIP)
                # static friction coefficients
                contact.mu0 = value_vector(0.0 * np.ones(len(contact.cell_ids)))
                contact.mu = contact.mu0

                # setup friction model
                contact.friction_model = friction_model
                # setup friction criterion
                contact.friction_criterion = critical_stress.TERZAGHI
                # setup normal condition
                contact.normal_condition = normal_condition.ZERO_GAP_CHANGE

                # Slip dependent model
                if (friction_model == friction.SLIP_DEPENDENT):
                    prop = sd_props()
                    prop.crit_distance = 0.02
                    prop.mu_dyn = 0.4
                    contact.sd_props = prop
                # RSF model
                if (friction_model == friction.RSF):
                    prop = rsf_props()
                    theta = 10.0 / 86400.0 * np.ones(len(contact.cell_ids))
                    prop.theta_n = value_vector(theta)
                    prop.theta = value_vector(theta)
                    prop.a = 0.01#0.0008#0.0078
                    prop.b = 0.02
                    prop.crit_distance = 0.01 * 1.E-3
                    prop.ref_velocity = 0.001 * 1.E-6 * 86400
                    prop.law = state_law.MIXED
                    contact.rsf = prop

                # Damping term
                for i in range(len(contact.eta)):
                    contact.eta[i] *= 1.0

                # init local solver in the case of local iterations
                if contact_algorithm == contact_solver.LOCAL_ITERATIONS:
                    contact.init_local_iterations()

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