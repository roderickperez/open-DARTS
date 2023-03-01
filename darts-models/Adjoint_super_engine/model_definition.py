from darts.engines import *
from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params
import numpy as np
from darts.models.physics_sup.properties_basic import *
from darts.models.physics_sup.property_container import *
from darts.models.physics_sup.physics_comp_sup import Compositional

from darts.models.opt.opt_module_settings import OptModuleSettings
from darts.tools.keyword_file_tools import get_table_keyword


class Model(DartsModel, OptModuleSettings):

    def __init__(self, T, report_step=120, perm=300, poro=0.2, customize_new_operator=False, Peaceman_WI=False):
        # call base class constructor
        super().__init__()
        OptModuleSettings.__init__(self)

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.T = T
        self.report_step = report_step
        self.customize_new_operator = customize_new_operator

        # initialize global data to record the well location in vtk output file
        self.global_data = {'well location': 0}  # will be updated later in "run"



        """Reservoir construction"""
        self.nx = 20
        self.ny = 10
        self.nz = 2

        # self.nx = 3
        # self.ny = 3
        # self.nz = 1

        # reservoir geometry： for realistic case, one just needs to load the data and input it
        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=30, dy=30, dz=12, permx=perm, permy=perm,
                                         permz=perm, poro=poro, depth=2000)

        self.inj_list = [[5, 5]]
        self.prod_list = [[15, 3], [15, 8]]

        # self.inj_list = [[2, 2]]
        # self.prod_list = [[1, 2], [3, 2]]


        # well index setting
        if Peaceman_WI:
            WI = -1  # use Peaceman function; check the function "add_perforation" for more details
        else:
            WI = 200

        n_perf = self.reservoir.nz
        perf_list = list(range(n_perf))
        for i, inj in enumerate(self.inj_list):
            self.reservoir.add_well('I' + str(i + 1))
            for n in perf_list:
                self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=inj[0], j=inj[1], k=n + 1,
                                               well_radius=0.1,
                                               well_index=WI, multi_segment=False, verbose=True)

        for p, prod in enumerate(self.prod_list):
            self.reservoir.add_well('P' + str(p + 1))
            for n in perf_list:
                self.reservoir.add_perforation(self.reservoir.wells[-1], i=prod[0], j=prod[1], k=n + 1, well_radius=0.1,
                                               well_index=WI, multi_segment=False, verbose=True)


        self.zero = 1e-8
        """Physical properties"""
        # Create property containers:
        components_name = ['CO2', 'C1', 'H2O']
        self.thermal = 0
        Mw = [44.01, 16.04, 18.015]
        self.property_container = property_container(phases_name=['gas', 'oil'],
                                                     components_name=components_name,
                                                     Mw=Mw, min_z=self.zero / 10)
        self.components = self.property_container.components_name
        self.phases = self.property_container.phases_name

        """ properties correlations """
        self.property_container.flash_ev = Flash(self.components, [4, 2, 1e-1], self.zero)
        self.property_container.density_ev = dict([('gas', Density(compr=1e-3, dens0=200)),
                                                   ('oil', Density(compr=1e-5, dens0=600))])
        self.property_container.viscosity_ev = dict([('gas', ViscosityConst(0.05)),
                                                     ('oil', ViscosityConst(0.5))])
        self.property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                                    ('oil', PhaseRelPerm("oil"))])

        """ Activate physics """
        self.physics = Compositional(self.property_container, self.timer, n_points=200, min_p=1, max_p=300,
                                     min_z=self.zero/10, max_z=1-self.zero/10)

        self.inj_stream = [1.0 - 2 * self.zero, self.zero]
        self.ini_stream = [0.1, 0.2]

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 0.001
        self.params.max_ts = 1
        self.params.mult_ts = 2

        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-3
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop

        # self.params.linear_type = self.params.linear_solver_t.cpu_superlu

        self.runtime = 1000

        self.timer.node["initialization"].stop()


    def set_initial_conditions(self):
        """ initialize conditions for all scenarios"""
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, 50, self.ini_stream)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if "I" in w.name:
                w.control = self.physics.new_bhp_inj(140, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(50)

    def properties(self, state):
        (sat, x, rho, rho_m, mu, kr, ph) = self.property_container.evaluate(state)
        return sat[0]


    def export_pro_vtk(self, file_name='Saturation', global_cell_data={}):
        Xn = np.array(self.physics.engine.X, copy=False)
        P = Xn[0:self.reservoir.nb * 2:2]
        z1 = Xn[1:self.reservoir.nb * 2:2]

        so = np.zeros(len(P))
        sw = np.zeros(len(P))

        for i in range(len(P)):
            values = value_vector([0] * self.physics.n_ops)
            state = value_vector((P[i], z1[i]))
            self.physics.property_itor.evaluate(state, values)
            sw[i] = values[0]
            so[i] = 1 - sw[i]

        self.export_vtk(file_name, local_cell_data={'OilSat': so, 'WatSat': sw}, global_cell_data=global_cell_data)

    def set_op_list(self):
        if self.customize_new_operator:
            water_component_etor = customized_etor_specific_component()
            water_component_itor = self.physics.create_interpolator(water_component_etor, self.physics.n_vars, 1,
                                                                    self.physics.n_axes_points, self.physics.n_axes_min,
                                                                    self.physics.n_axes_max,
                                                                    platform='cpu', algorithm='multilinear',
                                                                    mode='adaptive', precision='d')
            self.physics.create_itor_timers(water_component_itor, "customized component interpolation")
            self.physics.engine.customize_operator = self.customize_new_operator

            self.op_list = [self.physics.acc_flux_itor[0], water_component_itor]

            # specify the index of blocks of customized operator
            idx_in_op_list = 1
            op_num_new = np.array(self.reservoir.mesh.op_num, copy=True)
            op_num_new[:] = idx_in_op_list  # set the second interpolator (i.e. "water_component_itor") from "self.op_list" to all blocks
            self.physics.engine.idx_customized_operator = idx_in_op_list
            self.physics.engine.customize_op_num = index_vector(op_num_new)
        else:
            # self.op_list = [self.physics.acc_flux_itor]

            self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
            n_res = self.reservoir.mesh.n_res_blocks
            self.op_num[n_res:] = 1
            self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]


    def run(self, export_to_vtk=False, file_name='data'):
        import random
        # use seed to generate the same random values every run
        random.seed(3)
        if export_to_vtk:

            well_loc = np.zeros(self.reservoir.n)
            for inj in self.inj_list:
                well_loc[(inj[1] - 1) * self.reservoir.nx + inj[0] - 1] = -1

            for prod in self.prod_list:
                well_loc[(prod[1] - 1) * self.reservoir.nx + prod[0] - 1] = 1

            self.global_data = {'well location': well_loc}

            self.export_pro_vtk(global_cell_data=self.global_data, file_name=file_name)

        # now we start to run for the time report--------------------------------------------------------------
        time_step = self.report_step
        even_end = int(self.T / time_step) * time_step
        time_step_arr = np.ones(int(self.T / time_step)) * time_step
        if self.T - even_end > 0:
            time_step_arr = np.append(time_step_arr, self.T - even_end)

        for ts in time_step_arr:
            for i, w in enumerate(self.reservoir.wells):
                if "I" in w.name:
                    w.control = self.physics.new_bhp_inj(140, self.inj_stream)
                else:
                    w.control = self.physics.new_bhp_prod(50)

            self.physics.engine.run(ts)
            self.physics.engine.report()
            if export_to_vtk:
                self.export_pro_vtk(global_cell_data=self.global_data, file_name=file_name)



class model_properties(property_container):
    def __init__(self, phases_name, components_name, pvt, min_z=1e-11):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z)
        self.x = np.zeros((self.nph, self.nc))
        self.pvt = pvt
        self.surf_dens = get_table_keyword(self.pvt, 'DENSITY')[0]
        self.surf_oil_dens = self.surf_dens[0]
        self.surf_wat_dens = self.surf_dens[1]

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

        zc = np.append(vec_state_as_np[1:], 1 - np.sum(vec_state_as_np[1:]))

        self.clean_arrays()
        # two-phase flash - assume water phase is always present and water component last
        for i in range(self.nph):
            self.x[i, i] = 1

        ph = [0, 1]

        for j in ph:
            M = 0
            # molar weight of mixture
            for i in range(self.nc):
                M += self.Mw[i] * self.x[j][i]
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(state)  # output in [kg/m3]
            self.dens_m[j] = self.dens[j] / M
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(state)  # output in [cp]

        self.nu = zc
        self.compute_saturation(ph)

        for j in ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[0])
            self.pc[j] = 0

        return self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, ph

    def evaluate_at_cond(self, pressure, zc):

        self.sat[:] = 0

        state = value_vector([1, 0])

        ph = [0, 1]
        for j in ph:
            self.dens_m[j] = self.density_ev[self.phases_name[j]].evaluate(state)

        self.dens_m = [self.surf_wat_dens, self.surf_oil_dens]  # to match DO based on PVT

        self.nu = zc
        self.compute_saturation(ph)

        return self.sat, self.dens_m


class customized_etor_specific_component(operator_set_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """

        # temp = self.temperature.evaluate(state)

        # values[0] = state[0]  # pressure
        values[0] = 1 - state[1]  # oil
        return 0
