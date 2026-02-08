from reservoir import UnstructReservoir
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh
import numpy as np
from dataclasses import dataclass, field

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.base.operators_base import PropertyOperators

from darts.physics.properties.basic import ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
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
    def __init__(self,  discr_type, mesh_file, platform='cpu', pres_in=1.01325, temp_in=293., thermal=False):
        # call base class constructor
        super().__init__()

        self.pres_in = pres_in

        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.discr_type = discr_type
        self.set_reservoir(mesh_file)

        if thermal:
            temp_init = None
        else:
            temp_init = temp_in
        self.set_physics(temperature=temp_init, temp_in=temp_in, n_points=2001, platform=platform)

        """Configuration"""
        self.bound_cond = 'wells'
        self.physics_type = 'compositional'

        self.discr_type = discr_type
        if discr_type == 'mpfa':
            self.reservoir.P_VAR = 0

        # max_ts - 10 minutes reporting time
        self.set_sim_params(first_ts=1e-8, mult_ts=2, max_ts=0.00694, tol_newton=1e-3, tol_linear=1e-4, it_newton=6, it_linear=100)
        self.params.newton_type = sim_params.newton_local_chop

        # self.add_wells_unstructured_cpp()

        self.timer.node["initialization"].stop()

    def init(self, platform='cpu'):
        DartsModel.init(self, discr_type=self.discr_type, platform=platform)

    def add_wells_unstructured_cpp(self):
        layers_num = 1
        is_center = False
        Lx = max([pt.values[0] for pt in self.reservoir.discr_mesh.nodes])
        Ly = max([pt.values[1] for pt in self.reservoir.discr_mesh.nodes])
        Lz = max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])
        dx = np.sqrt(np.mean(self.reservoir.volume_all_cells) / \
                ( max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes]) - min([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])))

        n_cells = self.reservoir.discr_mesh.n_cells
        pt_x = np.array([c.values[0] for c in self.reservoir.discr_mesh.centroids])[:n_cells]
        pt_y = np.array([c.values[1] for c in self.reservoir.discr_mesh.centroids])[:n_cells]
        pt_z = np.array([c.values[2] for c in self.reservoir.discr_mesh.centroids])[:n_cells]

        x0 = 800.0
        y0 = -800.0
        id1 = ((pt_x - x0) ** 2 + (pt_y - y0) ** 2 + pt_z ** 2).argmin()
        self.reservoir.add_well("PROD001", depth=0)
        for kk in range(layers_num):
            self.reservoir.add_perforation(self.reservoir.wells[-1], int(id1 + kk), well_index=self.reservoir.well_index)

    def set_physics(self, temperature=None, temp_in=None, n_points=2001, platform='cpu'):
        self.zero = 1e-8

        """Physical properties"""
        # Fluid components, ions and solid
        components = ["CO2", "H2O"]
        phases = ["V", "Aq"]
        nc = len(components)
        comp_data = CompData(components, setprops=True)

        """ Initialize flash """
        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                               AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                               AQEoS.CompType.ion: AQEoS.Jager2003
                               })

        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["PR", "AQ"]

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3
        # flash_params.split_tol = 1e-14

        flash_ev = NegativeFlash(flash_params, flash_params.eos_order, [InitialGuess.Henry_AV])

        if temperature is None:  # if None, then thermal=True
            thermal = True
            self.ini_temp = temp_in
        else:
            thermal = False
            self.ini_temp = None

        self.ini_comp = [self.zero]  # z1
        self.inj_comp = [1 - self.zero]
        self.inj_temp = self.ini_temp
        self.inj_rate = 10 * 24 * 60 / 1E6  # ml/min to m3/day
        self.p_atm = 1.01325

        state_spec = Compositional.StateSpecification.PT if thermal else Compositional.StateSpecification.P
        self.physics = CustomPhysics(components=components, phases=phases, timer=self.timer,
                                     n_points=n_points, min_p=0.1, max_p=300., min_z=self.zero/10, max_z=1-self.zero/10,
                                     state_spec=state_spec, min_t=0.8 * temperature, max_t=1.2 * temperature,
                                     discr_type=self.reservoir.discr_type, cache=False)

        corey = {
            0: Corey(nw=2.0, ng=1.5, swc=0.11, sgc=0.06, krwe=0.80, krge=0.85, labda=2., p_entry=0.),
            # 1: Corey(nw=2.0, ng=1.5, swc=0.12, sgc=0.08, krwe=0.93, krge=0.95, labda=2., p_entry=1e-3),
            # 2: Corey(nw=2.5, ng=2.0, swc=0.14, sgc=0.10, krwe=0.93, krge=0.95, labda=2., p_entry=3e-3),
            3: Corey(nw=2.5, ng=2.0, swc=0.32, sgc=0.14, krwe=0.71, krge=0.75, labda=2., p_entry=0.1)
        }
        self.property_regions = [key for key in corey.keys()]
        for region in self.property_regions:
            property_container = PropertyContainer(components_name=components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=self.zero / 10, temperature=temperature) #diff_coef=8.64e-6,

            property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])

            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])

            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])

            property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                       ('Aq', ConstFunc(0.)), ])

            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey[region], 'V')),
                                                   ('Aq', ModBrooksCorey(corey[region], 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey[region])

            self.physics.add_property_region(property_container, region)

        #output_props = PropertyEvaluator(self.physics.vars, property_container)
        self.physics.add_property_region(property_container)
        self.physics.init_physics()

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)
        rcond.fill(181.44)

        return

    def set_reservoir(self, mesh_file):
        self.reservoir = UnstructReservoir(self.discr_type, mesh_file)

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        input_depth = [self.reservoir.depth_all_cells.max(), self.reservoir.depth_all_cells.max() - 10.]
        pressure_gradient = -100.
        input_distribution = {'pressure': [self.pres_in, self.pres_in + (input_depth[1] - input_depth[0]) * pressure_gradient / 1000.],
                              'CO2': [self.ini_comp[0], self.ini_comp[0]]}
        input_distribution.update({'temperature': self.ini_temp} if self.ini_temp is not None else {})
        self.physics.set_initial_conditions_from_depth_table(self.reservoir.mesh,
                                                             input_depth=input_depth,
                                                             input_distribution=input_distribution)

    def set_boundary_conditions(self):
        """
        Class method called in the init() class method of parents class
        :return:
        """
        # Takes care of well controls, argument of the function is (in case of bhp) the bhp pressure and (in case of
        # rate) water/oil rate:
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            # if i == 0:
            #     # For BHP control in injection well we usually specify pressure and composition (upstream) but here
            #     # the method is wrapped such  that we only need to specify bhp pressure (see lambda for more info)
            #     #w.control = self.physics.new_bhp_inj(self.p_atm + 0.1)
            #     # w.control = self.physics.new_rate_inj(30.0, self.inj_stream, 0)
            #     w.control = self.physics.new_bhp_inj(2.5, self.inj_stream)
            # else:
                # Add controls for production well:
                # Specify bhp for particular production well:
                # w.control = self.physics.new_bhp_prod(325)
            self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                           is_inj=False, target=50.)
        return 0

    def set_op_list(self):
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        n_res = self.reservoir.mesh.n_res_blocks

        if 1:
            if self.discr_type == 'mpfa':
                Lz = max([pt.values[2] for pt in self.reservoir.discr_mesh.nodes])
                centroids = np.array(self.reservoir.discr_mesh.centroids)[:self.op_num.size]
                cell_ids = [cell_id for cell_id, c in enumerate(centroids) if c.values[2] > Lz / 2]
            else:
                Lz = np.max(self.reservoir.unstr_discr.mesh_data.points[:, 2])
                cell_ids = [cell_id for cell_id, cell in self.reservoir.unstr_discr.mat_cell_info_dict.items() if cell.centroid[2] > Lz / 2]
            self.op_num[cell_ids] = 1
            self.op_num[n_res:] = len(self.property_regions)
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

'''
class PropertyEvaluator():
    def __init__(self, variables, property_container):
        super().__init__(variables, property_container)  # Initialize base-class

        self.props = ['sat0', 'xCO2', 'rho_v', 'rho_m_Aq']
        self.n_props = len(self.props)

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        values[0] = sat[0]
        values[1] = x[1, 0]
        values[2] = rho[0]
        values[3] = rho_m[1]

        return 0
'''

class CustomPhysics(Compositional):
    def __init__(self, components, phases, timer,
                 n_points, min_p, max_p, min_z, max_z, min_t=None, max_t=None,
                 state_spec=Compositional.StateSpecification.P, discr_type='tpfa', cache=False):
        super().__init__(components=components, phases=phases, timer=timer,
                         n_points=n_points, min_p=min_p, max_p=max_p, min_z=min_z, max_z=max_z,
                         min_t=min_t, max_t=max_t, state_spec=state_spec, cache=cache)#discr_type=discr_type,

    def set_initial_conditions_from_depth_table(self, mesh, input_depth, input_distribution):
        """ Uniform Initial conditions """
        # set initial pressure
        super().set_initial_conditions_from_depth_table(mesh, input_depth=input_depth, input_distribution=input_distribution)

        # Set bounds
        pressure = np.array(mesh.initial_state[::self.nc], copy=False)
        pz_bounds = np.array(mesh.pz_bounds, copy=False)
        pz_bounds[::self.nc] = np.mean(pressure)

        for c in range(self.nc - 1):
            uniform_composition = np.array(mesh.initial_state[1+c::self.nc], copy=False)
            pz_bounds[1+c::self.nc] = np.mean(uniform_composition)

    # def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
    #     assert isinstance(mesh, conn_mesh)
    #
    #     # Class methods which can create constant pressure and composition boundary condition:
    #     pressure = np.array(mesh.pressure, copy=False)
    #     pressure.fill(uniform_pressure)
    #
    #     mesh.composition.resize(mesh.n_blocks * (self.nc - 1))
    #     composition = np.array(mesh.composition, copy=False)
    #     for c in range(self.nc - 1):
    #         composition[c::(self.nc - 1)] = uniform_composition[c]


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

        pc = self.p_entry
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