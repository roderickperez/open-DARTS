import numpy as np
from dataclasses import dataclass, field

from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData


# region Dataclasses
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
    anisotropy: list = None
    hcap: float = 2200
    rcond: float = 181.44

# endregion


class Model(DartsModel):
    def set_physics(self, corey: dict = {}, zero: float = 1e-8, temperature: float = None, n_points: int = 10001):
        """Physical properties"""
        # Fluid components, ions and solid
        epsilon = zero/10
        components = ["H2O", "CO2"]
        phases = ["V", "Aq"]
        comp_data = CompData(components, setprops=True)

        from dartsflash.libflash import EoS
        from dartsflash.mixtures import DARTSFlash, VLAq
        vl_phases = False
        nc = len(components)

        flash_ev = VLAq(comp_data, hybrid=True)
        flash_ev.set_vl_eos("PR", root_order=[EoS.STABLE],
                            trial_comps=[InitialGuess.Yi.Wilson, components.index("CO2")],
                            stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False
                            )
        flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pr = flash_ev.eos["VL"]
        aq = flash_ev.eos["Aq"]

        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["VL", "Aq"],
                            # nf_initial_guess=[InitialGuess.Henry_VA],
                            # t_min=270., t_max=500., t_init=300., t_tol=1e-3,
                            verbose=False
                            )

        pres_in = 1.01325
        state_spec = Compositional.StateSpecification.PT if temperature is None else Compositional.StateSpecification.P
        self.physics = Compositional(components, phases, timer=self.timer,
                                     n_points=n_points, min_p=0.8 * pres_in, max_p=1.2 * pres_in,
                                     min_z=0., max_z=1., epsilon_z=epsilon, extrapolation_flag=True,
                                     min_t=0.8 * temperature, max_t=1.2 * temperature,
                                     state_spec=state_spec, cache=False)

        for i, (region, corey_params) in enumerate(corey.items()):
            diff = 8.64e-6
            property_container = PropertyContainer(components_name=components, phases_name=phases, Mw=comp_data.Mw,
                                                   eps_z=epsilon, temperature=temperature)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = flash_ev
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])
            property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(len(components)) * diff)),
                                                    ('Aq', ConstFunc(np.ones(len(components)) * diff))])

            # property_container.enthalpy_ev = dict([('V', Enthalpy(hcap=0.035)),
            #                                        ('L', Enthalpy(hcap=0.035)),
            #                                        ('Aq', Enthalpy(hcap=0.00418*18.015)),]
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])

            property_container.conductivity_ev = dict([('V', ConstFunc(0.)),
                                                        ('Aq', ConstFunc(0.)), ])

            property_container.rel_perm_ev = dict([('V', ModBrooksCorey(corey_params, 'V')),
                                                   ('Aq', ModBrooksCorey(corey_params, 'Aq'))])
            property_container.capillary_pressure_ev = ModCapillaryPressure(corey_params)

            self.physics.add_property_region(property_container, i)

            property_container.output_props = {"satV": lambda ii=i: self.physics.property_containers[ii].sat[0],
                                               "xCO2": lambda ii=i: self.physics.property_containers[ii].x[1, 1],
                                               "rhoV": lambda ii=i: self.physics.property_containers[ii].dens[0],
                                               "rho_mA": lambda ii=i: self.physics.property_containers[ii].dens_m[1]}

    def set_initial_conditions(self):
        input_depth = [0., 1.e-3]
        input_distribution = {'pressure': [self.pres_in, self.pres_in + input_depth[1] * 0.09995],
                              'H2O': 1. - self.zero}

        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh,
                                                             input_depth=input_depth, input_distribution=input_distribution)
        return

    def set_well_controls(self):
        from darts.engines import well_control_iface
        self.physics.set_well_controls(wctrl=self.reservoir.wells[0].control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                       target=0., phase_name='V', inj_composition=self.inj_comp, inj_temp=self.inj_temp)
        self.physics.set_well_controls(wctrl=self.reservoir.wells[1].control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                       target=0., phase_name='V', inj_composition=self.inj_comp, inj_temp=self.inj_temp)
        self.physics.set_well_controls(wctrl=self.reservoir.wells[2].control, is_inj=False, control_type=well_control_iface.BHP,
                                       target=self.p_prod)
        return

    def run_python_my(self, days=0, restart_dt=0, verbose=0, max_res=1e20):
        runtime = days

        mult_dt = self.params.mult_ts
        max_dt = self.params.max_ts

        # get current engine time
        t = self.physics.engine.t

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

        counter_cut = 0
        counter_add = 0
        while t < runtime:
            converged = self.run_timestep_python(dt, t, verbose=0, max_res=max_res)

            if converged:
                t += dt
                ts = ts + 1
                if verbose:
                    print("# %d \tT = %3g\tDT = %2g\tNI = %d\tLI=%d"
                          % (ts, t, dt, self.physics.engine.n_newton_last_dt, self.physics.engine.n_linear_last_dt))

                dt *= mult_dt
                if dt > max_dt:
                    dt = max_dt

                if t + dt > runtime:
                    dt = runtime - t

                if self.physics.engine.n_newton_last_dt < 5:
                    counter_add += 1

                if counter_add > 20 and max_dt < self.params.max_ts:
                    max_dt *= 2
                    counter_add = 0
                    counter_cut = 0
                    print("Increase max timestep to %g" % max_dt)

            else:
                dt /= mult_dt
                if verbose:
                    print("Cut timestep to %g" % dt)
                counter_cut += 1
                counter_add = 0
                if counter_cut > 5:
                    max_dt /= 4
                    counter_cut = 0
                    print("Cut max timestep to %g" % max_dt)

                if dt < 1e-10:
                    break
        # update current engine time
        self.physics.engine.t = runtime

        print("TS = %d(%d), NI = %d(%d), LI = %d(%d)" % (self.physics.engine.stat.n_timesteps_total, self.physics.engine.stat.n_timesteps_wasted,
                                                         self.physics.engine.stat.n_newton_total, self.physics.engine.stat.n_newton_wasted,
                                                         self.physics.engine.stat.n_linear_total, self.physics.engine.stat.n_linear_wasted))
        self.max_dt = max_dt

    def run_timestep_python(self, dt, t, verbose=0, max_res=1e20):
        max_newt = self.params.max_i_newton
        max_residual = np.zeros(max_newt + 1)
        self.physics.engine.n_linear_last_dt = 0
        well_tolerance_coefficient = 1e1
        self.timer.node['simulation'].start()
        for i in range(max_newt+1):
            self.physics.engine.assemble_linear_system(dt)
            self.physics.engine.newton_residual_last_dt = self.physics.engine.calc_newton_residual()

            max_residual[i] = self.physics.engine.newton_residual_last_dt
            counter = 0
            for j in range(i):
                if abs(max_residual[i] - max_residual[j])/max_residual[i] < 1e-3:
                    counter += 1
            if counter > 2:
                if verbose:
                    print("Stationary point detected!")
                break

            if max_residual[i] / max_residual[0] > max_res:
                if verbose:
                    print("Residual growing over the limit!")
                break

            self.physics.engine.well_residual_last_dt = self.physics.engine.calc_well_residual()
            self.physics.engine.n_newton_last_dt = i
            #  check tolerance if it converges
            if ((self.physics.engine.newton_residual_last_dt < self.params.tolerance_newton and
                 self.physics.engine.well_residual_last_dt < well_tolerance_coefficient * self.params.tolerance_newton )
                    or self.physics.engine.n_newton_last_dt == self.params.max_i_newton):
                if (i > 0):  # min_i_newton
                    break
            r_code = self.physics.engine.solve_linear_equation()
            self.timer.node["newton update"].start()
            self.physics.engine.apply_newton_update(dt)
            self.timer.node["newton update"].stop()
        # End of newton loop
        converged = self.physics.engine.post_newtonloop(dt, t)
        self.timer.node['simulation'].stop()
        return converged


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
