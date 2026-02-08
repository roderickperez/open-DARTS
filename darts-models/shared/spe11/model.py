import numpy as np
from dataclasses import dataclass, field

from darts.models.darts_model import DartsModel
from darts.engines import value_vector, index_vector, sim_params, conn_mesh

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer

from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, RockEnergyEvaluator, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.flash import ConstantK

from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData, EnthalpyIdeal
from dartsflash.eos_properties import EoSDensity, EoSEnthalpy


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
        components = ["H2O", "CO2"]
        phases = ["V", "Aq"]
        comp_data = CompData(components, setprops=True)

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                               AQEoS.CompType.solute: AQEoS.Ziabakhsh2012
                               })

        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3

        if temperature is None:  # if None, then thermal=True
            thermal = True
        else:
            thermal = False

        pres_in = 1.01325 # 1.01325 for 11a, 210 for 11b,11c (pressure at depth of well 1 will be 300 bar) unsure if this line does anything
        min_t = 273.15 if temperature is None else None
        max_t = 373.15 if temperature is None else None # 373.15 for a, higher for b/c
        self.physics = Compositional(components, phases, timer=self.timer,
                                     n_points=n_points, min_p=0.8, max_p=1.3,
                                     min_z=zero / 10, max_z=1 - zero / 10, min_t=min_t, max_t=max_t,
                                     thermal=thermal, cache=False)

        for i, (region, corey_params) in enumerate(corey.items()):
            diff = 8.64e-6
            property_container = PropertyContainer(components_name=components, phases_name=phases, Mw=comp_data.Mw,
                                                   min_z=zero / 10, diff_coef=diff, temperature=temperature)

            # property_container.flash_ev = ConstantK(nc=2, ki=[0.001, 100])
            property_container.flash_ev = NegativeFlash(flash_params, ["PR", "AQ"], [InitialGuess.Henry_VA])
            property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                                  ('Aq', Garcia2001(components)), ])
            property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                    ('Aq', Islam2012(components)), ])

            if thermal:
                # property_container.enthalpy_ev = dict([('V', Enthalpy(hcap=0.035)),
                #                                        ('L', Enthalpy(hcap=0.035)),
                #                                        ('Aq', Enthalpy(hcap=0.00418*18.015)),]
                h_ideal = EnthalpyIdeal(components)
                property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr, h_ideal=h_ideal)),
                                                       ('Aq', EoSEnthalpy(eos=aq, h_ideal=h_ideal)), ])

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



    def set_well_controls(self):
        self.reservoir.wells[0].control = self.physics.new_rate_inj(0, self.inj_stream, 0)
        self.reservoir.wells[1].control = self.physics.new_rate_inj(0, self.inj_stream, 0)
        self.reservoir.wells[2].control = self.physics.new_bhp_prod(self.p_prod) # turn on for 11a and off for 11b
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

from darts.physics.super.physics import Compositional
class VolumeRateCompositional(Compositional):
    def set_operators(self):
        super().set_operators()
        self.rate_operators = VolumeRateOperators(self.property_containers[self.regions[0]])


from darts.physics.super.operator_evaluator import RateOperators
class VolumeRateOperators(RateOperators):
    def evaluate(self, state: value_vector, values: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        for i in range(self.n_ops):
            values[i] = 0

        ph, sat, x, rho, rho_m, mu, kr, pc, mass_source = self.property.evaluate(state)

        flux = np.zeros(self.nc)
        # step-1
        for j in ph:
            for i in range(self.nc):
                flux[i] += rho_m[j] * kr[j] * x[j][i] / mu[j]
        # step-2
        flux_sum = np.sum(flux)

        # (sat_sc, rho_m_sc) = self.property.evaluate_at_cond(1, self.flux/flux_sum)
        sat_sc = sat
        rho_m_sc = rho_m

        # step-3
        total_density = np.sum(sat_sc * rho_m_sc)
        # step-4
        for j in ph:
            # values[j] = rho_m[j] * kr[j] / mu[j]
            values[j] = sat_sc[j] * flux_sum / total_density

        # print(state, values)
        return 0