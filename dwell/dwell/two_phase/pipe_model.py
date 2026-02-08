import math
import numpy as np
import pandas as pd

from dwell.two_phase.define_pipe_geometry import PipeGeometry
from dwell.two_phase.define_fluid_model import FluidModel, PropertyEvaluator
from dwell.utilities.units import *

class PipeModel:
    g = 9.80665 * meter() / second()**2  # Gravitational acceleration
    Cku = 142
    Cw = 0.008

    def __init__(self, pipe_geometry: PipeGeometry, fluid_model: FluidModel, property_evaluator: PropertyEvaluator,
                 isothermal, system_temperature: float = None, Cmax: float = 1.2, Fv: float = 1,
                 eps_p: float = 10, eps_temp: float = 0.1, eps_z: float = 0.001, verbose: bool = False):
        """
        :param pipe_geometry: Pipe geometry object
        :type pipe_geometry: PipeGeometry
        :param fluid_model: Fluid model object
        :type fluid_model: FluidModel
        :param property_evaluator: The object that evaluates phase properties during the simulation
        :type property_evaluator: PropertyEvaluator
        :param isothermal: Whether or not the system is isothermal or non-isothermal
        :type isothermal: bool
        :param system_temperature: If the system is isothermal, the system temperature must be entered here.
        :type system_temperature: float
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
        could have a value between 1.0 and 1.5. It is set to:
        --> 1.2 in ECLIPSE according to Shi et al. paper (Drift-Flux Modeling of Two-Phase Flow in Wellbores)
        --> 1 in a wellbore simulator in Tonken et al. paper (A transient geothermal wellbore simulator)
        :type Cmax: float
        :param Fv: A multiplier on the flooding velocity fraction, set to be 1 by default, and its value can be tuned
        to fit the observations.
        :type Fv: float
        :param eps_p: A very small number used for numerically differentiating residual equations with respect to pressure.
        :type eps_p: float
        :param eps_temp: A very small number used for numerically differentiating residual equations with respect to temperature.
        :type eps_temp: float
        :param eps_z: A very small number used for numerically differentiating residual equations with respect to composition
        :type eps_z: float
        :param verbose: Whether to display extra info about PipeModel
        :type verbose: boolean
        """
        self.pipe_geometry = pipe_geometry
        self.fluid_model = fluid_model
        assert self.pipe_geometry.pipe_name == self.fluid_model.pipe_name, \
            "The names of the pipes for the geometry and fluid model are not identical!"

        self.property_evaluator = property_evaluator
        self.isothermal = isothermal
        if isothermal:
            assert system_temperature is not None, "If model is isothermal, system_temperature must be specified!"
        elif not isothermal:
            assert system_temperature is None, "If model is non-isothermal, system_temperature must not be specified!"
        self.system_temperature = system_temperature

        self.Cmax = Cmax
        self.B = 2 / Cmax - 1.0667
        self.Fv = Fv

        # I did not see anywhere to tell if I can use linear interp and extra here or not.
        if Cmax == 1:
            a1 = 0.06
            a2 = 0.21
            m0 = 1.85
            n1 = 0.21
            n2 = 0.95
        elif 1 < Cmax < 1.2:
            # Linear interpolation
            a1 = 0.06
            a2 = np.interp(Cmax, [1, 1.2], [0.21, 0.12])
            m0 = np.interp(Cmax, [1, 1.2], [1.85, 1.27])
            n1 = np.interp(Cmax, [1, 1.2], [0.21, 0.24])
            n2 = np.interp(Cmax, [1, 1.2], [0.95, 1.08])
        elif Cmax == 1.2:
            a1 = 0.06
            a2 = 0.12
            m0 = 1.27
            n1 = 0.24
            n2 = 1.08
        elif 1.2 < Cmax <= 1.5:
            # Linear extrapolation
            a1 = 0.06
            a2 = np.interp(Cmax, [1, 1.2], [0.21, 0.12])
            m0 = np.interp(Cmax, [1, 1.2], [1.85, 1.27])
            n1 = np.interp(Cmax, [1, 1.2], [0.21, 0.24])
            n2 = np.interp(Cmax, [1, 1.2], [0.95, 1.08])
        else:
            raise ValueError("Cmax value is out of the allowed range [1 to 1.5]")

        self.m = m0 * ((math.cos(pipe_geometry.inclination_angle_radian)) ** n1) * (
                    1 + math.sin(pipe_geometry.inclination_angle_radian)) ** n2

        self.a1 = a1
        self.a2 = a2

        self.g_cos_theta = self.g * math.cos(pipe_geometry.inclination_angle_radian)

        # Epsilon values for numerical differentiation with respect to pressure, temperature, and overall composition
        self.eps_p = eps_p
        self.eps_temp = eps_temp
        self.eps_z = eps_z

        if verbose:
            print("** Model of the pipe \"%s\" is created!" % self.pipe_geometry.pipe_name)

    def init_pipe_model(self):
        pass

    def calc_mass_energy_residuals(self, vars0, vars, dt, bc, simulation_timer, iter_counter, total_iter_counter, flag,
                                   phase_props_df: pd.DataFrame = None):
        source_sink = bc['source/sink']
        nodes = bc['nodes']
        extra_nodes = bc['extra_nodes']

        pipe_geom = self.pipe_geometry
        V = pipe_geom.segments_volumes
        A = pipe_geom.pipe_internal_A
        num_segments = pipe_geom.num_segments

        p0 = vars0[0:num_segments:1]
        p = vars[0:num_segments:1]

        zc0 = vars0[num_segments:num_segments * self.fluid_model.num_components]
        zc = vars[num_segments:num_segments * self.fluid_model.num_components]

        if self.isothermal:
            # When the system is isothermal, a single system temperature is used in the calculations.
            T0 = self.system_temperature
            T = T0
        elif not self.isothermal:
            T0 = vars0[num_segments * self.fluid_model.num_components:num_segments * self.fluid_model.num_components + num_segments]
            T = vars[num_segments * self.fluid_model.num_components:num_segments * self.fluid_model.num_components + num_segments]

        "================================= Calculate phase props of previous time step ================================"
        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            sG0 = np.zeros(num_segments)
            rhoG0 = np.zeros(num_segments)
            rhoL0 = np.zeros(num_segments)
            miuG0 = np.zeros(num_segments)
            miuL0 = np.zeros(num_segments)
            xG_mass0 = np.zeros((num_segments, self.fluid_model.num_components))
            xL_mass0 = np.zeros((num_segments, self.fluid_model.num_components))

            if not self.isothermal:
                Ug0 = np.zeros(num_segments)
                Ul0 = np.zeros(num_segments)

            for i in range(num_segments):
                zc_full_segment_i0 = []
                for c in range(self.fluid_model.num_components - 1):
                    zc_full_segment_i0.append(zc0[c * num_segments + i])
                last_component_mole_fraction = 1 - sum(zc_full_segment_i0)
                zc_full_segment_i0.append(last_component_mole_fraction)

                if self.isothermal:
                    ph0, s0, x_mass0, rho0, miu0 = self.property_evaluator.evaluate(p0[i], T0, zc_full_segment_i0, self.isothermal)
                elif not self.isothermal:
                    ph0, s0, x_mass0, rho0, miu0, h0, U0 = self.property_evaluator.evaluate(p0[i], T0[i], zc_full_segment_i0, self.isothermal)

                    # Enthalpies of the previous time step are used for calculating internal energies of the previous
                    # time step. They're not used directly in the equations.
                    Ug0[i], Ul0[i] = U0[0], U0[1]

                sG0[i] = s0[0]

                xG_mass0[i, :], xL_mass0[i, :] = x_mass0[0, :], x_mass0[1, :]

                rhoG0[i], rhoL0[i] = rho0[0], rho0[1]
                miuG0[i], miuL0[i] = miu0[0], miu0[1]

            if self.isothermal:
                self.iter_phases_props0 = [xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, miuG0, miuL0]
            elif not self.isothermal:
                self.iter_phases_props0 = [xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, miuG0, miuL0, Ug0, Ul0]

        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            if self.isothermal:
                self.iter_phases_props0 = self.iter_phases_props
            elif not self.isothermal:
                # Just from the first column to 9th column of this must be put in iter_phases_props0
                self.iter_phases_props0 = self.iter_phases_props[0:9]

        if self.isothermal:
            xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, miuG0, miuL0 = self.iter_phases_props0
        elif not self.isothermal:
            xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, miuG0, miuL0, Ug0, Ul0 = self.iter_phases_props0

        "================================= Calculate phase props of current time step ================================="
        sG = np.zeros(num_segments)
        rhoG = np.zeros(num_segments)
        rhoL = np.zeros(num_segments)
        miuG = np.zeros(num_segments)
        miuL = np.zeros(num_segments)
        xG_mass = np.zeros((num_segments, self.fluid_model.num_components))
        xL_mass = np.zeros((num_segments, self.fluid_model.num_components))

        if not self.isothermal:
            hG = np.zeros(num_segments)
            hL = np.zeros(num_segments)
            Ug = np.zeros(num_segments)
            Ul = np.zeros(num_segments)

        for i in range(num_segments):
            zc_full_segment_i = []
            for c in range(self.fluid_model.num_components - 1):
                zc_full_segment_i.append(zc[c * num_segments + i])
            last_component_mole_fraction = 1 - sum(zc_full_segment_i)
            zc_full_segment_i.append(last_component_mole_fraction)

            if self.isothermal:
                ph, s, x_mass, rho, miu = self.property_evaluator.evaluate(p[i], T, zc_full_segment_i, self.isothermal)
            elif not self.isothermal:
                ph, s, x_mass, rho, miu, h, U = self.property_evaluator.evaluate(p[i], T[i], zc_full_segment_i, self.isothermal)

                Ug[i], Ul[i] = U[0], U[1]
                hG[i], hL[i] = h[0], h[1]

            sG[i] = s[0]

            xG_mass[i, :], xL_mass[i, :] = x_mass[0, :], x_mass[1, :]

            rhoG[i], rhoL[i] = rho[0], rho[1]
            miuG[i], miuL[i] = miu[0], miu[1]

        if self.isothermal:
            self.iter_phases_props = [xG_mass, xL_mass, sG, rhoG, rhoL, miuG, miuL]

        elif not self.isothermal:
            self.iter_phases_props = [xG_mass, xL_mass, sG, rhoG, rhoL, miuG, miuL, Ug, Ul, hG, hL]

        "========================================= Calculate phase velocities ========================================="
        # self.velocities0 will be used in the method self.calc_phase_velocities and also for energy conservation eq
        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            # Initial velocities in the wellbore are zero
            [rhoM0_vM0, vM0, vG0, vL0] = [np.array([0]), np.array([0]), np.array([0]), np.array([0])]
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])
        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            [rhoM0_vM0, vM0, vG0, vL0] = [self.rhoM_vM, self.vM, self.vG, self.vL]
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])

        [_, _, vG0, vL0] = self.velocities0

        self.calc_phase_velocities(p, dt, source_sink, simulation_timer, iter_counter, total_iter_counter, flag)

        # Store phase velocities if flag is -1
        if flag == -1:
            # Pad vG and vL to match the length of the other phase props
            vG_padded = self.vG.tolist() + [np.nan]  # Convert to list and append
            vL_padded = self.vL.tolist() + [np.nan]

            if self.isothermal:
                phase_props_df = pd.concat([phase_props_df, pd.DataFrame(list(zip(*self.iter_phases_props, vG_padded, vL_padded)),
                                                                         columns=["xG_mass", "xL_mass", "sG",
                                                                                  "rhoG", "rhoL", "miuG", "miuL",
                                                                                  "vG", "vL"])])
            elif not self.isothermal:
                phase_props_df = pd.concat([phase_props_df, pd.DataFrame(list(zip(*self.iter_phases_props, vG_padded, vL_padded)),
                                                                         columns=["xG_mass", "xL_mass", "sG",
                                                                                  "rhoG", "rhoL", "miuG", "miuL",
                                                                                  "Ug", "Ul", "hG", "hL", "vG", "vL"])])

        """----------------------------------------------------------------------------------------------------------"""
        """-------------------------------------- Mass conservation equation ----------------------------------------"""
        """----------------------------------------------------------------------------------------------------------"""
        residuals = np.zeros(len(vars))

        for i in range(self.fluid_model.num_components):
            """ Accumulation term """
            residuals[i*num_segments:(i+1)*num_segments] = V * (
                                                    (sG * rhoG * xG_mass[:,i] + (1 - sG) * rhoL * xL_mass[:,i])
                                                    - (sG0 * rhoG0 * xG_mass0[:,i] + (1 - sG0) * rhoL0 * xL_mass0[:,i]))

            """ Flux term """
            # Pre-allocate upwinded props
            xG_mass_up = np.zeros((pipe_geom.num_interfaces, self.fluid_model.num_components))
            sG_up = np.zeros(pipe_geom.num_interfaces)
            rhoG_up = np.zeros(pipe_geom.num_interfaces)
            xL_mass_up = np.zeros((pipe_geom.num_interfaces, self.fluid_model.num_components))
            sL_up = np.zeros(pipe_geom.num_interfaces)
            rhoL_up = np.zeros(pipe_geom.num_interfaces)
            for j in range(pipe_geom.num_interfaces):
                if self.vG[j] > 0:
                    xG_mass_up[j, :] = xG_mass[j, :]
                    sG_up[j] = sG[j]
                    rhoG_up[j] = rhoG[j]

                elif self.vG[j] < 0:
                    xG_mass_up[j, :] = xG_mass[j + 1, :]
                    sG_up[j] = sG[j + 1]
                    rhoG_up[j] = rhoG[j + 1]

                if self.vL[j] > 0:
                    xL_mass_up[j, :] = xL_mass[j, :]
                    sL_up[j] = 1 - sG[j]
                    rhoL_up[j] = rhoL[j]

                elif self.vL[j] < 0:
                    xL_mass_up[j, :] = xL_mass[j + 1, :]
                    sL_up[j] = 1 - sG[j + 1]
                    rhoL_up[j] = rhoL[j + 1]

            flux_c = dt * (self.vG * A * sG_up * rhoG_up * xG_mass_up[:,i]
                           + self.vL * A * sL_up * rhoL_up * xL_mass_up[:,i])
            for j in range(pipe_geom.num_interfaces):
                residuals[i*num_segments + j] += flux_c[j]
                residuals[i*num_segments + j + 1] -= flux_c[j]

            """ Add mass boundary conditions """
            for source_sink_name, source_sink_object in source_sink.items():
                if source_sink_name.startswith("ConstantMassRateSource"):
                    if source_sink_object.flow_direction[0] == "inflow":
                        z_segment = None
                    elif source_sink_object.flow_direction[0] == "outflow":
                        z_segment = np.append(zc[source_sink_object.segment_index::pipe_geom.num_segments], 1 - sum(zc[source_sink_object.segment_index::pipe_geom.num_segments]))

                    components_mass_rate = source_sink_object.evaluate_components_mass_rate(simulation_timer, z_segment)
                    residuals[source_sink_object.segment_index + i*num_segments] += - dt * components_mass_rate[i]

                if source_sink_name.startswith("Perforation"):
                    segment_pressure = p[source_sink_object.segment_index]

                    # The following props are stored for when fluid flows from the reservoir into the wellbore
                    if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
                        # The properties of the reservoir fluid are considered the same as those of the fluid in the
                        # segment connected to the reservoir at the first iteration of the first time step. The
                        # reservoir fluid properties are stored in the object of each perforation.
                        source_sink_object.reservoir_sG = sG[source_sink_object.segment_index]
                        source_sink_object.reservoir_gas_rho = rhoG[source_sink_object.segment_index]
                        source_sink_object.reservoir_liquid_rho = rhoL[source_sink_object.segment_index]
                        if 0 < sG[source_sink_object.segment_index] <= 1:
                            source_sink_object.reservoir_gas_miu = miuG[source_sink_object.segment_index]
                        elif sG[source_sink_object.segment_index] == 0:
                            source_sink_object.reservoir_gas_miu = 0
                        if 0 <= sG[source_sink_object.segment_index] < 1:
                            source_sink_object.reservoir_liquid_miu = miuL[source_sink_object.segment_index]
                        elif sG[source_sink_object.segment_index] == 1:
                            source_sink_object.reservoir_liquid_miu = 0

                        if self.isothermal:
                            source_sink_object.reservoir_temp = self.system_temperature
                        elif not self.isothermal:
                            source_sink_object.reservoir_temp = T[source_sink_object.segment_index]

                        if not self.isothermal:
                            source_sink_object.reservoir_gas_specific_enthalpy = hG[source_sink_object.segment_index]
                            source_sink_object.reservoir_liquid_specific_enthalpy = hL[source_sink_object.segment_index]

                    if source_sink_object.reservoir_pressure > segment_pressure:
                        print("Fluid is flowing from the reservoir.")
                        flow_direction = "inflow"

                        xG_mass_up_perf = "ReservoirGasComposition"   # Reservoir composition is available in Perforation
                        xL_mass_up_perf = "ReservoirLiquidComposition"   # Reservoir composition is available in Perforation
                        rhoG_up_perf = source_sink_object.reservoir_gas_rho
                        rhoL_up_perf = source_sink_object.reservoir_liquid_rho
                        miuG_up_perf = source_sink_object.reservoir_gas_miu
                        miuL_up_perf = source_sink_object.reservoir_liquid_miu
                        sG_up_perf = source_sink_object.reservoir_sG
                        KrG_up_perf = self.fluid_model.rel_perm_eval['gas'].evaluate(sG_up_perf)
                        KrL_up_perf = self.fluid_model.rel_perm_eval['liquid'].evaluate(1 - sG_up_perf)

                        if not self.isothermal:
                            hG_up_perf = source_sink_object.reservoir_gas_specific_enthalpy
                            hL_up_perf = source_sink_object.reservoir_liquid_specific_enthalpy

                    # The second condition also includes an equality check (=) without a specific reason.
                    # This equality check could alternatively be placed in the first condition.
                    # The main idea is that if the reservoir pressure equals the segment pressure,
                    # the flow rate will be calculated as zero. Consequently, it does not matter
                    # whether the fluid properties of the reservoir cell or the well segment are used.
                    elif source_sink_object.reservoir_pressure <= segment_pressure:
                        flow_direction = "outflow"
                        xG_mass_up_perf = xG_mass[source_sink_object.segment_index, :]
                        xL_mass_up_perf = xL_mass[source_sink_object.segment_index, :]
                        rhoG_up_perf = rhoG[source_sink_object.segment_index]
                        rhoL_up_perf = rhoL[source_sink_object.segment_index]
                        sG_up_perf = sG[source_sink_object.segment_index]

                        miuG_up_perf = miuG[source_sink_object.segment_index]
                        miuL_up_perf = miuL[source_sink_object.segment_index]
                        KrG_up_perf = self.fluid_model.rel_perm_eval['gas'].evaluate(sG_up_perf)
                        KrL_up_perf = self.fluid_model.rel_perm_eval['liquid'].evaluate(1 - sG_up_perf)

                        if not self.isothermal:
                            hG_up_perf = hG[source_sink_object.segment_index]
                            hL_up_perf = hL[source_sink_object.segment_index]

                    components_mass_rates = source_sink_object.evaluate_components_mass_rate(simulation_timer,
                                                        iter_counter, total_iter_counter, flag, segment_pressure,
                                                        xG_mass_up_perf, xL_mass_up_perf, rhoG_up_perf, rhoL_up_perf,
                                                        miuG_up_perf, miuL_up_perf, KrG_up_perf, KrL_up_perf, flow_direction)

                    residuals[source_sink_object.segment_index + i*num_segments] -= dt * components_mass_rates[i]

        if not self.isothermal:
            """------------------------------------------------------------------------------------------------------"""
            """------------------------------------- Energy conservation equation -----------------------------------"""
            """------------------------------------------------------------------------------------------------------"""
            # Calculate kinetic energy at segments centroids using an arithmetic average of the kinetic energy
            # at the neighboring interfaces (This method is used in "A transient geothermal wellbore simulator (2023)".)
            if iter_counter == 0 and flag == 1:
                if np.array_equal(vG0, np.array([0])):
                    self.gas_specific_kinetic_energy_seg0 = np.zeros(pipe_geom.num_segments)
                else:
                    gas_specific_kinetic_energy_faces0 = vG0 ** 2 / 2

                    """ Add energy (kinetic) boundary conditions for gas """
                    for source_sink_name, source_sink_object in source_sink.items():
                        if source_sink_name.startswith("ConstantMassRateSource"):
                            simulation_timer0 = simulation_timer   # simulation_timer0 should be used correctly later.
                            # The props of the fluid of the segment on which ConstantMassRateSource is defined are used.
                            specific_KE_gas_at_bc_interface0, _ = source_sink_object.evaluate_specific_kinetic_energy0(simulation_timer0,
                                                                                        sG0[source_sink_object.segment_index],
                                                                                        rhoG0[source_sink_object.segment_index],
                                                                                        rhoL0[source_sink_object.segment_index])

                            gas_specific_kinetic_energy_faces0 = np.insert(gas_specific_kinetic_energy_faces0,
                                                                           source_sink_object.segment_index,
                                                                           specific_KE_gas_at_bc_interface0)

                        if source_sink_name.startswith("Perforation"):
                            # For injection scenarios
                            sG0_up = sG0[source_sink_object.segment_index]
                            sL0_up = 1 - sG0[source_sink_object.segment_index]
                            rhoG0_up = rhoG0[source_sink_object.segment_index]
                            rhoL0_up = rhoL0[source_sink_object.segment_index]
                            specific_KE_gas_at_perf0, _ = source_sink_object.evaluate_specific_kinetic_energy0(sG0_up, sL0_up, rhoG0_up, rhoL0_up)

                            gas_specific_kinetic_energy_faces0 = np.insert(gas_specific_kinetic_energy_faces0,
                                                                           source_sink_object.segment_index,
                                                                           specific_KE_gas_at_perf0)

                    self.gas_specific_kinetic_energy_seg0 = (gas_specific_kinetic_energy_faces0[0:-1]
                                                             + gas_specific_kinetic_energy_faces0[1:])/2

                    # self.gas_specific_kinetic_energy_seg0 = (vG0[0:-1] ** 2 / 2 + vG0[1:] ** 2 / 2) / 2

                if np.array_equal(vL0, np.array([0])):
                    self.liquid_specific_kinetic_energy_seg0 = np.zeros(pipe_geom.num_segments)
                else:
                    liquid_specific_kinetic_energy_faces0 = vL0 ** 2 / 2

                    """ Add energy (kinetic) boundary conditions for liquid """
                    for source_sink_name, source_sink_object in source_sink.items():
                        if source_sink_name.startswith("ConstantMassRateSource"):
                            simulation_timer0 = simulation_timer  # simulation_timer0 should be used correctly later.
                            # The props of the fluid of the segment on which ConstantMassRateSource is defined are used.
                            _, specific_KE_liquid_at_bc_interface0 = source_sink_object.evaluate_specific_kinetic_energy0(simulation_timer0,
                                                                                        sG0[source_sink_object.segment_index],
                                                                                        rhoG0[source_sink_object.segment_index],
                                                                                        rhoL0[source_sink_object.segment_index])

                            liquid_specific_kinetic_energy_faces0 = np.insert(liquid_specific_kinetic_energy_faces0,
                                                                              source_sink_object.segment_index,
                                                                              specific_KE_liquid_at_bc_interface0)

                        if source_sink_name.startswith("Perforation"):
                            sG0_up = sG0[source_sink_object.segment_index]
                            sL0_up = 1 - sG0[source_sink_object.segment_index]
                            rhoG0_up = rhoG0[source_sink_object.segment_index]
                            rhoL0_up = rhoL0[source_sink_object.segment_index]
                            _, specific_KE_liquid_at_perf0 = source_sink_object.evaluate_specific_kinetic_energy0(sG0_up, sL0_up, rhoG0_up, rhoL0_up)

                            liquid_specific_kinetic_energy_faces0 = np.insert(liquid_specific_kinetic_energy_faces0,
                                                                              source_sink_object.segment_index,
                                                                              specific_KE_liquid_at_perf0)

                    self.liquid_specific_kinetic_energy_seg0 = (liquid_specific_kinetic_energy_faces0[0:-1]
                                                                + liquid_specific_kinetic_energy_faces0[1:]) / 2
                    # self.liquid_specific_kinetic_energy_seg0 = (vL0[0:-1] ** 2 / 2 + vL0[1:] ** 2 / 2) / 2

            if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
                # Specific potential energy is independent of the state, so it is only calculated at the first iteration
                # of the first time step.
                self.specific_potential_energy = self.g_cos_theta * pipe_geom.z

            """ Accumulation term """
            residuals[num_segments*self.fluid_model.num_components:num_segments*self.fluid_model.num_components + num_segments] = (
                                           V*((sG * rhoG * (Ug + self.gas_specific_kinetic_energy_seg0
                                                        + self.specific_potential_energy) +
                                          (1 - sG) * rhoL * (Ul + self.liquid_specific_kinetic_energy_seg0
                                                             + self.specific_potential_energy)) -
                                          (sG0 * rhoG0 * (Ug0 + self.gas_specific_kinetic_energy_seg0
                                                          + self.specific_potential_energy) +
                                          (1 - sG0) * rhoL0 * (Ul0 + self.liquid_specific_kinetic_energy_seg0
                                                               + self.specific_potential_energy))))
            # residuals[num_segments:] = V * ((sG * rhoG * Ug + (1 - sG) * rhoL * Ul)
            #                                 - (sG0 * rhoG0 * Ug0 + (1 - sG0) * rhoL0 * Ul0))

            """ Flux term (Advection) """
            # Pre-allocate upwinded enthalpy, and kinetic and potential energies
            hG_up = np.zeros(pipe_geom.num_interfaces)
            hL_up = np.zeros(pipe_geom.num_interfaces)
            gas_specific_kinetic_energy_up0 = np.zeros(pipe_geom.num_interfaces)
            liquid_specific_kinetic_energy_up0 = np.zeros(pipe_geom.num_interfaces)
            specific_potential_energy_up_gas = np.zeros(pipe_geom.num_interfaces)
            specific_potential_energy_up_liquid = np.zeros(pipe_geom.num_interfaces)

            for j in range(pipe_geom.num_interfaces):
                if self.vG[j] > 0:
                    hG_up[j] = hG[j]
                    gas_specific_kinetic_energy_up0[j] = self.gas_specific_kinetic_energy_seg0[j]
                    specific_potential_energy_up_gas[j] = self.specific_potential_energy[j]

                elif self.vG[j] < 0:
                    hG_up[j] = hG[j + 1]
                    gas_specific_kinetic_energy_up0[j] = self.gas_specific_kinetic_energy_seg0[j + 1]
                    specific_potential_energy_up_gas[j] = self.specific_potential_energy[j + 1]

                if self.vL[j] > 0:
                    hL_up[j] = hL[j]
                    liquid_specific_kinetic_energy_up0[j] = self.liquid_specific_kinetic_energy_seg0[j]
                    specific_potential_energy_up_liquid[j] = self.specific_potential_energy[j]

                elif self.vL[j] < 0:
                    hL_up[j] = hL[j + 1]
                    liquid_specific_kinetic_energy_up0[j] = self.liquid_specific_kinetic_energy_seg0[j + 1]
                    specific_potential_energy_up_liquid[j] = self.specific_potential_energy[j + 1]

            flux_energy_adv = dt * (A * sG_up * rhoG_up * (self.vG * hG_up + vG0 * gas_specific_kinetic_energy_up0 +
                                                           vG0 * specific_potential_energy_up_gas) +
                                    A * sL_up * rhoL_up * (self.vL * hL_up + vL0 * liquid_specific_kinetic_energy_up0 +
                                                           vL0 * specific_potential_energy_up_liquid))
            # flux_energy_adv = dt * (A * sG_up * rhoG_up * (self.vG * hG_up) +
            #                         A * sL_up * rhoL_up * (self.vL * hL_up))

            """ Flux term (Conduction) """
            # sg_m = sg[0:-1]
            # sg_p = sg[1:]
            # sg_interface = (sg_m + sg_p) / 2
            # fluid_conductivity = sg_interface * condG + (1 - sg_interface) * condL   # I'm not sure if this averaging method is correct
            # T_m = T[0:-1]
            # T_p = T[1:]
            # flux_heat_cond = fluid_conductivity * A * (T_p - T_m) / pipe_geom.D   # I'm not sure if it should be (T_p - T_m) or (T_m - T_p)
            #
            # flux_energy = flux_energy_adv - flux_heat_cond

            flux_energy = flux_energy_adv   # Ignore the heat conduction term for now (axial heat conduction, not lateral)
            for j in range(pipe_geom.num_interfaces):
                residuals[num_segments*self.fluid_model.num_components + j] += flux_energy[j]
                residuals[num_segments*self.fluid_model.num_components + j + 1] -= flux_energy[j]

            """ Add energy (enthalpy) boundary conditions and lateral heat transfer """
            for source_sink_name, source_sink_object in source_sink.items():
                if source_sink_name.startswith("ConstantMassRateSource"):
                   residuals[num_segments*self.fluid_model.num_components + source_sink_object.segment_index] += - dt * source_sink_object.evaluate_enthalpy_rate(simulation_timer)

                if source_sink_name.startswith("Perforation"):
                    gas_enthalpy_rate, liquid_enthalpy_rate = source_sink_object.evaluate_enthalpy_rate(hG_up_perf, hL_up_perf)
                    residuals[num_segments*self.fluid_model.num_components + source_sink_object.segment_index] += - dt * (gas_enthalpy_rate + liquid_enthalpy_rate)

                if source_sink_name == "LateralHeatTransfer":
                    q_lateral_heat = source_sink_object.evaluate(T, simulation_timer)
                    residuals[num_segments*self.fluid_model.num_components:num_segments*self.fluid_model.num_components + num_segments] += - q_lateral_heat * dt

            """ Add energy (potential) boundary conditions """
            for source_sink_name, source_sink_object in source_sink.items():
                if source_sink_name.startswith("ConstantMassRateSource"):
                    residuals[num_segments*self.fluid_model.num_components + source_sink_object.segment_index] += - dt * source_sink_object.evaluate_potential_energy_rate(simulation_timer, self.specific_potential_energy)

                if source_sink_name.startswith("Perforation"):
                    gas_PE_rate, liquid_PE_rate = source_sink_object.evaluate_potential_energy_rate(self.specific_potential_energy)
                    residuals[num_segments*self.fluid_model.num_components + source_sink_object.segment_index] += - dt * (gas_PE_rate + liquid_PE_rate)

        """----------------------------------------------------------------------------------------------------------"""
        """ ------------------------------------------- Replace nodes -----------------------------------------------"""
        """----------------------------------------------------------------------------------------------------------"""
        for node_name, node_object in nodes.items():
            if node_name == "ConstantPressureNode":
                residuals[node_object.segment_index] = vars[node_object.segment_index] - node_object.pressure

            if node_name == "ConstantTempNode":
                residuals[num_segments*self.fluid_model.num_components + node_object.segment_index] = (vars[num_segments*self.fluid_model.num_components + node_object.segment_index] -
                                                                       node_object.temperature)

        """----------------------------------------------------------------------------------------------------------"""
        """ ------------------------------------------ Add extra nodes ----------------------------------------------"""
        """----------------------------------------------------------------------------------------------------------"""
        for extra_node_name, extra_node_object in extra_nodes.items():
            if extra_node_name == "ConstantPressureExtraNode":
                p_segment = p[extra_node_object.segment_index]

                if extra_node_object.flow_direction == "inflow":
                    fluid_composition = "composition_of_ghost_cell_from_user_in_extra_node_object"
                    # viscosity_up = vis_func(p_ghost_segment)
                    # density_up = den_func(p_ghost_segment)
                    # total_mass_rate = extra_node_object.evaluate_total_mass_rate(p_segment, viscosity, density)

                elif extra_node_object.flow_direction == "outflow":
                    miu_gas_up, miu_liquid_up = miuG[extra_node_object.segment_index], miuL[extra_node_object.segment_index]
                    rho_gas_up, rho_liquid_up = rhoG[extra_node_object.segment_index], rhoL[extra_node_object.segment_index]
                    s_gas_up = sG[extra_node_object.segment_index]
                    Kr_gas_up = self.fluid_model.rel_perm_eval['gas'].evaluate(s_gas_up)
                    Kr_liquid_up = self.fluid_model.rel_perm_eval['liquid'].evaluate(1 - s_gas_up)
                    if sG[extra_node_object.segment_index] > 0:
                        gas_mass_rate = extra_node_object.evaluate_phase_mass_rate(p_segment, miu_gas_up, rho_gas_up, Kr_gas_up)
                    elif sG[extra_node_object.segment_index] == 0:
                        gas_mass_rate = 0
                    if sG[extra_node_object.segment_index] < 1:
                        liquid_mass_rate = extra_node_object.evaluate_phase_mass_rate(p_segment, miu_liquid_up, rho_liquid_up, Kr_liquid_up)
                    elif sG[extra_node_object.segment_index] == 1:
                        liquid_mass_rate = 0
                    comps_mass_rates = (gas_mass_rate * xG_mass[extra_node_object.segment_index]
                                        + liquid_mass_rate * xL_mass[extra_node_object.segment_index])

                    if not self.isothermal:
                        gas_enthalpy_rate = gas_mass_rate * hG[extra_node_object.segment_index]
                        liquid_enthalpy_rate = liquid_mass_rate * hL[extra_node_object.segment_index]
                        total_enthalpy_rate = gas_enthalpy_rate + liquid_enthalpy_rate

                for i in range(self.fluid_model.num_components):
                    residuals[extra_node_object.segment_index + i * num_segments] -= dt * comps_mass_rates[i]

                if not self.isothermal:
                    residuals[num_segments * self.fluid_model.num_components + extra_node_object.segment_index] -= dt * total_enthalpy_rate

                target_value = extra_node_object.ghost_segment_pressure

            residuals[extra_node_object.variable_index] = vars[extra_node_object.variable_index] - target_value

        " Either return the phase props or residuals "
        if flag == -1:   # or if phase_props_df is not None
            return phase_props_df
        else:
            return residuals

    def calc_phase_velocities(self, p, dt, source_sink, simulation_timer, iter_counter, total_iter_counter, flag):
        if self.isothermal:
            xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0
        elif not self.isothermal:
            xG_mass0, xL_mass0, sG0, rhoG0, rhoL0, _, _, _, _ = self.iter_phases_props0

        if iter_counter == 0 and flag == 1:
            # Method 1
            # # xG_mass0_face and xL_mass0_face for IFT calculation
            # # xG_mass0_face = (xG_mass0[0:-1] + xG_mass0[1:]) / 2
            # # xL_mass0_face = (xL_mass0[0:-1] + xL_mass0[1:]) / 2
            # xG_mass0_face = xG_mass0[0:-1]
            # xL_mass0_face = xL_mass0[0:-1]
            #
            # sG0_face = (sG0[0:-1] + sG0[1:]) / 2
            # rhoG0_face = (rhoG0[0:-1] + rhoG0[1:]) / 2
            # rhoL0_face = (rhoL0[0:-1] + rhoL0[1:]) / 2

            # Method 2
            sG0_face = (sG0[0:-1] + sG0[1:]) / 2

            # Initialize arrays to store interface properties
            rhoG0_face = np.zeros(self.pipe_geometry.num_segments - 1)
            rhoL0_face = np.zeros(self.pipe_geometry.num_segments - 1)
            xG_mass0_face = np.zeros((self.pipe_geometry.num_segments - 1, self.fluid_model.num_components))
            xL_mass0_face = np.zeros((self.pipe_geometry.num_segments - 1, self.fluid_model.num_components))

            # Compute interface values using conditional averaging
            for i in range(self.pipe_geometry.num_segments - 1):
                if sG0[i] == 0:
                    # If no gas in segment i, use properties from segment i+1
                    rhoG0_face[i] = rhoG0[i + 1]
                    xG_mass0_face[i] = xG_mass0[i + 1]
                elif sG0[i + 1] == 0:
                    # If no gas in segment i+1, use properties from segment i
                    rhoG0_face[i] = rhoG0[i]
                    xG_mass0_face[i] = xG_mass0[i]
                else:
                    # If both segments have gas, use arithmetic averaging
                    rhoG0_face[i] = (rhoG0[i] + rhoG0[i + 1]) / 2
                    xG_mass0_face[i] = (xG_mass0[i] + xG_mass0[i + 1]) / 2

                if sG0[i] == 1:
                    # If no liquid in segment i, use properties from segment i+1
                    rhoL0_face[i] = rhoL0[i + 1]
                    xL_mass0_face[i] = xL_mass0[i + 1]
                elif sG0[i + 1] == 1:
                    # If no liquid in segment i+1, use properties from segment i
                    rhoL0_face[i] = rhoL0[i]
                    xL_mass0_face[i] = xL_mass0[i]
                else:
                    # If both segments have liquid, use arithmetic averaging
                    rhoL0_face[i] = (rhoL0[i] + rhoL0[i + 1]) / 2
                    xL_mass0_face[i] = (xL_mass0[i] + xL_mass0[i + 1]) / 2

            self.iter_phases_props0_face = [xG_mass0_face, xL_mass0_face, sG0_face, rhoG0_face, rhoL0_face]

        [_, vM0, vG0, vL0] = self.velocities0

        p_m = p[0:-1:1]
        p_p = p[1::1]

        if self.isothermal:
            _, _, sG, rhoG, rhoL, _, _ = self.iter_phases_props
        elif not self.isothermal:
            _, _, sG, rhoG, rhoL, _, _, _, _, _, _ = self.iter_phases_props

        # Method 1
        # sG_face = (sG[0:-1] + sG[1:])/2
        # rhoG_face = (rhoG[0:-1] + rhoG[1:]) / 2
        # rhoL_face = (rhoL[0:-1] + rhoL[1:]) / 2

        # Method 2
        sG_face = (sG[0:-1] + sG[1:]) / 2

        # Initialize arrays to store interface properties
        rhoG_face = np.zeros(self.pipe_geometry.num_segments - 1)
        rhoL_face = np.zeros(self.pipe_geometry.num_segments - 1)

        # Compute interface values using conditional averaging
        for i in range(self.pipe_geometry.num_segments - 1):
            if sG[i] == 0:
                # If no gas in segment i, use properties from segment i+1
                rhoG_face[i] = rhoG[i + 1]
            elif sG[i + 1] == 0:
                # If no gas in segment i+1, use properties from segment i
                rhoG_face[i] = rhoG[i]
            else:
                # If both segments have gas, use arithmetic averaging
                rhoG_face[i] = (rhoG[i] + rhoG[i + 1]) / 2

            if sG[i] == 1:
                # If no liquid in segment i, use properties from segment i+1
                rhoL_face[i] = rhoL[i + 1]
            elif sG[i + 1] == 1:
                # If no liquid in segment i+1, use properties from segment i
                rhoL_face[i] = rhoL[i]
            else:
                # If both segments have liquid, use arithmetic averaging
                rhoL_face[i] = (rhoL[i] + rhoL[i + 1]) / 2


        self.iter_phases_props_face = [sG_face, rhoG_face, rhoL_face]
        self.calc_mixture_densities(iter_counter, total_iter_counter, flag)

        pg = self.pipe_geometry

        if iter_counter == 0 and flag == 1:
            # delta_segment = (pg.pipe_internal_A * delta_interface / pg.D + pg.pipe_internal_A * delta_interface / pg.D) / (pg.pipe_internal_A / pg.D + pg.pipe_internal_A / pg.D)
            # To increase the numerical stability, you may need to use an upwind scheme for the momentum flux like
            # in the paper "A transient gothermal wellbore simulator (2023)
            delta_interface0 = pg.pipe_internal_A * (rhoG0_face * sG0_face * vG0 ** 2 +
                                                     rhoL0_face * (1 - sG0_face) * vL0 ** 2)

            """ Add momentum boundary conditions """
            momentum_at_first_last_exterfaces = [0, 0]
            for source_sink_name, source_sink_object in source_sink.items():
                if source_sink_name.startswith("ConstantMassRateSource"):
                    if source_sink_object.flow_direction[1] == "along the pipe":
                        # The props of the fluid of the segment on which ConstantMassRateSource is defined are used.
                        delta_at_bc_interface0 = source_sink_object.evaluate_momentum0(simulation_timer,
                                                                                       sG0[source_sink_object.segment_index],
                                                                                       rhoG0[source_sink_object.segment_index],
                                                                                       rhoL0[source_sink_object.segment_index])

                        if source_sink_object.segment_index == 0:
                            momentum_at_first_last_exterfaces[0] = delta_at_bc_interface0
                        elif source_sink_object.segment_index == self.pipe_geometry.num_segments - 1:
                            momentum_at_first_last_exterfaces[1] = delta_at_bc_interface0
                        # delta_at_bc_interface0 = 0
                        # delta_interface0 = np.insert(delta_interface0, source_sink_object.segment_index, delta_at_bc_interface0)

                # if source_sink_name == "Perforation":
                #     # if total_iter_counter == 0:
                #     #     # There won't be any fluid movement at the perforation once the well is opened for injection
                #     #     # or production due the wellbore storage effect.
                #     #     delta_at_perf0 = 0
                #     #
                #     # else:
                #     #     # For injection scenario
                #     #     sG0_up = sG0[source_sink_object.segment_index]
                #     #     rhoG0_up = rhoG0[source_sink_object.segment_index]
                #     #     rhoL0_up = rhoL0[source_sink_object.segment_index]
                #     #
                #     #     delta_at_perf0 = source_sink_object.evaluate_momentum0(sG0_up, rhoG0_up, rhoL0_up)
                #     #     # delta_at_perf0 = 0
                #
                #     # For the perforation, because the fluid enters or exits the well perpendicular to the direction
                #     # of flow, there is no net transfer of momentum along the wellbore due to the perforation.
                #     delta_at_perf0 = 0
                #
                #     delta_interface0 = np.insert(delta_interface0, source_sink_object.segment_index, delta_at_perf0)

            delta_interface0 = np.insert(delta_interface0, 0, momentum_at_first_last_exterfaces[0])
            delta_interface0 = np.insert(delta_interface0, self.pipe_geometry.num_segments, momentum_at_first_last_exterfaces[1])

            delta_segment0 = (delta_interface0[0:-1:1] + delta_interface0[1::1]) / 2
            self.delta_m0 = delta_segment0[0:-1:1]
            self.delta_p0 = delta_segment0[1::1]

            ff0 = self.calc_Fanning_friction_factor()
            self.w0 = 1 / (1 / dt + pg.perimeter * ff0 * abs(vM0) / (2 * pg.pipe_internal_A))

        self.rhoM_vM = (- self.w0 * (p_p - p_m) / (pg.z_p - pg.z_m)
                        - self.w0 * self.g_cos_theta * self.rhoM_face
                        - self.w0 * ((self.delta_p0 - self.delta_m0) / (pg.pipe_internal_A * (pg.z_p - pg.z_m)) - self.rhoM0_face * vM0 / dt))

        self.vM = self.rhoM_vM / self.rhoM_face

        if iter_counter == 0 and flag == 1 and total_iter_counter != 0:
            self.calc_drift_velocity()
        elif iter_counter == 0 and flag == 1 and total_iter_counter == 0:
            self.vD0 = np.zeros(self.pipe_geometry.num_interfaces)

        # Gas velocity at wellbore interfaces
        # self.vG = self.C00 * self.rhoM_vM / self.rhoM_adjusted_face + rhoL_face * self.vD0 / self.rhoM_adjusted_face
        self.vG = np.zeros(self.pipe_geometry.num_interfaces)
        for i in range(self.pipe_geometry.num_interfaces):
            if sG_face[i] != 0:
                self.vG[i] = self.C00[i] * self.rhoM_vM[i] / self.rhoM_adjusted_face[i] + rhoL_face[i] * self.vD0[i] / self.rhoM_adjusted_face[i]

        # Liquid velocity at wellbore interfaces
        self.vL = np.zeros(self.pipe_geometry.num_interfaces)
        for i in range(self.pipe_geometry.num_interfaces):
            if sG_face[i] != 1:
                self.vL[i] = ((1 - self.C00[i] * sG_face[i]) * self.rhoM_vM[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i])
                         - sG_face[i] * rhoG_face[i] * self.vD0[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i]))

    def calc_mixture_densities(self, iter_counter, total_iter_counter, flag):
        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            if self.isothermal:
                _, _, sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0
            elif not self.isothermal:
                _, _, sG0, rhoG0, rhoL0, _, _, _, _ = self.iter_phases_props0

            rhoM0 = sG0 * rhoG0 + (1 - sG0) * rhoL0
            self.rhoM0_face = (rhoM0[0:-1] + rhoM0[1:]) / 2
            # _, _, sG0_face, rhoG0_face, rhoL0_face = self.iter_phases_props0_face
            # self.rhoM0_face = sG0_face * rhoG0_face + (1 - sG0_face) * rhoL0_face

        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            self.rhoM0_face = self.rhoM_face

        # Calculate mixture density
        if self.isothermal is True:
            _, _, sG, rhoG, rhoL, _, _ = self.iter_phases_props
        elif self.isothermal is False:
            _, _, sG, rhoG, rhoL, _, _, _, _, _, _ = self.iter_phases_props

        [sG_face, rhoG_face, rhoL_face] = self.iter_phases_props_face
        rhoM = sG * rhoG + (1 - sG) * rhoL
        self.rhoM_face = (rhoM[0:-1] + rhoM[1:]) / 2
        # self.rhoM_face = sG_face * rhoG_face + (1 - sG_face) * rhoL_face

        # Calculate adjusted-mixture density
        if iter_counter == 0 and flag == 1 and total_iter_counter != 0:
            self.calc_profile_parameter()
        elif iter_counter == 0 and flag == 1 and total_iter_counter == 0:
            # At the beginning, there is no flow, so C00 is considered 1 everywhere.
            self.C00 = np.ones(self.pipe_geometry.num_interfaces)
        self.rhoM_adjusted_face = self.C00 * sG_face * rhoG_face + (1 - self.C00 * sG_face) * rhoL_face

    def calc_Fanning_friction_factor(self):
        pg = self.pipe_geometry
        Re0 = self.calc_Reynolds_number()
        ff0 = []
        for i in range(pg.num_interfaces):
            if Re0[i] == 0:
                ff0.append(0)

            elif Re0[i] != 0:
                if Re0[i] < 2400:
                    ff0.append(16 / Re0[i])
                elif Re0[i] > 2400:
                    # T2Well
                    # ff0.append((1 / (-4 * math.log10(2 * pg.wall_roughness / (3.7 * pg.pipe_ID) - (5.02 / Re0[i])
                    #             * math.log10(2 * pg.wall_roughness / (3.7 * pg.pipe_ID) + 13 / Re0[i])))) ** 2)

                    # # Chen's correlation (The explicit form of Colebrook-White's correlation)
                    # relative_roughness = pg.wall_roughness / pg.pipe_ID
                    # Fanning_friction_factor = (1 / (-4 * math.log10(relative_roughness / 3.7065 - 5.0452 / Re0[i] * math.log10(relative_roughness ** 1.1098 / 2.8257 + (7.149 / Re0[i]) ** 0.8981)))) ** 2
                    # ff0.append(Fanning_friction_factor)

                    # Define Colebrook-White correlation
                    from scipy.optimize import fsolve
                    def colebrook(f, Re, relative_roughness):
                        if f <= 0:  # Ensure the friction factor doesn't go negative or zero
                            return 1e6  # Return a large value to prevent sqrt of negative number
                        return 1 / math.sqrt(f) + 4 * math.log10(
                            relative_roughness / 3.7065 + (1.2613 / (Re * math.sqrt(f))))

                    # Initial guess for f
                    initial_guess = 0.005

                    # Solve for f using fsolve
                    Fanning_friction_factor = \
                    fsolve(colebrook, initial_guess, args=(Re0[i], pg.wall_roughness / pg.pipe_ID))[0]
                    ff0.append(Fanning_friction_factor)

        self.ff0 = np.array(ff0)

        return self.ff0

    def calc_Reynolds_number(self):
        if self.isothermal:
            _, _, _, rhoG0, rhoL0, miuG0, miuL0 = self.iter_phases_props0
        elif not self.isothermal:
            _, _, _, rhoG0, rhoL0, miuG0, miuL0, _, _ = self.iter_phases_props0

        _, _, sG0_face, _, _ = self.iter_phases_props0_face
        [_, vM0, _, _] = self.velocities0

        miuG0_face = (miuG0[0:-1] + miuG0[1:]) / 2 if len(miuG0) != 1 else miuG0
        miuL0_face = (miuL0[0:-1] + miuL0[1:]) / 2 if len(miuL0) != 1 else miuL0

        # Saturation-weighted average is used to calculate the mixture viscosity of two phases. The method is used in
        # Beggs and Brill's book: Eq. 1.38
        # My production engineering notebook: Pressure drop calc in wellbore for 2-phase flow with Beggs and Brill's method
        self.miuM0 = sG0_face * miuG0_face + (1 - sG0_face) * miuL0_face

        pg = self.pipe_geometry

        self.Re0 = self.rhoM0_face * abs(vM0) * pg.pipe_ID / self.miuM0

        return self.Re0

    def calc_profile_parameter(self):
        pg = self.pipe_geometry
        [rhoM0_vM0, _, _, _] = self.velocities0
        [xG_mass0_face, xL_mass0_face, sG0_face, rhoG0_face, rhoL0_face] = self.iter_phases_props0_face

        mask = (sG0_face > 0) & (sG0_face < 1)
        at_least_one_true = any(mask)
        if at_least_one_true:
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            rhoM0_vM0_filtered = rhoM0_vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]
            # For IFT calculation
            xG_mass0_face_filtered = xG_mass0_face[indices]
            xL_mass0_face_filtered = xL_mass0_face[indices]

            # Constant or variable IFT using FluidModel depending on the class used by the user
            IFT0_face = np.zeros(len(indices))
            for i in range(len(indices)):
                IFT0_face[i] = self.fluid_model.IFT_eval.evaluate(rhoG0_face_filtered[i], rhoL0_face_filtered[i],
                                                                  xG_mass0_face_filtered[i], xL_mass0_face_filtered[i])

            # self.Ku0_filtered = np.zeros(len(indices))
            # self.vC0_filtered = np.zeros(len(indices))

            # Calculate C00 from the solution of the previous time step
            vM0 = rhoM0_vM0_filtered / rhoM0_face_filtered
            NB0 = (pg.pipe_ID ** 2) * (self.g * (rhoL0_face_filtered - rhoG0_face_filtered) / IFT0_face)
            self.Ku0_filtered = np.sqrt(self.Cku / np.sqrt(NB0) * (np.sqrt(1 + NB0 / (self.Cku ** 2 * self.Cw)) - 1))
            self.vC0_filtered = (self.g * IFT0_face * (rhoL0_face_filtered - rhoG0_face_filtered) / rhoL0_face_filtered ** 2) ** 0.25
            v_sgf0 = self.Ku0_filtered * np.sqrt(rhoL0_face_filtered / rhoG0_face_filtered) * self.vC0_filtered
            beta0 = np.maximum(sG0_face_filtered, self.Fv * sG0_face_filtered * abs(vM0) / v_sgf0)
            beta0 = np.clip(beta0, 0, 1)   # beta0 is subject to limits 0 <= beta0 <= 1
            eta0 = (beta0 - self.B) / (1 - self.B)   # B is calculated in the constructor
            C00_filtered = self.Cmax / (1 + (self.Cmax - 1) * eta0 ** 2)

            C00 = np.ones(self.pipe_geometry.num_interfaces)   # C00 all ones first
            self.C00 = np.ones(self.pipe_geometry.num_interfaces)
            self.C00_filtered = np.ones(len(indices))

            for i, idx in enumerate(indices):
                C00[idx] = C00_filtered[i]

            self.C00_filtered = C00_filtered
            self.C00 = C00
            # Set all profile parameters equal to 1
            # self.C00 = np.ones(self.pipe_geometry.num_interfaces)

        else:
            self.C00_filtered = 1
            # self.C00 = np.ones(self.pipe_geometry.num_interfaces)

    def calc_drift_velocity(self):
        # if np.all(self.C00 == 1):
        #     vD0 = np.zeros(self.pipe_geometry.num_interfaces)
        # else:
        if any(0 < sG < 1 for sG in self.iter_phases_props0_face[2]):
            pg = self.pipe_geometry
            [_, _, sG0_face, rhoG0_face, rhoL0_face] = self.iter_phases_props0_face
            [_, vM0, _, _] = self.velocities0

            mask = (sG0_face > 0) & (sG0_face < 1)
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            vM0_filtered = vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]

            # Calculate the K function to make a smooth transition of drift velocity between
            # the bubble-rise and film-flooding stages
            K0_filtered = np.zeros(len(self.C00_filtered))
            if type(self.Ku0_filtered) is float:   # If type of self.Ku0_filtered is float, this turns it into a list.
                self.Ku0_filtered = [self.Ku0_filtered]

            for index, value in enumerate(indices):
                if sG0_face[value] <= self.a1:
                    K0_filtered[index] = 1.53
                elif self.a1 < sG0_face[value] < self.a2:
                    K0_filtered[index] = 1.53 + (self.C00_filtered[index] * self.Ku0_filtered[index] - 1.53) / 2 * (1 - np.cos(math.pi * (sG0_face[value] - self.a1) / (self.a2 - self.a1)))
                elif sG0_face[value] >= self.a2:
                    K0_filtered[index] = self.C00_filtered[index] * self.Ku0_filtered[index]

            # Calculate the adjustment function for the mist flow regime
            # I'm not sure if X should be multiplied by C0 or not.
            Xm1 = 1
            Xm2 = 0.94
            Gm1 = 300
            Gm2 = 700
            alpha = 0.001
            lambdaa = 199
            X0 = sG0_face_filtered * rhoG0_face_filtered / (sG0_face_filtered * rhoG0_face_filtered + (1 - sG0_face_filtered) * rhoL0_face_filtered)   # gas mass fraction [dimensionless]
            G0 = rhoM0_face_filtered * abs(vM0_filtered)   # Total mass flux (or total mass flow rate per unit cross-sectional area) [kg/m2/s]
            numerator = np.zeros(len(X0))
            for i in range(len(X0)):
                numerator[i] = np.linalg.det(np.array([[X0[i], alpha * G0[i], 1], [Xm1, alpha * Gm1, 1], [Xm2, alpha * Gm2, 1]]))

            denominator = np.sqrt((Xm2 - Xm1)**2 + (alpha*Gm2 - alpha*Gm1)**2)
            Dm = numerator/denominator
            f0 = np.maximum(0, 1 - np.minimum(1, G0/Gm1) * np.exp(-lambdaa * Dm*abs(Dm)))

            # Calculate drift velocity
            vD0 = np.zeros(self.pipe_geometry.num_interfaces)   # vD0 all zeros first
            for index, value in enumerate(indices):
                # Ignore the consideration of the adjustment function for the mist flow regime for now
                vD0[value] = (1 - self.C00_filtered[index] * sG0_face_filtered[index]) * self.vC0_filtered[index] * K0_filtered[index] * self.m * f0[index] / (self.C00_filtered[index] * sG0_face_filtered[index] * np.sqrt(rhoG0_face_filtered[index] / rhoL0_face_filtered[index]) + 1 - self.C00_filtered[index] * sG0_face_filtered[index])
                # vD0[value] = (1 - self.C00_filtered[index] * sG0_face_filtered[index]) * self.vC0_filtered[index] * K0_filtered[index] * self.m / (self.C00_filtered[index] * sG0_face_filtered[index] * np.sqrt(rhoG0_face_filtered[index] / rhoL0_face_filtered[index]) + 1 - self.C00_filtered[index] * sG0_face_filtered[index])
        else:
            vD0 = np.zeros(self.pipe_geometry.num_interfaces)
        self.vD0 = vD0
