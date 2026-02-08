import math
import numpy as np

from dwell.single_phase_non_isothermal.define_pipe_geometry import PipeGeometry
from dwell.single_phase_non_isothermal.define_fluid_model import FluidModel
from dwell.utilities.units import *

class PipeModel:
    g = 9.80665 * meter() / second()**2  # Gravitational acceleration
    Cku = 142
    Cw = 0.008

    def __init__(self, pipe_geometry: PipeGeometry, fluid_model: FluidModel, verbose: bool = False,
                 Cmax=1.2, Fv=1, eps_p=10 * Pascal(), eps_temp=0.1):
        """
        :param pipe_geometry: Pipe geometry object
        :type pipe_geometry: PipeGeometry
        :param fluid_model: Fluid model object
        :type fluid_model: FluidModel
        :param Cmax: A user-specified maximum profile parameter that can be tuned to match the observations and
        could have a value between 1.0 and 1.5. It is set to
        1.2 in ECLIPSE according to Shi et al. paper (Drift-Flux Modeling of Two-Phase Flow in Wellbores)
        1 in a wellbore simulator in Tonken et al. paper (A transient geothermal wellbore simulator)
        :type Cmax: float
        :param Fv: A multiplier on the flooding velocity fraction, set to be 1 by default, and its value can be tuned
        to fit the observations.
        :type Fv: float
        :param eps_p: A very small number used for numerically differentiating residual equations with respect to pressure.
        :type eps_p: float
        :param eps_temp: A very small number used for numerically differentiating residual equations with respect to temperature.
        :type eps_temp: float
        :param verbose: Whether to display extra info about PipeModel
        :type verbose: boolean
        """
        self.pipe_geometry = pipe_geometry
        self.fluid_model = fluid_model
        assert self.pipe_geometry.pipe_name == self.fluid_model.pipe_name, \
            "The names of the pipes for the geometry and fluid model are not identical!"

        self.Cmax = Cmax
        self.B = 2 / self.Cmax - 1.0667
        self.Fv = Fv
        self.g_cos_theta = self.g * math.cos(pipe_geometry.inclination_angle_radian)
        self.eps_p = eps_p
        self.eps_temp = eps_temp

        if verbose:
            print("** Model of the pipe \"%s\" is created!" % self.pipe_geometry.pipe_name)

    def init_pipe_model(self):
        pass

    def calc_mass_energy_residuals(self, vars0, vars, dt, bc, simulation_timer, iter_counter, total_iter_counter, flag):
        source_sink = bc['source/sink']
        nodes = bc['nodes']
        extra_nodes = bc['extra_nodes']

        pipe_geom = self.pipe_geometry
        V = pipe_geom.segments_volumes
        A = pipe_geom.pipe_internal_A
        num_segments = pipe_geom.num_segments

        x0 = [1.]
        x = [1.]
        p0 = vars0[0:num_segments:1]
        p = vars[0:num_segments:1]
        T0 = vars0[num_segments::1]
        T = vars[num_segments::1]

        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            sG0 = np.zeros(num_segments)
            rhoG0 = np.zeros(num_segments)
            rhoL0 = np.zeros(num_segments)
            hG0 = np.zeros(num_segments)
            hL0 = np.zeros(num_segments)
            Ug0 = np.zeros(num_segments)
            Ul0 = np.zeros(num_segments)
            for i in range(num_segments):
                sG0[i] = 1   # The code can be used for both single-phase gas and single-phase liquid flow, no matter this is equal to 1 or 0.
                if 0.001 < sG0[i] < 0.999:
                    # This PT-based PVT model cannot give different densities if the system is 2-phase (on the saturation line).
                    rhoG0[i] = self.fluid_model.density_obj.evaluate(p0[i], T0[i], x0)
                    rhoL0[i] = self.fluid_model.density_obj.evaluate(p0[i], T0[i], x0)

                    # Enthalpies of the previous time step are used for calculating internal energies of the previous
                    # time step. They're not used directly in the equations.
                    hG0[i] = self.fluid_model.enthalpy_obj.evaluate(p0[i], T0[i], x0)
                    hL0[i] = self.fluid_model.enthalpy_obj.evaluate(p0[i], T0[i], x0)

                    Ug0[i] = self.fluid_model.internal_energy_obj.evaluate(p0[i], hG0[i], rhoG0[i])
                    Ul0[i] = self.fluid_model.internal_energy_obj.evaluate(p0[i], hL0[i], rhoL0[i])

                elif sG0[i] >= 0.999:
                    rhoG0[i] = self.fluid_model.density_obj.evaluate(p0[i], T0[i], x0)
                    hG0[i] = self.fluid_model.enthalpy_obj.evaluate(p0[i], T0[i], x0)
                    Ug0[i] = self.fluid_model.internal_energy_obj.evaluate(p0[i], hG0[i], rhoG0[i])

                elif sG0[i] <= 0.001:
                    rhoL0[i] = self.fluid_model.density_obj.evaluate(p0[i], T0[i], x0)
                    hL0[i] = self.fluid_model.enthalpy_obj.evaluate(p0[i], T0[i], x0)
                    Ul0[i] = self.fluid_model.internal_energy_obj.evaluate(p0[i], hL0[i], rhoL0[i])

            self.iter_phases_props0 = [sG0, rhoG0, rhoL0, Ug0, Ul0]

        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            # Just from the first column to 5th column of this must be put in iter_phases_props0
            self.iter_phases_props0 = self.iter_phases_props[0:5]

        sG0, rhoG0, rhoL0, Ug0, Ul0 = self.iter_phases_props0

        # Because the heat conduction term is ignored, there is no need to calculate the phase conductivities for now
        # condG, condL = self.fluid_model.conductivity_obj(p, T)
        # self.iter_phases_props = [sG, rhoG, rhoL, Ug, Ul, hG, hL, condG, condL]

        sG = np.zeros(num_segments)
        rhoG = np.zeros(num_segments)
        rhoL = np.zeros(num_segments)
        hG = np.zeros(num_segments)
        hL = np.zeros(num_segments)
        Ug = np.zeros(num_segments)
        Ul = np.zeros(num_segments)
        for i in range(num_segments):
            sG[i] = 1   # The code can be used for both single-phase gas and single-phase liquid flow, no matter this is equal to 1 or 0.
            if 0.001 < sG[i] < 0.999:
                rhoG[i] = self.fluid_model.density_obj.evaluate(p[i], T[i], x)
                rhoL[i] = self.fluid_model.density_obj.evaluate(p[i], T[i], x)

                hG[i] = self.fluid_model.enthalpy_obj.evaluate(p[i], T[i], x)
                hL[i] = self.fluid_model.enthalpy_obj.evaluate(p[i], T[i], x)

                Ug[i] = self.fluid_model.internal_energy_obj.evaluate(p[i], hG[i], rhoG[i])
                Ul[i] = self.fluid_model.internal_energy_obj.evaluate(p[i], hL[i], rhoL[i])

            elif sG0[i] >= 0.999:
                rhoG[i] = self.fluid_model.density_obj.evaluate(p[i], T[i], x)
                hG[i] = self.fluid_model.enthalpy_obj.evaluate(p[i], T[i], x)
                Ug[i] = self.fluid_model.internal_energy_obj.evaluate(p[i], hG[i], rhoG[i])

            elif sG0[i] <= 0.001:
                rhoL[i] = self.fluid_model.density_obj.evaluate(p[i], T[i], x)
                hL[i] = self.fluid_model.enthalpy_obj.evaluate(p[i], T[i], x)
                Ul[i] = self.fluid_model.internal_energy_obj.evaluate(p[i], hL[i], rhoL[i])

        self.iter_phases_props = [sG, rhoG, rhoL, Ug, Ul, hG, hL]

        # self.velocities0 will be used in the method self.calc_phase_velocities and also for energy conservation eq
        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            [rhoM0_vM0, vM0, vG0, vL0] = [np.array([0]), np.array([0]), np.array([0]), np.array([0])]  # Initial velocities in the wellbore are zero
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])
        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            [rhoM0_vM0, vM0, vG0, vL0] = [self.rhoM_vM, self.vM, self.vG, self.vL]
            self.velocities0 = np.array([rhoM0_vM0, vM0, vG0, vL0])

        [_, _, vG0, vL0] = self.velocities0

        self.calc_phase_velocities(T0, p0, p, dt, source_sink, simulation_timer, iter_counter, total_iter_counter, flag)

        # Pre-allocate upwinded props
        sG_up = np.zeros(pipe_geom.num_interfaces)
        rhoG_up = np.zeros(pipe_geom.num_interfaces)
        sL_up = np.zeros(pipe_geom.num_interfaces)
        rhoL_up = np.zeros(pipe_geom.num_interfaces)
        for j in range(pipe_geom.num_interfaces):
            if self.vG[j] > 0:
                sG_up[j] = sG[j]
                rhoG_up[j] = rhoG[j]

            elif self.vG[j] < 0:
                sG_up[j] = sG[j + 1]
                rhoG_up[j] = rhoG[j + 1]

            if self.vL[j] > 0:
                sL_up[j] = 1 - sG[j]
                rhoL_up[j] = rhoL[j]

            elif self.vL[j] < 0:
                sL_up[j] = 1 - sG[j + 1]
                rhoL_up[j] = rhoL[j + 1]

        """----------------------------------------------------------------------------------------------------------"""
        """-------------------------------------- Mass conservation equation ----------------------------------------"""
        """----------------------------------------------------------------------------------------------------------"""
        num_eqs = int(len(vars0)/num_segments)
        residuals = np.zeros(num_segments * num_eqs)
        """ Accumulation term """
        residuals[0:num_segments] = V*((sG * rhoG + (1 - sG) * rhoL) - (sG0 * rhoG0 + (1 - sG0) * rhoL0))

        """ Flux term """
        flux_c = dt * (self.vG * A * sG_up * rhoG_up + self.vL * A * sL_up * rhoL_up)
        for j in range(pipe_geom.num_interfaces):
            residuals[j] += flux_c[j]
            residuals[j + 1] -= flux_c[j]

        """ Add mass boundary conditions """
        for source_sink_name, source_sink_object in source_sink.items():
            if source_sink_name == "ConstantMassRateSource":
               residuals[source_sink_object.segment_index] += - dt * source_sink_object.evaluate_mass_rate(simulation_timer)

            if source_sink_name == "Perforation":
                segment_pressure = p[source_sink_object.segment_index]

                # The following props are stored for when fluid flows from the reservoir into the wellbore
                if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
                    # The properties of the reservoir fluid are considered the same as those of the fluid in the
                    # segment connected to the reservoir at the first iteration of the first time step.
                    self.rho_reservoir_gas = rhoG[source_sink_object.segment_index]
                    self.rho_reservoir_liquid = rhoL[source_sink_object.segment_index]
                    self.reservoir_temp = T[source_sink_object.segment_index]

                    self.specific_enthalpy_reservoir_gas = hG[source_sink_object.segment_index]
                    self.specific_enthalpy_reservoir_liquid = hL[source_sink_object.segment_index]

                if source_sink_object.reservoir_pressure > segment_pressure:
                    print("Fluid is flowing from the reservoir.")

                    rhoG_up_perf = self.rho_reservoir_gas
                    rhoL_up_perf = self.rho_reservoir_liquid
                    T_up_perf = self.reservoir_temp
                    p_up_perf = source_sink_object.reservoir_pressure
                    miuG_up_perf = self.fluid_model.viscosity_obj.evaluate(p_up_perf, T_up_perf, [1], rhoG_up_perf)
                    miuL_up_perf = self.fluid_model.viscosity_obj.evaluate(p_up_perf, T_up_perf, [1], rhoL_up_perf)
                    KrG_up_perf = 1  # For single-phase flow
                    KrL_up_perf = 1  # For single-phase flow

                    hG_up_perf = self.specific_enthalpy_reservoir_gas
                    hL_up_perf = self.specific_enthalpy_reservoir_liquid

                # The second condition also includes an equality check (=) without a specific reason.
                # This equality check could alternatively be placed in the first condition.
                # The main idea is that if the reservoir pressure equals the segment pressure,
                # the flow rate will be calculated as zero. Consequently, it does not matter
                # whether the fluid properties of the reservoir cell or the well segment are used.
                elif source_sink_object.reservoir_pressure <= segment_pressure:
                    rhoG_up_perf = rhoG[source_sink_object.segment_index]
                    rhoL_up_perf = rhoL[source_sink_object.segment_index]
                    p_up_perf = segment_pressure
                    T_up_perf = T[source_sink_object.segment_index]
                    miuG_up_perf = self.fluid_model.viscosity_obj.evaluate(p_up_perf, T_up_perf, [1], rhoG_up_perf)
                    miuL_up_perf = self.fluid_model.viscosity_obj.evaluate(p_up_perf, T_up_perf, [1], rhoL_up_perf)
                    KrG_up_perf = 1   # For single-phase flow
                    KrL_up_perf = 1   # For single-phase flow

                    hG_up_perf = hG[source_sink_object.segment_index]
                    hL_up_perf = hL[source_sink_object.segment_index]

                gas_mass_rate_at_perf, liquid_mass_rate_at_perf = source_sink_object.evaluate_mass_rate(
                    simulation_timer, iter_counter, total_iter_counter, flag, segment_pressure,
                    rhoG_up_perf, rhoL_up_perf, miuG_up_perf, miuL_up_perf, KrG_up_perf, KrL_up_perf)

                residuals[source_sink_object.segment_index] -= dt * (gas_mass_rate_at_perf + liquid_mass_rate_at_perf)

        """----------------------------------------------------------------------------------------------------------"""
        """------------------------------------- Energy conservation equation ---------------------------------------"""
        """----------------------------------------------------------------------------------------------------------"""
        # Calculate kinetic energy at segments centroids using an arithmetic average of the kinetic energy
        # at the neighboring interfaces (This method is used in "A transient geothermal wellbore simulator (2023)".)
        if iter_counter == 0 and flag == 1:
            if np.array_equal(vG0, np.array([0])):
                self.gas_specific_kinetic_energy_seg0 = np.zeros(pipe_geom.num_segments)
            else:
                gas_specific_kinetic_energy_faces0 = vG0 ** 2 / 2

                """ Add energy (kinetic) boundary conditions for gas """
                for source_sink_name, source_sink_object in source_sink.items():
                    if source_sink_name == "ConstantMassRateSource":
                        simulation_timer0 = simulation_timer   # simulation_timer0 should be used correctly later.
                        # The props of the fluid of the segment on which ConstantMassRateSource is defined are used.
                        specific_KE_gas_at_bc_interface0, _ = source_sink_object.evaluate_specific_kinetic_energy0(simulation_timer0,
                                                                                    sG0[source_sink_object.segment_index],
                                                                                    rhoG0[source_sink_object.segment_index],
                                                                                    rhoL0[source_sink_object.segment_index])

                        gas_specific_kinetic_energy_faces0 = np.insert(gas_specific_kinetic_energy_faces0,
                                                                       source_sink_object.segment_index,
                                                                       specific_KE_gas_at_bc_interface0)

                    if source_sink_name == "Perforation":
                        sG0_up = sG0[source_sink_object.segment_index]   # For injection scenario
                        sL0_up = 1 - sG0[source_sink_object.segment_index]   # For injection scenario
                        rhoG0_up = rhoG0[source_sink_object.segment_index]   # For injection scenario
                        rhoL0_up = rhoL0[source_sink_object.segment_index]   # For injection scenario
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
                    if source_sink_name == "ConstantMassRateSource":
                        simulation_timer0 = simulation_timer  # simulation_timer0 should be used correctly later.
                        # The props of the fluid of the segment on which ConstantMassRateSource is defined are used.
                        _, specific_KE_liquid_at_bc_interface0 = source_sink_object.evaluate_specific_kinetic_energy0(simulation_timer0,
                                                                                    sG0[source_sink_object.segment_index],
                                                                                    rhoG0[source_sink_object.segment_index],
                                                                                    rhoL0[source_sink_object.segment_index])

                        liquid_specific_kinetic_energy_faces0 = np.insert(liquid_specific_kinetic_energy_faces0,
                                                                          source_sink_object.segment_index,
                                                                          specific_KE_liquid_at_bc_interface0)

                    if source_sink_name == "Perforation":
                        sG0_up = sG0[source_sink_object.segment_index]   # For injection scenario
                        sL0_up = 1 - sG0[source_sink_object.segment_index]   # For injection scenario
                        rhoG0_up = rhoG0[source_sink_object.segment_index]   # For injection scenario
                        rhoL0_up = rhoL0[source_sink_object.segment_index]   # For injection scenario
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
        residuals[num_segments:] = V*((sG * rhoG * (Ug + self.gas_specific_kinetic_energy_seg0
                                                    + self.specific_potential_energy) +
                                      (1 - sG) * rhoL * (Ul + self.liquid_specific_kinetic_energy_seg0
                                                         + self.specific_potential_energy)) -
                                      (sG0 * rhoG0 * (Ug0 + self.gas_specific_kinetic_energy_seg0
                                                      + self.specific_potential_energy) +
                                      (1 - sG0) * rhoL0 * (Ul0 + self.liquid_specific_kinetic_energy_seg0
                                                           + self.specific_potential_energy)))
        # residuals[num_segments:] = V * ((sG * rhoG * Ug + (1 - sG) * rhoL * Ul)
        #                                 - (sG0 * rhoG0 * Ug0 + (1 - sG0) * rhoL0 * Ul0))

        """ Flux term (Advection) """
        # Pre-allocate upwinded kinetic and potential energies
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
            residuals[num_segments + j] += flux_energy[j]
            residuals[num_segments + j + 1] -= flux_energy[j]

        """ Add energy (enthalpy) boundary conditions and lateral heat transfer """
        for source_sink_name, source_sink_object in source_sink.items():
            if source_sink_name == "ConstantMassRateSource":
               residuals[num_segments + source_sink_object.segment_index] += - dt * source_sink_object.evaluate_enthalpy_rate(simulation_timer)

            if source_sink_name == "Perforation":
                gas_enthalpy_rate, liquid_enthalpy_rate = source_sink_object.evaluate_enthalpy_rate(hG_up_perf, hL_up_perf)
                residuals[num_segments + source_sink_object.segment_index] += - dt * (gas_enthalpy_rate + liquid_enthalpy_rate)

            if source_sink_name == "LateralHeatTransfer":
                q_lateral_heat = source_sink_object.evaluate(T, simulation_timer)
                residuals[num_segments::] += - q_lateral_heat * dt

        """ Add energy (kinetic) boundary conditions """   # Kinetic energy boundary condition must not be applied more than one time unless the results won't be correct.
        # for source_sink_name, source_sink_object in source_sink.items():
        #     if source_sink_name == "ConstantMassRateSource":
        #         gas_KE_rate_at_bc_interface, liquid_KE_rate_at_bc_interface = source_sink_object.evaluate_kinetic_energy_rate(simulation_timer, sG[source_sink_object.segment_index], rhoG[source_sink_object.segment_index], rhoL[source_sink_object.segment_index])
        #         residuals[num_segments + source_sink_object.segment_index] += - dt * (gas_KE_rate_at_bc_interface + liquid_KE_rate_at_bc_interface)
        #
        #     if source_sink_name == "Perforation":
        #         sG_up = sG[source_sink_object.segment_index]   # For injection scenarios
        #         sL_up = 0   # For injection scenarios
        #         rhoG_up = rhoG[source_sink_object.segment_index]   # For injection scenarios
        #         rhoL_up = rhoL[source_sink_object.segment_index]   # For injection scenarios
        #         gas_KE_rate, liquid_KE_rate = source_sink_object.evaluate_kinetic_energy_rate(sG_up, sL_up, rhoG_up, rhoL_up)
        #         residuals[num_segments + source_sink_object.segment_index] += - dt * (gas_KE_rate + liquid_KE_rate)

        """ Add energy (potential) boundary conditions """
        for source_sink_name, source_sink_object in source_sink.items():
            if source_sink_name == "ConstantMassRateSource":
                residuals[num_segments + source_sink_object.segment_index] += - dt * source_sink_object.evaluate_potential_energy_rate(simulation_timer, self.specific_potential_energy)

            if source_sink_name == "Perforation":
                gas_PE_rate, liquid_PE_rate = source_sink_object.evaluate_potential_energy_rate(self.specific_potential_energy)
                residuals[num_segments + source_sink_object.segment_index] += - dt * (gas_PE_rate + liquid_PE_rate)

        """----------------------------------------------------------------------------------------------------------"""
        """ --------------------------------------------- Add nodes -------------------------------------------------"""
        """----------------------------------------------------------------------------------------------------------"""
        for node_name, node_object in nodes.items():
            if node_name == "ConstantPressureNode":
                residuals[node_object.segment_index] = p[node_object.segment_index] - node_object.pressure

            if node_name == "ConstantTempNode":
                residuals[num_segments + node_object.segment_index] = T[node_object.segment_index] - node_object.temperature

        return residuals

    def calc_phase_velocities(self, T0, p0, p, dt, source_sink, simulation_timer, iter_counter, total_iter_counter, flag):
        sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0
        sG0_face = (sG0[0:-1] + sG0[1:]) / 2
        rhoG0_face = (rhoG0[0:-1] + rhoG0[1:]) / 2
        rhoL0_face = (rhoL0[0:-1] + rhoL0[1:]) / 2
        self.iter_phases_props0_face = [sG0_face, rhoG0_face, rhoL0_face]

        [_, vM0, vG0, vL0] = self.velocities0

        p_m = p[0:-1]
        p_p = p[1:]
        sG, rhoG, rhoL, _, _, _, _ = self.iter_phases_props
        sG_face = (sG[0:-1] + sG[1:])/2
        rhoG_face = (rhoG[0:-1] + rhoG[1:]) / 2
        rhoL_face = (rhoL[0:-1] + rhoL[1:]) / 2
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
            for source_sink_name, source_sink_object in source_sink.items():
                if source_sink_name == "ConstantMassRateSource":
                    # The props of the fluid of the segment on which ConstantMassRateSource is defined are used.
                    delta_at_bc_interface0 = source_sink_object.evaluate_momentum0(simulation_timer,
                                                                                   sG0[source_sink_object.segment_index],
                                                                                   rhoG0[source_sink_object.segment_index],
                                                                                   rhoL0[source_sink_object.segment_index])
                    delta_interface0 = np.insert(delta_interface0, source_sink_object.segment_index, delta_at_bc_interface0)

                if source_sink_name == "Perforation":
                    # if total_iter_counter == 0:
                    #     # There won't be any fluid movement at the perforation once the well is opened for injection
                    #     # or production due the wellbore storage effect.
                    #     delta_at_perf0 = 0
                    #
                    # else:
                    #     # For injection scenario
                    #     sG0_up = sG0[source_sink_object.segment_index]
                    #     rhoG0_up = rhoG0[source_sink_object.segment_index]
                    #     rhoL0_up = rhoL0[source_sink_object.segment_index]
                    #
                    #     # delta_at_perf0 = source_sink_object.evaluate_momentum0(sG0_up, rhoG0_up, rhoL0_up)
                    #     delta_at_perf0 = 0

                    # For the perforation, because the fluid enters or exits the well perpendicular to the direction
                    # of flow, there is no net transfer of momentum along the wellbore due to the perforation.
                    delta_at_perf0 = 0

                    delta_interface0 = np.insert(delta_interface0, source_sink_object.segment_index, delta_at_perf0)

            delta_segment0 = (delta_interface0[0:-1:1] + delta_interface0[1::1]) / 2
            self.delta_m0 = delta_segment0[0:-1:1]
            self.delta_p0 = delta_segment0[1::1]

            # T0 will be ultimately used to calculate miuG0 and miuL0 for the Raynolds number
            ff0 = self.calc_Fanning_friction_factor(T0, p0)
            self.w0 = 1 / (1 / dt + pg.perimeter * ff0 * abs(vM0) / (2 * pg.pipe_internal_A))

        self.rhoM_vM = (- self.w0 * (p_p - p_m) / (pg.z_p - pg.z_m)
                        - self.w0 * self.g_cos_theta * self.rhoM_face
                        - self.w0 * ((self.delta_p0 - self.delta_m0) / (pg.pipe_internal_A * (pg.z_p - pg.z_m)) - self.rhoM0_face * vM0 / dt))

        self.vM = self.rhoM_vM / self.rhoM_face

        # Gas velocity at wellbore interfaces
        if iter_counter == 0 and flag == 1:
             self.calc_drift_velocity()
        self.vG = self.C00 * self.rhoM_vM / self.rhoM_adjusted_face + rhoL_face * self.vD0 / self.rhoM_adjusted_face

        # Liquid velocity at wellbore interfaces
        vL = np.zeros(len(sG_face))
        for i in range(len(sG_face)):
            if sG_face[i] >= 0.999:    # For single-phase gas
                vL[i] = 0
            elif sG_face[i] == 0:    # For single-phase liquid
                vL[i] = ((1 - self.C00 * sG_face[i]) * self.rhoM_vM[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i])
                         - sG_face[i] * rhoG_face[i] * self.vD0 / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i]))
            else:
                vL[i] = ((1 - self.C00[i] * sG_face[i]) * self.rhoM_vM[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i])
                           - sG_face[i] * rhoG_face[i] * self.vD0[i] / ((1 - sG_face[i]) * self.rhoM_adjusted_face[i]))
        self.vL = vL

    def calc_mixture_densities(self, iter_counter, total_iter_counter, flag):
        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            # sG0, rhoG0, rhoL0, _, _ = self.iter_phases_props0
            # rhoM0 = sG0 * rhoG0 + (1 - sG0) * rhoL0
            # self.rhoM0_face = (rhoM0[0:-1] + rhoM0[1:]) / 2
            sG0_face, rhoG0_face, rhoL0_face = self.iter_phases_props0_face
            self.rhoM0_face = sG0_face * rhoG0_face + (1 - sG0_face) * rhoL0_face

        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            self.rhoM0_face = self.rhoM_face

        # Calculate mixture density
        # sG, rhoG, rhoL, _, _, _, _ = self.iter_phases_props
        # rhoM = sG * rhoG + (1 - sG) * rhoL
        # self.rhoM_face = (rhoM[0:-1] + rhoM[1:]) / 2
        [sG_face, rhoG_face, rhoL_face] = self.iter_phases_props_face
        self.rhoM_face = sG_face * rhoG_face + (1 - sG_face) * rhoL_face

        # Calculate adjusted-mixture density
        if iter_counter == 0 and flag == 1:
            self.calc_profile_parameter()
        self.rhoM_adjusted_face = self.C00 * sG_face * rhoG_face + (1 - self.C00 * sG_face) * rhoL_face

    def calc_Fanning_friction_factor(self, T0, p0):
        pg = self.pipe_geometry
        Re0 = self.calc_Reynolds_number(T0, p0)
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
                    # Fanning_friction_factor = (1 / (-4 * math.log10(relative_roughness/3.7065 - 5.0452/Re0[i] * math.log10(relative_roughness ** 1.1098 / 2.8257 + (7.149/Re0[i]) ** 0.8981))) ) ** 2
                    # ff0.append(Fanning_friction_factor)

                    # Define Colebrook-White correlation
                    from scipy.optimize import fsolve
                    def colebrook(f, Re, relative_roughness):
                        if f <= 0:   # Ensure the friction factor doesn't go negative or zero
                            return 1e6   # Return a large value to prevent sqrt of negative number
                        return 1 / math.sqrt(f) + 4 * math.log10(relative_roughness / 3.7065 + (1.2613 / (Re * math.sqrt(f))))

                    # Initial guess for f
                    initial_guess = 0.005

                    # Solve for f using fsolve
                    Fanning_friction_factor = fsolve(colebrook, initial_guess, args=(Re0[i], pg.wall_roughness/pg.pipe_ID))[0]
                    ff0.append(Fanning_friction_factor)

                    # # Von Karman correlation
                    # Darcy_friction_factor = (1 / (2 * math.log10(pg.pipe_ID / pg.wall_roughness) + 1.74)) ** 2
                    # ff0.append(Darcy_friction_factor / 4)

                    # # Kiarash
                    # ff0.append(1 / (-3.6 * math.log10(6.9 / Re0[i] + (pg.wall_roughness / (3.7 * pg.pipe_ID) ** (10 / 9)))) ** 2)

                # # Moody correlation: This correlation is valid for all ranges of the Reynolds number
                # Darcy_friction_factor = 5.5e-3 * (1 + (2e4 * (pg.wall_roughness / pg.pipe_ID) + 10e6 / Re0[i]) ** 1 / 3)
                # ff0.append(Darcy_friction_factor / 4)

        self.ff0 = np.array(ff0)

        return self.ff0

    def calc_Reynolds_number(self, T0, p0):
        _, rhoG0, rhoL0, _, _ = self.iter_phases_props0
        sG0_face, _, _ = self.iter_phases_props0_face
        [_, vM0, _, _] = self.velocities0

        miuG0 = self.fluid_model.viscosity_obj.evaluate(p0, T0, [1], rhoG0)
        miuL0 = self.fluid_model.viscosity_obj.evaluate(p0, T0, [1], rhoL0)
        miuG0_face = (miuG0[0:-1] + miuG0[1:]) / 2
        miuL0_face = (miuL0[0:-1] + miuL0[1:]) / 2

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
        [sG0_face, rhoG0_face, rhoL0_face] = self.iter_phases_props0_face

        # Using only gas saturation values that are larger than 0.001 and smaller than 0.999
        # For this, sG0_face must be a numpy array or be converted to that.
        mask = (sG0_face > 0.001) & (sG0_face < 0.999)
        at_least_one_true = any(mask)
        if at_least_one_true:
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            rhoM0_vM0_filtered = rhoM0_vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]

            # Constant or variable IFT using FluidModel depending on the class used by the user
            IFT0_face = self.fluid_model.IFT_obj.evaluate(rhoG0_face_filtered, rhoL0_face_filtered)

            # Calculate C00 from the solution of the previous time step
            vM0 = rhoM0_vM0_filtered / rhoM0_face_filtered
            NB0 = (pg.pipe_ID ** 2) * (self.g * (rhoL0_face_filtered - rhoG0_face_filtered) / IFT0_face)
            self.Ku0_filtered = math.sqrt(self.Cku / math.sqrt(NB0) * (math.sqrt(1 + NB0 / (self.Cku ** 2 * self.Cw)) - 1))
            self.vC0_filtered = (self.g * IFT0_face * (rhoL0_face_filtered - rhoG0_face_filtered) / rhoL0_face_filtered ** 2) ** 0.25
            v_sgf0 = self.Ku0_filtered * math.sqrt(rhoL0_face_filtered / rhoG0_face_filtered) * self.vC0_filtered
            beta0 = max(sG0_face_filtered, self.Fv * sG0_face_filtered * abs(vM0) / v_sgf0)
            eta0 = (beta0 - self.B) / (1 - self.B)
            C00_filtered = self.Cmax / (1 + (self.Cmax - 1) * eta0 ** 2)

            C00 = np.ones(self.pipe_geometry.num_interfaces)   # C00 all ones first
            for i, idx in enumerate(indices):
                C00[idx] = C00_filtered[i]

            self.C00_filtered = C00_filtered
            self.C00 = C00

        else:
            self.C00_filtered = 1
            self.C00 = 1

    def calc_drift_velocity(self):
        if self.C00 == 1:
            vD0 = 0
        else:
            pg = self.pipe_geometry
            [sG0_face, rhoG0_face, rhoL0_face] = self.iter_phases_props0_face
            [_, vM0, _, _] = self.velocities0

            # Using only gas saturation values that are larger than 0.001 and smaller than 0.999
            # For this, sG0_face must be a numpy array or be converted to one.
            mask = (sG0_face > 0.001) & (sG0_face < 0.999)
            indices = np.where(mask)[0]
            sG0_face_filtered = sG0_face[indices]
            rhoG0_face_filtered = rhoG0_face[indices]
            rhoL0_face_filtered = rhoL0_face[indices]
            vM0_filtered = vM0[indices]
            rhoM0_face_filtered = self.rhoM0_face[indices]

            """ You should store this if statement and also m in order to prevent unnecessary repetitive calculations """
            # I did not see anywhere to tell if I can use linear interp and extra here or not.
            if self.Cmax == 1:
                a1 = 0.06
                a2 = 0.21
                m0 = 1.85
                n1 = 0.21
                n2 = 0.95
            elif 1 < self.Cmax < 1.2:
                # Linear interpolation
                a1 = 0.06
                a2 = np.interp(self.Cmax, [1, 1.2], [0.21, 0.12])
                m0 = np.interp(self.Cmax, [1, 1.2], [1.85, 1.27])
                n1 = np.interp(self.Cmax, [1, 1.2], [0.21, 0.24])
                n2 = np.interp(self.Cmax, [1, 1.2], [0.95, 1.08])
            elif self.Cmax == 1.2:
                a1 = 0.06
                a2 = 0.12
                m0 = 1.27
                n1 = 0.24
                n2 = 1.08
            elif 1.2 < self.Cmax <= 1.5:
                # Linear extrapolation
                a1 = 0.06
                a2 = np.interp(self.Cmax, [1, 1.2], [0.21, 0.12])
                m0 = np.interp(self.Cmax, [1, 1.2], [1.85, 1.27])
                n1 = np.interp(self.Cmax, [1, 1.2], [0.21, 0.24])
                n2 = np.interp(self.Cmax, [1, 1.2], [0.95, 1.08])
            else:
                raise ValueError("Cmax value is out of the allowed range [1 to 1.5]")

            m = m0 * (math.cos(pg.inclination_angle_radian)) ** n1 * (1 + math.sin(pg.inclination_angle_radian)) ** n2

            # Calculate the K function to make a smooth transition of drift velocity between
            # the bubble-rise and film-flooding stages
            K0_filtered = np.zeros(len(self.C00_filtered))
            for i in range(self.pipe_geometry.num_interfaces):
                if sG0_face[i] <= a1:
                    K0_filtered[i] = 1.53
                elif a1 < sG0_face[i] < a2:
                    K0_filtered[i] = 1.53 + (self.C00_filtered[i] * self.Ku0_filtered[i] - 1.53) / 2 * (1 - math.cos(math.pi * (sG0_face[i] - a1) / (a2 - a1)))
                elif sG0_face[i] >= a2:
                    K0_filtered[i] = self.C00_filtered[i] * self.Ku0_filtered[i]

            # Calculate the adjustment function for the mist flow regime
            # I'm not sure if X should be multiplied by C0 or not.
            Xm1 = 1
            Xm2 = 0.94
            Gm1 = 300
            Gm2 = 700
            alpha = 0.001
            lambdaa = 199
            X0 = sG0_face_filtered * rhoG0_face_filtered / (sG0_face_filtered * rhoG0_face_filtered + (1 - sG0_face_filtered) * rhoL0_face_filtered)   # gas mass fraction [dimensionless]
            G0 = rhoM0_face_filtered * vM0_filtered   # Total mass flux (or total mass flow rate per unit cross-sectional area) [kg/m2/s]
            numerator = np.linalg.det(np.array([[X0, alpha*G0, 1], [Xm1, alpha*Gm1, 1], [Xm2, alpha*Gm2, 1]]))
            denominator = math.sqrt((Xm2 - Xm1)**2 + (alpha*Gm2 - alpha*Gm1)**2)
            Dm = numerator/denominator
            f0 = max(0, 1 - min(1, G0/Gm1) * math.exp(-lambdaa * Dm*abs(Dm)))

            # Calculate drift velocity
            vD0 = np.zeros(self.pipe_geometry.num_interfaces)   # vD0 all zeros first
            for i, idx in enumerate(indices):
                vD0[idx] = (1 - self.C00_filtered[i] * sG0_face_filtered[i]) * self.vC0_filtered[idx] * K0_filtered[i] * m * f0[i] / (self.C00_filtered[i] * sG0_face_filtered[i] * math.sqrt(rhoG0_face_filtered[i] / rhoL0_face_filtered[i]) + 1 - self.C00_filtered[i] * sG0_face_filtered[i])

        self.vD0 = vD0
