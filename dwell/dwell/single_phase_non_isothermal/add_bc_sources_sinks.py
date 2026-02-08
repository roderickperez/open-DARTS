"""
Sources and sinks are considered as boundary conditions that are directly added to the governing
equations in the intended segment(s)
"""
from dwell.single_phase_non_isothermal.define_pipe_geometry import PipeGeometry
from dwell.utilities.units import *

import numpy as np

class WellLateralHeatTransfer:
    def __init__(self, pipe_name: str, pipe_geometry: PipeGeometry, earth_thermal_props: dict, outermost_layer_OD: float,
                 Ui: float = None, well_layers_props: dict = None, time_function_name: str = "Chiu&Thakur",
                 verbose: bool = False):
        """
        This class defines lateral heat transfer between the wellbore the geometry of which is entered as the first
        input argument of the constructor and the surrounding rock/soil using a semi-analytical lateral heat
        transfer model.
        Note that from the input args "U" and "well_layers_props", only one must be specified.

        :param pipe_name: Name of the pipe (well) for which WellLateralHeatTransfer is added.
        :type pipe_name: str
        :param pipe_geometry: The geometry of the pipe (well) for which lateral heat transfer is intended to be defined
        :type pipe_geometry: PipeGeometry
        :param earth_thermal_props: A dictionary containing earth thermal properties including these keys:
        "T": Earth temperature with the number of elements equal to the number of segments of the wellbore (list)
        "c": Earth specific heat capacity (float or list)
        "K": Earth thermal conductivity (float or list)
        "rho": Earth density (float or list)
        :type earth_thermal_props: dict
        :param Ui: Overall heat transfer coefficient based on the inner pipe diameter. If Ui is not specified,
        well_layers_props must be specified.
        :type Ui: float
        :param well_layers_props: The properties of the layers surrounding the fluid in the wellbore to thermal
        calculations. If well_layers_props is not specified, Ui must be specified.
        :type well_layers_props: dict
        :param time_function_name: The name of the time function used for transient calculation of heat transfer
        Available options are "Ramey" and "Chiu&Thakur". Default is "Chiu&Thakur"
        :type time_function_name: str
        :param verbose: Whether to display extra info about WellLateralHeatTransfer
        :type verbose: boolean
        """
        assert pipe_geometry.pipe_name == pipe_name, \
            "The names of the pipes in PipeGeometry and WellLateralHeatTransfer are not identical!"
        self.well_name = pipe_name

        T_earth = earth_thermal_props["T"]
        assert len(T_earth) == pipe_geometry.num_segments
        self.T_earth = np.array(T_earth)

        c_earth = earth_thermal_props["c"]
        if isinstance(c_earth, float) or isinstance(c_earth, int):
            c_earth = [c_earth] * pipe_geometry.num_segments
        self.c_earth = np.array(c_earth)

        K_earth = earth_thermal_props["K"]
        if isinstance(K_earth, float) or isinstance(K_earth, int):
            K_earth = [K_earth] * pipe_geometry.num_segments
        self.K_earth = np.array(K_earth)

        rho_earth = earth_thermal_props["rho"]
        if isinstance(rho_earth, float) or isinstance(rho_earth, int):
            rho_earth = [rho_earth] * pipe_geometry.num_segments
        self.rho_earth = np.array(rho_earth)

        # Calculate earth thermal diffusivity
        self.alpha = self.K_earth / (self.rho_earth * self.c_earth)

        if well_layers_props is not None:
            # The ID of the smallest pipe specified in well_layers_props
            # must be the same value as the pip_IR in the class PipeGeometry
            self.well_layers_props = well_layers_props
            # Calculate U

        if Ui is not None:
            self.tubing_IR = pipe_geometry.pipe_IR
            self.Ui = Ui

        self.time_function_name = time_function_name
        self.outermost_layer_OD = outermost_layer_OD

        self.segments_lengths = pipe_geometry.segments_lengths

        self.q_lateral_heat = []

        if verbose:
            print("** WellLateralHeatTransfer for the well \"%s\" is added!" % pipe_name)

    def evaluate(self, T_segments, simulation_timer):
        """
        :param T_segments: Fluid temperature inside the segment
        :param simulation_timer: Simulation timer in seconds
        :return Lateral heat rate
        """
        # Time function evaluation
        # outermost_layer_OD is the outside diameter of the outermost layer of the wellbore before the formation, so
        # it could be a casing, a cement sheath, etc.
        if self.time_function_name == "Ramey":
            # Ramey's time function: Gives reasonably good results for long times but fails for times less than seven days.
            f_t = 1 / (- np.log((self.outermost_layer_OD/2) / (2 * np.sqrt(self.alpha * simulation_timer))) - 0.29)
        elif self.time_function_name == "Chiu&Thakur":
            # Chiu and Thakur time function: Provides a reasonable approximation of transient wellbore-formation heat
            # exchange while avoiding the early time discontinuity that results from using Ramey’s time function.
            f_t = 0.982 * np.log(1 + 1.81 * np.sqrt(self.alpha * simulation_timer) / (self.outermost_layer_OD))
        else:
            raise TypeError("Unrecognized time function name " + self.time_function_name)

        # Lateral heat rate evaluation
        if self.Ui is not None:
            # For constant overall heat transfer coefficient
            # I should see if U is based on ID or OD of the pipe. I think it's based on ID.
            self.q_lateral_heat = (2 * np.pi * self.tubing_IR * self.segments_lengths) * self.Ui * (self.T_earth - T_segments) / f_t
            return self.q_lateral_heat
        elif self.well_layers_props is not None:
            # Calculate the overall heat transfer coefficient using Willhite's formula
            U_to = "Willhite's formula"
            r_to = "tubing_outside_radius"
            self.q_lateral_heat = (2 * np.pi * self.K_earth * self.segments_lengths * (self.T_earth - T_segments)
                                   / (f_t + self.K_earth / (r_to * U_to)))
            return self.q_lateral_heat


class ConstantMassRateSource:
    def __init__(self, pipe_name: str, pipe_geom: PipeGeometry, segment_index: int, start_time: float, stop_time,
                 flow_direction: str, mass_rate: float, specific_enthalpy: float, verbose: bool = False):
        """
        If you intend to define ConstantMassRateSource for multiple segments, it must be defined for each of them separately.
        :param pipe_name: Name of the pipe for which ConstantMassRateSource is going to be added.
        :type pipe_name: str
        :param pipe_geom:
        :type pipe_geom: PipeGeometry
        :param segment_index: Index of the segment on which ConstantMassRateSource is going to be added. Index starts from zero.
        :type segment_index: int
        :param start_time: The time at which ConstantMassRateSource is going to be added to the segment [second]
        :type start_time: float
        :param stop_time: The time at which ConstantMassRateSource is going to be stopped. The input must be a float in seconds,
        but if the source is not stopped till the end of the simulation, "end" can be assigned to the variable.
        :type stop_time: float or str ("end")
        :param flow_direction: "inflow" or "outflow"
        :type flow_direction: str
        :param mass_rate:
        :type mass_rate: float
        :param specific_enthalpy:
        :type specific_enthalpy: float
        :param verbose: Whether to display extra info about ConstantMassRateSource
        :type verbose: boolean
        """
        self.pipe_name = pipe_name
        self.pipe_geom = pipe_geom
        self.segment_index = segment_index
        self.start_time = start_time
        self.stop_time = stop_time
        self.flow_direction = flow_direction

        self.specific_enthalpy = specific_enthalpy
        if flow_direction == 'inflow':
            self.mass_rate = mass_rate
            self.enthalpy_rate = mass_rate * specific_enthalpy
        elif flow_direction == 'outflow':
            self.mass_rate = - mass_rate
            self.enthalpy_rate = - mass_rate * specific_enthalpy
        else:
            raise Exception('Invalid flow direction for ConstantMassRateSource!')

        if verbose:
            print("** ConstantMassRateSource for the segment %d of the pipe \"%s\" is added!" % (segment_index, pipe_name))

    def evaluate_mass_rate(self, simulation_timer):
        source_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                       (self.start_time <= simulation_timer <= self.stop_time))
        if source_bool == True:
            return self.mass_rate
        else:
            return 0

    def evaluate_enthalpy_rate(self, simulation_timer):
        source_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                       (self.start_time <= simulation_timer <= self.stop_time))
        if source_bool == True:
            return self.enthalpy_rate
        else:
            return 0

    def evaluate_momentum0(self, simulation_timer, sG0, rhoG0, rhoL0):   # I'm not sure if this simulation_timer with previous time step (0) is going to work correctly
        source_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                       (self.start_time <= simulation_timer <= self.stop_time))

        if source_bool == True:
            mass_rate = self.mass_rate
            pipe_internal_A = self.pipe_geom.pipe_internal_A

            if sG0 <= 0.001:
                vG0 = 0
            else:
                gas_mass_fraction = 1   # This should be modified later for 2-phase flow
                gas_mass_rate = mass_rate * gas_mass_fraction
                vG0 = gas_mass_rate / rhoG0 / (pipe_internal_A * sG0)

            if sG0 >= 0.999:
                vL0 = 0
            else:
                liquid_mass_fraction = 1   # This should be modified later for 2-phase flow
                liquid_mass_rate = mass_rate * liquid_mass_fraction
                vL0 = liquid_mass_rate / rhoL0 / (pipe_internal_A * (1 - sG0))

            delta_at_bc_face0 = pipe_internal_A * (rhoG0 * sG0 * vG0 ** 2 + rhoL0 * (1 - sG0) * vL0 ** 2)
            return delta_at_bc_face0
        else:
            return 0

    def evaluate_specific_kinetic_energy0(self, simulation_timer, sG0, rhoG0, rhoL0):   # I'm not sure if this simulation_timer with previous time step (0) is going to work correctly
        """
        :param simulation_timer:
        :param sG0: The saturation of the gas at the previous time step in the segment for which the source/sink is defined.
        :param rhoG0: The density of the gas at the previous time step in the segment for which the source/sink is defined.
        :param rhoL0: The density of the liquid at the previous time step in the segment for which the source/sink is defined.
        :return:
        """
        source_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                       (self.start_time <= simulation_timer <= self.stop_time))

        if source_bool == True:
            mass_rate = self.mass_rate
            wellbore_area = self.pipe_geom.pipe_internal_A

            if sG0 <= 0.001:
                vG0 = 0
            else:
                gas_mass_fraction = 1  # This should be modified later for 2-phase flow
                gas_mass_rate = mass_rate * gas_mass_fraction
                vG0 = gas_mass_rate / rhoG0 / (wellbore_area * sG0)

            if sG0 >= 0.999:
                vL0 = 0
            else:
                liquid_mass_fraction = 1  # This should be modified later for 2-phase flow
                liquid_mass_rate = mass_rate * liquid_mass_fraction
                vL0 = liquid_mass_rate / rhoL0 / (wellbore_area * (1 - sG0))

            specific_KEg0 = vG0 ** 2 / 2
            specific_KEl0 = vL0 ** 2 / 2

            return specific_KEg0, specific_KEl0
        else:
            return 0, 0

    def evaluate_kinetic_energy_rate(self, simulation_timer, sG, rhoG, rhoL):
        source_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                       (self.start_time <= simulation_timer <= self.stop_time))

        if source_bool == True:
            mass_rate = self.mass_rate
            wellbore_area = self.pipe_geom.pipe_internal_A

            if sG <= 0.001:
                vG = 0
                gas_mass_rate = 0
            else:
                gas_mass_fraction = 1  # This should be modified later for 2-phase flow
                gas_mass_rate = mass_rate * gas_mass_fraction
                vG = gas_mass_rate / rhoG / (wellbore_area * sG)

            if sG >= 0.999:
                vL = 0
                liquid_mass_rate = 0
            else:
                liquid_mass_fraction = 1  # This should be modified later for 2-phase flow
                liquid_mass_rate = mass_rate * liquid_mass_fraction
                vL = liquid_mass_rate / rhoL / (wellbore_area * (1 - sG))

            specific_KEg = vG ** 2 / 2
            specific_KEl = vL ** 2 / 2

            gas_KE_rate = specific_KEg * gas_mass_rate
            liquid_KE_rate = specific_KEl * liquid_mass_rate

            return gas_KE_rate, liquid_KE_rate
        else:
            return 0, 0


    def evaluate_potential_energy_rate(self, simulation_timer, specific_potential_energy):
        source_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                       (self.start_time <= simulation_timer <= self.stop_time))
        if source_bool == True:
            PE_rate = self.mass_rate * specific_potential_energy[self.segment_index]
            return PE_rate
        else:
            return 0

class PressureDrivenSource:
    def __init__(self, pipe_name, verbose: bool = False):
        """
        :param pipe_name: Name of the pipe for which PressureDrivenSource is going to be added.
        :type pipe_name: str
        :param verbose: Whether to display extra info about PressureDrivenSource
        :type verbose: boolean
        """
        pass

    def evaluate(self):
        pass


class Perforation:
    """
    This class is used to define a perforation between a wellbore and a reservoir.
    """
    def __init__(self, pipe_name, segment_index, start_time, stop_time, pipe_geom, reservoir_pressure,
                 reservoir_cell_perm_list, reservoir_cell_size_list, perforation_hole_diameter, skin=0, verbose: bool = False):
        """
        :param pipe_name: Name of the pipe (well) for which Perforation is going to be added.
        :type pipe_name: str
        :param segment_index: Index of the segment for which Perforation is going to be added.
        :type segment_index: int
        :param start_time:
        :param stop_time:
        :param pipe_geom: Geometry of the wellbore for which Perforation is going to be added.
        :type pipe_geom: PipeGeometry
        :param reservoir_pressure: Pressure of the reservoir cell in which the perforated wellbore segment is located [Pa]
        :type reservoir_pressure: float
        :param reservoir_cell_perm_list: Permeabilities of the reservoir cell in which the perforated wellbore segment is located [meter^2]
        :type reservoir_cell_perm_list: list ---> [Kx, Ky]
        :param reservoir_cell_size_list: Size of the reservoir cell in which the perforated wellbore segment is located [meter]
        in each direction (the third element of the list, which is the thickness of the reservoir cell, must be equal to
        or smaller than the length of the wellbore segment that is connected to that reservoir cell)
        :type reservoir_cell_size_list: list ---> [dx, dy, dz]
        :param perforation_hole_diameter: Diameter of the perforation [meter]
        :type perforation_hole_diameter: float
        :param skin: Skin factor of the formation [dimensionless]
        :type skin: float or int
        :param verbose: Whether to display extra info about Perforation
        :type verbose: boolean
        """
        assert pipe_geom.pipe_name == pipe_name, \
            "The names of the pipes for the geometry and Perforation are not identical!"
        self.pipe_name = pipe_name

        self.segment_index = segment_index
        self.start_time = start_time
        self.stop_time = stop_time

        self.well_radius = pipe_geom.pipe_ID / 2
        self.reservoir_pressure = reservoir_pressure
        self.reservoir_cell_perm_list = reservoir_cell_perm_list

        assert reservoir_cell_size_list[2] <= pipe_geom.segments_lengths[segment_index], \
            "The thickness of the reservoir cell must not be larger than the length of the pipe segment!"
        self.reservoir_cell_size_list = reservoir_cell_size_list

        self.perforation_hole_diameter = perforation_hole_diameter
        self.skin = float(skin)

        if verbose:
            print("** Perforation for the segment %d of the well \"%s\" is added!" % (segment_index, pipe_name))

    def evaluate_mass_rate(self, simulation_timer, iter_counter, total_iter_counter, flag, segment_pressure, rhoG_up_at_perf, rhoL_up_at_perf, miuG_up_at_perf,
                      miuL_up_at_perf, KrG_up_at_perf, KrL_up_at_perf):
        """
        :param simulation_timer: Simulation time up to now
        :param segment_pressure: Pressure of the segment
        :param rhoG_up_at_perf: Upwinded gas density at the perforation [kg/m3]
        :param rhoL_up_at_perf: Upwinded liquid density at the perforation [kg/m3]
        :param miuG_up_at_perf: Upwinded gas viscosity at the perforation [Pa.s]
        :param miuL_up_at_perf: Upwinded liquid viscosity at the perforation [Pa.s]
        :param KrG_up_at_perf: Upwinded gas relative permeability at the perforation [dimensionless]
        :param KrL_up_at_perf: Upwinded liquid relative permeability at the perforation [dimensionless]
        :return: Gas and liquid mass rates [kg/second]
        """
        if iter_counter == 0 and total_iter_counter == 0 and flag == 1:
            self.gas_mass_rate_at_perf0 = 0
            self.liquid_mass_rate_at_perf0 = 0
        elif iter_counter == 0 and total_iter_counter != 0 and flag == 1:
            self.gas_mass_rate_at_perf0 = self.gas_mass_rate_at_perf
            self.liquid_mass_rate_at_perf0 = self.liquid_mass_rate_at_perf

        perforation_bool = ((self.stop_time == "end" and self.start_time <= simulation_timer) or
                            (self.start_time <= simulation_timer <= self.stop_time))

        if perforation_bool == True:
            Kx = self.reservoir_cell_perm_list[0]
            Ky = self.reservoir_cell_perm_list[1]
            dx = self.reservoir_cell_size_list[0]
            dy = self.reservoir_cell_size_list[1]
            dz = self.reservoir_cell_size_list[2]

            # Use Peaceman's model to calculate the equivalent radius
            if Kx != Ky:
                re = 0.28 * (np.sqrt(Ky/Kx) * dx ** 2 + np.sqrt(Kx/Ky) * dy ** 2) ** 0.5 / ((Ky/Kx) ** 0.25 + (Kx/Ky) ** 0.25)
                K = np.sqrt(Kx * Ky)
            elif Kx == Ky:
                re = 0.14 * np.sqrt(dx ** 2 + dy ** 2)
                K = Kx

            rw = self.well_radius
            # This paper "Advanced well model for superhot and saline geothermal reservoirs" does not use either -1/2
            # or -3/4 in the denominator
            # open-DARTS also does not use either -1/2 or -3/4 in the denominator
            # MRST also does not use either -1/2 or -3/4 in the denominator
            well_index_gas = (rhoG_up_at_perf * KrG_up_at_perf / miuG_up_at_perf) * 2 * np.pi * K * dz / (np.log(re / rw) + self.skin)
            well_index_liquid = (rhoL_up_at_perf * KrL_up_at_perf / miuL_up_at_perf) * 2 * np.pi * K * dz / (np.log(re / rw) + self.skin)

            # PI = well_index_gas / rhoG_up_at_perf * 24 * 60 * 60 * 1e5   # gives PI in m3/day/bar

            gas_mass_rate_at_perf = well_index_gas * (self.reservoir_pressure - segment_pressure)
            liquid_mass_rate_at_perf = well_index_liquid * (self.reservoir_pressure - segment_pressure)

            # Store mass rates in self in order to use in evaluate_enthalpy_rate method
            self.gas_mass_rate_at_perf = gas_mass_rate_at_perf
            self.liquid_mass_rate_at_perf = liquid_mass_rate_at_perf

        elif perforation_bool == False:
            # Store mass rates in self in order to use in evaluate_enthalpy_rate method
            self.gas_mass_rate_at_perf = 0
            self.liquid_mass_rate_at_perf = 0

        return self.gas_mass_rate_at_perf, self.liquid_mass_rate_at_perf

    def evaluate_enthalpy_rate(self, hG_up_at_perf, hL_up_at_perf):
        """
        :param hG_up_at_perf: Upwinded specific enthalpy of the gas at the perforation [Joule/kg]
        :param hL_up_at_perf: Upwinded specific enthalpy of the liquid at the perforation [Joule/kg]
        :return Gas and liquid enthalpy rates [Joule/second]
        """
        # Because evaluate_mass_rate is called before this method, gas_mass_rate_at_perf and liquid_mass_rate_at_perf are already stored in the instance of the class.
        gas_enthalpy_rate = hG_up_at_perf * self.gas_mass_rate_at_perf
        liquid_enthalpy_rate = hL_up_at_perf * self.liquid_mass_rate_at_perf
        return gas_enthalpy_rate, liquid_enthalpy_rate

    # def evaluate_momentum0(self, sG0_up, rhoG0_up, rhoL0_up):
    #     gas_mass_rate0 = self.gas_mass_rate_at_perf0
    #     liquid_mass_rate0 = self.liquid_mass_rate_at_perf0
    #
    #     perforation_hole_area = np.pi / 4 * self.perforation_hole_diameter ** 2
    #
    #     if sG0_up <= 0.001:
    #         vG0 = 0
    #     else:
    #         vG0 = gas_mass_rate0 / rhoG0_up / (perforation_hole_area * sG0_up)
    #     if sG0_up >= 0.999:
    #         vL0 = 0
    #     else:
    #         vL0 = liquid_mass_rate0 / rhoL0_up / (perforation_hole_area * (1 - sG0_up))   # I should think if 1 - sG0_up is correct or not. Maybe I need to get sL0_up separately depending on the sign of the liquid velocity.
    #
    #     delta_at_perf0 = perforation_hole_area * (rhoG0_up * sG0_up * vG0 ** 2 +
    #                                               rhoL0_up * (1 - sG0_up) * vL0 ** 2)
    #     return delta_at_perf0

    def evaluate_specific_kinetic_energy0(self, sG0_up, sL0_up, rhoG0_up, rhoL0_up):
        gas_mass_rate0 = self.gas_mass_rate_at_perf0
        liquid_mass_rate0 = self.liquid_mass_rate_at_perf0

        perforation_hole_area = np.pi / 4 * self.perforation_hole_diameter ** 2

        if sG0_up <= 0.001:
            vG0 = 0
        else:
            vG0 = gas_mass_rate0 / rhoG0_up / (perforation_hole_area * sG0_up)
        if sG0_up >= 0.999:
            vL0 = 0
        else:
            vL0 = liquid_mass_rate0 / rhoL0_up / (perforation_hole_area * sL0_up)  # I should think if 1 - sG0_up is correct or not. Maybe I need to get sL0_up separately depending on the sign of the liquid velocity.

        specific_KE_gas0 = vG0 ** 2 / 2
        specific_KE_liquid0 = vL0 ** 2 / 2

        return specific_KE_gas0, specific_KE_liquid0

    def evaluate_kinetic_energy_rate(self, sG_up, sL_up, rhoG_up, rhoL_up):
        gas_mass_rate = self.gas_mass_rate_at_perf
        liquid_mass_rate = self.liquid_mass_rate_at_perf

        perforation_hole_area = np.pi / 4 * self.perforation_hole_diameter ** 2

        if sG_up <= 0.001:
            vG = 0
        else:
            vG = gas_mass_rate / rhoG_up / (perforation_hole_area * sG_up)
        if sG_up >= 0.999:
            vL = 0
        else:
            vL = liquid_mass_rate / rhoL_up / (perforation_hole_area * sL_up)  # I should think if 1 - sG0_up is correct or not. Maybe I need to get sL0_up separately depending on the sign of the liquid velocity.

        specific_KE_gas = vG ** 2 / 2
        specific_KE_liquid = vL ** 2 / 2

        gas_KE_rate = specific_KE_gas * gas_mass_rate
        liquid_KE_rate = specific_KE_liquid * liquid_mass_rate

        return gas_KE_rate, liquid_KE_rate

    def evaluate_potential_energy_rate(self, specific_potential_energy):
        gas_mass_rate = self.gas_mass_rate_at_perf
        liquid_mass_rate = self.liquid_mass_rate_at_perf
        gas_PE_rate = gas_mass_rate * specific_potential_energy[self.segment_index]
        liquid_PE_rate = liquid_mass_rate * specific_potential_energy[self.segment_index]
        return gas_PE_rate, liquid_PE_rate

class NearWell:
    """
    This class defines a link between a section in DWell and a boundary/grid block in a reservoir model. DWell delivers
    boundary conditions to the reservoir model, and the injection into (or production from) the reservoir is received
    by DWell from the reservoir model as mass rate for each phase.

    Applications: The link provides coupled transient simulation of near-wellbore reservoir flow and well flow.
    Dynamic phenomena not accurately predicted or even not seen with the steady-state inflow performance relationship
    model may therefore be simulated in a more realistic way. Typical examples are well shut-in and start-up, dynamic
    gas and water coning, cross-flow between layers, etc.

    Note: There is no automatic check of the correspondence in positioning and size between segments in DWell and
    boundary grid blocks in the reservoir simulators. You must therefore discretize both grids in such a way that this
    is satisfied.
    """
    def __init__(self, pipe_name, verbose: bool = False):
        """
        :param pipe_name: Name of the pipe (well) for which NearWell is going to be added.
        :type pipe_name: str
        :param verbose: Whether to display extra info about NearWell
        :type verbose: boolean
        """
        pass

    def evaluate(self):
        pass


class TracerSource:
    def __init__(self, pipe_name, verbose: bool = False):
        """
        :param pipe_name: Name of the pipe for which TracerSource is going to be added.
        :type pipe_name: str
        :param verbose: Whether to display extra info about TracerSource
        :type verbose: boolean
        """
        pass

    def evaluate(self):
        pass
