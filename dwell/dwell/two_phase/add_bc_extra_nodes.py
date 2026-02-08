"""
Using the classes defined in this module, you can define nodes and node-type equipment in the flow network system.
The difference between these nodes and sources/sinks defined in add_bc_sources_sinks.py is that these nodes
have their own residual equations in order to be defined.

The different between nodes and extra nodes is that nodes are specified on the current segments of the pipe, and so
the residual equation of the segment will be replaced with the new residual equation of the boundary condition. On the
other hand, for extra nodes, an extra node will be connected to the current intended segment of the pipe, and a residual
equation will be written for the new node, so in this case, the total number of residual equations increases.
"""

import numpy as np

from dwell.two_phase.define_pipe_geometry import PipeGeometry

# class ConstantMassRateExtraNode:
#     """
#     The node represents a mass flow boundary.
#     ConstantMassRateExtraNode is very similar to ConstantMassRateSource in add_bc_sources_sinks.py. The main exception
#     is that ConstantMassRateExtraNode has an internal pressure calculation for obtaining the correct pressure
#     that will set up the user given mass flow rate into the connected pipe.
#     """
#     def __init__(self, pipe_name: str, pipe_geom: PipeGeometry, segment_index: int, start_time: float, stop_time,
#                  flow_direction: str, mass_rate: float, specific_enthalpy: float, verbose: bool = False):
#         self.pipe_name = pipe_name
#         self.pipe_geom = pipe_geom
#         self.segment_index = segment_index
#         self.start_time = start_time
#         self.stop_time = stop_time
#         self.flow_direction = flow_direction
#
#         self.specific_enthalpy = specific_enthalpy
#         if flow_direction == 'inflow':
#             self.mass_rate = mass_rate
#             self.enthalpy_rate = mass_rate * specific_enthalpy
#         elif flow_direction == 'outflow':
#             self.mass_rate = - mass_rate
#             self.enthalpy_rate = - mass_rate * specific_enthalpy
#         else:
#             raise Exception('Invalid flow direction for MassFlowSource!')
#
#     def evaluate_residual(self, residuals):
#         # The pressure of the node must be added to the list the primary variables.
#         # The mass node residual must be added to the list of the residuals.
#         # Maybe, it's a good idea if the topmost segment of the wellbore is considered as the wellhead.
#         A = self.pipe_geom.pipe_internal_A
#         vG, vL = "calculated from the momentum equation"
#         # residuals[] = self.mass_rate - (vG * A * sG * rhoG + vL * A * sL * rhoL)

class ConstantPressureExtraNode:
    def __init__(self, pipe_name: str, segment_index: int, start_time: float, stop_time, pressure: float,
                 flow_direction: str, initial_conditions: dict, verbose: bool = False):
        """
        This class is used to add a ghost segment with a constant pressure to a segment of the pipe.
        :param pipe_name: Name of the pipe (well) for which ConstantPressureExtraNode is going to be added.
        :type pipe_name: str
        :param segment_index: Index of the segment to which a ghost segment with a constant pressure is going to be added.
        :type segment_index: int
        :param start_time:
        :type start_time: float
        :param stop_time:
        :type stop_time: float or str
        :param pressure: Pressure value that is intended to be constant over the course of the simulation [Pa]
        :type pressure: float
        :param flow_direction: Whether the fluid is flowing in (inflow) or out of the pipe (outflow)
        :type flow_direction: str
        :param initial_conditions: A dict containing all the required initial conditions of the wellbore/pipe.
        :type initial_conditions: dict
        :param verbose: Whether to display extra info about ConstantPressureNode
        :type verbose: boolean
        """
        self.pipe_name = pipe_name
        self.segment_index = segment_index
        self.start_time = start_time  # This is not used yet
        self.stop_time = stop_time  # This is not used yet
        self.ghost_segment_pressure = pressure
        self.flow_direction = flow_direction

        initial_conditions["ConstantPressureExtraNode"] = pressure
        self.initial_conditions = initial_conditions

        count = 0
        for key, value in initial_conditions.items():
            if isinstance(value, np.ndarray):
                count += len(value)
            elif isinstance(value, (int, float)):
                count += 1
            else:
                raise Exception("Unexpected data type!")
        self.variable_index = count - 1

        if verbose:
            print("** ConstantPressureExtraNode on the segment %d of the pipe \"%s\" is added!" % (segment_index, pipe_name))

    def evaluate_phase_mass_rate(self, p_segment, phase_miu, phase_rho, Kr):
        phase_mass_rate = 1e-9 * phase_rho * Kr / phase_miu * (self.ghost_segment_pressure - p_segment)
        return phase_mass_rate

class ConstantTempExtraNode:
    def __init__(self):
        pass
