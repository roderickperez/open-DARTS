"""
Using the classes defined in this module, you can define nodes and node-type equipment in the flow network system.
The difference between these nodes and sources/sinks defined in add_bc_sources_sinks.py is that these nodes
have their own residual equations in order to be defined.

The different between nodes and extra nodes is that nodes are specified on the current segments of the pipe, and so
the residual equation of the segment will be replaced with the new residual equation of the boundary condition. On the
other hand, for extra nodes, an extra node will be connected to the current intended segment of the pipe, and a residual
equation will be written for the new node, so in this case, the total number of residual equations increases.
"""

from dwell.single_phase_non_isothermal.define_pipe_geometry import PipeGeometry

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
    def __init__(self):
        pass

class ConstantTempExtraNode:
    def __init__(self):
        pass
