"""
Using the classes defined in this module, you can define nodes and node-type equipment in the flow network system.
The difference between these nodes and sources/sinks defined in add_bc_sources_sinks.py is that these nodes
have their own residual equations in order to be defined.

The different between nodes and extra nodes is that nodes are specified on the current segments of the pipe, and so
the residual equation of the segment will be replaced with the new residual equation of the boundary condition. On the
other hand, for extra nodes, an extra node will be connected to the current intended segment of the pipe, and a residual
equation will be written for the new node, so in this case, the total number of residual equations increases.
"""

class ClosedNode:
    """
    This node represents a closed boundary (no-flow boundary).
    I think here, this node is not necessary to be defined in order to have a no-flow boundary condition, but
    in OLGA because the inlet or outlet of the pipe must be connected somewhere, this node is used if there is a
    no-flow inlet or outlet.
    """
    def __init__(self):
        pass

class ConstantPressureNode:
    def __init__(self, pipe_name: str, segment_index: int, start_time: float, stop_time, pressure: float,
                 verbose: bool = False):
        """
        This class is used to keep the pressure of a segment of the pipe at a particular constant pressure.
        :param pipe_name: Name of the pipe (well) for which ConstantPressureNode is going to be added.
        :type pipe_name: str
        :param segment_index: Index of the segment the pressure of which is going to be kept constant
        :type segment_index: int
        :param start_time:
        :type start_time: float
        :param stop_time:
        :type stop_time: float or str
        :param pressure: Pressure value that is intended to be constant over the course of the simulation [Pa]
        :type pressure: float
        :param verbose: Whether to display extra info about ConstantPressureNode
        :type verbose: boolean
        """
        self.pipe_name = pipe_name
        self.segment_index = segment_index
        self.start_time = start_time  # This is not used yet
        self.stop_time = stop_time  # This is not used yet
        self.pressure = pressure

        if verbose:
            print("** ConstantPressureNode on the segment %d of the pipe \"%s\" is added!" % (segment_index, pipe_name))

class ConstantTempNode:
    def __init__(self, pipe_name: str, segment_index: int, start_time: float, stop_time, temperature: float,
                 verbose: bool = False):
        """
        This class is used to keep the temperature of a segment of the pipe at a particular constant temperature.
        :param pipe_name: Name of the pipe (well) for which ConstantTempNode is going to be added.
        :type pipe_name: str
        :param segment_index: Index of the segment the temperature of which is going to be kept constant
        :type segment_index: int
        :param start_time:
        :type start_time: float
        :param stop_time:
        :type stop_time: float or str
        :param temperature: Temperature value that is intended to be constant over the course of the simulation [Kelvin]
        :type temperature: float
        :param verbose: Whether to display extra info about ConstantTempNode
        :type verbose: boolean
        """
        self.pipe_name = pipe_name
        self.segment_index = segment_index
        self.start_time = start_time   # This is not used yet
        self.stop_time = stop_time   # This is not used yet
        self.temperature = temperature

        if verbose:
            print("** ConstantTempNode on the segment %d of the pipe \"%s\" is added!" % (segment_index, pipe_name))

class InternalNode:
    def __init__(self):
        pass

class JunctionNode:
    def __init__(self):
        pass

class Separator:
    def __init__(self):
        pass

class PhaseSplitNode:
    def __init__(self):
        pass

