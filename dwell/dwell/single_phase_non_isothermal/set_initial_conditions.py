from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from dwell.single_phase_non_isothermal.define_pipe_geometry import PipeGeometry
from dwell.single_phase_non_isothermal.define_fluid_model import FluidModel

from dwell.utilities.units import *

g = 9.80665 * meter() / second()**2  # Gravitational acceleration

class LinearAmbientTemperature:
    """
    This class is used when the ambient temperature along the pipe changes linearly, and so the temperature of
    the fluid in the pipe changes linearly.
    """
    def __init__(self, pipe_name: str, pipe_geom: PipeGeometry, fluid_model: FluidModel, WHP: float,
                 fluid_comp: list, WHT: float, GG: float, verbose: bool = False):
        """
        Set initial conditions for a pipe with the name "pipe_name" filled with a static
        column of a particular fluid

        :param pipe_name: Name of the pipe for which the initial conditions are going to be set
        :type pipe_name: str
        :param pipe_geom: Pipe geometry object
        :type pipe_geom: PipeGeometry
        :param fluid_model: Fluid model object
        :type fluid_model: FluidModel
        :param WHP: Initial wellhead pressure [Pascal]
        :type WHP: float
        :param fluid_comp: Initial fluid composition in the pipe in mole fractions
        :type fluid_comp: list of floats [x1, x2, ...], which is used for all the segments or
        list of list of floats [[x1, x2, ...],[x1, x2, ...] , ...], each element of which is used for each segment
        :param WHT: Initial wellhead temperature, defined at the top of the last segment of the pipe [Kelvin]
        :type WHT: float
        :param GG: Geothermal gradient [Kelvin/meter]
        :type GG: float
        :param verbose: Whether to display extra info about LinearAmbientTemperature
        :type verbose: boolean
        """
        assert pipe_name == pipe_geom.pipe_name, \
            "Pipe names for PipeGeometry and LinearAmbientTemperature are not identical!"
        self.pipe_name = pipe_name

        self.pipe_geom = pipe_geom
        self.fluid_model = fluid_model
        self.WHP = WHP

        if all(isinstance(x, float) for x in fluid_comp):
            self.fluid_comp = [fluid_comp for _ in range(pipe_geom.num_segments)]
        elif all(isinstance(x, list) for x in fluid_comp):
            self.fluid_comp = fluid_comp
        else:
            raise TypeError("The initial fluid composition in the pipe is not entered correctly!")

        self.WHT = WHT
        self.GG = GG

        self.measured_depths_segments = sum(self.pipe_geom.segments_lengths) - self.pipe_geom.z
        self.measured_depths_interfaces = sum(self.pipe_geom.segments_lengths) - np.cumsum(self.pipe_geom.segments_lengths[:-1])
        self.true_vertical_depths_segments = self.measured_depths_segments * np.cos(pipe_geom.inclination_angle_radian)
        self.true_vertical_depths_interfaces = self.measured_depths_interfaces * np.cos(pipe_geom.inclination_angle_radian)

        # Get initial conditions
        self.get_initial_temperature_profile()
        self.get_initial_pressure_profile()

        if verbose:
            print("** Initial conditions (LinearAmbientTemperature) of the pipe \"%s\" are set!" % pipe_name)

    def get_initial_temperature_profile(self):
        self.temp_init_segments = np.array(self.WHT + self.GG * self.true_vertical_depths_segments)
        self.temp_init_interfaces = np.array(self.WHT + self.GG * self.true_vertical_depths_interfaces)   # Temperatures at interfaces are calculated even though they're not used in any part of the code.

        temp_init_seg_interfaces = np.zeros(self.pipe_geom.num_segments + self.pipe_geom.num_interfaces)
        temp_init_seg_interfaces[0::2] = self.temp_init_segments
        temp_init_seg_interfaces[1::2] = self.temp_init_interfaces
        self.temp_init_seg_interfaces = temp_init_seg_interfaces   # This is used to calculate pressures at segments and interfaces together even though the pressure values at interfaces are not used in any part of the code, but this variable is used for calculating initial pressure profile along the wellbore more easily.

    def get_initial_pressure_profile(self):
        def dpdz(TVE, p, g):
            temp = temp_func(TVE)
            return - g * self.fluid_model.density_obj.evaluate(p, temp, [1])

        p_top = self.WHP
        TVE_bottom = self.pipe_geom.z[0] * np.cos(self.pipe_geom.inclination_angle_radian)   # TEV stands for true vertical elevation relative to the bottom of the wellbore
        TVE_top = sum(self.pipe_geom.segments_lengths) * np.cos(self.pipe_geom.inclination_angle_radian)
        TVE_seg_interfaces = self.pipe_geom.z_seg_interfaces * np.cos(self.pipe_geom.inclination_angle_radian)
        TVE_seg_face = np.append(TVE_seg_interfaces, TVE_top)
        TVE_seg_face = TVE_seg_face[::-1]
        temp_init_seg_interfaces = np.append(self.temp_init_seg_interfaces, self.WHT)
        temp_init_seg_interfaces = temp_init_seg_interfaces[::-1]

        temp_func = interp1d(TVE_seg_face, temp_init_seg_interfaces, fill_value="extrapolate")
        # Seg and face together
        sol_seg_face = solve_ivp(dpdz, [TVE_top, TVE_bottom], [p_top], args=(g,), t_eval=TVE_seg_face)
        p_seg_interfaces = sol_seg_face.y[0]

        p_seg_interfaces = p_seg_interfaces[::-1]
        self.p_init_segments = p_seg_interfaces[0::2]
        p_init_interfaces = p_seg_interfaces[1::2]   # Pressures at interfaces are calculated. Maybe, they'll be used later.


class SingleAmbientTemperature:
    def __init__(self, pipe_name: str, pipe_geom: PipeGeometry, fluid_model: FluidModel, ambient_temperature: float,
                 pipe_head_pressure: float, pipe_head_segment_index: int, fluid_comp: list, verbose: bool = False):
        """
        This class is used when the ambient temperature along the pipe is a single value, and so the temperature of
        the fluid in the pipe does not change along the pipe.

        :param pipe_name: Name of the pipe for which the initial conditions are going to be set
        :type pipe_name: str
        :param pipe_geom: Pipe geometry object
        :type pipe_geom: PipeGeometry
        :param fluid_model: Fluid model object
        :type fluid_model: FluidModel
        :param ambient_temperature: The temperature of the fluid surrounding the pipe
        :type ambient_temperature: float
        :param pipe_head_pressure: Pressure at one of the heads of the pipe the index of which is specified in pip_head_segment_index
        :type pipe_head_pressure: float
        :param pipe_head_segment_index: The index of the segment of the pipe at which the pipe head pressure is specified. The index starts from 0.
        :type pipe_head_segment_index: int
        :param fluid_comp: Initial fluid composition in the pipe in mole fractions
        :type fluid_comp: list of floats [x1, x2, ...], which is used for all the segments or
        list of list of floats [[x1, x2, ...],[x1, x2, ...] , ...], each element of which is used for each segment
        :param verbose: Whether to display extra info about SingleAmbientTemperature
        :type verbose: boolean
        :return: Initial pressure and temperature profile along the pipe
        """
        assert pipe_name == pipe_geom.pipe_name, \
            "Pipe names for PipeGeometry and SingleAmbientTemperature are not identical!"
        self.pipe_name = pipe_name
        self.pipe_geom = pipe_geom
        self.fluid_model = fluid_model
        self.ambient_temperature = ambient_temperature
        self.pipe_head_pressure = pipe_head_pressure
        self.pipe_head_segment_index = pipe_head_segment_index

        if all(isinstance(x, float) for x in fluid_comp):
            self.fluid_comp = [fluid_comp for _ in range(pipe_geom.num_segments)]
        elif all(isinstance(x, list) for x in fluid_comp):
            self.fluid_comp = fluid_comp
        else:
            raise TypeError("The initial fluid composition in the pipe is not entered correctly!")

        self.measured_depths_segments = sum(self.pipe_geom.segments_lengths) - self.pipe_geom.z
        self.measured_depths_interfaces = sum(self.pipe_geom.segments_lengths) - np.cumsum(self.pipe_geom.segments_lengths[:-1])
        self.true_vertical_depths_segments = self.measured_depths_segments * np.cos(pipe_geom.inclination_angle_radian)
        self.true_vertical_depths_interfaces = self.measured_depths_interfaces * np.cos(pipe_geom.inclination_angle_radian)

        # Get initial conditions
        self.get_initial_temperature_profile()
        self.get_initial_pressure_profile()

        if verbose:
            print("** Initial conditions (SingleAmbientTemperature) of the pipe \"%s\" are set!" % pipe_name)

    def get_initial_temperature_profile(self):
        num_segments = self.pipe_geom.num_segments
        self.temp_init_segments = self.ambient_temperature * np.ones(num_segments)

    def get_initial_pressure_profile(self):
        def dpdz(TVE, p, g, temp):
            return - g * self.fluid_model.density_obj.evaluate(p, temp, [1])

        p_head2 = self.pipe_head_pressure   # Initial solution for the ODE
        if self.pipe_head_segment_index == self.pipe_geom.num_segments - 1:
            TVE_head1 = self.pipe_geom.z[0] * np.cos(self.pipe_geom.inclination_angle_radian)   # TEV stands for true vertical elevation
            TVE_head2 = sum(self.pipe_geom.segments_lengths) * np.cos(self.pipe_geom.inclination_angle_radian)   # TVE of the initial solution p_head2
            TVE_seg_interfaces = self.pipe_geom.z_seg_interfaces * np.cos(self.pipe_geom.inclination_angle_radian)
            TVE_seg_face = np.append(TVE_seg_interfaces, TVE_head2)

            TVE_seg_face = TVE_seg_face[::-1]

            temp = self.ambient_temperature
            # Seg and face together
            sol_seg_face = solve_ivp(dpdz, [TVE_head2, TVE_head1], [p_head2], args=(g, temp), t_eval=TVE_seg_face)
            p_seg_interfaces = sol_seg_face.y[0]

            p_seg_interfaces = p_seg_interfaces[::-1]

        elif self.pipe_head_segment_index == 0:
            TVE_head1 = sum(self.pipe_geom.segments_lengths) * np.cos(self.pipe_geom.inclination_angle_radian)
            TVE_head2 = self.pipe_geom.z[0] * np.cos(self.pipe_geom.inclination_angle_radian)   # TVE of the initial solution (p_head2)
            TVE_seg_interfaces = self.pipe_geom.z_seg_interfaces * np.cos(self.pipe_geom.inclination_angle_radian)
            TVE_seg_face = np.append(TVE_seg_interfaces, TVE_head1)

            temp = self.ambient_temperature
            # Seg and face together
            sol_seg_face = solve_ivp(dpdz, [TVE_head2, TVE_head1], [p_head2], args=(g, temp), t_eval=TVE_seg_face)
            p_seg_interfaces = sol_seg_face.y[0]

        self.p_init_segments = p_seg_interfaces[0::2]
        p_init_interfaces = p_seg_interfaces[1::2]   # Pressures at interfaces are calculated. Maybe, they'll be used later.
