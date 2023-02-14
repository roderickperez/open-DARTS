import numpy as np
from darts.engines import *
from darts.models.physics.physics_base import PhysicsBase

from .operator_evaluator_sup import *

# Define our own operator evaluator class
class Compositional(PhysicsBase):
    class operators_storage:
        def __init__(self):
            self.reservoir_operators = {}
            self.wellbore_operators = {}
            self.rate_operators = {}
            self.property_operators = {}

    def __init__(self, property_container, timer, n_points, min_p, max_p, min_z, max_z, min_t=-1, max_t=-1, thermal=0,
                 platform='cpu', itor_type='multilinear', itor_mode='adaptive', itor_precision='d', cache=False):
        super().__init__(cache)
        # Obtain properties from user input during initialization:
        self.timer = timer.node["simulation"]
        self.components = property_container.components_name
        # self.nc = property_container.nc
        self.nc = property_container.nelem
        self.phases = property_container.phases_name
        self.nph = property_container.nph
        self.n_vars = self.nc + thermal
        NE = self.n_vars
        self.vars = ['pressure', 'Temp']
        self.n_points = n_points

        self.n_axes_points = index_vector([n_points] * self.n_vars)

        self.phases = property_container.phases_name

        operators = self.set_operators(property_container, thermal)


        """ Name of interpolation method and engine used for this physics: """
        # engine including gravity term

        self.n_ops = NE + self.nph * NE + self.nph + self.nph * NE + NE + 3 + 2 * self.nph + 1

        if thermal:
            self.vars = ['pressure'] + self.components[:-1] + ['temperature']
            self.n_axes_min = value_vector([min_p] + [min_z] * (self.nc - 1) + [min_t])
            self.n_axes_max = value_vector([max_p] + [max_z] * (self.nc - 1) + [max_t])
            self.engine = eval("engine_super_%s%d_%d_t" % (platform, self.nc, self.nph))()
        else:
            self.vars = ['pressure'] + self.components[:-1]
            self.n_axes_min = value_vector([min_p] + [min_z] * (self.nc - 1))
            self.n_axes_max = value_vector([max_p] + [max_z] * (self.nc - 1))
            self.engine = eval("engine_super_%s%d_%d" % (platform, self.nc, self.nph))()

        self.acc_flux_itor = {}
        for i in operators.reservoir_operators.keys():
            self.acc_flux_itor[i] = self.create_interpolator(operators.reservoir_operators[i], self.n_vars,
                                                             self.n_ops, self.n_axes_points,
                                                             self.n_axes_min, self.n_axes_max, platform=platform)
            self.create_itor_timers(self.acc_flux_itor[i], 'reservoir %d interpolation' % i)

        self.acc_flux_w_itor = self.create_interpolator(operators.wellbore_operators, self.n_vars, self.n_ops,
                                                        self.n_axes_points,
                                                        self.n_axes_min, self.n_axes_max, platform=platform)

        self.property_itor = self.create_interpolator(operators.property_operators, self.n_vars, self.n_ops,
                                                      self.n_axes_points, self.n_axes_min, self.n_axes_max,
                                                      platform=platform)


        self.rate_itor = self.create_interpolator(operators.rate_operators, self.n_vars, self.nph, self.n_axes_points,
                                              self.n_axes_min, self.n_axes_max, platform=platform)

        self.create_itor_timers(self.acc_flux_w_itor, 'wellbore interpolation')
        self.create_itor_timers(self.rate_itor, 'well controls interpolation')

        # define well control factories
        # Injection wells (upwind method requires both bhp and inj_stream for bhp controlled injection wells):
        self.new_bhp_inj = lambda bhp, inj_stream: bhp_inj_well_control(bhp, value_vector(inj_stream))
        self.new_rate_inj = lambda rate, inj_stream, iph: rate_inj_well_control(self.phases, iph, self.n_vars,
                                                                               self.n_vars, rate,
                                                                               value_vector(inj_stream), self.rate_itor)
        # Production wells:
        self.new_bhp_prod = lambda bhp: bhp_prod_well_control(bhp)
        self.new_rate_prod = lambda rate, iph: rate_prod_well_control(self.phases, iph, self.n_vars,
                                                                      self.n_vars,
                                                                     rate, self.rate_itor)

    # Define some class methods:
    def init_wells(self, wells):
        for w in wells:
            assert isinstance(w, ms_well)
            w.init_rate_parameters(self.n_vars, self.phases, self.rate_itor)

    def set_uniform_initial_conditions(self, mesh, uniform_pressure, uniform_composition: list):
        assert isinstance(mesh, conn_mesh)

        nb = mesh.n_blocks
        """ Uniform Initial conditions """
        # set initial pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

         # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        # composition[:] = np.array(uniform_composition)
        if self.nc == 2:
            for c in range(self.nc - 1):
                composition[c::(self.nc - 1)] = uniform_composition[:]
        else:
            for c in range(self.nc - 1):  # Denis
                composition[c::(self.nc - 1)] = uniform_composition[c]


    def set_uniform_T_initial_conditions(self, mesh, uniform_pressure, uniform_composition, uniform_temp: list):
        """""
        Function to set uniform initial reservoir condition
        Arguments:
            -mesh: mesh object
            -uniform_pressure: uniform pressure setting
            -uniform_composition: uniform uniform_composition setting
        """
        assert isinstance(mesh, conn_mesh)
        nb = mesh.n_blocks

        # set inital pressure
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        temperature = np.array(mesh.temperature, copy=False)
        temperature.fill(uniform_temp)

        # set initial composition
        mesh.composition.resize(nb * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.nc - 1):
            composition[c::(self.nc - 1)] = uniform_composition[c]

    def set_boundary_conditions(self, mesh, uniform_pressure, uniform_composition):
        assert isinstance(mesh, conn_mesh)

        # Class methods which can create constant pressure and composition boundary condition:
        pressure = np.array(mesh.pressure, copy=False)
        pressure.fill(uniform_pressure)

        mesh.composition.resize(mesh.n_blocks * (self.nc - 1))
        composition = np.array(mesh.composition, copy=False)
        for c in range(self.nc - 1):
            composition[c::(self.nc - 1)] = uniform_composition[c]

    def set_operators(self, property_container, thermal):  # default definition of operators

        operators = self.operators_storage()

        if thermal:
            operators.reservoir_operators[0] = ReservoirThermalOperators(property_container)
            operators.wellbore_operators = ReservoirThermalOperators(property_container)
        else:
            operators.reservoir_operators[0] = ReservoirOperators(property_container)
            operators.wellbore_operators = WellOperators(property_container)

        operators.rate_operators = RateOperators(property_container)

        operators.property_operators = DefaultPropertyEvaluator(property_container)

        return operators