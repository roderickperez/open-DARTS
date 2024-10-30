# from reservoir import StructReservoir
from conversions import convert_composition, correct_composition, calculate_injection_stream, \
    get_mole_fractions, convert_rate
from own_physics import OwnPhysicsClass
from own_properties import *

from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from darts.engines import *
from darts.physics import *

import numpy as np
import pickle
import os
from math import fabs


# Definition of your input parameter data structure, change as you see fit (when you need more constant values, etc.)!!
class MyOwnDataStruct:
    def __init__(self, nc, zmin, temp, stoich_matrix, pressure_init, c_r, kin_fact,  exp_w=1, exp_g=1):
        """
        Data structure class which holds various input parameters for simulation
        :param nc: number of components used in simulation
        :param zmin: actual 0 used for composition (ussualy >0, around some small epsilon)
        :param temp: temperature
        """
        self.num_comp = nc
        self.min_z = zmin
        self.temperature = temp
        self.stoich_matrix = stoich_matrix
        self.exp_w = exp_w
        self.exp_g = exp_g
        self.pressure_init = pressure_init
        self.c_r = c_r
        self.kin_fact = kin_fact


# Actual Model class creation here!
class Model(DartsModel):
    def __init__(self, nx):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        # Static parameters input:
        self.nx = nx
        self.ny = 1
        self.nz = 1
        self.dx = 0.1 / nx      # m
        self.dy = 0.001000      # m
        self.dz = 0.058905      # m
        self.volume = self.dx * self.dy * self.dz

        self.depth = 1                      # m
        self.poro = 1                       # [-]
        self.temperature = 323.15           # K
        self.pressure_init = 100            # bar

        self.solid_sat = np.ones(self.nx) * 0.7
        self.const_perm = 1.25e4 * self.poro ** 4

        self.inj_rate = convert_rate(0.153742)     # input: ml/min; output: m3/day

        self.c_r = 1e-6
        # self.kin_fact = (1 + self.c_r * (self.pressure_init - 1)) * 2710 / 0.1000869 * np.mean(self.solid_sat)
        self.kin_fact = 1

        self.comp_min = 1e-11
        self.obl_min = self.comp_min / 10

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=self.dx, dy=self.dy,
                                         dz=self.dz, permx=self.const_perm, permy=self.const_perm,
                                         permz=self.const_perm, poro=self.poro, depth=self.depth)

        # Several parameters here related to components used, OBL limits, and injection composition:
        self.cell_property = ['pressure', 'H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+',
                              'CaHCO3+', 'Solid']
        self.phases = ['liq', 'gas']
        self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid']
        self.elements = ['Solid', 'Ca', 'C', 'O', 'H']
        self.num_vars = len(self.elements)
        self.n_points = 201
        self.min_p = 95
        self.max_p = 105
        self.min_z = self.obl_min
        self.max_z = 1 - self.obl_min

        # Rate annihilation matrix
        E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                      [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0],
                      [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]])

        # Several parameters related to kinetic reactions:
        stoich_matrix = np.array([-1, 1, 1, 3, 0])
        trans_mult_exp = 4

        # Create instance of data-structure for simulation (and chemical) input parameters:
        input_data_struct = MyOwnDataStruct(len(self.elements), self.comp_min, self.temperature, stoich_matrix,
                                            self.pressure_init, self.c_r, self.kin_fact)

        # Create property containers:
        self.property_container = property_container(self.phases, self.components)

        self.property_container.phase_name = ['liq', 'gas']

        self.property_container.rel_perm_ev = dict([('liq', custom_rel_perm(2)), ('gas', custom_rel_perm(2))])

        # PHREEQC
        self.phreeqc_db_path = 'phreeqc.dat'
        self.property_container.flash_ev = custom_flash(self.temperature, self.phreeqc_db_path, self.comp_min)
        self.property_container.init_flash_ev = init_flash(self.temperature, self.phreeqc_db_path, self.comp_min)

        self.property_container.kin_rate_ev = custom_kinetic_rate(self.temperature, self.comp_min)

        # Create instance of (own) physics class:
        self.physics = OwnPhysicsClass(self.timer, self.elements, self.n_points, self.min_p, self.max_p,
                                       self.min_z, input_data_struct, self.property_container)

        # Compute injection stream
        mole_water, mole_co2 = calculate_injection_stream(1.1, 0.1, self.temperature, self.pressure_init)
        mole_fraction_water, mole_fraction_co2 = get_mole_fractions(mole_water, mole_co2)

        # Define injection stream composition,
        # ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid']
        self.inj_stream_components = np.array([mole_fraction_water, 0, 0, mole_fraction_co2, 0, 0, 0, 0, 0, 0, 0])
        self.inj_stream = convert_composition(self.inj_stream_components, E)
        self.inj_stream = correct_composition(self.inj_stream, self.comp_min)

        # ====================================== Initialize reservoir composition ======================================
        print('\nInitializing compositions...')

        # Component-defined composition of a non-solid phase (pure water here)
        self.initial_comp_components = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.solid_frac = np.zeros(len(self.solid_sat))
        self.initial_comp = np.zeros((self.nx * self.ny * self.nz + 4, self.num_vars - 1))

        # Interpolated values of non-solid volume (second value always 0 due to no (5,1) interpolator)
        values = value_vector([0] * 2)

        # Iterate over solid saturation and call interpolator
        for i in range(len(self.solid_sat)):
            # There are 5 values in the state
            composition_full = convert_composition(self.initial_comp_components, E)
            composition = correct_composition(composition_full, self.comp_min)
            init_state = value_vector(np.hstack((self.solid_sat[i], composition)))

            # Call interpolator
            self.physics.comp_itor.evaluate(init_state, values)

            # Assemble initial composition
            self.solid_frac[i] = values[0]
            initial_comp_with_solid = np.multiply(composition_full, 1 - self.solid_frac[i])
            initial_comp_with_solid[0] = self.solid_frac[i]
            self.initial_comp[i, :] = correct_composition(initial_comp_with_solid, self.comp_min)

        # Define initial composition for wells
        for i in range(self.nx * self.ny * self.nz, self.nx * self.ny * self.nz + 2):
            self.initial_comp[i, :] = np.array(self.inj_stream)

        for i in range(self.nx * self.ny * self.nz + 2, self.nx * self.ny * self.nz + 4):
            self.initial_comp[i, :] = np.array(self.initial_comp[0, :])

        print('\tNegative composition occurrence while initializing:', self.physics.comp_etor.counter, '\n')
        # ====================================== Initialize reservoir composition ======================================

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-7
        self.params.max_ts = 1e-4

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4
        self.params.max_i_newton = 20
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        self.params.newton_params[0] = 0.2
        self.params.trans_mult_exp = trans_mult_exp
        # self.params.norm_type = 1

        self.runtime = 1
        self.timer.node["initialization"].stop()

    def set_wells(self):
        d_w = 1.5
        r_w = d_w / 2
        well_index = 5

        self.reservoir.add_well("I1", wellbore_diameter=d_w)
        for idx in range(self.ny):
            self.reservoir.add_perforation(well_name='I1', cell_index=(1, 1, 1), multi_segment=False,
                                           verbose=True, well_radius=r_w, well_index=well_index,
                                           well_indexD=well_index)

        self.reservoir.add_well("P1", wellbore_diameter=d_w)
        for idx in range(self.ny):
            self.reservoir.add_perforation(well_name='P1', cell_index=(self.nx, 1, 1), multi_segment=False,
                                           verbose=True, well_radius=r_w, well_index=well_index,
                                           well_indexD=well_index)

    # Initialize reservoir and set boundary conditions:
    def set_initial_conditions(self):
        initial_pressure = self.pressure_init
        initial_composition = self.initial_comp

        nb = self.reservoir.mesh.n_blocks
        nc = self.physics.n_components

        # set initial pressure
        pressure = np.array(self.reservoir.mesh.pressure, copy=False)
        pressure.fill(initial_pressure)

        # set initial composition
        self.reservoir.mesh.composition.resize(nb * (nc - 1))
        composition = np.array(self.reservoir.mesh.composition, copy=False)
        for c in range(nc - 1):
            composition[c::nc-1] = initial_composition[:, c]

    def set_boundary_conditions(self):
        # New boundary condition by adding wells:
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                # w.control = self.physics.new_bhp_inj(100, self.inj_stream)
                w.control = self.physics.new_rate_oil_inj(self.inj_rate, self.inj_stream)
            else:
                w.control = self.physics.new_bhp_prod(100)

    def set_op_list(self):
        """
        Function to define list of operator interpolators for accumulation-flux regions and wells.

        Operator list is in order [acc_flux_itor[0], ..., acc_flux_itor[n-1], acc_flux_w_itor]
        """
        self.op_list = [self.physics.acc_flux_itor] + [self.physics.acc_flux_itor]
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.op_num[self.reservoir.mesh.n_res_blocks:] = len(self.op_list) - 1

    def evaluate_porosity(self, Xm = None):
        # Initial porosity
        poro_init = 1 - self.solid_sat

        poro = np.zeros(self.reservoir.n)
        values = value_vector([0] * 2)
        if Xm is None:
            Xm = np.copy(self.physics.engine.X[:self.reservoir.n*self.physics.n_components])

        for i in range(self.reservoir.n):
            state = value_vector(
                Xm[i * self.physics.n_components:i * self.physics.n_components + self.physics.n_components])
            self.physics.results_itor.evaluate(state, values)
            poro[i] = 1 - values[0]

        poro_diff = poro - poro_init
        print('\tNegative composition while evaluating results:', self.physics.results_etor.counter, '\n')
        return poro_init, poro, poro_diff

    def plot_1d(self, map_data, name):
        """
        Function to plot the 1d parameter.
        :param map_data: data array
        :param name: parameter name
        """
        import plotly.graph_objs as go
        import numpy as np

        nx = self.reservoir.nx
        fig = go.Figure()
        data = [go.Scatter(x=np.linspace(0, 1, nx), y=map_data[1:nx])]
        fig.add_trace(data[0])
        fig.update_layout(
            xaxis_title="X block",
            yaxis_title=name,
            font=dict(size=12)
        )
        fig.show()
        # plotly.offline.plot(data, filename='%s_surf.html' % name)
