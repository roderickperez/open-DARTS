# from reservoir import StructReservoir
from phreeqc_dissolution.conversions import convert_composition, correct_composition, calculate_injection_stream, \
    get_mole_fractions, convert_rate, bar2atm
from phreeqc_dissolution.physics import PhreeqcDissolution

from darts.models.cicd_model import CICDModel
from darts.reservoirs.struct_reservoir import StructReservoir

from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.density import DensityBasic
from darts.physics.properties.basic import ConstFunc

from darts.engines import *

import numpy as np
import pickle, h5py
import os
from math import fabs
import warnings

try:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

# Definition of your input parameter data structure,
# change as you see fit (when you need more constant values, etc.)!!
class MyOwnDataStruct:
    def __init__(self, nc, zmin, temp, stoich_matrix, pressure_init, kin_fact,  exp_w=1, exp_g=1):
        """
        Data structure class which holds various input parameters for simulation
        :param nc: number of components used in simulation
        :param zmin: actual 0 used for composition (usually >0, around some small epsilon)
        :param temp: temperature
        """
        self.num_comp = nc
        self.min_z = zmin
        self.temperature = temp
        self.stoich_matrix = stoich_matrix
        self.exp_w = exp_w
        self.exp_g = exp_g
        self.pressure_init = pressure_init
        self.kin_fact = kin_fact
        self.n_prop_ops = 19

# Actual Model class creation here!
class Model(CICDModel):
    def __init__(self, domain: str = '1D', nx: int = 200, poro_filename: str = None):
        # Call base class constructor
        super().__init__()

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir(domain=domain, nx=nx, poro_filename=poro_filename)
        self.set_physics()

        # Some newton parameters for non-linear solution:
        self.params.first_ts = 1e-5
        self.params.max_ts = 1e-3

        self.params.tolerance_newton = 1e-3
        self.params.tolerance_linear = 1e-4
        self.params.max_i_newton = 20
        self.params.max_i_linear = 50
        self.params.newton_type = sim_params.newton_local_chop
        self.params.newton_params[0] = 0.2
        # self.params.norm_type = 1

        self.runtime = 1
        self.timer.node["initialization"].stop()

    def set_physics(self):
        # some properties
        self.temperature = 323.15           # K
        self.pressure_init = 100            # bar

        if self.domain == '1D':
            self.inj_rate = self.volume * 24     # output: m3/day
        else:
            self.inj_rate = 10 * self.volume * 24 / self.inj_cells.size     # output: m3/day

        self.min_z = 1e-11
        self.obl_min = self.min_z / 10

        # Several parameters here related to components used, OBL limits, and injection composition:
        self.cell_property = ['pressure', 'H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+',
                              'CaHCO3+', 'Solid']
        self.phases = ['liq', 'gas']
        self.components = ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid']
        self.elements = ['Solid', 'Ca', 'C', 'O', 'H']
        self.fc_mask = np.array([False, True, True, True, True], dtype=bool)
        Mw = {'Solid': 100.0869, 'Ca': 40.078, 'C': 12.0096, 'O': 15.999, 'H': 1.007} # molar weights in kg/kmol
        self.num_vars = len(self.elements)
        self.nc = len(self.elements)
        self.n_points = [101, 201, 101, 101, 101]
        self.axes_min = [self.pressure_init - 1] + [self.obl_min, self.obl_min, self.obl_min, 0.3] #[self.obl_min, self.obl_min, self.obl_min, self.obl_min]
        self.axes_max = [self.pressure_init + 2] + [1 - self.obl_min, 0.01, 0.02, 0.37]

        # Rate annihilation matrix
        self.E = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
                      [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
                      [1, 0, 1, 2, 3, 3, 3, 0, 1, 3, 0],
                      [2, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0]])

        # Several parameters related to kinetic reactions:
        stoich_matrix = np.array([-1, 1, 1, 3, 0])

        # Create property containers:
        property_container = ModelProperties(phases_name=self.phases, components_name=self.elements, Mw=Mw,
                                             min_z=self.obl_min, temperature=self.temperature, fc_mask=self.fc_mask)
        rock_compressibility = 1e-6
        property_container.rock_compr_ev = ConstFunc(rock_compressibility)
        property_container.density_ev['solid'] = DensityBasic(compr=rock_compressibility, dens0=2710., p0=1.)

        # self.kin_fact = self.property.density_ev['solid'].evaluate(pressure) / self.property.Mw['Solid'] * np.mean(self.solid_sat)
        self.kin_fact = 1

        # Create instance of data-structure for simulation (and chemical) input parameters:
        input_data_struct = MyOwnDataStruct(nc=self.nc, zmin=self.obl_min, temp=self.temperature,
                                            stoich_matrix=stoich_matrix, pressure_init=self.pressure_init,
                                            kin_fact=self.kin_fact)

        # Create instance of (own) physics class:
        self.physics = PhreeqcDissolution(timer=self.timer, elements=self.elements, n_points=self.n_points, 
                                          axes_min=self.axes_min, axes_max=self.axes_max,
                                          input_data_struct=input_data_struct, properties=property_container, cache=False)

        self.physics.add_property_region(property_container, 0)

        # Compute injection stream
        mole_water, mole_co2 = calculate_injection_stream(1.1, 0.1, self.temperature, self.pressure_init) # input - m3 of water, co2
        mole_fraction_water, mole_fraction_co2 = get_mole_fractions(mole_water, mole_co2)

        # Define injection stream composition,
        # ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2', 'CaCO3', 'Ca+2', 'CaOH+', 'CaHCO3+', 'Solid']
        self.inj_stream_components = np.array([mole_fraction_water, 0, 0, mole_fraction_co2, 0, 0, 0, 0, 0, 0, 0])
        self.inj_stream = convert_composition(self.inj_stream_components, self.E)
        self.inj_stream = correct_composition(self.inj_stream, self.min_z)

        # prepare arrays for evaluation of properties
        nb = np.prod(self.domain_cells)
        n_prop_ops = self.physics.input_data_struct.n_prop_ops
        n_vars = self.physics.n_vars
        self.prop_states = value_vector([0.] * nb * (n_vars + 1))
        self.prop_states_np = np.asarray(self.prop_states)
        self.prop_values = value_vector([0.] * n_prop_ops * nb)
        self.prop_values_np = np.asarray(self.prop_values)
        self.prop_dvalues = value_vector([0.] * n_prop_ops * nb * n_vars)

    def set_reservoir(self, domain, nx, poro_filename):
        self.domain = domain

        if self.domain == '1D':
            # grid
            self.domain_sizes = np.array([0.1, 0.001 * 7, 0.058905 / 7])
            self.domain_cells = np.array([nx, 1, 1])
            self.cell_sizes = self.domain_sizes / self.domain_cells

            # properties
            depth = 1                      # m
            self.poro = 1                            # [-]
            self.params.trans_mult_exp = 4
            perm = 1.25e4 * self.poro ** self.params.trans_mult_exp
            self.solid_sat = np.ones(self.domain_cells[0]) * 0.7
            self.inj_cells = np.array([0])
        elif self.domain == '2D':
            # grid
            self.domain_sizes = np.array([0.09, 0.09, 0.006])
            self.domain_cells = np.array([nx, nx, 1])
            self.cell_sizes = self.domain_sizes / self.domain_cells

            # properties
            depth = 1                      # m
            self.poro = 1                       # [-]
            self.params.trans_mult_exp = 4
            perm = 1.25e4 * self.poro ** self.params.trans_mult_exp

            # porosity
            if poro_filename == None:
                poro = 0.3 + np.random.uniform(-0.1, 0.1, np.prod(self.domain_cells))
            else:
                poro = 0.3 + 0.05 * np.loadtxt(poro_filename).flatten()
                assert np.prod(self.domain_cells) == poro.size
            poro[poro < 1.e-4] = 1.e-4
            poro[poro > 1 - 1.e-4] = 1 - 1.e-4
            self.solid_sat = 1 - poro
            self.inj_cells = self.domain_cells[0] * np.arange(self.domain_cells[1])
        else:
            print(f'domain={self.domain} is not supported')
            exit(-1)

        self.volume = np.prod(self.domain_sizes)
        self.reservoir = StructReservoir(self.timer,
                                         nx=self.domain_cells[0], ny=self.domain_cells[1], nz=self.domain_cells[2],
                                         dx=self.cell_sizes[0], dy=self.cell_sizes[1], dz=self.cell_sizes[2],
                                         permx=perm, permy=perm, permz=perm, poro=self.poro, depth=depth)

    def set_initial_conditions(self):
        # ====================================== Initialize reservoir composition ======================================
        print('\nInitializing compositions...')

        n_matrix = np.prod(self.domain_cells)

        # Component-defined composition of a non-solid phase (pure water here)
        self.initial_comp_components = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.solid_frac = np.zeros(len(self.solid_sat))
        self.initial_comp = np.zeros((n_matrix + 2, self.num_vars - 1))

        # Interpolated values of non-solid volume (second value always 0 due to no (5,1) interpolator)
        values = value_vector([0] * 2)

        # Iterate over solid saturation and call interpolator
        for i in range(len(self.solid_sat)):
            # There are 5 values in the state
            composition_full = convert_composition(self.initial_comp_components, self.E)
            composition = correct_composition(composition_full, self.min_z)
            init_state = value_vector(np.hstack((self.physics.input_data_struct.pressure_init, self.solid_sat[i], composition[1:])))

            # Call interpolator
            self.physics.comp_itor[0].evaluate(init_state, values)

            # Assemble initial composition
            self.solid_frac[i] = values[0]
            initial_comp_with_solid = composition_full # np.multiply(composition_full, 1 - self.solid_frac[i])
            initial_comp_with_solid[0] = self.solid_frac[i]
            self.initial_comp[i, :] = initial_comp_with_solid[:-1] # correct_composition(initial_comp_with_solid, self.min_z)

        # Define initial composition for wells
        # for i in range(n_matrix, n_matrix + 2):
        #     self.initial_comp[i, :] = np.array(self.inj_stream)

        for i in range(n_matrix, n_matrix + 2):
            self.initial_comp[i, :] = np.array(self.initial_comp[0, :])

        # print('\tNegative composition occurrence while initializing:', self.physics.comp_itor[0].counter, '\n')

        initial_pressure = self.pressure_init
        initial_composition = self.initial_comp

        nb = self.reservoir.mesh.n_blocks
        nc = self.physics.nc

        # set initial pressure
        pressure = np.array(self.reservoir.mesh.pressure, copy=False)
        pressure.fill(initial_pressure)

        # set initial composition
        self.reservoir.mesh.composition.resize(nb * (nc - 1))
        composition = np.array(self.reservoir.mesh.composition, copy=False)
        for c in range(nc - 1):
            composition[c::nc-1] = initial_composition[:, c]

    def set_wells(self):
        d_w = 1.5
        r_w = d_w / 2
        well_index = 5

        # self.reservoir.add_well("I1", wellbore_diameter=d_w)
        # for idx in range(self.domain_cells[1]):
        #     self.reservoir.add_perforation(well_name='I1', cell_index=(1, idx + 1, 1), multi_segment=False,
        #                                    verbose=True, well_radius=r_w, well_index=well_index,
        #                                    well_indexD=well_index)

        self.reservoir.add_well("P1", wellbore_diameter=d_w)
        for idx in range(self.domain_cells[1]):
            self.reservoir.add_perforation(well_name='P1', cell_index=(self.domain_cells[0], idx + 1, 1), multi_segment=False,
                                           verbose=True, well_radius=r_w, well_index=well_index,
                                           well_indexD=well_index)

    def set_rhs_flux(self, t: float = None):
        nv = self.physics.n_vars
        nb = self.reservoir.mesh.n_res_blocks
        rhs_flux = np.zeros(nb * nv)

        rho_m_h20 = 1000 / 18.015 # kmol/m3
        for cell_id in self.inj_cells:
            for i in range(nv - 1):
                rhs_flux[cell_id * nv + i] = -self.inj_rate * rho_m_h20 * self.inj_stream[i]
            rhs_flux[cell_id * nv + nv - 1] = -self.inj_rate * rho_m_h20 * (1 - np.sum(self.inj_stream))

        return rhs_flux

    def set_boundary_conditions(self):
        # New boundary condition by adding wells:
        self.reservoir.wells[0].control = self.physics.new_bhp_prod(self.pressure_init)

        # for i, w in enumerate(self.reservoir.wells):
        #     if i == 0:
        #         # w.control = self.physics.new_bhp_inj(100, self.inj_stream)
        #         w.control = self.physics.new_rate_oil_inj(self.inj_rate, self.inj_stream)
        #     else:
        #         w.control = self.physics.new_bhp_prod(100)

    def output_properties(self, output_properties: list = None, timestep: int = None) -> tuple:
        timesteps = [timestep] if timestep is not None else [0]
        if output_properties is None:
            prop_names = self.physics.property_operators[next(iter(self.physics.property_operators))].props_name
        else:
            prop_names = output_properties

        X = np.asarray(self.physics.engine.X)
        nb = np.prod(self.domain_cells)
        nv = self.physics.n_vars
        nops = len(prop_names)
        n_interp_size = self.physics.input_data_struct.n_prop_ops

        # unknowns
        property_array = {var: np.array([X[i:nb * nv:nv]]) for i, var in enumerate(self.physics.vars)}
        # properties
        self.physics.property_itor[0].evaluate_with_derivatives(self.physics.engine.X, self.physics.engine.region_cell_idx[0],
                                                                self.prop_values, self.prop_dvalues)

        for i, prop in enumerate(prop_names):
            property_array[prop] = np.array([self.prop_values_np[i::n_interp_size]])

        # porosity
        n_cells = self.reservoir.n
        n_vars = self.physics.nc
        op_vals = np.asarray(self.physics.engine.op_vals_arr).reshape(self.reservoir.mesh.n_blocks, self.physics.n_ops)
        poro = op_vals[:self.reservoir.mesh.n_res_blocks, self.physics.reservoir_operators[0].PORO_OP]
        property_array['porosity'] = poro[np.newaxis]

        # hydrogen
        property_array['H'] = 1. - property_array['C'] - property_array['Ca'] - property_array['O']

        # write to *.h5
        if self.domain == '1D':
            path = os.path.join(self.output_folder, self.sol_filename)
            with h5py.File(path, "a") as f:
                current_index = f["dynamic/time"].shape[0] - 1
                written_vars = list(f["dynamic/variable_names"].asstr())
                new_keys = [prop for prop in property_array.keys() if prop not in written_vars]
                new_vars_num = len(new_keys)

                if "properties" not in f["dynamic"]:
                    f["dynamic"].create_dataset("properties", shape=(0, n_cells, new_vars_num),
                                                maxshape=(None, n_cells, new_vars_num), dtype=np.float64)

                extra_dataset = f["dynamic/properties"]
                if extra_dataset.shape[0] <= current_index:
                    extra_dataset.resize((current_index + 1, n_cells, new_vars_num))

                for i, key in enumerate(new_keys):
                    extra_dataset[current_index, :, i] = property_array[key]

                if "properties_name" not in f["dynamic"]:
                    datatype = h5py.special_dtype(vlen=str)  # dtype for variable-length strings
                    var_names = f["dynamic"].create_dataset('properties_name', (new_vars_num,), dtype=datatype)
                    var_names[:] = new_keys

        return timesteps, property_array

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, nc_sol=0, np_sol=0, min_z=1e-11, rate_ann_mat=None,
                 temperature=None, fc_mask=None):
        super().__init__(phases_name=phases_name, components_name=components_name, Mw=Mw, nc_sol=nc_sol, np_sol=np_sol,
                         min_z=min_z, rate_ann_mat=rate_ann_mat, temperature=temperature)
        self.components_name = np.array(self.components_name)

        # Define primary fluid constituents
        if fc_mask is None:
            self.fc_mask = self.nc * [True]
        else:
            self.fc_mask = fc_mask
        self.fc_idx = {comp: i for i, comp in enumerate(self.components_name[self.fc_mask])}

        # Define custom evaluators
        self.flash_ev = self.Flash(min_z=self.min_z, fc_mask=self.fc_mask, fc_idx=self.fc_idx, temperature=self.temperature)
        self.kinetic_rate_ev = self.CustomKineticRate(self.temperature, self.min_z)
        self.rel_perm_ev = {ph: self.CustomRelPerm(2) for ph in phases_name[:2]}  # Relative perm for first two phases

    # default flash working with molar fractions
    class Flash:
        def __init__(self, min_z, fc_mask, fc_idx, temperature=None):
            """
            :param min_z: minimal composition value
            :param fc_mask: boolean mask for extraction of fluid components from all components
            :param fc_idx: dictionary for mapping names of fluid components to filtered (via mask) state
            :param temperature: temperature for isothermal case
            """
            self.fc_mask = fc_mask
            self.fc_idx = fc_idx

            if temperature is None:
                self.thermal = True
            else:
                self.thermal = False
                self.temperature = temperature - 273.15
            self.min_z = min_z
            self.total_moles = 1000
            self.molar_weight_h2o = 0.018016
            self.phreeqc = IPhreeqc()
            self.load_database(self.phreeqc, "phreeqc.dat")
            self.pitzer = IPhreeqc()
            self.load_database(self.pitzer, "pitzer.dat")
            # self.phreeqc.phreeqc.OutputFileOn = True
            # self.phreeqc.phreeqc.SelectedOutputFileOn = True

            self.phreeqc_species = ["OH-", "H+", "H2O", "C(-4)", "CH4", "C(4)", "HCO3-", "CO2", "CO3-2", "CaHCO3+", "CaCO3", "(CO2)2", "Ca+2", "CaOH+", "H(0)", "H2", "O(0)", "O2"]
            self.species_2_element_moles = np.array([2, 1, 3, 1, 5, 1, 5, 3, 4, 6, 5, 6, 1, 3, 1, 2, 1, 2])
            species_headings = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])
            species_punch = " ".join([f'MOL("{sp}")' for sp in self.phreeqc_species])

            self.phreeqc_template = f"""
            USER_PUNCH            
            -headings    H(mol)      O(mol)      C(mol)      Ca(mol)      Vol_aq   SI            SR            ACT("H+") ACT("CO2") ACT("H2O") {species_headings}
            10 PUNCH    TOTMOLE("H") TOTMOLE("O") TOTMOLE("C") TOTMOLE("Ca") SOLN_VOL SI("Calcite") SR("Calcite") ACT("H+") ACT("CO2") ACT("H2O") {species_punch}
        
            SELECTED_OUTPUT
            -selected_out    true
            -user_punch      true
            -reset           false
            -high_precision  true
            -gases           CO2(g) H2O(g)

            SOLUTION 1
            temp      {{temperature:.2f}}
            pressure  {{pressure:.4f}}
            pH        7 charge
            -water    {{water_mass:.10f}} # kg
            REACTION 1
            H         {{hydrogen:.10f}}
            O         {{oxygen:.10f}}
            C         {{carbon:.10f}}
            Ca        {{calcium:.10f}}
            1
            KNOBS
            -convergence_tolerance  1e-10
            END
            """

            self.phreeqc_gas_template = """
            USER_PUNCH
            -headings    H(mol)      O(mol)      C(mol)      Ca(mol)      Vol_aq   SI            SR            ACT("H+") ACT("CO2") ACT("H2O")
            10 PUNCH    TOTMOLE("H") TOTMOLE("O") TOTMOLE("C") TOTMOLE("Ca") SOLN_VOL SI("Calcite") SR("Calcite") ACT("H+") ACT("CO2") ACT("H2O")

            SELECTED_OUTPUT
            -selected_out    true
            -user_punch      true
            -reset           false
            -high_precision  true
            -gases           CO2(g) H2O(g)

            SOLUTION 1
            temp      {temperature:.2f}
            pressure  {pressure:.4f}
            pH        7 charge
            -water    {water_mass:.10f} # kg
            
            GAS_PHASE
            -temp     {temperature:.2f}
            -fixed_pressure
            -pressure {pressure:.4f} 
            CO2(g)    0
            H2O(g)    0
            
            REACTION 1
            H         {hydrogen:.10f}
            O         {oxygen:.10f}
            C         {carbon:.10f}
            Ca        {calcium:.10f}
            1       
            
            KNOBS
            -convergence_tolerance  1e-10
            END
            """

        def load_database(self, database, db_path):
            try:
                database.load_database(db_path)
            except Exception as e:
                warnings.warn(f"Failed to load '{db_path}': {e}.", Warning)

        def interpret_results(self, database):
            results_array = np.array(database.get_selected_output_array()[2])

            co2_gas_mole = results_array[3]
            h2o_gas_mole = results_array[4]

            # interpret aqueous phase
            hydrogen_mole_aq = results_array[5]
            oxygen_mole_aq = results_array[6]
            carbon_mole_aq = results_array[7]
            calcium_mole_aq = results_array[8]

            volume_aq = results_array[9] / 1000  # liters to m3
            total_mole_aq = (hydrogen_mole_aq + oxygen_mole_aq + carbon_mole_aq + calcium_mole_aq)  # mol
            rho_aq = total_mole_aq / volume_aq / 1000  # kmol/m3

            # molar fraction of elements in aqueous phase
            x = np.array([0,
                          calcium_mole_aq / total_mole_aq,
                          carbon_mole_aq / total_mole_aq,
                          oxygen_mole_aq / total_mole_aq,
                          hydrogen_mole_aq / total_mole_aq])

            # suppress gaseous phase
            y = np.zeros(len(x))
            rho_g = 0
            total_mole_gas = 0

            # molar densities
            rho_phases = {'aq': rho_aq, 'gas': rho_g}
            # molar fraction of gaseous phase in fluid
            nu_v = total_mole_gas / (total_mole_aq + total_mole_gas)

            # interpret kinetic parameters
            kin_state = {'SI': results_array[10],
                         'SR': results_array[11],
                         'Act(H+)': results_array[12],
                         'Act(CO2)': results_array[13],
                         'Act(H2O)': results_array[14]}
            species_molalities = results_array[15:]

            return nu_v, x, y, rho_phases, kin_state, volume_aq, species_molalities

        def get_fluid_composition(self, state):
            if self.thermal:
                z = state[1:-1][self.fc_mask[:-1]]
            else:
                z = state[1:][self.fc_mask[:-1]]
            z = np.append(z, 1 - np.sum(z))
            return z

        def evaluate(self, state):
            """
            :param state: state vector with fluid composition accessible by fc_mask
            :type state: np.ndarray
            :return: phase molar fraction, molar composition of aqueous and vapour phases, kinetic params, solution volume
            """
            # extract pressure and fluid composition
            pressure_atm = bar2atm(state[0])

            # check for negative composition occurrence
            fluid_composition = self.get_fluid_composition(state)

            # calculate amount of moles of each component in 1000 moles of mixture
            fluid_moles = self.total_moles * fluid_composition

            # adjust oxygen and hydrogen moles for water formation
            init_h_moles, init_o_moles = fluid_moles[self.fc_idx['H']], fluid_moles[self.fc_idx['O']]
            if init_h_moles / 2 <= init_o_moles:
                water_mass = init_h_moles / 2 * self.molar_weight_h2o
                fluid_moles[self.fc_idx['H']] = 0
                fluid_moles[self.fc_idx['O']] = init_o_moles - init_h_moles / 2
            else:
                water_mass = init_o_moles * self.molar_weight_h2o
                fluid_moles[self.fc_idx['H']] = init_h_moles - 2 * init_o_moles
                fluid_moles[self.fc_idx['O']] = 0

            # Check if solvent (water) is enough
            ion_strength = np.sum(fluid_moles) / (water_mass + 1.e-8)
            if ion_strength > 8:
                print(f'ion_strength = {ion_strength}')
            # assert ion_strength < 7, "Not enough water to form a realistic brine"

            # Generate and execute PHREEQC input
            input_string = self.phreeqc_template.format(
                temperature=self.temperature,
                pressure=pressure_atm,
                water_mass=water_mass,
                hydrogen=fluid_moles[self.fc_idx['H']],
                oxygen=fluid_moles[self.fc_idx['O']],
                carbon=fluid_moles[self.fc_idx['C']],
                calcium=fluid_moles[self.fc_idx['Ca']]
            )

            try:
                self.phreeqc.run_string(input_string)
                nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molalities = self.interpret_results(self.phreeqc)
            except Exception as e:
                warnings.warn(f"Failed to run PHREEQC: {e}", Warning)
                print(f"h20_mass={water_mass}, p={state[0]}, Ca={fluid_moles[self.fc_idx['Ca']]}, C={fluid_moles[self.fc_idx['C']]}, O={fluid_moles[self.fc_idx['O']]}, H={fluid_moles[self.fc_idx['H']]}")
                self.pitzer.run_string(input_string)
                nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molalities = self.interpret_results(self.pitzer)

            species_molar_fractions = species_molalities * water_mass * self.species_2_element_moles / self.total_moles
            return nu_v, x, y, rho_phases, kin_state, fluid_volume, species_molar_fractions

    class CustomKineticRate:
        def __init__(self, temperature, min_z):
            self.temperature = temperature
            self.min_z = min_z

        def evaluate(self, kin_state, solid_saturation, rho_s, min_z, kin_fact):
            # Define constants
            specific_sa = 0.925         # [m2/mol], default = 0.925
            k25a = 0.501187234          # [mol * m-2 * s-1]
            k25n = 1.54882e-06          # [mol * m-2 * s-1]
            Eaa = 14400                 # [J * mol-1]
            Ean = 23500                 # [J * mol-1]
            na = 1                      # reaction order with respect to H+
            R = 8.314472                # gas constant [J/mol/Kelvin]
            p = 1
            q = 1
            sat_ratio_threshold = 100

            # Define rate parameters
            sat_ratio = min(kin_state['SR'], sat_ratio_threshold)
            hydrogen_act = kin_state['Act(H+)']
            KTa = k25a * np.exp((-Eaa / R) * (1 / self.temperature - 1 / 298.15)) * hydrogen_act ** na
            KTn = k25n * np.exp((-Ean / R) * (1 / self.temperature - 1 / 298.15))

            # # [kmol/d]
            # kinetic_rate = -specific_sa * (
            #         (solid_saturation * rho_s * 1000) ** n) * (KTa + KTn) * (1 - sat_ratio) / (kin_fact ** (n - 1)) * 86.400

            # [mol/s/m3]
            kinetic_rate = -specific_sa * solid_saturation * (rho_s * 1000) * (KTa + KTn) * (1 - sat_ratio ** p) ** q

            # [kmol/d/m3]
            kinetic_rate *= 60 * 60 * 24 / 1000
            return kinetic_rate

    class CustomRelPerm:
        def __init__(self, exp, sr=0):
            self.exp = exp
            self.sr = sr

        def evaluate(self, sat):
            return (sat - self.sr) ** self.exp


