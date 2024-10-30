from conversions import bar2atm
from darts.engines import *
import numpy as np
import warnings

# Ensure correct import based on OS
try:
    from phreeqpy.iphreeqc.phreeqc_com import IPhreeqc
except ImportError:
    from phreeqpy.iphreeqc.phreeqc_dll import IPhreeqc

class property_container(property_evaluator_iface):
    def __init__(self, phase_name, component_name):
        super().__init__()
        # This class contains all the property evaluators required for simulation
        self.n_phases = len(phase_name)
        self.nc = len(component_name)
        self.component_name = component_name
        self.phase_name = phase_name

        # Allocate (empty) evaluators
        self.density_ev = []
        self.viscosity_ev = []
        self.rel_perm_ev = []
        self.enthalpy_ev = []
        self.kin_rate_ev = []
        self.flash_ev = 0
        self.state_ev = []
        self.init_flash_ev = []


def comp_out_of_bounds(vec_composition, min_z):
    # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
    temp_sum = 0
    count_corr = 0
    check_vec = np.zeros((len(vec_composition),))

    for ith_comp in range(len(vec_composition)):
        if vec_composition[ith_comp] < min_z:
            vec_composition[ith_comp] = min_z
            count_corr += 1
            check_vec[ith_comp] = 1
        elif vec_composition[ith_comp] > 1 - min_z:
            vec_composition[ith_comp] = 1 - min_z
            temp_sum += vec_composition[ith_comp]
        else:
            temp_sum += vec_composition[ith_comp]

    for ith_comp in range(len(vec_composition)):
        if check_vec[ith_comp] != 1:
            vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * min_z)
    return vec_composition


class custom_rel_perm(property_evaluator_iface):
    def __init__(self, exp, sr=0):
        super().__init__()
        self.exp = exp
        self.sr = sr

    def evaluate(self, sat):
        return (sat - self.sr)**self.exp


class custom_flash:
    def __init__(self, temperature, phreeqc_db_path, comp_min):
        self.temperature = temperature - 273.15
        self.comp_min = comp_min

        self.phreeqc_database_loaded = False
        self.phreeqc_template = """
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
        self.phreeqc = IPhreeqc()
        self.load_database(phreeqc_db_path)

    def load_database(self, db_path):
        if not self.phreeqc_database_loaded:
            try:
                self.phreeqc.load_database(db_path)
                self.phreeqc_database_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load '{db_path}': {e}.", Warning)

    @staticmethod
    def get_moles(composition):
        # Assume 1000 mol of solution
        total_mole = 1000
        hydrogen_mole = total_mole * composition[-1]
        oxygen_mole = total_mole * composition[-2]
        carbon_mole = total_mole * composition[-3]
        calcium_mole = total_mole * composition[-4]
        return hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole

    def get_composition(self, state):
        comp = state[2:]
        # comp[comp <= self.comp_min / 10] = 0
        comp = np.divide(comp, np.sum(comp))
        return comp

    @staticmethod
    def generate_input(*args):
        hydrogen, oxygen, carbon, calcium = args[0], args[1], args[2], args[3]
        if hydrogen / 2 <= oxygen:
            water_mass = hydrogen / 2 * 0.018016
            hydrogen_mole = 0
            oxygen_mole = oxygen - hydrogen / 2
        else:
            water_mass = oxygen * 0.018016
            hydrogen_mole = hydrogen - 2 * oxygen
            oxygen_mole = 0
        carbon_mole = carbon
        calcium_mole = calcium
        return water_mass, hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole

    def interpret_results(self):
        results_array = np.array(self.phreeqc.get_selected_output_array()[2])

        # Interpret aqueous phase
        hydrogen_mole_aq = results_array[5]
        oxygen_mole_aq = results_array[6]
        carbon_mole_aq = results_array[7]
        calcium_mole_aq = results_array[8]

        volume_aq = results_array[9] / 1000                                                     # m3
        total_mole_aq = (hydrogen_mole_aq + oxygen_mole_aq + carbon_mole_aq + calcium_mole_aq)  # mol
        rho_aq = total_mole_aq / volume_aq / 1000                                               # kmol/m3

        x = np.array([0,
                      calcium_mole_aq / total_mole_aq,
                      carbon_mole_aq / total_mole_aq,
                      oxygen_mole_aq / total_mole_aq,
                      hydrogen_mole_aq / total_mole_aq])

        # Interpret gaseous phase
        y = np.zeros(len(x))
        rho_g = 0
        total_mole_gas = 0

        rho_phases = {'aq': rho_aq, 'gas': rho_g}
        vap = total_mole_gas / (total_mole_aq + total_mole_gas)

        # Interpret kinetic parameters
        kin_state = {'SI': results_array[10],
                     'SR': results_array[11],
                     'Act(H+)': results_array[12],
                     'Act(CO2)': results_array[13],
                     'Act(H2O)': results_array[14]}
        return vap, x, y, rho_phases, kin_state

    def evaluate(self, state):
        pressure_atm = bar2atm(state[0])
        comp = self.get_composition(state)
        hydrogen, oxygen, carbon, calcium = self.get_moles(comp)
        water_mass, hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole = self.generate_input(hydrogen, oxygen,
                                                                                                carbon, calcium)
        # Generate and execute PHREEQC input
        input_string = self.phreeqc_template.format(
            temperature=self.temperature,
            pressure=pressure_atm,
            water_mass=water_mass,
            hydrogen=hydrogen_mole,
            oxygen=oxygen_mole,
            carbon=carbon_mole,
            calcium=calcium_mole
        )

        try:
            self.phreeqc.run_string(input_string)
        except Exception as e:
            warnings.warn(f"Failed to run PHREEQC: {e}", Warning)

        vap, x, y, rho_phases, kin_state = self.interpret_results()
        vap = vap * (1 - state[1])
        return vap, x, y, rho_phases, kin_state

class init_flash(custom_flash):
    def __init__(self, temperature, phreeqc_db_path, comp_min):
        super().__init__(temperature, phreeqc_db_path, comp_min)

    @staticmethod
    def get_moles(comp_data):
        # Assume 1000 mol of solution
        total_mole = comp_data[-1]
        hydrogen_mole = total_mole * comp_data[-2]
        oxygen_mole = total_mole * comp_data[-3]
        carbon_mole = total_mole * comp_data[-4]
        calcium_mole = total_mole * comp_data[-5]
        return hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole

    def evaluate(self, state):
        # Convert pressure to atm
        pressure_atm = bar2atm(state[0])

        # Normalize the non-solid composition values
        non_solid_composition = np.divide(state[3:], np.sum(state[3:]))
        comp_data = np.hstack((non_solid_composition, state[1]))

        # Calculate moles for each element based on comp_data
        hydrogen, oxygen, carbon, calcium = self.get_moles(comp_data)
        water_mass, hydrogen_mole, oxygen_mole, carbon_mole, calcium_mole = self.generate_input(
            hydrogen, oxygen, carbon, calcium
        )

        # Create and run the input string using phreeqpy
        input_string = f"""
        USER_PUNCH
        -headings	Vol_aq(L)
        10 PUNCH	SOLN_VOL

        SELECTED_OUTPUT
            -selected_out    true
            -user_punch      true
            -reset           false
            -high_precision  true
        SOLUTION 1
            temp      {self.temperature:.2f}
            pressure  {pressure_atm:.4f}
            pH        7 charge
            -water    {water_mass:.3f} # kg
        REACTION 1
            H         {hydrogen_mole:.8f}
            O         {oxygen_mole:.8f}
            C         {carbon_mole:.8f}
            Ca        {calcium_mole:.8f}
            1
        END
        """

        try:
            # Run PHREEQC with the formatted input string
            self.phreeqc.run_string(input_string)
        except Exception as e:
            warnings.warn(f"Failed to run PHREEQC: {e}", Warning)

        # Retrieve the non-solid volume output from PHREEQC
        non_solid_volume = np.array(self.phreeqc.get_selected_output_array()[2]) / 1000  # Convert to m3
        return non_solid_volume

class custom_kinetic_rate:
    def __init__(self, temperature, comp_min):
        self.temperature = temperature
        self.comp_min = comp_min

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
        sat_ratio = kin_state['SR']
        hydrogen_act = kin_state['Act(H+)']

        if sat_ratio > sat_ratio_threshold:
            sat_ratio = sat_ratio_threshold

        KTa = k25a * np.exp((-Eaa / R) * (1 / self.temperature - 1 / 298.15)) * hydrogen_act ** na
        KTn = k25n * np.exp((-Ean / R) * (1 / self.temperature - 1 / 298.15))

        # [mol/s]
        kinetic_rate = -specific_sa * solid_saturation * (rho_s * 1000) * (KTa + KTn) * (1 - sat_ratio ** p) ** q

        # [kmol/d]
        kinetic_rate *= 60 * 60 * 24 / 1000
        return kinetic_rate


class custom_state(property_evaluator_iface):
    def __init__(self):
        super().__init__()

    def evaluate(self, zc, V):
        solid_present = zc[-1] > 1e-8
        vapor_present = V > 1e-8
        liquid_present = V < 1 - 1e-8 - zc[-1]
        if liquid_present and vapor_present and solid_present:
            # Three phase system:
            state = '111'
        elif liquid_present and vapor_present:
            # Liquid and Vapor:
            state = '110'
        elif liquid_present and solid_present:
            # Liquid and Solid:
            state = '101'
        elif vapor_present and solid_present:
            # Vapor and Solid:
            state = '011'
        elif liquid_present:
            # Single phase liquid:
            state = '100'
        elif vapor_present:
            # Single phase vapor:
            state = '010'
        elif solid_present:
            # Single phase solid:
            state = '001'
        else:
            # Something went wrong:
            state = '000'
        return state
