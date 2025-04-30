from darts.physics.base.operators_base import OperatorsBase
from darts.engines import *
import os.path as osp
import numpy as np

physics_name = osp.splitext(osp.basename(__file__))[0]

class my_own_acc_flux_etor(OperatorsBase):
    """
    Reservoir operators working with the following state:
    state: (pressure, overall mineral molar fractions, overall fluid molar fractions)
    """
    def __init__(self, input_data, properties):
        super().__init__(properties, thermal=properties.thermal)
        # Store your input parameters in self here, and initialize other parameters here in self
        self.input_data = input_data
        self.min_z = input_data.min_z
        self.temperature = input_data.temperature
        self.exp_w = input_data.exp_w
        self.exp_g = input_data.exp_g
        self.kin_fact = input_data.kin_fact
        self.property = properties
        self.counter = 0

        # Operator order
        self.ACC_OP = 0  # accumulation operator - ne
        self.FLUX_OP = self.ACC_OP + self.ne  # flux operator - ne * nph
        self.UPSAT_OP = self.FLUX_OP + self.ne * self.nph  # saturation operator (diffusion/conduction term) - nph
        self.GRAD_OP = self.UPSAT_OP + self.nph  # gradient operator (diffusion/conduction term) - ne * nph
        self.KIN_OP = self.GRAD_OP + self.ne * self.nph  # kinetic operator - ne

        # extra operators
        self.GRAV_OP = self.KIN_OP + self.ne  # gravity operator - nph
        self.PC_OP = self.GRAV_OP + self.nph  # capillary operator - nph
        self.PORO_OP = self.PC_OP + self.nph  # porosity operator - 1
        self.ENTH_OP = self.PORO_OP + 1  # enthalpy operator - nph
        self.TEMP_OP = self.ENTH_OP + self.nph  # temperature operator - 1
        self.PRES_OP = self.TEMP_OP + 1
        self.n_ops = self.PRES_OP + 1

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp in range(len(vec_composition)):
            if vec_composition[ith_comp] < self.min_z:
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif vec_composition[ith_comp] > 1 - self.min_z:
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp in range(len(vec_composition)):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = vec_composition[ith_comp] / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def get_overall_composition(self, state):
        if self.thermal:
            z = state[1:-1]
        else:
            z = state[1:]
        z = np.append(z, 1 - np.sum(z[self.property.flash_ev.fc_mask[:-1]]))
        return z

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # state and values numpy vectors:
        state_np = state.to_numpy()
        values_np = values.to_numpy()

        # pore pressure
        pressure = state_np[0]
        # get overall molar composition
        z = self.get_overall_composition(state_np)

        # call property:
        self.property.evaluate(state_np)

        # Densities
        rho_t = np.sum(self.property.dens_m * self.property.sat_overall[:self.property.nph]) + \
                self.property.dens_m_solid * self.property.sat_overall[self.property.nph]
        rho_f = np.sum(self.property.dens_m * self.property.sat)

        nc = self.property.nc
        nph = 2
        ne = nc

        """ CONSTRUCT OPERATORS HERE """
        values_np[:] = 0.
        
        """ Alpha operator represents accumulation term: """
        values_np[self.ACC_OP] = z[0] * rho_t
        values_np[self.ACC_OP + 1:self.ACC_OP + nc] = (1 - self.property.sat_overall[self.property.nph]) * z[1:] * rho_f

        """ Beta operator represents flux term: """
        for j in range(nph):
            values_np[self.FLUX_OP + j * self.ne:self.FLUX_OP + j * self.ne + self.nc] = self.property.x[j] * self.property.dens_m[j] * self.property.kr[j] / self.property.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in range(nph):
            values_np[self.UPSAT_OP + j] = self.property.rock_compr[0] * self.property.sat[j]

        """ Chi operator for diffusion """
        dif_coef = np.array([0, 1, 1, 1, 1]) * 5.2e-10 * 86400
        for i in range(nc):
            for j in range(nph):
                values_np[self.GRAD_OP + i * nph + j] = dif_coef[i] * self.property.dens_m[j] * self.property.x[j][i]
                # values[shift + ne * j + i] = 0

        """ Delta operator for reaction """
        for i in range(ne):
            values_np[self.KIN_OP + i] = self.input_data.stoich_matrix[i] * self.property.kin_rate

        """ Gravity and Capillarity operators """
        # E3-> gravity
        for i in range(nph):
            values_np[self.GRAV_OP + i] = 0

        # E4-> capillarity
        for i in range(nph):
            values_np[self.PC_OP + i] = 0

        # E5_> porosity
        values_np[self.PORO_OP] = 1 - self.property.sat_overall[self.property.nph]

        # values[shift + 3 + 2 * nph + 1] = kin_state['SR']
        # values[shift + 3 + 2 * nph + 2] = kin_state['Act(H+)']

        return 0

class my_own_comp_etor(my_own_acc_flux_etor):
    """
    Operator required for initialization, to convert given volume fraction to molar one
    DIFFERENT state: (pressure, overall mineral volume fractions, fluid molar fractions)
    """
    def __init__(self, input_data, properties):
        super().__init__(input_data, properties)  # Initialize base-class
        self.fluid_mole = 1
        self.counter = 0
        self.props_name = ['z_solid']

    def evaluate(self, state, values):
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        pressure = state_np[0]
        ss = state_np[1] # volume fraction in initialization

        # initial flash
        _, _, _, _, _, fluid_volume, _ = self.property.flash_ev.evaluate(state_np)

        # evaluate molar fraction
        solid_volume = fluid_volume * ss / (1 - ss)         # m3
        solid_mole = solid_volume * self.property.rock_density_ev['Solid_CaCO3'].evaluate(pressure) / self.property.Mw['Solid_CaCO3']
        nu_s = solid_mole / (solid_mole + self.fluid_mole)
        values_np[0] = nu_s

        return 0

class my_own_property_evaluator(operator_set_evaluator_iface):
    def __init__(self, input_data, properties):
        # Initialize base-class
        super().__init__()
        self.input_data = input_data
        self.property = properties
        self.props_name = ['z' + prop for prop in properties.flash_ev.phreeqc_species]

    def evaluate(self, state, values):
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        _, _, _, _, _, _, molar_fractions = self.property.flash_ev.evaluate(state_np)
        values_np[:molar_fractions.size] = molar_fractions

        return 0
