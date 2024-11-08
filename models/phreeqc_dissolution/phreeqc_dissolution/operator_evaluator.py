from darts.physics.operators_base import OperatorsBase
from phreeqc_dissolution.conversions import bar2pa
from darts.engines import *
import CoolProp.CoolProp as CP
import os.path as osp
import numpy as np

physics_name = osp.splitext(osp.basename(__file__))[0]

# Define our own operator evaluator class
class my_own_acc_flux_etor(OperatorsBase):
    def __init__(self, input_data, properties):
        super().__init__(properties, thermal=properties.thermal)
        # Store your input parameters in self here, and initialize other parameters here in self
        self.input_data = input_data
        self.min_z = input_data.min_z
        self.temperature = input_data.temperature
        self.exp_w = input_data.exp_w
        self.exp_g = input_data.exp_g
        self.c_r = input_data.c_r
        self.kin_fact = input_data.kin_fact
        self.property = properties
        self.counter = 0

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

    def evaluate(self, state, values):
        """
        Class methods which evaluates the state operators for the element based physics
        :param state: state variables [pres, comp_0, ..., comp_N-1]
        :param values: values of the operators (used for storing the operator values)
        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        state_np = state.to_numpy()
        values_np = state.to_numpy()
        pressure = state_np[0]

        zc = np.append(state_np[1:], 1 - np.sum(state_np[1:]))

        # # # ================================ Flash ================================ # # #
        # Check for negative composition occurrence
        if (zc < 0).any():
            self.counter += 1
            zc_copy = np.array(zc)
            zc_copy = self.comp_out_of_bounds(zc_copy)
            flash_state = np.append(pressure, zc_copy)
        else:
            flash_state = np.append(pressure, zc)

        # Perform Flash procedure here:
        vap, x, y, rho_phases, kin_state = self.property.flash_ev.evaluate(flash_state)

        # NOTE: z_CaCO3 = zc[-1] = 1 - V - L (since only CaCO3 appears in solid phase)
        sol = zc[0]

        # Calculate liquid phase fraction:
        liq = 1 - vap - sol

        # Note: officially three phases are present now
        rho_w = rho_phases['aq']
        mu_w = CP.PropsSI('V', 'T', self.temperature, 'P|liquid', bar2pa(pressure), 'Water') * 1000

        rho_g = rho_phases['gas']

        try:
            mu_g = CP.PropsSI('V', 'T', self.temperature, 'P|gas', bar2pa(pressure), 'CarbonDioxide') * 1000
        except ValueError:
            mu_g = 0.05

        # in kmol/m3
        rho_s = (1 + self.c_r * (pressure - 1)) * 2710 / 100.0869
        # # # ================================ Flash ================================ # # #

        # Get saturations
        if vap > 0:
            sg = vap / rho_g / (vap / rho_g + liq / rho_w + sol / rho_s)
            sw = liq / rho_w / (vap / rho_g + liq / rho_w + sol / rho_s)
            ss = sol / rho_s / (vap / rho_g + liq / rho_w + sol / rho_s)
        else:
            sg = 0
            sw = liq / rho_w / (liq / rho_w + sol / rho_s)
            ss = sol / rho_s / (liq / rho_w + sol / rho_s)

        # Need to normalize to get correct Brook-Corey relative permeability
        sg_norm = sg / (sg + sw)
        sw_norm = sw / (sg + sw)

        kr_w = self.property.rel_perm_ev['liq'].evaluate(sw_norm)
        kr_g = self.property.rel_perm_ev['gas'].evaluate(sg_norm)

        # all properties are in array, and can be separate
        self.x = np.array([y, x])
        self.rho_m = np.array([rho_g, rho_w])
        self.kr = np.array([kr_g, kr_w])
        self.mu = np.array([mu_g, mu_w])
        self.compr = 1 + self.c_r * (pressure - 1)
        self.sat = np.array([sg_norm, sw_norm])

        # Total density
        rho_t = rho_w * sw + rho_s * ss + rho_g * sg

        # Kinetic reaction rate
        kin_rate = self.property.kinetic_rate_ev.evaluate(kin_state, ss, rho_s, self.min_z, self.kin_fact)

        nc = self.property.nc
        nph = 2
        ne = nc

        #       al + bt        + gm + dlt + chi     + rock_temp por    + gr/cap  + por
        total = ne + ne * nph + nph + ne + ne * nph + 3 + 2 * nph + 1

        values_np[:] = 0.

        """ CONSTRUCT OPERATORS HERE """
        """ Alpha operator represents accumulation term: """
        for i in range(nc):
            values[i] = zc[i] * rho_t

        """ Beta operator represents flux term: """
        for j in range(nph):
            shift = ne + ne * j
            for i in range(nc):
                values[shift + i] = self.x[j][i] * self.rho_m[j] * self.kr[j] / self.mu[j]

        """ Gamma operator for diffusion (same for thermal and isothermal) """
        shift = ne + ne * nph
        for j in range(nph):
            values[shift + j] = self.compr * self.sat[j]
            # values[shift + j] = 0

        """ Chi operator for diffusion """
        dif_coef = np.array([0, 1, 1, 1, 1]) * 5.2e-10 * 86400
        shift += nph
        for i in range(nc):
            for j in range(nph):
                values[shift + i * nph + j] = dif_coef[i] * self.rho_m[j] * self.x[j][i]
                # values[shift + ne * j + i] = 0

        """ Delta operator for reaction """
        shift += nph * ne
        for i in range(ne):
            values[shift + i] = self.input_data.stoich_matrix[i] * kin_rate

        """ Gravity and Capillarity operators """
        shift += ne
        # E3-> gravity
        for i in range(nph):
            values[shift + 3 + i] = 0

        # E4-> capillarity
        for i in range(nph):
            values[shift + 3 + nph + i] = 0

        # E5_> porosity
        values[shift + 3 + 2 * nph] = 1 - ss

        return 0

# Define our own operator evaluator class for initialization
class my_own_comp_etor(my_own_acc_flux_etor):
    def __init__(self, input_data, properties):
        super().__init__(input_data, properties)  # Initialize base-class
        self.non_solid_mole = 1000
        self.counter = 0

    def evaluate(self, state, values):
        state_np = state.to_numpy()
        values_np = values.to_numpy()
        pressure = state_np[0]
        # state_np[1] is solid saturation (volume fraction)
        sol_saturation = state_np[1]

        # Check for negative composition occurrence
        zc = np.append(state_np[2:], 1 - np.sum(state_np[2:]))
        if (zc < 0).any():
            self.counter += 1
            zc_copy = np.array(zc)
            zc_copy = self.comp_out_of_bounds(zc_copy)
            init_flash_state = np.hstack((pressure, self.non_solid_mole, zc_copy))
            flash_state = np.append(pressure, zc_copy)
        else:
            init_flash_state = np.hstack((pressure, self.non_solid_mole, zc))
            flash_state = np.append(pressure, zc)

        # init flash, for initial properties
        non_sol_volume = self.property.init_flash_ev.evaluate(init_flash_state)    # m3

        # initial composition
        sol_volume = non_sol_volume * sol_saturation / (1 - sol_saturation)         # m3
        sol_mole = sol_volume * (1 + 1e-6 * (pressure - 1)) * 2710 / 0.1000869
        sol = sol_mole / (sol_mole + self.non_solid_mole)
        values_np[0] = sol

        # porosity
        # normal flash, for other properties
        vap, _, _, rho_phases, _ = self.property.flash_ev.evaluate(flash_state)

        # NOTE: z_CaCO3 = zc[0] = 1 - V - L (since only CaCO3 appears in solid phase)
        sol = zc[0]

        # Calculate liquid phase fraction:
        liq = 1 - vap - sol

        # Note: officially three phases are present now
        rho_w = rho_phases['aq']
        rho_g = rho_phases['gas']
        rho_s = (1 + 1e-6 * (pressure - 1)) * 2710 / 0.1000869 / 1000

        # Get saturations
        if vap > 0:
            ss = sol / rho_s / (vap / rho_g + liq / rho_w + sol / rho_s)
        else:
            ss = sol / rho_s / (liq / rho_w + sol / rho_s)

        values_np[1] = ss
        return 0

class my_own_rate_evaluator(operator_set_evaluator_iface):
    # Simplest class existing to mankind:
    def __init__(self, properties, temperature, c_r):
        # Initialize base-class
        super().__init__()
        self.property = properties
        self.temperature = temperature
        self.c_r = c_r

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

    def evaluate(self, state, values):
        # Composition vector and pressure from state:
        state_np = state.to_numpy()
        values_np = state.to_numpy()
        pressure = state_np[0]

        zc = np.append(state_np[1:], 1 - np.sum(state_np[1:]))

        # # # ================================ Flash ================================ # # #
        # Check for negative composition occurrence
        if (zc < 0).any():
            self.counter += 1
            zc_copy = np.array(zc)
            zc_copy = self.comp_out_of_bounds(zc_copy)
            flash_state = np.append(pressure, zc_copy)
        else:
            flash_state = np.append(pressure, zc)

        # Perform Flash procedure here:
        vap, x, y, rho_phases, _ = self.property.flash_ev.evaluate(flash_state)

        # NOTE: z_CaCO3 = zc[-1] = 1 - V - L (since only CaCO3 appears in solid phase)
        sol = zc[0]

        # Calculate liquid phase fraction:
        liq = 1 - vap - sol

        # Note: officially three phases are present now
        rho_w = rho_phases['aq']
        mu_w = CP.PropsSI('V', 'T', self.temperature, 'P|liquid', bar2pa(pressure), 'Water') * 1000        # Pa * s

        rho_g = rho_phases['gas']

        try:
            mu_g = CP.PropsSI('V', 'T', self.temperature, 'P|gas', bar2pa(pressure), 'CarbonDioxide') * 1000       # Pa * s
        except ValueError:
            mu_g = 16.14e-6 * 1000     # Pa * s, for 50 C

        # in kmol/m3
        rho_s = (1 + self.c_r * (pressure - 1)) * 2710 / 0.1000869 / 1000
        # # # ================================ Flash ================================ # # #

        # Get saturations
        if vap > 0:
            sg = vap / rho_g / (vap / rho_g + liq / rho_w + sol / rho_s)
            sw = liq / rho_w / (vap / rho_g + liq / rho_w + sol / rho_s)
            ss = sol / rho_s / (vap / rho_g + liq / rho_w + sol / rho_s)
        else:
            sg = 0
            sw = liq / rho_w / (liq / rho_w + sol / rho_s)
            ss = sol / rho_s / (liq / rho_w + sol / rho_s)

        # Need to normalize to get correct Brook-Corey relative permeability
        sg_norm = sg / (sg + sw)
        sw_norm = sw / (sg + sw)

        kr_w = self.property.rel_perm_ev['liq'].evaluate(sw_norm)
        kr_g = self.property.rel_perm_ev['gas'].evaluate(sg_norm)

        # Total density
        rho_t = rho_w * sw + rho_s * ss + rho_g * sg

        # Easiest example, constant volumetric phase rate:
        values[0] = 0   # vapor phase
        values[1] = 1 / mu_w    # liquid phase
        #values[1] = (rho_w * kr_w / mu_w + rho_g * kr_g / mu_g) / rho_t  #
        # Usually some steps will be executed to estimate unit volumetric flow rate based on current state (when
        # multiplied with a pressure difference one obtains actual volumetric flow rate)
        return 0

