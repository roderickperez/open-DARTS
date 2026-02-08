import numpy as np
from numba import jit
import warnings
from abc import abstractmethod, ABC

from dartsflash.components import CompData
from dartsflash.libflash import CubicEoS, AQEoS

from dwell.utilities.units import *
import dwell.utilities.library as library

# Constants
NA = 6.02214076e23 * 1 / mol()  # Avogadro's number
kB = 1.380649e-23 * Joule() / Kelvin()  # Boltzmann constant
R = NA * kB * Joule() / (mol() * Kelvin())  # Universal as constant

class FluidModel:
    def __init__(self, pipe_name: str, components_names: list, phases_names: list,
                 flash_calcs_eval, density_eval, viscosity_eval, IFT_eval, rel_perm_eval, enthalpy_eval=None, internal_energy_eval=None, verbose: bool = False):
        """
        :param pipe_name: Name of the pipe for which the fluid model is going to be defined
        :type pipe_name: str
        :param components_names: List of names of components
        :type components_names: list
        :param phases_names: List of names of phases
        :type phases_names: list of str
        :param flash_calcs_eval: Object for flash calculations
        :type flash_calcs_eval: Classes available for flash calculations
        :param density_eval: A dict of an object or objects for density evaluation
        :type density_eval: dict
        :param viscosity_eval: A dict of an object or objects for viscosity evaluation
        :type viscosity_eval: dict
        :param IFT_eval: Object for IFT evaluation
        :type IFT_eval: Classes available for IFT evaluation
        :param rel_perm_eval: A dict of objects for relative permeability evaluation
        :type rel_perm_eval: dict
        :param enthalpy_eval: A dict of an object or objects for enthalpy evaluation
        :type enthalpy_eval: dict
        :param internal_energy_eval: Object for internal energy evaluation
        :type internal_energy_eval: Classes available for internal energy evaluation
        :param verbose: Whether to display extra info about FluidModel
        :type verbose: boolean
        """
        self.pipe_name = pipe_name

        self.components_names = components_names
        self.num_components = len(components_names)

        assert phases_names == ['gas', 'liquid'], ('2 phases must be specified, gas and liquid must be the names and '
                                                   'order of the phases!')
        self.phases_names = phases_names
        self.max_num_phases = len(phases_names)

        # Get components molecular weights from the library and store them in MW
        MW = []
        for i in range(self.num_components):
            try:
                MW.append(library.components_molecular_weights[components_names[i]])
            except:
                raise Exception(f"Molecular weight of {components_names[i]} is not in the library!")
        self.MW = MW

        self.flash_calcs_eval = flash_calcs_eval
        self.density_eval = density_eval
        self.viscosity_eval = viscosity_eval
        self.IFT_eval = IFT_eval
        self.rel_perm_eval = rel_perm_eval

        # Non-isothermal
        self.enthalpy_eval = enthalpy_eval
        self.internal_energy_eval = internal_energy_eval

        if verbose:
            print("** Fluid model of the pipe \"%s\" is defined!" % self.pipe_name)

#%% Property evaluator
class PropertyEvaluator:
    def __init__(self, fluid_model, min_z=1e-11):
        self.fluid_model = fluid_model
        self.min_z = min_z

    def evaluate(self, pressure, temperature, zc_full, isothermal):
        self.x = np.zeros((self.fluid_model.max_num_phases, self.fluid_model.num_components))
        self.x_mass = np.zeros((self.fluid_model.max_num_phases, self.fluid_model.num_components))
        self.rho = np.zeros(self.fluid_model.max_num_phases)
        self.rho_molar = np.zeros(self.fluid_model.max_num_phases)
        self.sat = np.zeros(self.fluid_model.max_num_phases)
        self.nu = np.zeros(self.fluid_model.max_num_phases)
        self.miu = np.zeros(self.fluid_model.max_num_phases)
        if not isothermal:
            self.h = np.zeros(self.fluid_model.max_num_phases)
            self.U = np.zeros(self.fluid_model.max_num_phases)

        zc_full = self.check_composition(zc_full)

        # ph is the index or indices of the phase or phases present in the given conditions
        self.ph = self.run_flash(pressure, temperature, zc_full)

        for j in self.ph:
            M = np.sum(self.fluid_model.MW * self.x[j][:])

            self.rho[j] = self.fluid_model.density_eval[self.fluid_model.phases_names[j]].evaluate(pressure, temperature, self.x[j, :])  # output in [kg/m3]
            self.rho_molar[j] = self.rho[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            self.miu[j] = self.fluid_model.viscosity_eval[self.fluid_model.phases_names[j]].evaluate(pressure, temperature, self.x[j, :], self.rho[j])

            if not isothermal:
                self.h[j] = self.fluid_model.enthalpy_eval[self.fluid_model.phases_names[j]].evaluate(pressure, temperature, self.x[j, :])
                self.U[j] = self.fluid_model.internal_energy_eval.evaluate(pressure, self.h[j], self.rho[j])
        self.compute_saturation(self.ph)

        # Calculate mass fractions of components in each phase
        for j in self.ph:
            self.x_mass[j, :] = (self.x[j, :] * self.fluid_model.MW) / sum(self.x[j, :] * self.fluid_model.MW)

        if isothermal:
            return self.ph, self.sat, self.x_mass, self.rho, self.miu
        elif not isothermal:
            return self.ph, self.sat, self.x_mass, self.rho, self.miu, self.h, self.U

    def check_composition(self, zc):
        if zc[-1] < self.min_z:
            zc = self.comp_out_of_bounds(zc)
        return zc

    def comp_out_of_bounds(self, vec_composition):
        # Check if composition sum is above 1 or element comp below 0, i.e. if point is unphysical:
        temp_sum = 0
        count_corr = 0
        check_vec = np.zeros((len(vec_composition),))

        for ith_comp, zi in enumerate(vec_composition):
            if zi < self.min_z:
                # print(vec_composition)
                vec_composition[ith_comp] = self.min_z
                count_corr += 1
                check_vec[ith_comp] = 1
            elif zi > 1 - self.min_z:
                # print(vec_composition)
                vec_composition[ith_comp] = 1 - self.min_z
                temp_sum += vec_composition[ith_comp]
            else:
                temp_sum += vec_composition[ith_comp]

        for ith_comp, zi in enumerate(vec_composition):
            if check_vec[ith_comp] != 1:
                vec_composition[ith_comp] = zi / temp_sum * (1 - count_corr * self.min_z)
        return vec_composition

    def run_flash(self, pressure, temperature, zc):
        pressure = convertTo(pressure, bar())
        error_output = self.fluid_model.flash_calcs_eval.evaluate(pressure, temperature, zc)
        flash_results = self.fluid_model.flash_calcs_eval.get_flash_results()
        self.nu = np.array(flash_results.nu)
        self.x = np.array(flash_results.X).reshape(self.fluid_model.max_num_phases, self.fluid_model.num_components)

        ph = []
        for j in range(self.fluid_model.max_num_phases):
            if self.nu[j] > 0:
                ph.append(j)

        if len(ph) == 1:
            self.x = np.zeros((self.fluid_model.max_num_phases, self.fluid_model.num_components))
            self.x[ph[0]] = zc

        return ph

    def compute_saturation(self, ph):
        # Get phase saturations [volume fractions]
        Vtot = 0
        for j in ph:
            Vtot += self.nu[j] / self.rho_molar[j]

        for j in ph:
            self.sat[j] = (self.nu[j] / self.rho_molar[j]) / Vtot

        return


#%% Constant evaluator
class ConstFunc:
    """
    This class is used as an evaluator if the property (IFT, viscosity, etc.) is assumed to be constant.
    """
    def __init__(self, value):
        self.value = value

    # The following two methods are equivalent.
    # def evaluate(self, dummy1=0, dummy2=0, dummy3=0, dummy4=0):   # These dummy input arguments are defined because the
    #     # method "evaluate" of the property (IFT, viscosity, etc.) in the code may have a number of input arguments, so
    #     # if this method is called in the code and if these dummy input arguments are not defined, it will give an
    #     # error. If there is any evaluate method that has more than 4 input arguments, we can easily increase
    #     # the number of these dummy input args.
    #     return self.value
    def evaluate(self, *args):
        """
        :param args: Arbitrary number of input arguments
        :return: Constant value
        """
        return self.value

#%% Flash calculations evaluators
class ConstantK:
    """
    # This class is used to do flash calculations with constant K values.
    """
    def __init__(self, K_values: list, flash_calcs_eps: float = 1e-11):
        """
        :param K_values: Equilibrium ratios of components [dimensionless]
        :type K_values: list of floats
        :param flash_calcs_eps: A very small number used for flash calculations
        :type flash_calcs_eps: float
        """
        self.K_values = K_values
        self.flash_calcs_eps = flash_calcs_eps

    def evaluate(self, pressure, temperature, zc):
        K_values = np.array(self.K_values)
        zc = np.array(zc)
        self.nu, self.X = RR2(K_values, zc, self.flash_calcs_eps)
        return 0

    def get_flash_results(self):
        return self

@jit(nopython=True)
def RR2(K, zc, flash_calcs_eps):   # Rachford-Rice equation

    a = 1 / (1 - np.max(K)) + flash_calcs_eps
    b = 1 / (1 - np.min(K)) - flash_calcs_eps

    # Use bisection method to find the root of the Rachford-Rice equation.
    max_iter = 200
    for i in range(1, max_iter):
        V = 0.5 * (a + b)
        r = np.sum(zc * (K - 1) / (V * (K - 1) + 1))
        if abs(r) < 1e-12:
            break

        if r > 0:
            a = V
        else:
            b = V

    if i >= max_iter:
        print("Flash warning!!!")

    x = zc / (V * (K - 1) + 1)
    y = K * x

    return [V, 1 - V], [y, x]


# NegativeFlash is another flash evaluator which is available in darts-flash. It does flash calculations with
# K values as functions of pressure, temperature, and composition.

# %% Density evaluators
class Density(ABC):
    """
    This abstract class is used to make the programmer use the same evaluate method for all the density evaluators.
    """
    @abstractmethod
    def evaluate(self, pressure, temperature, x):
        pass

class DensityBasic(Density):
    def __init__(self, rho_ref, compr, p_ref):
        self.rho_ref = rho_ref
        self.compr = compr
        self.p_ref = p_ref

    def evaluate(self, pressure, temperature: float = None, x: list = None):
        return self.rho_ref * (1 + self.compr * (pressure - self.p_ref))


class Density_EoS_PT(Density):
    """
    This class evaluates density using a cubic EoS by getting the pressure, temperature (PT),
    and composition of the mixture.
    """
    def __init__(self, components_names: list, EoS_name: str):
        """
        :param components_names: Names of the components in the fluid model
        :type components_names: list
        :param EoS_name: Name of the cubic EoS which you want to use to evaluate density. Two options are available:
        "PR" ---> Peng-Robinson EoS
        "SRK" ---> Soave-Redlich-Kwong EoS
        """
        self.EoS_name = EoS_name

        # Get components data
        self.comp_data = CompData(components_names, setprops=True)

        if self.EoS_name == "PR":
            # Instantiate the Peng-Robinson EoS object
            self.eos_density = CubicEoS(self.comp_data, CubicEoS.PR)
        elif self.EoS_name == "SRK":
            # Instantiate the Soave-Redlich-Kwong EoS object
            self.eos_density =  CubicEoS(self.comp_data, CubicEoS.SRK)
        else:
            raise Exception("The name of the EoS must be either \"PR\" or \"SRK\"")

    def evaluate(self, pressure: float, temperature: float, x: list):
        """
        :param pressure: Pressure in Pascal
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions
        :type x: list of floats

        :returns: Phase density in kg/m3
        :rtype: float
        """
        pressure = convertTo(pressure, bar())

        MW_apparent = np.sum(x * np.array(self.comp_data.Mw)) * 1e-3  # MW in kg/mol
        # Here, the Peng-Robinson (PR) EoS gives the molar volume from pressure, temperature, and composition
        molar_volume = self.eos_density.V(pressure, temperature, x)   # molar_volume in m3/mol
        mass_density = MW_apparent / molar_volume   # mass_density in kg/m3
        return mass_density

class Spivey2004(Density):
    """
    Correlation for brine density: Spivey et al. (2004) - Estimating Density, Formation Volume Factor, Compressibility,
                                                        Methane Solubility, and Viscosity for Oilfield Brines at
                                                        Temperatures From 0 to 275˚ C, Pressures to 200 MPa, and
                                                        Salinities to 5.7 mole/kg
    """
    aw = [[-0.127213, 0.645486, 1.03265, -0.070291, 0.639589],
          [4.221, -3.478, 6.221, 0.5182, -0.4405],
          [-11.403, 29.932, 27.952, 0.20684, 0.3768]]
    ab = [[-7.925e-5, -1.93e-6, -3.4254e-4, 0., 0.],
          [1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016],
          [-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685],
          [3.746e-4, -3.328e-4, -3.346e-4, 0., 0.],
          [0., 0., 0.1353, 0., 0.],
          [-1.409, -0.361, -0.2532, 0., 9.216],
          [0., 5.614, 4.6782, -0.307, 2.6069],
          [-0.1127, 0.2047, -0.0452, 0., 0.]]

    def __init__(self, components_names: list, ions: list = None):
        # super().__init__(components_names)

        self.H2O_idx = components_names.index("H2O") if "H2O" in components_names else None
        if self.H2O_idx is None:
            warnings.warn("H2O not present")

        self.ions = ions
        self.ni = len(ions) if ions is not None else 0

    def evaluate(self, pressure: float, temperature: float, x: list):
        """
        :param pressure: Pressure [Pascal]
        :type pressure: float
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param x: Composition or list of mole fractions
        :type x: list
        """
        pressure = convertTo(pressure, bar())

        tc = temperature - 273.15  # Temp in [Celcius]
        tc_100 = tc/100  # needed many times
        p0 = 700  # reference pressure of 70 MPa

        # Pure water density
        a_w = np.empty(3)
        for i in range(3):
            a_w[i] = (self.aw[i][0] * tc_100 ** 2 + self.aw[i][1] * tc_100 + self.aw[i][2]) / \
                     (self.aw[i][3] * tc_100 ** 2 + self.aw[i][4] * tc_100 + 1.)

        rho_w0 = a_w[0]
        Ew = a_w[1]
        Fw = a_w[2]

        if self.ni == 0:
            # Pure water
            Iw = (1. / Ew) * np.log(np.abs(Ew * (pressure / p0) + Fw))
            Iw0 = (1. / Ew) * np.log(np.abs(Ew * (p0 / p0) + Fw))
            rho = 1000 * rho_w0 * np.exp(Iw - Iw0)
        else:
            # Brine density
            Cm = 55.509 * x[self.nc] / x[self.H2O_idx]

            a_b = np.empty(8)
            for i in range(8):
                a_b[i] = (self.ab[i][0] * tc_100 ** 2 + self.ab[i][1] * tc_100 + self.ab[i][2]) / \
                         (self.ab[i][3] * tc_100 ** 2 + self.ab[i][4] * tc_100 + 1.)
            rho_b0 = rho_w0 + a_b[0] * Cm ** 2 + a_b[1] * Cm ** 1.5 + a_b[2] * Cm + a_b[3] * np.sqrt(Cm)
            Eb = Ew + a_b[4] * Cm
            Fb = Fw + a_b[5] * Cm ** 1.5 + a_b[6] * Cm + a_b[7] * np.sqrt(Cm)

            Ib = (1. / Eb) * np.log(np.abs(Eb * (pressure / p0) + Fb))
            Ib0 = (1. / Eb) * np.log(np.abs(Eb * (p0 / p0) + Fb))
            rho = 1000 * rho_b0 * np.exp(Ib - Ib0)

        return rho  # kg/m3

class Density_Garcia2001(Spivey2004):
    """
    Correlation for brine density with dissolved CO2: Garcia (2001) - Density of aqueous solutions of CO2
    """
    def __init__(self, components_names: list, ions: list = None):
        super().__init__(components_names, ions)

        self.CO2_idx = components_names.index("CO2") if "CO2" in components_names else None

    def evaluate(self, pressure: float, temperature: float, x: list):
        """
        :param pressure: Pressure [Pascal]
        :type pressure: float
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param x: Composition or list of mole fractions
        :type x: list
        """
        rho_b = super().evaluate(pressure, temperature, x)

        # If CO2 is present, correct density
        if self.CO2_idx is not None:
            # Apparent molar volume of dissolved CO2
            tc = temperature - 273.15  # Temp in [Celcius]
            V_app = (37.51 - 9.585e-2 * tc + 8.740e-4 * tc ** 2 - 5.044e-7 * tc ** 3) * 1e-6  # in [m3 / mol]

            mCO2 = 55.509 * x[self.CO2_idx] / (x[self.H2O_idx])
            MW = 44.01  # molecular weight of CO2
            rho = (1. + mCO2 * MW * 1e-3) / (mCO2 * V_app + 1. / rho_b)  # in [kg / m3]
        else:
            rho = rho_b

        return rho

# %% Enthalpy evaluators
class Enthalpy_cp:
    """
    This class calculates specific enthalpy based on the constant heat capacity at constant pressure.
    This method calculates specific enthalpy only as a function of temperature, but not pressure.
    """
    def __init__(self, Tref, cp):
        """
        :param Tref: Reference temperature [K]
        :param cp: Specific heat capacity at constant pressure [Joule / kg / K]
        """
        self.Tref = Tref
        self.cp = cp
    def evaluate(self, pressure: float, temperature: float, x: list):
        """
        :param temperature: Temperature [K]
        :type temperature: float
        :return: Specific enthalpy [Joule / kg]
        """
        enthalpy = self.cp * (temperature - self.Tref)
        return enthalpy

class Enthalpy_PR_PT:
    """
    This class calculates phase specific enthalpy using the Peng-Robinson (PR) EoS as a function of
    pressure, temperature (PT), and composition.
    This class can evaluate phase enthalpy. It evaluates ideal gas enthalpy and EoS-derived residual enthalpy.
    It Evaluates the EoS for residual enthalpy at given pressure, temperature and composition x.
    It Evaluates the ideal gas enthalpy at temperature and composition x.
    """
    def __init__(self, components_names: list):
        """
        :param components_names: Names of the components in the fluid model
        :type components_names: list
        """
        # Get components data
        self.comp_data = CompData(components_names, setprops=True)

        # Instantiate the Peng-Robinson EoS object
        self.eos_enthalpy = CubicEoS(self.comp_data, CubicEoS.PR)

    def evaluate(self, pressure: float, temperature: float, x: list):
        """
        :param pressure: Pressure in Pascal
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase specific enthalpy in Joule/kg
        :rtype: float
        """
        pressure = convertTo(pressure, bar())

        H = self.eos_enthalpy.H(pressure, temperature, x)  # H/R

        molar_enthalpy = H * R  # J/mol == kJ/kmol
        MW_apparent = sum(np.array(self.comp_data.Mw) * np.array(x)) * 1e-3   # in kg/mol
        specific_enthalpy = molar_enthalpy / MW_apparent
        return specific_enthalpy


class Enthalpy_AQ_PT:
    """
    This class calculates phase specific enthalpy using the aqueous fugacity model as a function of
    pressure, temperature (PT), and composition.
    This class can evaluate the aqueous phase enthalpy. It evaluates ideal gas enthalpy and EoS-derived residual enthalpy.
    It Evaluates the EoS for residual enthalpy at given pressure, temperature and composition x.
    It Evaluates the ideal gas enthalpy at temperature and composition x.
    """
    def __init__(self, components_names: list):
        """
        :param components_names: Names of the components in the fluid model
        :type components_names: list
        """
        # Get components data
        self.comp_data = CompData(components_names, setprops=True)

        # Instantiate the aqueous fugacity model
        self.eos_enthalpy = AQEoS(self.comp_data, AQEoS.Ziabakhsh2012)

    def evaluate(self, pressure: float, temperature: float, x: list):
        """
        :param pressure: Pressure in Pascal
        :type pressure: float
        :param temperature: Temperature in Kelvin
        :type temperature: float
        :param x: Phase composition in mole fractions/mole numbers
        :type x: list

        :returns: Phase specific enthalpy in Joule/kg
        :rtype: float
        """
        pressure = convertTo(pressure, bar())

        H = self.eos_enthalpy.H(pressure, temperature, x)  # H/R

        molar_enthalpy = H * R  # J/mol == kJ/kmol
        MW_apparent = sum(np.array(self.comp_data.Mw) * np.array(x)) * 1e-3   # in kg/mol
        specific_enthalpy = molar_enthalpy / MW_apparent
        return specific_enthalpy

# %% Internal energy evaluators
class InternalEnergy:
    """
    This class evaluates phase specific internal energy.
    """
    def __init__(self):
        pass  # It does not need any constructor.

    def evaluate(self, p, h, rho):
        """

        :param p: Pressure [Pa]
        :type p: float
        :param h: Specific enthalpy [Joule/kg]
        :param rho: Density [kg/m3]
        :return: Internal energy [Joule/kg]
        """
        U = h - p / rho
        return U

#%% Viscosity evaluators
# class Viscosity:
#     def __init__(self, components_names: list = None, ions_names: list = None):
#         self.nc = len(components_names) if components_names is not None else 0
#         self.ni = len(ions_names) if ions_names is not None else 0
#
#     def evaluate(self, pressure, temperature, x, rho):
#         pass
#
# class Viscosity_Fenghouretal1998(Viscosity):
#     """
#     Correlation for CO2 viscosity: Fenghour, Wakeham & Vesovic (1998)
#     """
#     def __init__(self):
#         # This correlation is only used for CO2.
#         self.component_name = "CO2"
#
#     def evaluate(self, temperature, rho):
#         """
#         :param temperature: Temperature in Kelvin
#         :type temperature: float
#         :param rho: Density in kg/m3
#         :type rho: float
#
#         :returns: CO2 viscosity in Pa.s
#         """
#         a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
#         d = [0.4071119e-2, 0.7198037e-4, 0.2411967e-16, 0.2971072e-22, -0.1627888e-22]  # d11, d21, d64, d81, d82
#         # Viscosity in zero density limit
#         eps_k = 251.196  # energy scaling parameter eps/k [K]
#         T_ = 1/eps_k * temperature
#         lnT_ = np.log(T_)
#
#         lnG = a[0]
#         for i in range(1, 5):
#             lnG += a[i] * lnT_ ** i
#         G = np.exp(lnG)
#
#         n0 = 1.00697*np.sqrt(temperature) / G
#
#         # Excess viscosity
#         dn = d[0] * rho + d[1] * rho ** 2 + d[2] * rho ** 6 / (T_ ** 3) + d[3] * rho ** 8 + d[4] * rho ** 8 / T_
#
#         # Correction of viscosity in vicinity of critical point
#         dnc = 0
#
#         miu = (n0 + dn + dnc) * 1e-6  # microPa.s to Pa.s
#         return miu

class Viscosity:
    def __init__(self, components: list = None, ions: list = None):
        self.nc = len(components) if components is not None else 0
        self.ni = len(ions) if ions is not None else 0

    def evaluate(self, pressure, temperature, x, rho):
        pass


class Viscosity_Fenghouretal1998(Viscosity):
    """
    Correlation for CO2 viscosity: Fenghour, Wakeham & Vesovic (1998) - The Viscosity of CO2
    """
    a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
    d = [0.4071119e-2, 0.7198037e-4, 0.2411967e-16, 0.2971072e-22, -0.1627888e-22]  # d11, d21, d64, d81, d82

    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        """
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param rho: Density [kg/m3]
        :type rho: float

        :returns: CO2 viscosity [Pa.s]
        """
        # Viscosity in zero density limit
        eps_k = 251.196  # energy scaling parameter eps/k [K]
        T_ = 1/eps_k * temperature
        lnT_ = np.log(T_)

        lnG = self.a[0]
        for i in range(1, 5):
            lnG += self.a[i] * lnT_ ** i
        G = np.exp(lnG)

        n0 = 1.00697*np.sqrt(temperature) / G

        # Excess viscosity
        dn = self.d[0] * rho + self.d[1] * rho ** 2 + self.d[2] * rho ** 6 / (T_ ** 3) + \
             self.d[3] * rho ** 8 + self.d[4] * rho ** 8 / T_

        # Correction of viscosity in vicinity of critical point
        dnc = 0

        n = (n0 + dn + dnc) * 1e-6  # microPa.s to Pa.s
        return n


class Viscosity_MaoDuan2009(Viscosity):
    """
    Correlation for brine + NaCl viscosity: Mao and Duan (2009) - The Viscosity of Aqueous Alkali-Chloride Solutions
                                                                  up to 623 K, 1,000 bar, and High Ionic Strength
    """
    # # Data from Mao and Duan (2009) for pure water viscosity
    # mu_d = [0.28853170e7, -0.11072577e5, -0.90834095e1, 0.30925651e-1, -0.27407100e-4,
    #         -0.19283851e7, 0.56216046e4, 0.13827250e2, -0.47609523e-1, 0.35545041e-4]

    # Data from Islam and Carlson (2012) for pure water viscosity
    mu_a = 9.03591045e1
    mu_b = [3.40285740e4, 8.23556123e8, -9.28022905e8]
    mu_c = [1.40090092e-2, 4.86126399e-2, 5.26696663e-2]
    mu_d = [-1.22757462e-1, 2.15995021e-2, -3.65253919e-4, 1.97270835e-6]

    # Data from Islam and Carlson (2012) for pure water density
    rho_a = 1.34136579e2
    rho_b = [-4.07743800e3, 1.63192756e4, 1.37091355e3]
    rho_c = [-5.56126409e-3, -1.07149234e-2, -5.46294495e-4]
    rho_d = [4.45861703e-1, -4.51029739e-4]

    # Data from Mao and Duan (2009) for relative viscosity of solutions of NaCl
    a = [-0.21319213, 0.13651589e-2, -0.12191756e-5]
    b = [0.69161945e-1, -0.27292263e-3, 0.20852448e-6]
    c = [-0.25988855e-2, 0.77989227e-5]

    def __init__(self, components_names: list, ions: list = None, combined_ions: list = None):
        super().__init__(components_names, ions)

        self.H2O_idx = components_names.index("H2O")
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x, rho):
        """
        :param pressure: Pressure [Pa]
        :type pressure: float
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param x: Composition [mole fractions]
        :type x: list
        :param rho: Density [kg/m3]
        :type rho: float

        :returns: CO2 viscosity [Pa.s]
        """
        pressure = convertTo(pressure, bar())

        # Density of pure water (Islam and Carlson, 2012)
        rhoH2O = self.rho_a
        for i in range(3):
            rhoH2O += self.rho_b[i] * 10 ** (self.rho_c[i] * temperature)
        for i in range(2):
            rhoH2O += self.rho_d[i] * pressure ** (i+1)

        # Viscosity of pure water (Islam and Carlson, 2012)
        muH2O = self.mu_a
        for i in range(3):
            muH2O += self.mu_b[i] * np.exp(-self.mu_c[i] * temperature)
        for i in range(4):
            muH2O += pressure * 0.1 * self.mu_d[i] * (temperature - 293.15) ** i

        # # Viscosity of pure water (Mao and Duan, 2009)
        # muH2O = 0.
        # for i in range(5):
        #     muH2O += self.mu_d[i] * temperature ** (i-3)
        # for i in range(5, 10):
        #     muH2O += self.mu_d[i] * rhoH2O * temperature ** (i-8)
        # muH2O = np.exp(muH2O)

        # Relative viscosity of solution (Mao and Duan, 2009)
        A = self.a[0] + self.a[1] * temperature + self.a[2] * temperature ** 2
        B = self.b[0] + self.b[1] * temperature + self.b[2] * temperature ** 2
        C = self.c[0] + self.c[1] * temperature
        if self.combined_ions is not None:
            m = 55.509 * x[self.nc] / x[self.H2O_idx]
        else:
            m = np.sum([55.509 * x[i] / x[self.H2O_idx] for i in range(self.nc, self.nc + self.ni)]) * 0.5  # half because sum of ions molality is double NaCl molality

        mu_r = np.exp(A * m + B * m ** 2 + C * m ** 3)
        mu = mu_r * muH2O

        return mu * 1e-6


class Viscosity_Islam2012(Viscosity_MaoDuan2009):
    """
    Correlation for H2O + NaCl + CO2 viscosity: Islam and Carlson (2012) - Viscosity Models and Effects of Dissolved CO2
    """
    def __init__(self, components_names: list, ions: list = None, combined_ions: list = None):
        super().__init__(components_names, ions, combined_ions)

        self.CO2_idx = components_names.index("CO2") if "CO2" in components_names else None

    def evaluate(self, pressure, temperature, x, rho):
        """
        :param pressure: Pressure [Pa]
        :type pressure: float
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param x: Composition [mole fractions]
        :type x: list
        :param rho: Density [kg/m3]
        :type rho: float

        :returns: CO2 viscosity [Pa.s]
        """
        mu_brine = super().evaluate(pressure, temperature, x, rho)

        # Viscosity of Aq + CO2
        if self.CO2_idx is not None:
            mu_brine *= 1. + 4.65 * x[self.CO2_idx] ** 1.0134

        return mu_brine


#%% IFT evaluators
class IFT_MCM:
    """
    Liquid-gas interfacial tension (IFT) or surface tension for pure fluids
    Macleod-Sugden surface tension model (MCS) (used in Multiflash of OLGA, PVTi, WinProp, and McCain's PVT book)
    This IFT model is for fluids containing a single component
    The parameters used in this correlation are as follows:
    parachor is the parachor of the component (parachors of different components are available in the library file)
    rhoG_molar is the molar density of the gaseous phase [mol/cm3]
    rhoL_molar is the molar density of the liquid phase [mol/cm3]
    rhoG is the molar density of the gaseous phase [gram/cm3]
    rhoL is the molar density of the liquid phase in gram/cm3
    MW is the molecular weight of the component [gram/mol]
    IFT that this correlation gives is in dyne/cm.
    """
    def __init__(self, component_name: str):
        """
        :param component_name: Name of the component
        :type component_name: str ---> The input must be a str, not a list because the class can be used only for pure components, not mixtures.
        """
        self.component_name = component_name

        # Get component MW and parachor from the library
        try:
            self.MW = library.components_molecular_weights[component_name]
        except:
            raise Exception(f"Molecular weight of {component_name} is not in the library!")

        try:
            self.parachor = library.components_parachors[component_name]
        except:
            raise Exception(f"Parachor of {component_name} is not in the library!")

    def evaluate(self, rhoG, rhoL):
        """
        :param rhoG: Gas density in kg/m3
        :param rhoL: Liquid density in kg/m3

        :returns IFT: Interfacial tension in N/m
        """
        # IFT = (parachor * (rhoL_molar - rhoG_molar)) ** 4   # for molar densities
        rhoG = convertTo(rhoG, gram() / (centi() * meter()) ** 3)
        rhoL = convertTo(rhoL, gram() / (centi() * meter()) ** 3)
        IFT = (self.parachor * (rhoL - rhoG) / self.MW) ** 4  # for mass densities
        # Convert IFT from MCS correlation (dyne/cm) to N/m
        IFT = IFT * dyne() / (centi() * meter())
        return IFT


class IFT_multicomponent_MCM:
    """
    Liquid-gas interfacial tension (IFT) or surface tension for multi-component fluids
    Macleod-Sugden surface tension model (MCS) (used in Multiflash of OLGA, PVTi, WinProp, and McCain's PVT book)
    The parameters used in this correlation are as follows:
    parachor is the parachor of the component (parachors of different components are available in the library file)
    rhoG_molar is the molar density of the gaseous phase [mol/cm3]
    rhoL_molar is the molar density of the liquid phase [mol/cm3]
    rhoG is the molar density of the gaseous phase [gram/cm3]
    rhoL is the molar density of the liquid phase in gram/cm3
    MW is the molecular weight of the component [gram/mol]
    IFT that this correlation gives is in dyne/cm.
    """
    def __init__(self, components_names: list):
        """
        :param components_names: Names of the components
        :type components_names: list
        """
        self.components_names = components_names
        num_components = len(components_names)

        # Get components MW and parachor from the library
        self.MW = np.zeros(num_components)
        self.parachor = np.zeros(num_components)

        for i in range(num_components):
            try:
                self.MW[i] = library.components_molecular_weights[components_names[i]]
            except:
                raise Exception(f"Molecular weight of {components_names[i]} is not in the library!")

            try:
                self.parachor[i] = library.components_parachors[components_names[i]]
            except:
                raise Exception(f"Parachor of {components_names[i]} is not in the library!")

    def evaluate(self, rhoG, rhoL, xG_mass, xL_mass):
        """
        :param rhoG: Gas density in kg/m3
        :param rhoL: Liquid density in kg/m3
        :param xL_mass: Mass fractions of components in the liquid phase
        :param xG_mass: Mass fractions of components in the gaseous phase

        :returns IFT: Interfacial tension in N/m
        """
        # IFT = (parachor * (rhoL_molar - rhoG_molar)) ** 4   # for molar densities
        rhoG = convertTo(rhoG, gram() / (centi() * meter()) ** 3)
        rhoL = convertTo(rhoL, gram() / (centi() * meter()) ** 3)
        IFT = (sum(self.parachor * (rhoL * xL_mass - rhoG * xG_mass) / self.MW)) ** 4  # for mass densities
        # Convert IFT from MCS correlation (dyne/cm) to N/m
        IFT = IFT * dyne() / (centi() * meter())
        return IFT

#%% Relative permeability evaluators
class PhaseRelPerm_BC:
    """
    This class gives phase relative permeabilities using the Brooks and Corey model
    """
    def __init__(self, phase, swc=0., sgr=0., kre=1., n=2.):
        self.phase = phase

        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
        else:  # water
            self.kre = kre
            self.sr = 0
            self.sr1 = 0
            self.n = n

    def evaluate(self, sat):
        if sat >= 1 - self.sr1:
            kr = self.kre
        elif sat <= self.sr:
            kr = 0
        else:
            # general Brooks-Corey
            kr = self.kre * ((sat - self.sr) / (1 - self.Sgr - self.Swc)) ** self.n

        return kr


class PhaseRelPerm_VG:
    """
    This class gives phase relative permeabilities using the Van Genuchten model
    """
    def __init__(self, phase, swc=0.20, sgr=0, kre=1., n=4.367):
        self.phase = phase
        self.Swc = swc
        self.Sgr = sgr
        if phase == "oil":
            self.kre = kre
            self.sr = swc
            self.sr1 = sgr
            self.n = n
            self.m = 1 - 1/n
        elif phase == 'gas':
            self.kre = kre
            self.sr = sgr
            self.sr1 = swc
            self.n = n
            self.m = 1 - 1/n
        else:  # water
            self.kre = 1
            self.sr = 0
            self.sr1 = 0
            self.n = n
            self.m = 1 - 1/n

    def evaluate(self, sat):
        if self.phase == "oil":
            Se = (sat - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = np.sqrt(Se) * (1 - (1 - Se ** (1/self.m)) ** self.m) ** 2

        elif self.phase == 'gas':
            Se = ((1-sat) - self.Swc) / (1 - self.Sgr - self.Swc)
            if sat >= 1 - self.sr1:
                kr = self.kre
            elif sat <= self.sr:
                kr = 0
            else:
                kr = (1 - Se) ** (1/3) * (1 - Se ** (1/self.m)) ** (2*self.m)
        return kr


#%% Conductivity evaluators
class Conductivity:
    def __init__(self):
        pass

    def evaluate(self):
        pass
