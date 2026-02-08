import numpy as np
import warnings
from abc import abstractmethod, ABC

from dartsflash.components import CompData
from dartsflash.libflash import CubicEoS

from dwell.utilities.units import *
import dwell.utilities.library as library

# Constants
NA = 6.02214076e23 * 1 / mol()  # Avogadro's number
kB = 1.380649e-23 * Joule() / Kelvin()  # Boltzmann constant
R = NA * kB * Joule() / (mol() * Kelvin())  # Universal as constant

class FluidModel:
    def __init__(self, pipe_name: str, components_names: list,
                 flash_calcs_obj, density_obj, enthalpy_obj, internal_energy_obj, viscosity_obj, IFT_obj, verbose: bool = False):
        """
        :param pipe_name: Name of the pipe for which the fluid model is going to be defined
        :type pipe_name: str
        :param components_names: List of names of components
        :type components_names: list
        :param flash_calcs_obj: Object for flash calculations
        :type flash_calcs_obj: Classes available for flash calculations
        :param density_obj: Object for density evaluation
        :type density_obj: Classes available for density evaluation
        :param enthalpy_obj: Object for enthalpy evaluation
        :type enthalpy_obj: Classes available for enthalpy evaluation
        :param internal_energy_obj: Object for internal energy evaluation
        :type internal_energy_obj: Classes available for internal energy evaluation
        :param viscosity_obj: Object for viscosity evaluation
        :type viscosity_obj: Classes available for viscosity evaluation
        :param IFT_obj: Object for IFT evaluation for 2-phase fluid flow
        :type IFT_obj: Classes available for IFT evaluation
        :param verbose: Whether to display extra info about FluidModel
        :type verbose: boolean
        """
        self.pipe_name = pipe_name

        self.components_names = components_names

        self.flash_calcs_obj = flash_calcs_obj
        self.density_obj = density_obj
        self.enthalpy_obj = enthalpy_obj
        self.internal_energy_obj = internal_energy_obj
        self.viscosity_obj = viscosity_obj
        self.IFT_obj = IFT_obj

        if verbose:
            print("** Fluid model of the pipe \"%s\" is defined!" % self.pipe_name)

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

# %% Density evaluators
class Density(ABC):
    """
    This abstract class is used to make the programmer use the same evaluate method for all the density evaluators.
    """
    @abstractmethod
    def evaluate(self, pressure, temperature, x):
        pass

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
        :param pressure: Pressure [Pascal]
        :type pressure: float
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param x: Phase composition
        :type x: list of floats

        :returns: Phase density [kg/m3]
        :rtype: float
        """
        pressure = convertTo(pressure, bar())

        MW_apparent = np.sum(x * np.array(self.comp_data.Mw)) * 1e-3  # MW in kg/mol
        # Here, the EoS gives the molar volume from pressure, temperature, and composition
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
        # super().__init__(components)

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
        :param pressure: Pressure [Pascal]
        :type pressure: float
        :param temperature: Temperature [Kelvin]
        :type temperature: float
        :param x: Phase composition
        :type x: list

        :returns: Phase specific enthalpy [Joule/kg]
        :rtype: float
        """
        pressure = convertTo(pressure, bar())

        H = self.eos_enthalpy.H(pressure, temperature, x)  # H/R

        molar_enthalpy = H * R  # J/mol == kJ/kmol
        MW_apparent = sum(np.array(self.comp_data.Mw) * np.array(x)) * 1e-3   # in kg/mol
        specific_enthalpy = molar_enthalpy / MW_apparent   # in J/kg
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
class Viscosity:
    def __init__(self, components_names: list = None, ions_names: list = None):
        self.nc = len(components_names) if components_names is not None else 0
        self.ni = len(ions_names) if ions_names is not None else 0

    def evaluate(self, pressure, temperature, x, rho):
        pass


class Viscosity_Fenghouretal1998(Viscosity):
    """
    Correlation for pure CO2 viscosity: Fenghour, Wakeham & Vesovic (1998) - The viscosity of CO2
    """
    a = [0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2]
    d = [0.4071119e-2, 0.7198037e-4, 0.2411967e-16, 0.2971072e-22, -0.1627888e-22]  # d11, d21, d64, d81, d82

    def __init__(self):
        super().__init__()

    def evaluate(self, pressure, temperature, x, rho):
        """
        :param temperature: Temperature [Kelvin]
        :type temperature: float or list
        :param rho: Density [kg/m3]
        :type rho: float or list

        :returns: CO2 viscosity in Pa.s
        """
        # Viscosity in zero density limit
        eps_k = 251.196  # energy scaling parameter eps/k [K]
        T_ = 1 / eps_k * temperature
        lnT_ = np.log(T_)

        lnG = self.a[0]
        for i in range(1, 5):
            lnG += self.a[i] * lnT_ ** i
        G = np.exp(lnG)

        n0 = 1.00697 * np.sqrt(temperature) / G

        # Excess viscosity
        dn = self.d[0] * rho + self.d[1] * rho ** 2 + self.d[2] * rho ** 6 / (T_ ** 3) + \
             self.d[3] * rho ** 8 + self.d[4] * rho ** 8 / T_

        # Correction of viscosity in vicinity of critical point
        dnc = 0

        n = (n0 + dn + dnc) * 1e-6  # muPa.s to cP
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

    def __init__(self, components_names: list, ions_names: list = None, combined_ions: list = None):
        super().__init__(components_names, ions_names)

        self.H2O_idx = components_names.index("H2O")
        self.combined_ions = combined_ions

    def evaluate(self, pressure, temperature, x, rho):
        """
        :param pressure: Pressure [Pa]
        :type pressure: float or list
        :param temperature: Temperature [Kelvin]
        :type temperature: float or list
        :param x: Composition (components mole fractions)
        :type rho: list

        :returns: CO2 viscosity in Pa.s
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
        mu = mu_r * muH2O  # Pa.s

        return mu * 1e-6


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
        :type component_name: str ---> The input must be a str because the class can be used only for pure components, not mixtures.
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

class Conductivity:
    def __init__(self):
        pass

    def evaluate(self):
        pass
