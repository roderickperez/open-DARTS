from enum import Enum
import numpy as np
import dartsflash.libflash
from dartsflash.libflash import Units


_Tc = {"H2O": 647.14, "CO2": 304.10, "N2": 126.20, "H2S": 373.53, "C1": 190.58, "C2": 305.32, "C3": 369.83, "iC4": 407.85, "nC4": 425.12, "iC5": 460.45, "nC5": 469.70, "nC6": 507.60, "nC7": 540.20, "nC8": 569.32, "nC9": 594.6, "nC10": 617.7, "H2" : 33.145, }
_Pc = {"H2O": 220.50, "CO2": 73.75, "N2": 34.00, "H2S": 89.63, "C1": 46.04, "C2": 48.721, "C3": 42.481, "iC4": 36.4, "nC4": 37.960, "iC5": 33.77, "nC5": 33.701, "nC6": 30.251, "nC7": 27.40, "nC8": 24.97, "nC9": 22.88, "nC10": 21.2, "H2": 12.93, }
_ac = {"H2O": 0.328, "CO2": 0.239, "N2": 0.0377, "H2S": 0.0942, "C1": 0.012, "C2": 0.0995, "C3": 0.1523, "iC4": 0.1844, "nC4": 0.2002, "iC5": 0.227, "nC5": 0.2515, "nC6": 0.3013, "nC7": 0.3495, "nC8": 0.396, "nC9": 0.445, "nC10": 0.489, "H2": -0.219, }
_Mw = {"H2O": 18.015, "CO2": 44.01, "N2": 28.013, "H2S": 34.10, "C1": 16.043, "C2": 30.07, "C3": 44.097, "iC4": 58.124, "nC4": 58.124, "iC5": 72.151, "nC5": 72.151, "nC6": 86.178, "nC7": 100.205, "nC8": 114.231, "nC9": 128.257, "nC10": 142.2848,
       "Na+": 22.99, "Ca2+": 40.08, "K+": 39.0983, "Cl-": 35.45, "H2": 2.016, }

_kij = {"H2O": {"H2O": 0., "CO2": 0.19014, "N2": 0.32547, "H2S": 0.105, "C1": 0.47893, "C2": 0.5975, "C3": 0.5612, "iC4": 0.508, "nC4": 0.5569, "iC5": 0.5, "nC5": 0.5260, "nC6": 0.4969, "nC7": 0.4880, "nC8": 0.48, "nC9": 0.48, "nC10": 0.48, "H2": 0.0, },
        "CO2": {"H2O": 0.19014, "CO2": 0., "N2": -0.0462, "H2S": 0.1093, "C1": 0.0936, "C2": 0.1320, "C3": 0.1300, "iC4": 0.13, "nC4": 0.1336, "iC5": 0.13, "nC5": 0.1454, "nC6": 0.1167, "nC7": 0.1209, "nC8": 0.1, "nC9": 0.1, "nC10": 0.1, "H2": 0.0, },
        "N2": {"H2O": 0.32547, "CO2": -0.0462, "N2": 0., "H2S": 0.1475, "C1": 0.0291, "C2": 0.0082, "C3": 0.0862, "iC4": 0.1, "nC4": 0.0596, "iC5": 0.1, "nC5": 0.0917, "nC6": 0.1552, "nC7": 0.1206, "nC8": 0.1, "nC9": 0.1, "nC10": 0.1, "H2": 0.0, },
        "H2S": {"H2O": 0.105, "CO2": 0.1093, "N2": 0.1475, "H2S": 0., "C1": 0.0912, "C2": 0.0846, "C3": 0.0874, "iC4": 0.06, "nC4": 0.0564, "iC5": 0.06, "nC5": 0.0655, "nC6": 0.0465, "nC7": 0.0191, "nC8": 0, "nC9": 0, "nC10": 0.1, },
        "C1": {"H2O": 0.47893, "CO2": 0.0936, "N2": 0.0291, "H2S": 0.0912, "C1": 0., "C2": 0.00518, "C3": 0.01008, "iC4": 0.026717, "nC4": 0.0152, "iC5": 0.0206, "nC5": 0.0193, "nC6": 0.0258, "nC7": 0.0148, "nC8": 0.037, "nC9": 0.03966, "nC10": 0.048388, "H2": -0.1622, },
        "C2": {"H2O": 0.5975, "CO2": 0.1320, "N2": 0.0082, "H2S": 0.0846, "C1": 0.00518, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "nC5": 0., "iC5": 0, "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "C3": {"H2O": 0.5612, "CO2": 0.1300, "N2": 0.0862, "H2S": 0.0874, "C1": 0.01008, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "nC5": 0., "iC5": 0, "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "iC4": {"H2O": 0.508, "CO2": 0.13, "N2": 0.1, "H2S": 0.06, "C1": 0.026717, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC4": {"H2O": 0.5569, "CO2": 0.1336, "N2": 0.0596, "H2S": 0.0564, "C1": 0.0152, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "iC5": {"H2O": 0.5, "CO2": 0.13, "N2": 0.1, "H2S": 0.06, "C1": 0.0206, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC5": {"H2O": 0.5260, "CO2": 0.1454, "N2": 0.0917, "H2S": 0.0655, "C1": 0.0193, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC6": {"H2O": 0.4969, "CO2": 0.1167, "N2": 0.1552, "H2S": 0.0465, "C1": 0.0258, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC7": {"H2O": 0.4880, "CO2": 0.1209, "N2": 0.1206, "H2S": 0.0191, "C1": 0.0148, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC8": {"H2O": 0.48, "CO2": 0.1, "N2": 0.1, "H2S": 0, "C1": 0.037, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC9": {"H2O": 0.48, "CO2": 0.1, "N2": 0.1, "H2S": 0, "C1": 0.03966, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "nC10": {"H2O": 0.48, "CO2": 0.1, "N2": 0.1, "H2S": 0.1, "C1": 0.048388, "C2": 0., "C3": 0., "iC4": 0, "nC4": 0., "iC5": 0, "nC5": 0., "nC6": 0., "nC7": 0., "nC8": 0, "nC9": 0, "nC10": 0, "H2": 0.0, },
        "H2": {"H2": 0., "C1": -0.1622, "H2O": 0.0, },
        }

_H0 = {"H2O": 0., "CO2": 33., "N2": 0.64, "H2S": 100., "C1": 1.4, "C2": 1.9, "C3": 1.5, "iC4": 0.91, "nC4": 1.2, "iC5": 0.7, "nC5": 0.8, "nC6": 0.61, "nC7": 0.44, "nC8": 0.31, "nC9": 0.2, "nC10": 0.14, "H2": 0.77, }
_dlnH0 = {"H2O": 0., "CO2": 2400., "N2": 1600., "H2S": 2100., "C1": 1900., "C2": 2400., "C3": 2700., "iC4": 2700., "nC4": 3100., "iC5": 3400., "nC5": 3400., "nC6": 3800., "nC7": 4100., "nC8": 4300., "nC9": 5000., "nC10": 5000., "H2": 500, }

_charge = {"Na+": 1, "Cl-": -1, "Ca2+": 2, "K+": 1}

_comp_labels = {"H2O": r"H$_2$O", "CO2": r"CO$_2$", "N2": r"N$_2$", "H2S": r"H$_2$S", "H2": r"H$_2$",
                "C1": r"C$_1$", "C2": r"C$_2$", "C3": r"C$_3$", "iC4": r"iC$_4$", "nC4": r"nC$_4$", "iC5": r"iC$_5$",
                "nC5": r"nC$_5$", "nC6": r"nC$_6$", "nC7": r"nC$_7$", "nC8": r"nC$_8$", "nC9": r"nC$_9$", "nC10": r"nC$_{10}$",
                "Na+": r"Na$^+$", "Ca2+": r"Ca$^{2+}$", "K+": r"K$^+$", "Cl-": r"Cl$^-$",
                "NaCl": r"NaCl", "CaCl2": r"CaCl$_2$", "KCl": r"KCl"
                }


def get_properties(property: dict, species: list):
    return np.array([property[i] if i in property.keys() else 0. for i in species])


class ConcentrationUnits(Enum):
    MOLALITY = 0
    WEIGHT = 1
cu = ConcentrationUnits


class CompData(dartsflash.libflash.CompData):
    """
    This class contains component properties and data.

    :ivar nc: Number of components
    :type nc: int
    :ivar ni: Number of ions
    :type ni: int
    :ivar ns: Number of species (components + ions)
    :type ns: int
    :ivar Pc: List of component critical pressures [bar]
    :type Pc: list
    :ivar Tc: List of component critical temperatures [K]
    :type Tc: list
    :ivar ac: List of component acentric factors [-]
    :type ac: list
    :ivar Mw: List of species molar weight [g/mol]
    :type Mw: list
    :ivar kij: List of component binary interaction coefficients (flattened 2D array)
    :type kij: list
    :ivar H0: List of component H0 (Sander, 2006)
    :type H0: list
    :ivar dlnH0: List of component dlnH0 (Sander, 2006)
    :type dlnH0: list
    :ivar charge: List of ion charges
    :type charge: list
    """
    def __init__(self, components: list, ions: list = None, setprops: bool = True):
        """        
        :param components: List of components
        :type components: list
        :param ions: List of ions, default is None, sets empty list
        :type ions: list
        :param setprops: Switch to get properties from pre-defined data, default is True
        :type setprops: bool
        :param units: Object that contains units and methods for unit conversion
        :type units: :class:`dartsflash.libflash.Units`
        """
        super().__init__(components, ions if ions is not None else [])

        self.components = components
        self.ions = ions if ions is not None else []
        self.species = components + ions if ions is not None else components
        self.comp_labels = get_properties(_comp_labels, self.species)

        self.H2O_idx = components.index("H2O") if "H2O" in components else None
        self.salt_stoich = {"NaCl": {0: 1, 1: 1}, "CaCl2": {0: 1, 1: 2}, "KCl": {0: 1, 1: 1}}
        self.salt_mass = {"NaCl": 53.99, "CaCl2": 110.98, "KCl": 74.5513}

        self.set_properties(setprops)

    def set_properties(self, setprops: bool = False):
        """
        Function to populate properties with pre-defined properties from data at the top of this file
        """
        self.Pc = get_properties(_Pc, self.components) if setprops else np.zeros(self.ns)
        self.Tc = get_properties(_Tc, self.components) if setprops else np.zeros(self.ns)
        self.ac = get_properties(_ac, self.components) if setprops else np.zeros(self.ns)
        self.Mw = get_properties(_Mw, self.species) if setprops else np.zeros(self.ns)
        self.kij = np.array([get_properties(_kij[i], self.components) for i in self.components]).flatten() if setprops \
                        else np.zeros(self.ns*self.ns)
        self.H0 = get_properties(_H0, self.components) if setprops else np.zeros(self.ns)
        self.dlnH0 = get_properties(_dlnH0, self.components) if setprops else np.zeros(self.ns)

        self.charge = get_properties(_charge, self.ions) if self.ions else []

        return

    def calculate_concentrations(self, ni: np.ndarray, mole_fractions: bool = False, concentrations: dict = None,
                                 concentration_unit: ConcentrationUnits = ConcentrationUnits.MOLALITY):
        # Translate concentration into composition of dissolved components
        nH2O = ni[self.H2O_idx]
        ni = np.append(ni, np.zeros(self.ni))
        if concentration_unit == ConcentrationUnits.WEIGHT:
            M_H2O = nH2O * self.Mw[self.H2O_idx]
            for comp, ci in concentrations.items():
                Mw_comp = self.salt_mass[comp]
                Ni = 1. / Mw_comp * M_H2O * (1./(1.-ci) - 1.)  # weight fraction to mole number conversion
                for i, stoich in self.salt_stoich[comp].items():
                    ni[self.nc + i] = Ni * stoich

        elif concentration_unit == ConcentrationUnits.MOLALITY:
            for comp, ci in concentrations.items():
                Ni = ci * nH2O / 55.509
                for i, stoich in self.salt_stoich[comp].items():
                    ni[self.nc + i] = Ni * stoich

        if mole_fractions:
            ni = ni / np.sum(ni)
        return ni