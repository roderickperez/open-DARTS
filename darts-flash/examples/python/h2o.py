import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoS, InitialGuess
from dartsflash.libflash import PXFlash
from dartsflash.libflash import CubicEoS, AQEoS

from dartsflash.dartsflash import DARTSFlash, CompData, R
from dartsflash.plot import *


""" CREATE INSTANCE OF IAPWS/V-L-Aq DARTSFLASH OBJECT """
comp_data = CompData(components=["H2O"], setprops=True)
z = [1.]
compositions = {"H2O": 1.}

if 1:
    from dartsflash.mixtures import IAPWS
    f = IAPWS(iapws_ideal=False, ice_phase=False)

    vl_eos = f.eos["IAPWS"]
    aq = f.eos["IAPWS"]

    eos_order = ["IAPWS"]
    suff = "-iapws"
else:
    from dartsflash.mixtures import VLAq
    f = VLAq(comp_data=comp_data, hybrid=True)

    f.set_vl_eos("PR", switch_tol=1e-3)
    f.set_aq_eos("Aq", use_gmix=True)

    vl_eos = f.eos["VL"]
    aq = f.eos["Aq"]

    eos_order = ["Aq", "VL"]
    suff = "-hybrid"

T_min = 250.
T_max = 750.

Trange = [250., 700.]
Prange = [10., 250.]
logy = False

if 0:
    flash_type = DARTSFlash.FlashType.PHFlash
    plot_method = PlotFlash.ph
    suff += "-ph"
    Xrange = [-40000., 10000.]
else:
    flash_type = DARTSFlash.FlashType.PSFlash
    plot_method = PlotFlash.ps
    suff += "-ps"
    Xrange = [-100, 0]
f.init_flash(flash_type=flash_type, t_min=T_min, t_max=T_max)

""" SPECIFY STATES """
logy = False
state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 100) if not logy
                        else np.logspace(np.log10(Prange[0]), np.log10(Prange[1]), 100),
              "enthalpy" if flash_type == DARTSFlash.FlashType.PHFlash else "entropy":
                  np.linspace(Xrange[0], Xrange[1], 100),
              }

flash_results = f.evaluate_flash_1c(state_spec=state_spec, print_state="Flash")

plot_pt = False
if plot_pt:
    state_pt = {"pressure": state_spec["pressure"],
                "temperature": np.linspace(Trange[0], Trange[1], 100),}
    pt_props = f.evaluate_properties_1p(state_spec=state_pt, compositions=compositions, mole_fractions=True,
                                        properties_to_evaluate={"H": vl_eos.H} if flash_type == DARTSFlash.FlashType.PHFlash
                                                          else {"S": vl_eos.S}
                                        )
else:
    pt_props = None

plot = plot_method(f, flash_results, composition=z, min_temp=T_min, max_temp=T_max, min_val=0., max_val=1.,
                   plot_phase_fractions=True, pt_props=pt_props)

plt.savefig(f.filename + "-" + "-".join(str(int(zi*100)) for zi in z) + suff + ".pdf")

plt.show()
