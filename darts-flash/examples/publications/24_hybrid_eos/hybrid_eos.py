import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS

from dartsflash.components import CompData
from dartsflash.mixtures import VLAq
from dartsflash.dartsflash import DARTSFlash
from dartsflash.plot import *


components = ["H2O", "CO2"]
comp_data = CompData(components, setprops=True)

""" HYBRID-EOS """
f = VLAq(comp_data=comp_data, hybrid=True)

# Add CubicEoS with preferred roots for actual flash calculations
ceos = CubicEoS(comp_data, CubicEoS.PR)
f.set_vl_eos(ceos, root_order=[EoS.RootFlag.MAX, EoS.RootFlag.MIN],
             trial_comps=[0, 1], switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)

# Add Aq EoS
aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                       AQEoS.solute: AQEoS.Ziabakhsh2012})
f.set_aq_eos(aq, stability_tol=1e-2, max_iter=10, use_gmix=True)

f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
             stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
             split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2)

""" PLOT GIBBS ENERGY SURFACES """
p, T = 30., 350.
state_spec = {"temperature": T,
              "pressure": p,
              }
dz = 0.001
min_z = [0.]
max_z = [1.]
compositions = {comp: np.arange(min_z[i], max_z[i]+0.1*dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.

""" Plot Gibbs energy surfaces and flash results """
composition = [0.95]
composition += [1. - sum(composition)]

props = {"G": EoS.GIBBS}
ge1p_results = f.evaluate_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                        print_state="GE 1P", properties_to_evaluate=props, mix=True)
flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                 print_state="Flash")
genp_results = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions, state_variables=['pressure', 'temperature'] + components,
                                        flash_results=flash_results, properties_to_evaluate=props, mix=True, print_state="GE NP")

plot = PlotProps.binary(f, state=state_spec, prop_name="G", variable_comp_idx=0, dz=dz, min_z=min_z, max_z=max_z,
                        flash_results=flash_results, props=ge1p_results, composition_to_plot=composition,
                        datalabels=["Aq", "CEOS-V", "CEOS-L"], ax_label=r"G$^m$/R")
plot.set_axes(ylim=[-50, 500],
              # xlim=[0.8, 1.]
              )
plt.savefig("GE-" + f.filename + "-" + str(int(p)) + "-" + str(int(T)) + ".pdf")

plt.show()
