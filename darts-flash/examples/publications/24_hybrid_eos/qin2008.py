import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess
from dartsflash.libflash import Flash
from dartsflash.libflash import CubicEoS, AQEoS

from dartsflash.dartsflash import DARTSFlash
from dartsflash.mixtures import VLAq, CompData
from dartsflash.plot import *


# Qin et al. (2008) and Dhima et al. (1999) H2O-CO2-C1 mixtures
components = ["H2O", "CO2", "C1"]
comp_data = CompData(components, setprops=True)
f = VLAq(comp_data=comp_data, hybrid=True)

f.set_vl_eos("PR", trial_comps=[0, 1, 2],
             switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
f.set_aq_eos("Aq", max_iter=10, use_gmix=True)

f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
             stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
             split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
             split_negative_flash_iter=3,
             )

# Ref data mixture 1
ref_T = [324.3, 375.6, 344.15]
suffix = ["_324", "_375", "_344"]

ref_p_375 = [499., 402., 303., 205., 107.]
ref_p_324 = [499., 306.]
ref_p_344 = [100., 200., 500., 750., 1000.]  # Dhima (1999)

ref_V_375 = np.array([[[0.0361, 0.9639, 0.], [0.02676, 0.70305, 0.27019], [0.02199, 0.51968, 0.46103], [0.01821, 0.39828, 0.58351], [0.0052, 0., 0.9948]],  # 500 bar
                      [[0.0337, 0.9663, 0.], [0.02520, 0.70550, 0.26930], [0.01923, 0.51593, 0.46484], [0.01695, 0.40332, 0.57973], [0.0057, 0., 0.9943]],  # 400 bar
                      [[0.0309, 0.9691, 0.], [0.02317, 0.71203, 0.26480], [0.01905, 0.53282, 0.44813], [0.01681, 0.41424, 0.56895], [0.0066, 0., 0.9934]],  # 300 bar
                      [[0.0274, 0.9726, 0.], [0.02296, 0.72455, 0.25429], [0.01765, 0.51552, 0.46683], [0.01581, 0.41846, 0.56573], [0.0084, 0., 0.9916]],  # 200 bar
                      [[0.0251, 0.9749, 0.], [0.02080, 0.72812, 0.25108], [0.01906, 0.56551, 0.41544], [0.01747, 0.43726, 0.54527], [0.0131, 0., 0.9869]],  # 100 bar
                      ])
ref_A_375 = np.array([[[1., 0.0275, 0.], [1., 0.01971, 0.00152], [1., 0.01549, 0.00215], [1., 0.01179, 0.00258], [1., 0., 0.0041]],
                      [[1., 0.0256, 0.], [1., 0.01882, 0.00127], [1., 0.01414, 0.00199], [1., 0.01172, 0.00222], [1., 0., 0.0035]],
                      [[1., 0.0234, 0.], [1., 0.01801, 0.00104], [1., 0.01369, 0.00166], [1., 0.01074, 0.00188], [1., 0., 0.0030]],
                      [[1., 0.0208, 0.], [1., 0.01524, 0.00077], [1., 0.01186, 0.00127], [1., 0.00940, 0.00136], [1., 0., 0.0023]],
                      [[1., 0.0153, 0.], [1., 0.01065, 0.00038], [1., 0.00855, 0.00074], [1., 0.00627, 0.00084], [1., 0., 0.0014]],
                      ])
ref_A_375[:, :, 0] -= np.sum(ref_A_375[:, :, 1:], axis=2)

ref_V_324 = np.array([[[0.0082, 0.9918, 0.], [0.00632, 0.74103, 0.25265], [0.00517, 0.59844, 0.39639], [0.00328, 0.32466, 0.67206], [0.0008, 0., 0.9992]],
                      [[0.0073, 0.9927, 0.], [0.00592, 0.76440, 0.22968], [0.00476, 0.58432, 0.41092], [0.00312, 0.33229, 0.66459], [0.0010, 0., 0.9990]],
                      ])
ref_A_324 = np.array([[[1., 0.0302, 0.], [1., 0.02231, 0.00125], [1., 0.01817, 0.00186], [1., 0.01131, 0.00258], [1., 0., 0.0039]],
                      [[1., 0.0247, 0.], [1., 0.01966, 0.00092], [1., 0.01645, 0.00144], [1., 0.01055, 0.00196], [1., 0., 0.0030]],
                      ])
ref_A_324[:, :, 0] -= np.sum(ref_A_324[:, :, 1:], axis=2)

ref_V_344 = np.array([[[None, None, 0.5670], [None, None, 0.8045]],
                      [[None, None, 0.5700], [None, None, 0.8102]],
                      [[None, None, 0.5850], [None, None, 0.8205]],
                      [[None, None, 0.5870], [None, None, 0.8260]],
                      [[None, None, 0.5940], [None, None, 1.]],
                      ])
ref_A_344 = np.array([[[1., 0.008346, 0.000776], [1., 0.003555, 0.001100]],
                      [[1., 0.011300, 0.001310], [1., 0.005390, 0.001820]],
                      [[1., 0.012670, 0.002434], [1., 0.006265, 0.003190]],
                      [[1., 0.014347, 0.003027], [1., 0.007105, 0.003893]],
                      [[1., 0.015071, 0.003610], [1., 0., 0.]],
                      ])
ref_A_344[:, :, 0] -= np.sum(ref_A_344[:, :, 1:], axis=2)

""" TERNARY DIAGRAM """
temp_idx = 1
pres_idx = 4
state_spec = {"pressure": eval("ref_p" + suffix[temp_idx])[pres_idx],
              "temperature": ref_T[temp_idx],
              }

dz = 0.0001
min_z = [0.97, 0.]
max_z = [1., 0.03]
compositions = {comp: np.arange(min_z[i], max_z[i]+0.1*dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.

x0 = compositions[components[0]]
x1 = compositions[components[1]]

flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True, print_state="Flash")

plot = PlotFlash.ternary(f, flash_results, state=state_spec, dz=dz, min_z=min_z, max_z=max_z, plot_phase_fractions=False)
ref_A = eval("ref_A" + suffix[temp_idx])[pres_idx, :]
ref_V = eval("ref_V" + suffix[temp_idx])[pres_idx, :]
for j, (xA, xV) in enumerate(zip(ref_A, ref_V)):
    plot.draw_compositions(compositions=[xA, xV], colours='r', connect_compositions=True)

if 0:
    plot.add_text("Aq - V", xloc=0.45, yloc=0.4, fontsize=12, colours='red', box_colour='k')
    plot.add_text("Aq - L", xloc=0.79, yloc=-0.025, fontsize=12, colours='red', box_colour='k')
    plot.add_text("Aq", xloc=0.15, yloc=-0.025, fontsize=12, colours='red', box_colour='k')

plt.savefig("qin" + suffix[temp_idx] + "_" + str(int(state_spec["pressure"])) + ".pdf")

plt.show()
