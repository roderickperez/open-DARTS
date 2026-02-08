import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess
from dartsflash.libflash import Flash
from dartsflash.libflash import CubicEoS, AQEoS

from dartsflash.components import CompData, ConcentrationUnits as cu
from dartsflash.mixtures import VLAq
from dartsflash.dartsflash import DARTSFlash
from dartsflash.plot import *
from dartsflash.diagram import Diagram


# Savary et al. (2012) H2O-CO2-H2S mixtures
components = ["H2O", "CO2", "H2S"]
salinity = False
ions = ["Na+", "Cl-"] if salinity else None

comp_data = CompData(components, ions, setprops=True)
f = VLAq(comp_data=comp_data, hybrid=True)

f.set_vl_eos("PR")
f.set_aq_eos("Aq", )

f.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash, eos_order=["Aq", "VL"], nf_initial_guess=[InitialGuess.Henry_AV])

# Ref data mixture 1
ref_T = 393.15
concentration_unit = cu.MOLALITY

if not salinity:
    # Table 3-4 -- Molality = 0 M
    concentration = None

    ref_d = np.array([[0., 1.], [0.785, 0.796], [0.639, 0.747], [0.681, 0.775], [0.752, 0.780], [0.667, 0.775], [0.667, 0.761], [1., 0.]])  # 1:0, 3:1, 3:1, 3:3, 3:3, 1:4, 2:4, 0:1
    ref_X = np.array([[0, 1], [1, 3], [1, 3], [3, 3], [3, 3], [4, 2], [4, 1], [1, 0]])
    ref_x = np.zeros(8)
    for i in range(8):
        n1 = ref_X[i, 0] * ref_d[i, 0] / comp_data.Mw[1]
        n2 = ref_X[i, 1] * ref_d[i, 1] / comp_data.Mw[2]
        ref_x[i] = n1/(n1+n2)

    calc_x = [True, True, False, True, False, True, True, True]
    labels = ["0:1", "1:3", "3:3", "4:2", "4:1", "1:0"]
    colours = [Diagram.colours[i] for i in [0, 1, 1, 2, 2, 3, 4, 5]]
    styles = ["solid" if calc_x[i] else "dashed" for i in range(8)]
    markers = ["o" if calc_x[i] else "^" for i in range(8)]

    ref_p = np.array([[17., 24., 325., 230., 202., None],
                      [350., 241., 39., None, None, None],
                      [338., 233., 173., 273., None, None],
                      [344., 285., 248., 156., None, None],
                      [346., 285., 310., 239., 195., None],
                      [349., 336., 315., 245., 186., 86.],
                      [350., 292., 204., 135., None, None],
                      [337., 212., 113., None, None, None],
                      ])

    ref_V = np.array([[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [None, None, None]],  # 0:1
                      [[0., 0.30, 0.70], [0., 0.26, 0.74], [0., 0.26, 0.74], [None, None, None], [None, None, None], [None, None, None]],  # 1:3
                      [[None, None, None], [0., 0.21, 0.79], [0., 0.23, 0.77], [0., 0.21, 0.79], [None, None, None], [None, None, None]],  # 1:3
                      [[0., 0.46, 0.54], [0., 0.44, 0.56], [0., 0.48, 0.52], [0., 0.50, 0.50], [None, None, None], [None, None, None]],  # 3:3
                      [[0., 0.48, 0.52], [0., 0.48, 0.52], [0., 0.49, 0.51], [0., 0.55, 0.45], [0., 0.44, 0.56], [None, None, None]],  # 3:3
                      [[0., 0.59, 0.41], [0., 0.69, 0.31], [0., 0.64, 0.36], [0., 0.68, 0.32], [0., 0.77, 0.23], [0., 0.66, 0.34]],  # 4:2
                      [[0., 0.64, 0.36], [0., 0.69, 0.31], [0., 0.73, 0.27], [0., 0.65, 0.35], [None, None, None], [None, None, None]],  # 4:1
                      [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [None, None, None], [None, None, None], [None, None, None]],  # 1:0
                      ])
    ref_A_m = np.array([[[55.509, np.nan, 0.511], [55.509, np.nan, 0.815], [55.509, np.nan, 3.10], [55.509, np.nan, 2.93], [55.509, np.nan, 2.89], [None, None, None]],
                        [[55.509, 0.25, 2.31], [55.509, 0.31, 2.11], [55.509, 0.01, 1.23], [None, None, None], [None, None, None], [None, None, None]],
                        [[55.509, 0.94, 2.57], [55.509, 0.49, 2.51], [55.509, 0.39, 2.41], [55.509, 0.46, 2.40], [None, None, None], [None, None, None]],
                        [[55.509, 0.67, 1.78], [55.509, 0.89, 1.81], [55.509, 0.90, 2.24], [55.509, 0.61, 2.20], [None, None, None], [None, None, None]],
                        [[55.509, 0.65, 1.63], [55.509, 0.82, 1.84], [55.509, 0.97, 2.13], [55.509, 0.90, 2.71], [55.509, 0.78, 1.32], [None, None, None]],
                        [[55.509, 1.72, 1.40], [55.509, 0.93, 1.69], [55.509, 1.06, 1.43], [55.509, 0.98, 1.75], [55.509, 0.71, 2.27], [55.509, 0.41, 1.31]],
                        [[55.509, 1.15, 1.42], [55.509, 1.08, 1.70], [55.509, 0.81, 1.89], [55.509, 0.73, 1.32], [None, None, None], [None, None, None]],
                        [[55.509, 1.36, np.nan], [55.509, 1.12, np.nan], [55.509, 0.508, np.nan], [None, None, None], [None, None, None], [None, None, None]],
                        ])
    ref_A = np.array([[ref_m/np.nansum(ref_m) if ref_m[0] is not None else ref_m for j, ref_m in enumerate(ref_A_m[i])] for i in range(8)])

else:
    # Table 3-5 -- Molality = 2 M
    concentration = {"NaCl": 2.}

    ref_d = np.array([[0., 1.], [0.792, 0.785], [0.792, 0.796], [0.792, 0.785], [1., 0.]])
    ref_X = np.array([[0, 1], [2, 4], [3, 3], [3, 2], [1, 0]])
    ref_x = np.zeros(5)
    for i in range(5):
        n1 = ref_X[i, 0] * ref_d[i, 0] / comp_data.Mw[1]
        n2 = ref_X[i, 1] * ref_d[i, 1] / comp_data.Mw[2]
        ref_x[i] = n1/(n1+n2)

    colours = [Diagram.colours[i] for i in [0, 1, 2, 3, 5]]
    calc_x = [True, True, True, True, True]
    labels = ["0:1", "2:4", "3:3", "3:2", "1:0"]
    styles = ["solid" if calc_x[i] else "dashed" for i in range(5)]
    markers = ["o" if calc_x[i] else "^" for i in range(5)]

    ref_p = np.array([[341., 273., 226., 305., 247., 166., 125.],
                      [323., 258., 285., 210., 161., 122., None],
                      [326., 260., 212., 176., 47., None, None],
                      [248., 306., 344., 181., None, None, None],
                      [323., 244., 168., 124., 86., None, None],
                      ])

    ref_V = np.array([[[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [None, None, None], [None, None, None]],  # 0:1
                      [[0., 0.35, 0.65], [0., 0.35, 0.65], [0., 0.34, 0.66], [0., 0.32, 0.68], [0., 0.31, 0.69], [0., 0.31, 0.69], [None, None, None]],  # 2:4
                      [[0., 0.53, 0.47], [0., 0.49, 0.51], [0., 0.46, 0.54], [0., 0.49, 0.51], [0., 0.49, 0.51], [None, None, None], [None, None, None]],  # 3:3
                      [[0., 0.65, 0.35], [0., 0.63, 0.37], [0., 0.62, 0.38], [0., 0.64, 0.36], [None, None, None], [None, None, None], [None, None, None]],  # 3:2
                      [[0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.], [0., 1., 0.]],  # 1:0
                      ])
    ref_A_m = np.array([[[59.509, np.nan, 2.79], [59.509, np.nan, 1.92], [59.509, np.nan, 2.78], [59.509, np.nan, 2.15], [59.509, np.nan, 1.60], [None, None, None], [None, None, None]],
                        [[59.509, 0.25, 2.16], [59.509, 0.22, 2.24], [59.509, 0.24, 2.14], [59.509, 0.26, 1.78], [59.509, 0.22, 1.45], [59.509, 0.16, 1.30], [None, None, None]],
                        [[59.509, 0.26, 1.78], [59.509, 0.54, 1.53], [59.509, 0.53, 1.18], [59.509, 0.55, 1.68], [59.509, 0.10, 1.11], [None, None, None], [None, None, None]],
                        [[59.509, 0.33, 1.30], [59.509, 0.40, 1.22], [59.509, 0.28, 1.08], [59.509, 0.33, 1.25], [None, None, None], [None, None, None], [None, None, None]],
                        [[59.509, 0.62, np.nan], [59.509, 0.87, np.nan], [59.509, 0.84, np.nan], [59.509, 0.89, np.nan], [59.509, 0.86, np.nan], [59.509, 0.82, np.nan], [59.509, 0.51, np.nan]],
                        ])  # H2O molality 59.509 because 2M Na+ and Cl-
    ref_A = np.array([[ref_m/np.nansum(ref_m) if ref_m[0] is not None else ref_m for j, ref_m in enumerate(ref_A_m[i])] for i in range(5)])

""" SOLUBILITY CURVES """
state_spec = {"pressure": np.arange(5., 400., 1.),
              "temperature": ref_T,
              }

compositions = {"H2O": np.array([0.5]), "CO2": ref_x[calc_x] * 0.5, "H2S": 1.}
flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                 concentrations=concentration, concentration_unit=concentration_unit)
flash_results.transpose("CO2", "pressure", ...)

logy = True
plot_co2_A = PlotFlash.solubility(f, flash_results, dissolved_comp_idx=1, phase_idx=0, x_var="pressure",
                                  state=state_spec, logy=logy, labels=labels, plot_1p=False,
                                  legend_loc='lower right', xlim=[0., 400.], ylim=[5e-4, 1e-1] if logy else [0., None])
plot_co2_A.draw_point(X=ref_p, Y=ref_A[:, :, 1], colours=colours, markers=markers)
plt.savefig("savary_sol_co2" + ("_s" if salinity else "") + ("_log" if logy else "") + ".pdf")

plot_h2s_A = PlotFlash.solubility(f, flash_results, dissolved_comp_idx=2, phase_idx=0, x_var="pressure",
                                  state=state_spec, logy=logy, labels=labels, plot_1p=False,
                                  legend_loc='lower right', xlim=[0., 400.], ylim=[5e-4, 1e-1] if logy else [0., None])
plot_h2s_A.draw_point(X=ref_p, Y=ref_A[:, :, 2], colours=colours, markers=markers)
plt.savefig("savary_sol_h2s" + ("_s" if salinity else "") + ("_log" if logy else "") + ".pdf")

plt.show()
