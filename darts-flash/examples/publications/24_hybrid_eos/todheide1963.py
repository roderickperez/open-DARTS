import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess

from dartsflash.components import CompData
from dartsflash.mixtures import VLAq
from dartsflash.dartsflash import DARTSFlash
from dartsflash.plot import *


components = ["H2O", "CO2"]
comp_data = CompData(components)

if 1:
    """ HYBRID-EOS """
    f = VLAq(comp_data=comp_data, hybrid=True)
    f.set_vl_eos("PR", trial_comps=[0, 1],
                 switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
    f.set_aq_eos("Aq", max_iter=10, use_gmix=True)

else:
    """ FULL-CUBIC """
    f = VLAq(comp_data=comp_data, hybrid=False)
    f.set_vl_eos("PR", trial_comps=[0, 1],
                 switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)

f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
             stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
             split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2)

""" PLOT P-X AND T-X DIAGRAMS """
if 0:
    # Reference data of H2O-CO2 mixture from Spycher et al. (2003)
    ref_T = np.array([25, 31.04, 50, 75, 100]) + 273.15
    ref_p = [[1.0, 22.7, 25.3, 29.8, 30.0, 37.3, 37.4, 48.3, 50.7, 50.7, 65.9, 70.9, 76.0, 76.0, 82.8, 91.2, 101.3, 101.3, 101.4, 103.4, 111.5, 126.7, 136.8, 141.9, 152.0, 152.0, 177.3, 202.7, 202.7, 202.7, 405.3, 456.0, 481.3, 506.6],
             [1.0, 25.3, 50.7, 76.0, 101.3, 152.0, 202.7, 405.3, 506.6, 532.0, 557.3, 6.9, 25.3, 50.7, 101.4, 202.7, 73.9],
             [1.0, 17.3, 25.3, 25.5, 25.8, 36.4, 36.4, 40.5, 46.3, 50.6, 50.7, 60.6, 60.8, 68.2, 70.8, 75.3, 76.0, 80.8, 87.2, 90.9, 100.6, 100.9, 101.0, 101.3, 101.3, 101.33, 111.0, 121.0,
              122.1, 126.7, 141.1, 147.5, 147.5, 152.0, 152.0, 176.8, 200., 201., 202.7, 301., 304., 344.8, 405.3, 500., 608., 709.3],
             [1.0, 6.9, 25.3, 25.3, 23.3, 37.4, 37.5, 50.7, 51.3, 51.5, 76.0, 101.3, 101.33, 101.4, 103.4, 111.5, 126.7, 152., 152., 153.1, 202.7, 202.7, 209.4, 304., 344.8, 405.3, 608., 709.3],
             [3.25, 6.00, 9.20, 11.91, 14.52, 18.16, 23.07, 25.3, 36.8, 37.2, 44.8, 44.8, 50.7, 51.5, 51.5, 76.0, 101.3, 152.0, 200.0, 202.7, 304., 405.3, 500., 709.3],
             ]
    ref_y = [np.array([28.6, 1.95, 1.64, 1.63, 1.67, 1.45, 1.49, 1.2787, 1.28, 1.29, 3.00, 3.07, np.nan, 3.09, 3.0152, 3.14, 3.27, 3.32, 3.36, 3.3739, 3.37, 3.41, np.nan, 3.44, 3.54, 3.60, 3.69, 3.76, 3.78, 3.77, np.nan, 4.01, 3.99, 3.97]) * 1e-3,
             np.array([39.8, 2.28, 1.61, np.nan, 3.65, np.nan, 4.21, 4.77, 4.80, 4.75, 4.78, 6.94, 2.39, 1.63, 4.08, 4.50, 2.1079]) * 1e-3,
             np.array([116., 8.41, 6.20, 5.95, 5.98, 4.66, 4.63, 4.6, 3.96, 3.6, 3.83, 3.7, 3.57, 3.39, 3.4, 3.45, 3.50, 3.4, 3.64, 4.1, 4.29, 4.5, 5.47, 4.36, 4.49, 5.5, 5.0, 5.5,
                       5.43, np.nan, 6.1, 6.08, np.nan, 6.10, 7.9, 6.43, np.nan, 6.82, 6.77, 7.82, np.nan, 7.5, 7.59, np.nan, 7.93, 8.01]) * 1e-3,
             np.array([301., 60.14, 18.16, 10.6, 20., 12.5, 12.6, 10.87, np.nan, 10.4, 10.2, np.nan, 8.29, 7.4, 7.27, 6.3, 8.11, 8.55, 9.56, 9.0, 7.5, 9.38, 11.3, 8.4, np.nan, 13.3, 13.19, 13.93, 14.0]) * 1e-3,
             np.array([288., 155., 107., 77., 69., 54., 45., np.nan, 32.8, 32.3, 27.7, 27.4, np.nan, 24.8, 25.1, np.nan, np.nan, np.nan, 29., np.nan, np.nan, np.nan, 30., np.nan]) * 1e-3]
    ref_x = [1.-np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 2.10, 2.142, np.nan, np.nan, 2.444, 2.445, np.nan, np.nan, 2.510, 2.488, 2.49, np.nan, np.nan, np.nan, 2.582, np.nan, 2.603, np.nan, 2.672, 2.57, 2.734, np.nan, 3.011, np.nan, np.nan, np.nan]) * 1e-2,
             1.-np.array([np.nan, 1.127, 1.904, 2.303, 2.368, 2.476, 2.567, 2.871, 3.014, np.nan, np.nan, 0.331, 1.056, 1.817, 2.41, 2.62, np.nan]) * 1e-2,
             1.-np.array([np.nan, np.nan, 0.774, np.nan, np.nan, np.nan, np.nan, 1.09, np.nan, 1.37, 1.367, 1.61, np.nan, 1.651, 1.76, 1.750, 1.779, 1.90, 1.768, 2.00, np.nan, 2.05, 2.075, 2.081, 2.018, 1.98, 2.10, 2.14,
                          2.096, 2.106, 2.17, 2.215, 2.207, 2.174, 2.10, 2.262, 2.3, 2.347, 2.289, 2.514, 2.457, np.nan, 2.606, 2.8, 2.868, 2.989]) * 1e-2,
             1.-np.array([np.nan, 0.149, 0.542, 0.545, np.nan, np.nan, np.nan, 1.006, 1.002, np.nan, np.nan, 1.351, 1.630, 1.56, 1.616, 1.91, np.nan, np.nan, 1.937, 1.88, 1.92, 2.09, 2.098, np.nan, 2.317, np.nan, 2.498, np.nan, 2.933]) * 1e-2,
             1.-np.array([0.045, 0.098, 0.159, 0.208, 0.261, 0.328, 0.414, 0.4294, np.nan, np.nan, np.nan, np.nan, 0.812, np.nan, np.nan, 1.135, 1.400, 1.794, 2.0, 2.023, 2.318, 2.537, 2.8, 3.002]) * 1e-2]
else:
    # Reference data of H2O-CO2 mixture from Todheide and Franck (1963)
    ref_T = np.array([50, 100, 150, 200, 250, 260, 265, 267, 268, 270, 275, 300, 350]) + 273.15
    ref_p = [200, 250, 300, 325, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500] + [2000, 2500, 3000, 3500]
    ref_y = np.array([
        # T, P
        1.-np.array([.971, np.nan, np.nan, np.nan, np.nan, .970, np.nan, np.nan, np.nan, np.nan, .970, np.nan, np.nan, np.nan, np.nan, .971, .971, .971, .972, .972]),
        1.-np.array([.940, np.nan, np.nan, np.nan, np.nan, .937, np.nan, np.nan, np.nan, np.nan, .935, np.nan, np.nan, np.nan, np.nan, .932, .932, .933, .935, .937]),
        1.-np.array([.845, np.nan, np.nan, np.nan, np.nan, .860, np.nan, np.nan, np.nan, np.nan, .855, np.nan, np.nan, np.nan, np.nan, .850, .855, .860, .867, .875]),
        1.-np.array([.665, np.nan, np.nan, np.nan, np.nan, .720, np.nan, np.nan, np.nan, np.nan, .695, np.nan, np.nan, np.nan, np.nan, .685, .675, .675, .693, .705]),
        1.-np.array([.601, np.nan, np.nan, np.nan, np.nan, .682, np.nan, np.nan, np.nan, np.nan, .632, np.nan, np.nan, np.nan, np.nan, .585, .575, .595, .615, .630]),
        1.-np.array([.577, np.nan, np.nan, np.nan, np.nan, .657, np.nan, np.nan, np.nan, np.nan, .600, np.nan, np.nan, np.nan, np.nan, .546, .508, .493, .550, .570]),
        1.-np.array([.567, np.nan, np.nan, np.nan, np.nan, .650, np.nan, np.nan, np.nan, np.nan, .575, np.nan, np.nan, np.nan, np.nan, .526, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.559, np.nan, .620, np.nan, .644, .642, .638, .622, .603, .586, .569, .556, .547, .530, .518, .497, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.547, np.nan, .614, np.nan, .635, .631, .628, .607, .585, .565, .545, .525, .503, .466, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.519, np.nan, .591, np.nan, .612, .611, .595, .572, .541, .507, .455, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.352, np.nan, .454, np.nan, .480, .460, .335, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.055, .120, .162, .160, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    ])
    ref_x = np.array([
        1.-np.array([.023, np.nan, np.nan, np.nan, np.nan, .028, np.nan, np.nan, np.nan, np.nan, .035, np.nan, np.nan, np.nan, np.nan, .038, .041, .045, .048, .052]),
        1.-np.array([.020, np.nan, np.nan, np.nan, np.nan, .028, np.nan, np.nan, np.nan, np.nan, .036, np.nan, np.nan, np.nan, np.nan, .041, .045, .049, .053, .056]),
        1.-np.array([.021, np.nan, np.nan, np.nan, np.nan, .034, np.nan, np.nan, np.nan, np.nan, .043, np.nan, np.nan, np.nan, np.nan, .050, .055, .060, .065, .070]),
        1.-np.array([.024, np.nan, np.nan, np.nan, np.nan, .045, np.nan, np.nan, np.nan, np.nan, .061, np.nan, np.nan, np.nan, np.nan, .073, .081, .085, .090, .095]),
        1.-np.array([.028, np.nan, np.nan, np.nan, np.nan, .070, np.nan, np.nan, np.nan, np.nan, .120, np.nan, np.nan, np.nan, np.nan, .149, .166, .170, .175, .177]),
        1.-np.array([.027, np.nan, np.nan, np.nan, np.nan, .075, np.nan, np.nan, np.nan, np.nan, .138, np.nan, np.nan, np.nan, np.nan, .185, .215, .235, .235, .235]),
        1.-np.array([.026, np.nan, np.nan, np.nan, np.nan, .080, np.nan, np.nan, np.nan, np.nan, .155, np.nan, np.nan, np.nan, np.nan, .215, .272, .298, .292, .280]),
        1.-np.array([.026, np.nan, np.nan, np.nan, np.nan, .082, np.nan, np.nan, np.nan, np.nan, .170, np.nan, np.nan, np.nan, np.nan, .240, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.026, np.nan, .048, np.nan, .066, .083, .095, .118, .138, .160, .179, .197, .211, .228, .248, .278, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.026, np.nan, .049, np.nan, .066, .087, .102, .124, .148, .170, .190, .207, .226, .250, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.025, np.nan, .049, np.nan, .070, .092, .114, .137, .163, .195, .241, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.023, np.nan, .049, np.nan, .079, .125, .225, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        1.-np.array([.008, .026, .051, .077, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
    ])
if 0:
    # Px
    T_idx = 11
    T = ref_T[T_idx]
    state_spec = {"pressure": np.arange(20, 1200, 20),
                  "temperature": T,
                  }
    filename = "Px-" + f.filename + "-" + str(int(T)) + ".pdf"
    ref_V = ref_y[T_idx, :]
    ref_A = ref_x[T_idx, :]
    ref_Y = ref_p
    # filename = "Px-" + f.filename + "-" + str(int(T)) + ".pdf"
else:
    # Tx
    p_idx = 5
    p = ref_p[p_idx]
    state_spec = {"temperature": np.arange(273.15, 673.15, 5),
                  "pressure": p,
                  }
    filename = "Tx-" + f.filename + "-" + str(int(p)) + ".pdf"
    ref_V = ref_y[:, p_idx]
    ref_A = ref_x[:, p_idx]
    ref_Y = ref_T
dz = 0.005
min_z = [0.]
max_z = [1.]
compositions = {comp: np.arange(min_z[i], max_z[i] + 0.1 * dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.

""" Plot Gibbs energy surfaces and flash results """
flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                 print_state="Flash")

plot = PlotFlash.binary(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z,
                        plot_phase_fractions=True)
plot.draw_point(X=ref_A, Y=ref_Y, colours=plot.colours[0])
plot.draw_point(X=ref_V, Y=ref_Y, colours=plot.colours[1])

plt.savefig(filename)

plt.show()
