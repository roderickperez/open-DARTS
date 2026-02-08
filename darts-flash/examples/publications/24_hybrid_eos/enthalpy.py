from cmath import phase

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoS, CubicEoS, AQEoS, InitialGuess

from dartsflash.dartsflash import DARTSFlash, CompData
from dartsflash.components import ConcentrationUnits as cu
from dartsflash.mixtures import VL, VLAq
from dartsflash.plot import *


""" DEFINE FLASH """
def define_flash(components: list, aqueous: bool, ions: list = None, flash_type: DARTSFlash.FlashType = DARTSFlash.FlashType.PTFlash) -> DARTSFlash:
    comp_data = CompData(components, ions, setprops=True)

    if not aqueous:
        f = VL(comp_data=comp_data)
        f.set_vl_eos("PR", trial_comps=[i for i, comp in enumerate(components)],
                     switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
    else:
        f = VLAq(comp_data=comp_data, hybrid=True)
        f.set_vl_eos("PR", trial_comps=[i for i, comp in enumerate(components)],
                     switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
        f.set_aq_eos("Aq", max_iter=10, use_gmix=True)

    f.init_flash(flash_type=flash_type, nf_initial_guess=[InitialGuess.Henry_AV],
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.lnK,
                 split_tol=1e-20, tpd_close_to_boundary=1e-2, tpd_tol=1e-11,
                 # split_switch_tol=1e-1,
                 # verbose=True
                 )
    return f


""" ENTHALPY OF MIXING OF TOTAL MIXTURE """
if 0:
    if 0:
        # C1 + N2 (Lewis, 1977)
        components = ["C1", "N2"]
        aqueous = False
        ref_p = np.array([21.4, 31.3, 40.8, 55.5, 71.6, 82.1])
        ref_T = np.array([183.])
        ref_x = [[0.279, 0.433, 0.557, 0.668, 0.780, np.nan, np.nan, np.nan],
                 [0.295, 0.440, 0.550, 0.667, np.nan, np.nan, np.nan, np.nan],
                 [0.334, 0.545, 0.582, 0.639, 0.676, 0.702, 0.823, 0.894],
                 [0.296, 0.436, 0.533, 0.638, 0.756, 0.841, np.nan, np.nan],
                 [0.147, 0.280, 0.467, 0.555, 0.654, 0.764, 0.849, np.nan],
                 [0.168, 0.222, 0.283, 0.411, 0.524, 0.641, 0.713, np.nan]
                 ]
        ref_Hm = np.array([[0.053, 0.069, 0.070, 0.067, 0.053, np.nan, np.nan, np.nan],
                           [0.131, 0.173, 0.187, 0.184, np.nan, np.nan, np.nan, np.nan],
                           [1.422, 2.337, 2.425, 2.688, 2.747, 2.823, 3.059, 2.006],
                           [0.942, 1.307, 1.542, 1.647, 1.407, 0.626, np.nan, np.nan],
                           [0.379, 0.725, 1.053, 1.068, 0.850, 0.330, 0.131, np.nan],
                           [0.396, 0.515, 0.610, 0.765, 0.639, 0.369, 0.251, np.nan],
        ]) / R * 1e3
    else:
        # CO2 + H2O (Chen, 1992)
        components = ["CO2", "H2O"]
        aqueous = True
        ref_p = np.array([104., 124., 150.])
        ref_T = np.array([523.15])
        ref_x = np.array(
            [[0.0650, 0.1195, 0.1403, 0.1626, 0.2405, 0.3043, 0.3807, 0.4250, 0.4489, 0.5289, 0.5589, 0.6245, 0.6605,
              0.6990, 0.7403, 0.7846, 0.8323, 0.8837, 0.9395, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
              np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             [0.0033, 0.0066, 0.0100, 0.0134, 0.0169, 0.0204, 0.0239, 0.0275, 0.0312, 0.0349, 0.0387, 0.1225, 0.1664,
              0.2457, 0.3102, 0.3470, 0.3873, 0.4318, 0.4558, 0.4810, 0.5359, 0.5973, 0.6486, 0.7049, 0.7456, 0.7893,
              0.8361, 0.8866, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
             [0.0068, 0.0103, 0.0139, 0.0174, 0.0211, 0.0248, 0.0285, 0.0323, 0.0361, 0.0440, 0.0520, 0.0689, 0.0869,
              0.1059, 0.1263, 0.1480, 0.1713, 0.1963, 0.2232, 0.2522, 0.2836, 0.3177, 0.3549, 0.3956, 0.4403, 0.4897,
              0.5164, 0.5445, 0.5743, 0.6057, 0.6390, 0.6744, 0.7121, 0.7522, 0.7950, 0.8408, 0.8900]
        ])
        ref_Hm = np.array([
            [1.843, 3.110, 3.591, 4.035, 6.047, 8.004, 10.127, 10.919, 10.827, 12.471, 11.511, 10.176, 9.356, 8.382,
             7.252, 5.949, 4.673, 3.312, 1.640, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.041, 0.090, 0.145, 0.183, 0.260, 0.322, 0.366, 0.457, 0.515, 0.593, 0.674, 2.229, 3.105, 4.811, 5.782,
             6.573, 7.846, 8.468, 8.922, 9.532, 10.681, 10.497, 9.403, 7.839, 7.025, 5.958, 4.427, 3.198,
             np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [0.054, 0.090, 0.209, 0.212, 0.312, 0.366, 0.429, 0.459, 0.538, 0.650, 0.794, 1.079, 1.332, 1.636, 1.968,
             2.363, 2.760, 3.101, 3.477, 4.067, 4.506, 4.961, 5.516, 6.368, 6.964, 7.563, 8.365, 8.873, 9.248, 9.896,
             9.126, 8.430, 7.371, 6.501, 5.490, 4.342, 3.047]
        ]) / R * 1e3  # kJ/mol -> H/R
        # yrange =

    """ DEFINE STATE """
    f = define_flash(components, aqueous)
    state_spec = {"temperature": ref_T,
                  "pressure": ref_p,
                  }
    dz = 0.001
    min_z = [0.]
    max_z = [1.]
    compositions = {comp: np.arange(min_z[i], max_z[i] + 0.1 * dz, dz) for i, comp in enumerate(components[:-1])}
    compositions[components[-1]] = 1.

    """ Plot enthalpy of mixing """
    mixing_props = ["H"]
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")
    results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions,
                                          state_variables=['pressure', 'temperature'] + components,
                                          flash_results=flash_results,
                                          total_properties_to_evaluate={"H": EoS.Property.ENTHALPY}, mix=True,
                                          print_state="NP props")

    plot = PlotProps.binary(flash=f, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z,
                            prop_name="H_total",
                            # title="Enthalpy of mixing of " + f.mixture_name,
                            plot_1p=False, props=results_np,
                            # flash_results=flash_results,
                            ax_label=r"$\Delta H^{mix}$/R",
                            datalabels=["P = {} bar".format(p) for p in ref_p]
                            )
    plot.draw_point(ref_x, ref_Hm, widths=15)
    plot.set_axes(ylim=[0., 1800.])
    plt.savefig("Hm-" + f.filename + ".pdf")


""" ENTHALPY OF MIXING OF AQUEOUS PHASE """
if 0:
    # CO2 + H2O + NaCl (Koschel, 2006)
    components = ["CO2", "H2O"]
    ions = ["NaCl"]
    ref_m = [0., 1., 3.]
    ref_T = [323.15, 373.15]
    ref_p = [[[20.6, 51.0, 105.3, 142.0, 202.0], [50.5, 100.8, 195.0, np.nan, np.nan]],
             [[51.0, 103.0, 143.8, 202.4, np.nan], [50.7, 104.0, 194.0, np.nan, np.nan]],
             [[50.0, 100.4, 144.1, 202.4, np.nan], [50.4, 100.3, 190.2, np.nan, np.nan]],
             ]
    ref_x = np.array([
            # m = 0
            [[[0.0032, 0.0037, 0.0040, 0.0042, 0.0048, 0.0058, 0.0060, 0.0064, 0.0079, 0.0079, 0.0106, 0.0126, 0.0158, 0.0210, 0.0311, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0052, 0.0064, 0.0084, 0.0092, 0.0101, 0.0105, 0.0110, 0.0120, 0.0125, 0.0136, 0.0138, 0.0147, 0.0151, 0.0155, 0.0156, 0.0172, 0.0181, 0.0190, 0.0201, 0.0208, 0.0254, np.nan, np.nan, np.nan],
              [0.0100, 0.0120, 0.0132, 0.0165, 0.0197, 0.0230, 0.0249, 0.0260, 0.0323, 0.0324, 0.0446, 0.0508, 0.0625, 0.0627, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0105, 0.0125, 0.0139, 0.0173, 0.0207, 0.0241, 0.0248, 0.0261, 0.0274, 0.0287, 0.0307, 0.0340, 0.0470, 0.0658, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0110, 0.0147, 0.0183, 0.0197, 0.0210, 0.0239, 0.0253, 0.0274, 0.0288, 0.0359, 0.0427, 0.0461, 0.0494, 0.0692, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],]],
            # m = 1
            [[[0.0039, 0.0050, 0.0062, 0.0089, 0.0112, 0.0219, 0.0220, 0.0255, 0.0428, 0.0529, 0.0839, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0080, 0.0100, 0.0107, 0.0113, 0.0132, 0.0146, 0.0165, 0.0197, 0.0230, 0.0324, 0.0388, 0.0448, 0.0450, 0.0510, 0.0705, 0.0747, 0.0803, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0071, 0.0078, 0.0085, 0.0107, 0.0121, 0.0142, 0.0156, 0.0176, 0.0177, 0.0189, 0.0193, 0.0204, 0.0210, 0.0211, 0.0231, 0.0252, 0.0278, 0.0280, 0.0312, 0.0414, 0.0477, 0.0542, 0.0545, 0.0637],
              [0.0060, 0.0068, 0.0074, 0.0112, 0.0149, 0.0185, 0.0199, 0.0221, 0.0221, 0.0243, 0.0257, 0.0292, 0.0432, 0.0569, 0.0701, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],],
            # m = 3
            [[[0.0032, 0.0039, 0.0045, 0.0053, 0.0056, 0.0064, 0.0074, 0.0078, 0.0084, 0.0086, 0.0089, 0.0095, 0.0100, 0.0111, 0.0130, 0.0148, 0.0155, 0.0165, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0048, 0.0055, 0.0062, 0.0069, 0.0082, 0.0089, 0.0095, 0.0102, 0.0109, 0.0122, 0.0136, 0.0150, 0.0169, 0.0203, 0.0235, 0.0269, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0037, 0.0045, 0.0052, 0.0060, 0.0067, 0.0074, 0.0111, 0.0147, 0.0169, 0.0183, 0.0211, 0.0218, 0.0232, 0.0289, 0.0290, 0.0290, 0.0360, 0.0378, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [0.0039, 0.0055, 0.0063, 0.0078, 0.0117, 0.0155, 0.0192, 0.0231, 0.0231, 0.0268, 0.0304, 0.0338, 0.0378, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],]
    ])

    ref_Hm = np.array([
            # m = 0
            [[[47.4, 54.7, 57.9, 63.8, 70.1, 83.3, 92.0, 90.3, 93.0, 92.5, 95.4, 93.9, 95.5, 93.2, 94.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [67.6, 83.5, 109.1, 121.4, 135.3, 138.5, 143.9, 156.1, 164.9, 172.2, 174.6, 179.3, 184.6, 183.3, 187.6, 183.3, 184.6, 189.9, 183.2, 189.6, 185.2, np.nan, np.nan, np.nan],
              [74.3, 86.9, 101.6, 125.9, 134.3, 145.6, 145.3, 154.9, 153.9, 153.8, 151.0, 150.0, 148.1, 149.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [59.4, 70.1, 78.9, 94.3, 104.8, 112.2, 113.1, 115.9, 116.7, 120.4, 121.2, 118.8, 118.2, 116.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [53.4, 70.0, 86.4, 90.2, 98.9, 98.6, 99.4, 103.1, 103.9, 104.1, 105.3, 104.0, 105.2, 103.3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]]],
            # m = 1
            [[[52.8, 67.7, 82.5, 116.9, 139.4, 144.3, 145.7, 146.2, 142.6, 139.2, 136.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [65.5, 79.2, 86.1, 92.9, 104.6, 112.3, 119.0, 122.5, 126.0, 129.7, 125.9, 129.0, 128.0, 128.1, 123.6, 121.2, 120.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [38.7, 41.7, 45.6, 54.4, 60.3, 70.0, 72.8, 81.5, 79.6, 83.4, 85.3, 86.2, 85.2, 84.2, 88.9, 89.7, 89.4, 87.5, 91.1, 89.1, 88.6, 87.0, 87.9, 86.1],
              [27.8, 29.8, 34.7, 48.5, 65.0, 69.7, 70.6, 73.3, 74.3, 76.1, 77.0, 79.6, 78.5, 78.3, 77.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
             [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,  np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],],
            # m = 3
            [[[37.9, 39.8, 49.8, 62.7, 65.6, 71.5, 83.4, 88.3, 90.2, 95.2, 92.2, 96.5, 98.0, 93.9, 96.1, 94.6, 95.2, 96.5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [34.8, 40.8, 43.7, 50.7, 56.5, 60.5, 66.4, 67.3, 72.2, 75.1, 77.9, 79.8, 79.6, 83.3, 83.0, 82.7, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [15.9, 18.9, 22.9, 25.8, 28.8, 29.8, 40.5, 47.3, 51.1, 51.1, 53.8, 51.8, 54.7, 52.4, 52.4, 52.4, 53.0, 52.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [13.9, 18.9, 21.9, 24.8, 34.6, 40.4, 43.2, 45.9, 45.9, 45.7, 45.6, 46.4, 46.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],],
             [[np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
              [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],]]
    ]) / -R
    aqueous = True

    """ DEFINE STATE """
    f = define_flash(components, aqueous)
    p_idx = None
    T_idx = 0
    m_idx = 0
    state_spec = {"temperature": ref_T[T_idx],
                  "pressure": ref_p[m_idx][T_idx],
                  }
    dz = 0.0001
    min_z = [0.]
    max_z = [0.08]
    compositions = {comp: np.arange(min_z[i], max_z[i] + 0.1 * dz, dz) for i, comp in enumerate(components[:-1])}
    compositions[components[-1]] = 1.

    """ Plot enthalpy of mixing """
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")
    results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions,
                                          state_variables=['pressure', 'temperature'] + components,
                                          flash_results=flash_results,
                                          properties_to_evaluate={"H": EoS.Property.ENTHALPY}, phase_idxs=[f.np_max-1], mix=True,
                                          print_state="NP props")

    plot = PlotProps.binary(flash=f, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z,
                            prop_name="H", props=results_np, plot_1p=False,
                            ax_label=r"$-\Delta H^{mix}$/R",
                            # title=r"Enthalpy of dissolution of CO$_2$ in H$_2$O",
                            # flash_results=flash_results,
                            datalabels=["P = {} bar".format(p) for p in state_spec["pressure"]],
                            )
    plot.draw_point(ref_x[m_idx][T_idx], ref_Hm[m_idx][T_idx], widths=15)
    plot.set_axes(ylim=[0., -22.5])
    plt.savefig("Hdiss-" + f.filename + ".pdf")

    # def calculate(self):
    #     # CO2 dissolution enthalpy Guo (2019) Fig. 9
    #     T_HdissCO2 = np.arange(273.15, 573.15, 1)
    #     HdissCO2 = np.zeros(len(T_HdissCO2))
    #     comp = ["H2O", "CO2"]
    #     comp_data = CompData(comp, setprops=True)
    #     aq = AQComposite(comp_data, Jager2003(comp_data), Ziabakhsh2012(comp_data))
    #     enthAq = EoSEnthalpy(aq, EnthalpyIdeal(comp))
    #     x = [0.99, 0.01]
    #     for i, T in enumerate(T_HdissCO2):
    #         Hbrine_pure = enthAq.evaluate(self.p_ref_HdissCO2[0], T, np.array([1., 0.])) * 1e-3
    #         HbrineCO2 = enthAq.evaluate(self.p_ref_HdissCO2[0], T, x) * 1e-3
    #         H_gas = HbrineCO2 - Hbrine_pure * (1 - x[1])
    #         MW = 44.03  # CO2
    #         HdissCO2[i] = H_gas / x[1] / (MW / 1000)  # kJ/mol -> kJ/kg
    #
    #     # CH4 dissolution enthalpy Duan (2003) Table 10
    #     T_HdissC1 = np.arange(273.15, 343.15, 1)
    #     HdissC1 = np.zeros(len(T_HdissC1))
    #     comp = ["H2O", "C1"]
    #     comp_data = CompData(comp, setprops=True)
    #     aq = AQComposite(comp_data, Jager2003(comp_data), Ziabakhsh2012(comp_data))
    #     enthAq = EoSEnthalpy(aq, EnthalpyIdeal(comp))
    #     x = [0.99, 0.01]
    #     for i, T in enumerate(T_HdissC1):
    #         Hbrine_pure = enthAq.evaluate(self.p_ref_HdissC1[0], T, np.array([1., 0.])) * 1e-3
    #         HbrineC1 = enthAq.evaluate(self.p_ref_HdissC1[0], T, x) * 1e-3
    #         H_gas = HbrineC1 - Hbrine_pure * (1 - x[1])
    #         MW = 16.043  # C1
    #         HdissC1[i] = H_gas / x[1]  # kJ/mol
    #
    #     # # Salting out coefficient Guo (2019) fig. 1
    #     return [T_HdissC1, T_HdissCO2], [HdissC1, HdissCO2]


""" WATER/BRINE ENTHALPY """
if 0:
    # Brine enthalpy Pitzer et al. (1984), data from Guo et al. (2019), Table 4
    components = ["H2O"]
    ions = ["Na+", "Cl-"]

    ref_p = np.array([10.])
    ref_T = np.array([298.15, 323.15, 348.15, 373.15, 423.15, 473.15, 523.15, 573.15])
    ref_Hm = np.array([[104.89, 209.33, 313.93, 419.04, 632.2, 852.45, 1085.4, 1344.],
                       [100.74, 201.84, 303.16, 404.88, 610.7, 822.2, 1043.2, 1279.4],
                       [97.12, 195.72, 294.55, 393.71, 594.0, 799.2, 1012.2, 1234.5],
                       [88.29, 181.88, 275.65, 369.55, 558.6, 751.3, 949.1, 1146.9],
                       [78.59, 167.8, 257.07, 346.28, 525.3, 707.3, 892.2, 1070.8],
                       [68.39, 153.81, 239.11, 324.17, 494.4, 667.0, 840.9, 1003.6]
                       ])
    ref_m = np.array([1e-8, 0.4708, 0.9006, 1.9012, 3.0197, 4.2777])

    """ Define flash object """
    aqueous = True
    f = define_flash(components, aqueous, ions)

    """ Calculate water/brine enthalpy """
    prop_array = np.empty(np.shape(ref_Hm))

    for i, m in enumerate(ref_m):
        x_Na = m / 55.509
        x = [1., x_Na, x_Na]
        x = x / np.sum(x)
        MWbrine = 18.015 * x[0] + 22.99 * x[1] + 35.453 * x[2]

        ref_Hm[i, :] *= MWbrine

        for j, T in enumerate(ref_T):
            prop_array[i, j] = f.eos["Aq"].H(ref_p[0], T, x) * R
        prop_array[i, :] += ref_Hm[i, 0] - prop_array[i, 0]

    plot = Diagram(figsize=(8, 5))

    plot.draw_line(X=ref_T, Y=prop_array,
                   datalabels=["m = {:.4f}".format(mi) for mi in ref_m],
                   widths=1.5
                   )
    plot.draw_point(X=ref_T, Y=ref_Hm, widths=15)
    plot.add_attributes(#title="Total enthalpy of " + f.mixture_name,
                        ax_labels=["temperature, K", "enthalpy, kJ/mol"],
                        legend=True, legend_loc='lower right', grid=True,
                        )
    plot.set_axes(ylim=[0., 25000.])

    plt.savefig("Hbrine.pdf")


""" ENTHALPY OF DISSOLUTION FOR GASES """
if 0:
    if 1:
        # CO2 dissolution enthalpy Guo (2019) Fig. 9
        components = ["H2O", "CO2"]
        ions = None

        ref_p = np.array([1.01325])
        ref_T = np.array([298.15, 313.15, 373.15, 373.15, 398.15, 423.15, 423.15, 448.15, 473.15, 473.15, 497.15, 523.15, 548.15])
        ref_Hm = np.array([-448.11, -370.27, -167.03, -132.43, -119.46, -58.919, 31.892, 5.946, 70.811, 140.00, 140.00, 209.19, 338.92])

    else:
        # C1 dissolution enthalpy Duan (2003) Table 10
        components = ["H2O", "C1"]
        ions = None

        ref_p = np.array([1.])
        ref_T = [[288.15, 298.15, 308.15],
                 [288.15, 298.15, 308.15],
                 [288.15, 298.15, 308.15, 313.15, 323.15, 333.15],
                 [288.15, 298.15, 308.15, 313.15, 323.15, 333.15]]
        ref_Hm = [[-15.45, -13.18, -11.09],
                  [-15.53, -13.06, -10.70],
                  [-15.60, -13.19, -10.87, -9.75, -7.59, -5.54],
                  [-14.56, -12.64, -10.75, -9.82, -7.97, -6.15]]

    # CO2 dissolution enthalpy Guo (2019) Fig. 9
    T_HdissCO2 = np.arange(273.15, 573.15, 1)
    HdissCO2 = np.zeros(len(T_HdissCO2))
    comp = ["H2O", "CO2"]
    comp_data = CompData(comp, setprops=True)
    aq = AQComposite(comp_data, Jager2003(comp_data), Ziabakhsh2012(comp_data))
    enthAq = EoSEnthalpy(aq, EnthalpyIdeal(comp))
    x = [0.99, 0.01]
    for i, T in enumerate(T_HdissCO2):
        Hbrine_pure = enthAq.evaluate(self.p_ref_HdissCO2[0], T, np.array([1., 0.])) * 1e-3
        HbrineCO2 = enthAq.evaluate(self.p_ref_HdissCO2[0], T, x) * 1e-3
        H_gas = HbrineCO2 - Hbrine_pure * (1 - x[1])
        MW = 44.03  # CO2
        HdissCO2[i] = H_gas / x[1] / (MW / 1000)  # kJ/mol -> kJ/kg

    # CH4 dissolution enthalpy Duan (2003) Table 10
    T_HdissC1 = np.arange(273.15, 343.15, 1)
    HdissC1 = np.zeros(len(T_HdissC1))
    comp = ["H2O", "C1"]
    comp_data = CompData(comp, setprops=True)
    aq = AQComposite(comp_data, Jager2003(comp_data), Ziabakhsh2012(comp_data))
    enthAq = EoSEnthalpy(aq, EnthalpyIdeal(comp))
    x = [0.99, 0.01]
    for i, T in enumerate(T_HdissC1):
        Hbrine_pure = enthAq.evaluate(self.p_ref_HdissC1[0], T, np.array([1., 0.])) * 1e-3
        HbrineC1 = enthAq.evaluate(self.p_ref_HdissC1[0], T, x) * 1e-3
        H_gas = HbrineC1 - Hbrine_pure * (1 - x[1])
        MW = 16.043  # C1
        HdissC1[i] = H_gas / x[1]  # kJ/mol

    # Enthalpy of CO2 dissolution
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    plt.rc('font', size=16)

    axs[0].grid(True, which='both', linestyle='-.')
    axs[0].tick_params(direction='in', length=1, width=1, colors='k',
                       grid_color='k', grid_alpha=0.2)

    axs[0].plot(T_HdissC1, HdissC1, linewidth=2)
    axs[0].scatter(self.T_ref_HdissC1[0][:], self.HdissC1_ref[0][:], linewidth=2)
    axs[0].scatter(self.T_ref_HdissC1[1][:], self.HdissC1_ref[1][:], linewidth=2)
    axs[0].scatter(self.T_ref_HdissC1[2][:], self.HdissC1_ref[2][:], linewidth=2)
    axs[0].scatter(self.T_ref_HdissC1[3][:], self.HdissC1_ref[3][:], linewidth=2)

    axs[0].set_title('CH4')
    axs[0].set(xlabel='temperature, K', ylabel='H, kJ/mol')

    # Enthalpy of C1 dissolution
    axs[1].grid(True, which='both', linestyle='-.')
    axs[1].tick_params(direction='in', length=1, width=1, colors='k',
                       grid_color='k', grid_alpha=0.2)

    axs[1].plot(T_HdissCO2, HdissCO2, linewidth=2)
    axs[1].scatter(self.T_ref_HdissCO2, self.HdissCO2_ref, linewidth=2)

    axs[1].set_title('CO2')
    axs[1].set(xlabel='temperature, K', ylabel='H, kJ/mol')

    plt.rc('font', size=16)
    plt.suptitle('Enthalpy of gas dissolution in water')

plt.show()
