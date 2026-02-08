import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import InitialGuess, EoS, AQEoS, CubicEoS

from dartsflash.dartsflash import DARTSFlash
from dartsflash.components import CompData, ConcentrationUnits as cu
from dartsflash.mixtures import VLAq
from dartsflash.plot import PlotFlash


if 0:
    """Validation of xCO2, Ziabaksh (2012)"""
    comp_data = CompData(components=["H2O", "CO2"], ions=["Na+", "Cl-"], setprops=True)
    f = VLAq(comp_data, hybrid=True)

    f.set_vl_eos("PR", trial_comps=[InitialGuess.Yi.Wilson])
    f.set_aq_eos("Aq", )

    # f = DARTSFlash(comp_data)
    # f.add_eos("Aq", AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
    #                                   AQEoS.solute: AQEoS.Ziabakhsh2012,
    #                                   }))
    # f.add_eos("VL", CubicEoS(comp_data, CubicEoS.PR))
    f.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash, eos_order=["Aq", "VL"], nf_initial_guess=[InitialGuess.Henry_AV],
                 # verbose=True
                 )

    state_spec = {"pressure": np.arange(1., 600., 1.),
                  "temperature": np.array([273.15 + 25, 273.15 + 90])
                  }
    compositions = {"H2O": 0.8,
                    "CO2": 0.2
                    }
    results_m0 = f.evaluate_flash(state_spec=state_spec, compositions=compositions,
                                  mole_fractions=True, concentrations={"NaCl": 0.})
    results_m2 = f.evaluate_flash(state_spec=state_spec, compositions=compositions,
                                  mole_fractions=True, concentrations={"NaCl": 2.})
    flash_results = xr.concat([results_m0, results_m2], dim='concentrations')

    ref_p = [[[51.51, 75.76, 103.03, 137.88, 153.03, 178.79, 204.55, 204.55, 406.06] + [None] * 23,
             [39.8374, 59.3496, 101.626, 160.976, 257.724, 342.276, 439.837, 508.943, 555.285] + [None] * 23],
             [[None] * 32,
             [9.5599e0, 2.0182e1, 4.0364e1, 6.0546e1, 7.0106e1, 7.9666e1, 9.9848e1, 1.2003e2, 1.4021e2, 1.6039e2, 1.7951e2,
              1.9969e2, 2.1987e2, 2.4006e2, 2.6024e2, 2.8042e2, 3.0060e2, 3.1972e2, 3.3990e2, 3.6115e2, 3.8027e2, 4.0045e2,
              4.2063e2, 4.4081e2, 4.5993e2, 4.8012e2, 5.0030e2, 5.1942e2, 5.4066e2, 5.6191e2, 5.8103e2, 6.0121e2]]]
    ref_data = [[[0.0212, 0.0249, 0.0255, 0.0260, 0.0262, 0.0269, 0.0274, 0.0259, 0.0298] + [None] * 23,
                [0.0053479, 0.0087771, 0.0135211, 0.0180282, 0.0214789, 0.023662, 0.0255634, 0.0266901, 0.0274648] + [None] * 23],
                [[None] * 32,
                [0.001453, 0.002688, 0.004796, 0.006686, 0.007558, 0.008284, 0.009520, 0.01053, 0.01133, 0.01191,
                 0.01250, 0.01286, 0.01322, 0.01366, 0.01395, 0.01424, 0.01446, 0.01475, 0.01497, 0.01526, 0.01547,
                 0.01577, 0.01598, 0.01620, 0.01642, 0.01664, 0.01693, 0.01707, 0.01729, 0.01751, 0.01765, 0.01795]]]
    labels = [['25$\degree$C, m=0', '90$\degree$C, m=0'], ['25$\degree$C, m=2', '90$\degree$C, m=2']]

    plot = PlotFlash.solubility(f, flash_results, x_var='pressure', dissolved_comp_idx=1, phase_idx=0,
                                labels=labels, concentrations=[0, 2], styles=['solid', 'dashed'],
                                xlim=[0, 600], ylim=[0., None], legend_loc='lower right')
    plot.draw_point(X=ref_p[0], Y=ref_data[0])  # m=0
    plot.draw_point(X=ref_p[1], Y=ref_data[1], markers='^')  # m=2

    plt.savefig("xCO2-salinity.pdf")

if 0:
    """XCO2 with IMPURITIES, Ziabaksh (2012)"""
    ref_p = [8.037, 18.70, 38.67, 79.85, 120.8, 159.2, 200.1, 299.3, 401.0, 498.8, 600.5]
    temperature = 273.15 + 50
    components = [["H2O", "CO2", "C1"],
                  ["H2O", "CO2", "C1"],
                  ["H2O", "CO2", "N2"],
                  ["H2O", "CO2", "H2S"]]
    composition = [[0.8, 0.2, 0.],
                   [0.8, 0.2 * (1 - 0.1262), 0.2 * 0.1262],
                   [0.8, 0.2 * (1 - 0.0764), 0.2 * 0.0764],
                   [0.8, 0.2 * (1 - 0.0636), 0.2 * 0.0636]]
    ref_data = [[0.002917, 0.006209, 0.011102, 0.018064, 0.020793, 0.021922, 0.022768, 0.024650, 0.025944, 0.027249, 0.02860],
                [0.003097, 0.005724, 0.010040, 0.016233, 0.018672, 0.019517, 0.020361, 0.021957, 0.023364, 0.024584, 0.025804],
                [0.003284, 0.006005, 0.010697, 0.016890, 0.019517, 0.020361, 0.021206, 0.022895, 0.024396, 0.025804, 0.027024],
                [0.003190, 0.006005, 0.011072, 0.017453, 0.019705, 0.020924, 0.021675, 0.023552, 0.025053, 0.026554, 0.027868]]
    labels = ["CO2", "CO2+C1", "CO2+N2", "CO2+H2S"]

    results = []
    for i, comps in enumerate(components):
        comp_data = CompData(components=comps, setprops=True)
        f = VLAq(comp_data, hybrid=True)

        f.set_vl_eos("PR", trial_comps=[InitialGuess.Yi.Wilson])
        f.set_aq_eos("Aq", )

        f.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash, eos_order=["Aq", "VL"],
                     nf_initial_guess=[InitialGuess.Henry_AV],
                     # verbose=True
                     )

        state_spec = {"pressure": np.arange(1., 600., 1.),
                      "temperature": temperature
                      }
        compositions = {comp: composition[i][j] for j, comp in enumerate(comps)}

        results.append(f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True))

    flash_results = xr.concat(results, dim='concentrations')

    plot = PlotFlash.solubility(f, flash_results, x_var='pressure', dissolved_comp_idx=1, phase_idx=0,
                                labels=labels, concentrations=[0, 1, 2, 3], styles=['solid' for i in range(4)],
                                xlim=[0, 600], ylim=[0., None], legend_loc='lower right')
    plot.draw_point(X=ref_p, Y=ref_data)  # m=0
    # plot.draw_point(X=ref_p[1], Y=ref_data[1], markers='^')  # m=2

    plt.savefig("xCO2-impurities.pdf")

if 0:
    """Validation of yH2O, Spycher (2003)"""
    comp_data = CompData(components=["H2O", "CO2"], setprops=True)
    f = VLAq(comp_data=comp_data, hybrid=True)

    f.set_vl_eos("PR", trial_comps=[1], root_order=[EoS.STABLE])
    f.set_aq_eos("Aq")
    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
                 nf_initial_guess=[InitialGuess.Henry_AV]
                 )

    state_spec = {"pressure": np.arange(1.1, 300., 1.),
                  "temperature": np.array([273.15 + 25., 273.15 + 50., 273.15 + 75., 273.15 + 100.])
                  }
    compositions = {"H2O": 0.8,
                    "CO2": 0.2
                    }

    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True)

    ref_p = [[1., 22.7, 25.3, 29.8, 30., 37.3, 37.4, 48.3, 50.7, 50.7, 65.9, 70.9, 76., 82.8, 91.2, 101.3, 101.3, 101.4,
              103.4, 111.5, 126.7, 141.9, 152., 152., 177.3, 202.7, 202.7, 202.7, 456., 481.3, 506.6],
             [1., 17.3, 25.3, 25.5, 25.8, 36.4, 36.4, 46.3, 50.7, 60.8, 68.2, 75.3, 76., 87.2, 100.6, 101., 101.3, 101.3,
              101.33, 122.1, 147.5, 152., 152., 176.8, 201., 202.7, 301., 344.8, 405.3, 608., 709.3],
             [1., 6.9, 25.3, 25.3, 23.3, 37.4, 37.5, 50.7, 51.3, 51.5, 101.3, 101.33, 101.4, 103.4, 111.5, 126.7, 152.,
              152., 153.1, 202.7, 202.7, 209.4, 344.8, 405.3, 608., 709.3] + [None] * 5,
             [3.25, 6., 9.2, 11.91, 14.52, 18.16, 23.07, 36.8, 37.2, 44.8, 44.8, 51.5, 51.5] + [None] * 18]
    ref_data = [[28.62e-3, 1.95e-3, 1.64e-3, 1.63e-3, 1.67e-3, 1.45e-3, 1.49e-3, 1.2787e-3, 1.28e-3, 1.29e-3, 3.e-3,
                 3.07e-3, 3.09e-3, 3.0152e-3, 3.14e-3, 3.27e-3, 3.32e-3, 3.36e-3, 3.3739e-3, 3.37e-3, 3.41e-3, 3.44e-3,
                 3.54e-3, 3.6e-3, 3.69e-3, 3.76e-3, 3.78e-3, 3.77e-3, 4.01e-3, 3.99e-3, 3.97e-3],
                [115.71e-3, 8.41e-3, 6.2e-3, 5.95e-3, 5.98e-3, 4.66e-3, 4.63e-3, 3.96e-3, 3.83e-3, 3.57e-3, 3.39e-3,
                 3.45e-3, 3.5e-3, 3.64e-3, 4.29e-3, 5.47e-3, 4.36e-3, 4.49e-3, 5.5e-3, 5.43e-3, 6.08e-3, 6.10e-3, 7.9e-3,
                 6.43e-3, 6.82e-3, 6.77e-3, 7.82e-3, 7.50e-3, 7.59e-3, 7.93e-3, 8.01e-3],
                [301.09e-3, 60.14e-3, 18.16e-3, 10.64e-3, 20.e-3, 12.5e-3, 12.6e-3, 10.87e-3, 10.4e-3, 10.2e-3, 8.29e-3,
                 7.4e-3, 7.27e-3, 6.3e-3, 8.11e-3, 8.55e-3, 9.56e-3, 9.e-3, 7.5e-3, 9.38e-3, 11.32e-3, 8.4e-3, 13.3e-3,
                 13.19e-3, 13.93e-3, 14.e-3] + [None] * 5,
                [288.e-3, 155.e-3, 107.e-3, 77.e-3, 69.e-3, 54.e-3, 45.e-3, 32.8e-3, 32.3e-3, 27.7e-3, 27.4e-3, 24.8e-3,
                 25.1e-3] + [None] * 18]
    labels = ["{}$\degree$C".format(temp-273.15) for temp in state_spec["temperature"]]

    plot = PlotFlash.solubility(f, flash_results, x_var='pressure', dissolved_comp_idx=0, phase_idx=1,
                                labels=labels, xlim=[0, 300], ylim=[1e-3, 0.4], logy=True)
    plot.draw_point(X=ref_p, Y=ref_data)

    plt.savefig("yH2O.pdf")

if 0:
    """Validation of yH2O, Spycher (2003)"""
    comp_data = CompData(components=["H2O", "CO2"], ions=["Na+", "Cl-"], setprops=True)
    f = VLAq(comp_data=comp_data, hybrid=True)

    f.set_vl_eos("PR", trial_comps=[InitialGuess.Yi.Wilson], root_order=[EoS.RootFlag.STABLE])
    f.set_aq_eos("Aq", )
    f.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash, eos_order=["Aq", "VL"],
                 nf_initial_guess=[InitialGuess.Henry_AV])

    state_spec = {"pressure": np.arange(1., 300., 1.),
                  "temperature": 273.15 + 50.}
    compositions = {"H2O": 0.8,
                    "CO2": 0.2,
                    }
    concentrations = [0., 2., 4., 6., 6.329]
    results = []
    for i, concentration in enumerate(concentrations):
        results.append(f.evaluate_flash(state_spec=state_spec, compositions=compositions,
                                        mole_fractions=True, concentrations={"NaCl": concentration},
                                        concentration_unit=cu.MOLALITY))
    flash_results = xr.concat(results, dim='concentrations')

    ref_p = np.array([[1., 17.3, 25.3, 25.5, 25.8, 36.4, 36.4, 46.3, 50.7, 60.8, 68.2, 75.3, 76., 87.2, 100.6, 101., 101.3,
                       101.3, 101.33, 122.1, 147.5, 152., 152., 176.8, 201., 202.7, 301., 344.8, 405.3, 608., 709.3],
                      [None] * 31, [None] * 31, [None] * 31, [None] * 31])
    ref_data = np.array([[115.71e-3, 8.41e-3, 6.2e-3, 5.95e-3, 5.98e-3, 4.66e-3, 4.63e-3, 3.96e-3, 3.83e-3, 3.57e-3, 3.39e-3,
                          3.45e-3, 3.5e-3, 3.64e-3, 4.29e-3, 5.47e-3, 4.36e-3, 4.49e-3, 5.5e-3, 5.43e-3, 6.08e-3, 6.10e-3,
                          7.9e-3, 6.43e-3, 6.82e-3, 6.77e-3, 7.82e-3, 7.50e-3, 7.59e-3, 7.93e-3, 8.01e-3],
                         [None] * 31, [None] * 31, [None] * 31, [None] * 31])
    labels = ['m = {:.3f}'.format(i) for i in [0., 2., 4., 6., 6.329]]

    plot = PlotFlash.solubility(f, flash_results, x_var='pressure', dissolved_comp_idx=0, phase_idx=1,
                                labels=labels, ylim=[2.5e-3, 0.01], xlim=[0, 300], logy=True, legend_loc='lower right'
                                )
    plot.draw_point(X=ref_p, Y=ref_data)

    plt.savefig("yH2O-salinity.pdf")

if 0:
    """Validation of brine-gas at surface conditions"""
    comp_data = CompData(components=["H2O", "CO2"], setprops=True)
    f = VLAq(comp_data=comp_data, hybrid=True)

    f.set_vl_eos("PR", trial_comps=[InitialGuess.Yi.Wilson])
    f.set_aq_eos("Aq", )
    f.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash, eos_order=["Aq", "VL"], nf_initial_guess=[InitialGuess.Henry_AV])

    state_spec = {"pressure": 1.01325,
                  "temperature": np.arange(273.15, 363.15, 1.)}
    compositions = {"H2O": 0.8,
                    "CO2": 0.2,
                    }
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=False)

    """Validation of xCO2 at surface conditions (Carroll, 1991)"""
    ref_T = np.arange(273.15, 373.15, 10)
    ref_data = np.array([13.426, 9.5106, 7.0213, 5.234, 4.0426, 3.234, 2.4681, 1.8085, 1.2340, 0.6170]) / 10000

    plot = PlotFlash.solubility(f, flash_results, x_var='temperature', dissolved_comp_idx=1, phase_idx=0,
                                # labels=labels
                                )
    plot.draw_point(X=ref_T, Y=ref_data)

    """Validation of yH2O at surface conditions (Spycher, 2003)"""
    ref_T = np.array([25., 31.04, 50., 75.]) + 273.15
    ref_data = np.array([0.0286, 0.0398, 0.116, 0.301])

    plot = PlotFlash.solubility(f, flash_results, x_var='temperature', dissolved_comp_idx=0, phase_idx=1,
                                # labels=labels
                                )
    plot.draw_point(X=ref_T, Y=ref_data)

if 0:
    """Validation of yH2O with Ih Ice"""
    ref_p = [
        # np.arange(1, 20, 1), np.arange(1, 27, 1),
        np.arange(1, 35, 1), np.arange(1, 46, 1), np.arange(1, 76, 1)
    ]
    temperature = [
        # 253.15, 263.15,
        273.15, 283.15, 298.15
    ]
    components = ["H2O", "CO2"]
    composition = [0.8, 0.2]
    ref_data = [
        np.array([0.12677090100064, 0.0638449563971339, 0.0428768366801168, 0.0323982386753862, 0.0261158543945171, 0.021931767686759,
                  0.0189467062363757, 0.0167112319731334, 0.0149757937254469, 0.0135903337166458, 0.0124596888983803, 0.0115203106288921,
                  0.0107280927062505, 0.0100518084441467, 0.00946829202553471, 0.00896045234165744, 0.0085149867822623,
                  0.00812177304036575, 0.00777277056489548]) * 0.01,
        np.array([0.289109626508199, 0.145633045143549, 0.0978224462517425, 0.0739281767912553, 0.0596010197989581, 0.0500577412426812,
                  0.0432484054992975, 0.0381478278970266, 0.0341870274897647, 0.031024173789738, 0.0284419878518956, 0.0262953496951502,
                  0.0244840736785028, 0.0229364447607547, 0.0216001342978248, 0.0204357659048625, 0.019413109205836, 0.0185088029039568,
                  0.0177045602714168, 0.016985647515714, 0.0163400745171351, 0.0157581508824306, 0.0152320300273719, 0.0147551126364554,
                  0.0143217439334377, 0.0139274986232379]) * 0.01,
        np.array([0.616677102329108, 0.310598359871442, 0.208600177521816, 0.157620761971019, 0.127051313584004, 0.106685834893473,
                  0.092152987065136, 0.0812646875702648, 0.0728076609039985, 0.0660518971270115, 0.0605346582085427, 0.0559458581699594,
                  0.0520723460651281, 0.0487604794026399, 0.0458989497732841, 0.0434031852335289, 0.041208998840837, 0.0392668996576587,
                  0.0375369175238028, 0.0359881296628151, 0.0345945356787622, 0.0333357274875105, 0.0321945664586127, 0.0311564526970052,
                  0.0302099243360278, 0.0293444889779, 0.0285519893375154, 0.0278251869362809, 0.0271576276497735, 0.0265443674045525,
                  0.0259809152429683, 0.025463190691466, 0.0249882718268698, 0.0245534832775758]) * 0.01,
        np.array([1.23947053502918, 0.624144337981624, 0.419076842407803, 0.316581611774703, 0.255112633567611, 0.214159965726946,
                  0.184931617593692, 0.163030463162276, 0.146014531294825, 0.132420486913392, 0.121314230343348, 0.112074334863912,
                  0.104271364901418, 0.0975978999278866, 0.091827874864761, 0.0867923970860641, 0.0823629064302283, 0.0784388960503012,
                  0.0749405207799994, 0.0718044431603302, 0.0689799108881149, 0.0664249581431584, 0.0641045137133454, 0.0619898000254448,
                  0.060057172097809, 0.0582862010659763, 0.0566591564846522, 0.0551612794562462, 0.0537802891553474, 0.0525052391610046,
                  0.0513262997941883, 0.0502354883964252, 0.0492255368589581, 0.0482906368088142, 0.047425443351613, 0.0466250085059171,
                  0.0458855673937672, 0.0452036162954508, 0.0445767325227516, 0.0440026978471602, 0.0434795483626686, 0.0430064036626312,
                  0.04258273276684, 0.0422092493488467, 0.0418871976189025]) * 0.01,
        np.array([3.20280467384792, 1.61234775215096, 1.08224928011749, 0.817273500998723, 0.658350514943481, 0.552458426655411,
                  0.476868087275901, 0.420219068877804, 0.376197724257231, 0.34101856874875, 0.312270567880035, 0.288347668630437,
                  0.268135387401825, 0.250840336025203, 0.235879081923608, 0.222816424311812, 0.211317377219527, 0.201122142562091,
                  0.192026329457634, 0.183864526742474, 0.17650472529013, 0.169838378044589, 0.163775254532745, 0.15824188921828,
                  0.153175012546586, 0.148521651382464, 0.144237234921394, 0.140282097190461, 0.136623655319374, 0.133233139056402,
                  0.13008514750606, 0.127158845339994, 0.124434698116691, 0.121895972034313, 0.119528344385288, 0.117317947702021,
                  0.115253695460238, 0.113325399297151, 0.111524025948916, 0.109841187242207, 0.108270585628957, 0.106805573112644,
                  0.105440657225053, 0.104171042690038, 0.102992932162566, 0.101902085740073, 0.100895932023728, 0.0999721231554912,
                  0.0991289090689897, 0.0983647861071949, 0.0976799808712729, 0.0970743722465674, 0.0965490219616258, 0.09610596306262,
                  0.0957484232333599, 0.0954811307161863, 0.0953111292427621, 0.0952471566304243, 0.0953028291172937, 0.0954972200228889,
                  0.0958586542305711, 0.0964314173528372, 0.0972904587712602, 0.098579820823059, 0.21095864160528, 0.215674010166149,
                  0.219926057497967, 0.223829488780566, 0.227458357956713, 0.230863845232368, 0.234082998945075, 0.237143694018352,
                  0.240067457984919, 0.242871386872749, 0.24556929291693]) * 0.01
    ]
    labels = [
        "-20$\degree$C", "-10$\degree$C",
        "0$\degree$C", "10$\degree$C", "25$\degree$C"
    ]

plt.show()
