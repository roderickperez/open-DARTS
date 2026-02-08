import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS, Ballard

from dartsflash.hyflash import HyFlash
from dartsflash.components import CompData, ConcentrationUnits as cu
from dartsflash.plot import PlotHydrate


if 0:
    """CO2 sI equilibrium PT"""
    components = ["H2O", "CO2"]
    ions = ["Na+", "Cl-"]
    trange = [271., 291.]
    state_spec = {"temperature": np.linspace(trange[0], trange[1], 40),
                  "pressure": None
                  }
    compositions = {"H2O": 0.8,
                    "CO2": 0.2,
                    }
    concentrations = [{"NaCl": m} for m in [0., 0.01, 0.05, 0.10, 0.15]]  #, 0.20, 0.25]]
    concentration_unit = cu.WEIGHT

    ref_T = [float(i) for i in range(271, 291)]
    ref_p = [[10.22278855301975, 10.916282619416473, 12.182293454706306, 13.745416208713321, 15.50946687924238,
              17.693363362654498, 19.74557345084955, 22.525432627269492, 25.697251071130427, 28.99550728193967,
              33.07799441785211, 39.00378586433987, 49.67893177327907, 132.37526067025257, 234.64316877596258,
              348.7097695267014, 490.4466824373712, 660.0559213976551, 868.9508558685552, 1169.4294015942921,
              # 1681.419738721254, 2214.590162102342, 2792.3803696708565]),
              ],
             np.ones(20) * np.nan, np.ones(20) * np.nan, np.ones(20) * np.nan, np.ones(20) * np.nan] #, None, None]
    labels = ["m = {:.1f} wt%".format(c["NaCl"]*100) for c in concentrations]
    pt_plot = True

elif 0:
    """C1 sI equilibrium PT"""
    components = ["H2O", "C1"]
    ions = ["Na+", "Cl-"]
    trange = [271., 291.]
    state_spec = {"temperature": np.linspace(trange[0], trange[1], 40),
                  "pressure": None
                  }
    compositions = {"H2O": 0.8,
                    "C1": 0.2,
                    }
    concentrations = [{"NaCl": m} for m in np.array([0., 2.001E-2, 3.611E-2, 5.994E-2, 8.014E-2]) * 55.509]
    concentration_unit = cu.MOLALITY
    labels = ["m = {:.2f} M".format(c["NaCl"]) for c in concentrations]

    ref_T = [
        [float(i) for i in range(273, 289)] + [291.86, 293.08, 293.46, 295.08, 295.94, 297.48, 297.58, 298.24, 298.68],
        [280.66, 286.00, 289.42, 291.71, 293.42, 293.51, 295.62, 296.30, 297.42, 298.42, 299.06] + [None] * 14,
        [279.16, 284.53, 287.50, 288.30, 289.23, 290.55, 292.15, 293.37, 294.58, 295.47, 296.03] + [None] * 14,
        [274.40, 280.25, 282.33, 284.67, 286.23, 287.47, 288.42, 289.21, 290.37, 291.00] + [None] * 15,
        [270.66, 275.22, 278.04, 279.20, 281.29, 282.39, 283.49, 284.36, 284.92, 285.76] + [None] * 15
        ]
    ref_p = [
        [25.582, 28.259, 31.226, 34.519, 38.178, 42.251, 46.796, 51.878, 57.576, 63.982, 71.208, 79.389, 88.686, 99.293, 111.44, 125.41] + [201.90, 229.10, 242.30, 304.60, 342.80, 434.60, 432.50, 456.40, 504.50],
        [66.00, 137.10, 213.80, 281.80, 359.40, 348.40, 455.20, 487.50, 562.00, 632.50, 678.10] + [None] * 14,
        [75.10, 140.70, 233.80, 238.80, 271.90, 346.90, 418.50, 490.00, 576.30, 646.00, 715.60] + [None] * 14,
        [79.20, 143.20, 207.40, 291.20, 359.30, 429.30, 495.30, 564.70, 641.40, 705.60] + [None] * 15,
        [78.50, 149.40, 229.80, 281.80, 378.60, 422.60, 505.70, 588.50, 640.30, 713.00] + [None] * 15
        ]
    pt_plot = True

else:
    """Gas mixture equilibrium Pressure"""
    components = ["H2O", "CO2", "C1"]
    ions = None
    concentrations = None
    concentration_unit = cu.MOLALITY

    state_spec = {"temperature": np.arange(271., 291., 1.),
                  "pressure": None
                  }
    dz = 0.05
    min_z = [0.]
    max_z = [0.5]
    compositions = {"H2O": np.array([1.-max_z[0]]),
                    components[1]: np.arange(min_z[0], max_z[0]+dz*0.1, dz),
                    components[2]: 1.,
                    }

    ref_T = [None]
    ref_p = [None]
    pt_plot = False

comp_data = CompData(components, ions, setprops=True)
f = HyFlash(comp_data=comp_data)

f.add_eos("CEOS", CubicEoS(comp_data, CubicEoS.SRK),
          trial_comps=[InitialGuess.Yi.Wilson])
f.add_eos("AQ", AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                                      AQEoS.solute: AQEoS.Ziabakhsh2012,
                                      AQEoS.ion: AQEoS.Jager2003}))
f.add_hydrate_eos("sI", Ballard(comp_data, "sI"))

f.flash_params.eos_order = ["AQ", "CEOS"]
f.init_flash(flash_type=HyFlash.FlashType.NegativeFlash, eos_order=["AQ", "CEOS"], nf_initial_guess=[InitialGuess.Henry_AV])

if concentrations is not None:
    results = []
    for concentration in concentrations:
        results.append(f.evaluate_equilibrium(state_spec, compositions, mole_fractions=True,
                                              concentrations=concentration, concentration_unit=concentration_unit))
    results = xr.concat(results, dim='concentrations')
else:
    results = f.evaluate_equilibrium(state_spec, compositions, mole_fractions=True, print_state="Hydrate PT")

if pt_plot:
    plot = PlotHydrate.pt(f, results, compositions_to_plot=compositions, concentrations=concentrations, xlim=trange, logy=False,
                          ref_t=ref_T, ref_p=ref_p, labels=labels, legend_loc="upper left")
    plt.savefig(f.filename + "-hydrate-equilibrium.pdf")
else:
    plot = PlotHydrate.binary(f, results, variable_comp_idx=1, dz=dz, min_z=min_z, max_z=max_z, state=state_spec, logy=True)

plt.show()
