import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import EoS, InitialGuess, FlashParams, EoSParams
from dartsflash.libflash import CubicEoS, AQEoS, PureSolid, Ballard, IAPWS95

from dartsflash.dartsflash import DARTSFlash, R
from dartsflash.components import CompData
from dartsflash.plot import *


if 0:
    components = ["CO2"]
    n = [1.]
    pressure = np.arange(25, 150, 0.1)
    # temperature_to_plot = np.array([304, 324, 344, 364, 384])
    temperature_to_plot = np.array([273.15, 293.15, 304.10, 313.15])
    temperature = np.arange(273.15, 373.15, 1)
elif 1:
    components = ["C1"]
    n = [1.]
    pressure = np.arange(25, 75, 0.1)
    # temperature_to_plot = np.array([304, 324, 344, 364, 384])
    temperature_to_plot = np.array([100., 150., 190., 290.])
    temperature = np.arange(100., 200., 1)
elif 1:
    components = ["H2O"]
    n = [1.]
    pressure = np.linspace(1e-5, 500, 1000)
    temperature_to_plot = np.array([273, 373, 473, 573])
    temperature = np.arange(273, 373, 1)
elif 1:
    components = ["H2O", "CO2"]
    n = [np.linspace(0, 1, 11), 1.]
    pressure = np.arange(1, 201, 1)
    # temperature_to_plot = np.array([304, 324, 344, 364, 384])
    temperature_to_plot = np.array([273.15, 293.15, 304.10, 313.15])
    temperature = np.arange(273, 373, 1)
elif 1:
    components = ["CO2", "C1"]
    n = [0.5, 0.5]
    pressure = np.linspace(1e-5, 500, 100000)
    temperature_to_plot = np.array([273, 373, 473, 573])
    temperature = np.arange(273, 373, 1)
else:
    components = ["CO2", "N2", "H2S", "C1", "C2", "C3"]
    n = [0.9, 0.03, 0.04, 0.06, 0.04, 0.03]
    pressure = np.arange(1, 201, 1)
    temperature_to_plot = np.array([273, 293, 304, 313, 330])
    temperature = np.arange(273, 373, 1)

comp_data = CompData(components, setprops=True)
f = DARTSFlash(comp_data=comp_data)

f.add_eos("CEOS", CubicEoS(comp_data, CubicEoS.PR, volume_shift=False),
          trial_comps=[InitialGuess.Yi.Wilson])
ceos = f.eos["CEOS"]

compositions = {comp: n[i] for i, comp in enumerate(components)}
pprops, vprops = None, None

if 0:
    state_spec = {"pressure": pressure,
                  "temperature": temperature
                  }
    properties = {"V": ceos.V,
                  "rho": ceos.rho,
                  "H": ceos.H,
                  "S": ceos.S,
                  "JT": ceos.JT,
                  "Z": ceos.Z,
                  "Cv": ceos.Cv,
                  "Cp": ceos.Cp,
                  "vs": ceos.vs,
                  }

    pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)

    PlotEoS.surf(f, pprops, composition=n, x_var="temperature", y_var="pressure", prop_names=["H", "S"])
    PlotEoS.plot(f, pprops, composition=n, x_var="pressure", prop_names=["JT"],
                 state={"temperature": temperature_to_plot},
                 datalabels=["T = {:.0f} K".format(t) for t in temperature_to_plot])

if 1:
    # vmin = ceos.V(p=pressure[-1], T=temperature[0], n=n)
    # vmax = ceos.V(p=pressure[0], T=temperature[-1], n=n)
    vmin, vmax = 4e-5, 6e-4

    state_spec = {"temperature": temperature,
                  "volume": np.linspace(vmin, vmax, 201)
                  }
    properties = {"Z": ceos.Z,
                  "P": ceos.P
                  }

    vprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)

    state_spec = {"temperature": temperature,
                  "pressure": np.linspace(1., 100., 201)
                  }
    properties = {"Z": ceos.Z,
                  "V": ceos.V
                  }
    pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)

    pv = PlotEoS.pressure_volume(f, temperatures=temperature_to_plot, compositions=[1.],
                                 p_props=pprops, v_props=vprops, v_range=[0, vmax], p_range=[0, 100])

    pz = PlotEoS.compressibility(f, temperatures=temperature_to_plot, compositions=[1.],
                                 p_props=pprops, v_props=vprops, z_range=[-0.1, 1.1], p_range=[-25, 200])

    # Evaluate the Z-factor at (Pc,Tc)
    cp = ceos.critical_point(n=[1.])

    pv.draw_point(X=cp.Vc, Y=cp.Pc, colours=pv.colours[2])
    pz.draw_point(X=cp.Pc, Y=cp.Zc, colours=pz.colours[2])

    plt.show()

if 0:
    comp_data = CompData(components=["CO2", "C1"], setprops=True)
    f = DARTSFlash(comp_data=comp_data)

    f.add_eos("PR", CubicEoS(comp_data, CubicEoS.PR),
              trial_comps=[InitialGuess.Yi.Wilson])
    ceos = f.eos["PR"]

    state_spec = {"pressure": np.arange(10, 301, 1),
                  "temperature": np.array([273.15 + 25, 273.15 + 50, 273.15 + 75]),
                  }
    compositions = {"CO2": np.array([1.-1e-10, 0.5, 1e-10]),
                    "C1": 1.
                    }
    properties = {"V": ceos.V,
                  "Cv": ceos.Cv,
                  "Cp": ceos.Cp,
                  "JT": ceos.JT,
                  "vs": ceos.vs,
                  }
    props = list(properties.keys())

    results = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                             properties_to_evaluate=properties, mole_fractions=True)
    results = results[props]

    num_curves = 3
    ref_p = [np.arange(25, 301, 25) for i in range(num_curves)]
    ref_Cv_CO2 = np.array([
        [32.047, 37.992, 43.558, 41.573, 40.916, 40.617, 40.467, 40.391, 40.357, 40.348, 40.355, 40.373],
        [32.061, 34.836, 39.008, 45.270, 43.077, 41.540, 40.831, 40.441, 40.220, 40.095, 40.028, 39.996],
        [32.493, 34.223, 36.241, 38.525, 40.606, 41.326, 41.005, 40.575, 40.259, 40.053, 39.923, 39.845]]) / R
    ref_Cp_CO2 = np.array([
        [48.005, 82.938, 172.52, 126.22, 110.64, 102.23, 96.804, 92.940, 90.012, 87.696, 85.807, 84.228],
        [45.528, 59.236, 95.319, 255.60, 185.54, 134.21, 114.71, 104.37, 97.856, 93.325, 89.960, 87.344],
        [44.580, 52.560, 65.648, 88.452, 122.37, 139.32, 128.65, 115.35, 105.71, 98.921, 93.957, 90.204]]) / R
    ref_JT_CO2 = [[1.1035, 1.0884, 0.14586, 0.087512, 0.061627, 0.045863, 0.034932, 0.026768, 0.020367, 0.015176, 0.010858, 0.0071951],
                  [0.89274, 0.87627, 0.80974, 0.56138, 0.24332, 0.14483, 0.10067, 0.074904, 0.057712, 0.045288, 0.035820, 0.028325],
                  [0.73793, 0.72154, 0.68380, 0.60928, 0.48363, 0.33536, 0.22619, 0.16054, 0.11992, 0.092761, 0.073363, 0.058842]]
    ref_vs_CO2 = [[248.53, 220.79, 347.29, 432.31, 487.35, 530.04, 565.68, 596.70, 624.44, 649.71, 673.04, 694.81],
                  [263.76, 246.78, 228.38, 218.24, 291.23, 362.47, 416.21, 459.90, 497.05, 529.56, 558.63, 585.07],
                  [276.98, 265.24, 254.37, 246.80, 249.30, 273.13, 313.83, 356.73, 396.22, 431.80, 464.00, 493.38]]
    ref_Cv_C1 = np.array([
        [27.800, 28.210, 28.605, 28.961, 29.259, 29.489, 29.656, 29.775, 29.863, 29.934, 29.997, 30.058],
        [28.716, 29.023, 29.316, 29.583, 29.818, 30.016, 30.180, 30.313, 30.423, 30.517, 30.599, 30.673],
        [29.807, 30.047, 30.274, 30.483, 30.672, 30.839, 30.985, 31.112, 31.222, 31.321, 31.409, 31.490]]) / R
    ref_Cp_C1 = np.array([
        [38.249, 41.282, 44.753, 48.424, 51.873, 54.634, 56.426, 57.721, 57.406, 57.107, 56.583, 55.965],
        [38.730, 40.992, 43.445, 45.963, 48.364, 50.459, 52.109, 53.262, 53.954, 54.273, 54.321, 54.190],
        [39.510, 41.276, 43.126, 44.988, 46.769, 48.377, 49.742, 50.826, 51.629, 52.175, 52.505, 52.668]]) / R
    ref_JT_C1 = [[0.42234, 0.39961, 0.36866, 0.32979, 0.28554, 0.23989, 0.19667, 0.15842, 0.12610, 0.099500, 0.077808, 0.060106],
                 [0.35408, 0.33455, 0.31041, 0.28201, 0.25059, 0.21802, 0.18617, 0.15650, 0.12987, 0.10660, 0.086592, 0.069549],
                 [0.29872, 0.28197, 0.26520, 0.24060, 0.21698, 0.19259, 0.16486, 0.14543, 0.12408, 0.10474, 0.087488, 0.072275]]
    ref_vs_C1 = [[441.73, 437.57, 437.49, 442.75, 454.35, 472.51, 496.53, 525.00, 556.27, 588.95, 622.02, 654.84],
                  [460.90, 459.03, 460.35, 465.50, 474.94, 488.78, 506.73, 528.19, 552.36, 578.39, 605.57, 633.29],
                  [478.43, 478.22, 480.59, 485.86, 494.27, 505.85, 520.45, 537.73, 557.27, 578.56, 601.14, 624.58]]
    labels = ['25$\degree$C', '50$\degree$C', '75$\degree$C']

    p = [state_spec["pressure"] for i in range(num_curves)]
    cv = [results.isel(temperature=0, CO2=0).Cv.values,
          results.isel(temperature=1, CO2=0).Cv.values,
          results.isel(temperature=2, CO2=0).Cv.values,
          ]
    cp = [results.isel(temperature=0, CO2=0).Cp.values,
          results.isel(temperature=1, CO2=0).Cp.values,
          results.isel(temperature=2, CO2=0).Cp.values,
          ]
    jt = [results.isel(temperature=0, CO2=0).JT.values,
          results.isel(temperature=1, CO2=0).JT.values,
          results.isel(temperature=2, CO2=0).JT.values,
          ]
    vs = [results.isel(temperature=0, CO2=0).vs.values,
          results.isel(temperature=1, CO2=0).vs.values,
          results.isel(temperature=2, CO2=0).vs.values,
          ]

    plot = PlotEoS()
    plot.plot(flash=f, props=results, prop_names=["JT"], x_var="pressure", composition=[1., 0.],
              title="Joule-Thomson coefficient CO2", ax_labels=["pressure", "JT [K/bar]"], datalabels=labels)
    # plot.draw_refdata(number_of_curves=num_curves, xref=ref_p, yref=ref_JT_CO2)
    plt.show()
