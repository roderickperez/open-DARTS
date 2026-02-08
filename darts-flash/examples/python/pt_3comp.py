import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import EoS, InitialGuess, FlashParams, EoSParams
from dartsflash.libflash import CubicEoS, AQEoS, PureSolid, Ballard

from dartsflash.dartsflash import DARTSFlash
from dartsflash.mixtures import Mixture
from dartsflash.plot import *


if 0:
    if 0:
        components = ["nC4", "nC10", "CO2"]
        pt = [(69.1, 344.15), (50., 450.)]
        np_max = 3
    elif 0:
        components = ["C1", "CO2", "nC10"]
        pt = [(100., 250.)]
        np_max = 3
    elif 0:
        components = ["C2", "CO2", "nC10"]
        pt = [(30., 300.)]
        np_max = 3
    elif 1:
        components = ["CO2", "C1", "H2S"]
        pt = [(60., 273.15)]
        np_max = 3
    else:
        components = ["nC4", "CO2", "C1"]
        pt = [(10., 200.)]
        np_max = 4

    mix = Mixture(components)
    f = DARTSFlash(mixture=mix)
    f.add_eos("CEOS", CubicEoS(mix.comp_data, CubicEoS.SRK),
              trial_comps=[InitialGuess.Wilson, components.index("CO2")],
              switch_tol=1e-2
              )
    eos_order = ["CEOS"]

    f.flash_params.stability_variables = FlashParams.alpha
    f.flash_params.split_variables = FlashParams.nik
    f.flash_params.split_tol = 1e-20
    f.flash_params.split_switch_tol = 1e-2
    f.flash_params.tpd_close_to_boundary = 1e-2
    # f.flash_params.negative_
    f.flash_params.tpd_tol = 1e-11

else:
    components = ["H2O", "CO2", "C1"]
    mix = Mixture(components)
    pt = [(600., 473.15)]

    f = DARTSFlash(mixture=mix)
    f.add_eos("CEOS", CubicEoS(mix.comp_data, CubicEoS.PR),
              trial_comps=[0, 1, 2], preferred_roots=[(0, 0.75, EoS.MAX)],
              switch_tol=1e-2, stability_tol=1e-20, max_iter=50, use_gmix=0)
    f.add_eos("AQ", AQEoS(mix.comp_data, {AQEoS.water: AQEoS.Jager2003,
                                          AQEoS.solute: AQEoS.Ziabakhsh2012}),
              switch_tol=1e-2, stability_tol=1e-16,
              trial_comps=[0], eos_range={0: [0.6, 1.]}, max_iter=10, use_gmix=True)

    if 0:
        f.add_eos("sI", Ballard(mix.comp_data, "sI"),
                  trial_comps=[0], max_iter=20, switch_tol=1e2)
        eos_order = ["AQ", "sI", "CEOS"]
        np_max = 4
    else:
        eos_order = ["AQ", "CEOS"]
        np_max = 3

    f.flash_params.stability_variables = FlashParams.alpha
    f.flash_params.split_variables = FlashParams.nik
    f.flash_params.split_tol = 1e-20
    f.flash_params.split_switch_tol = 1e-2
    f.flash_params.tpd_close_to_boundary = 1e-2
    # f.flash_params.negative_
    f.flash_params.tpd_tol = 1e-11
    f.flash_params.verbose = 0

f.flash_params.verbose = 0
f.init_flash(eos_order=eos_order, flash_type=DARTSFlash.FlashType.PTFlash, np_max=np_max)

""" Define state specifications and compositions """
state_spec = {"pressure": np.array([60.]),
              "temperature": np.linspace(273.15, 373.15, 3),
              }
dz = 0.05
min_z = [0., 0.]
max_z = [1., 1.]
compositions = {comp: np.arange(min_z[i], max_z[i]+0.1*dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.

x0 = compositions[components[0]]
x1 = compositions[components[1]]

if 0:
    """ Plot tpd """
    tpd_results = f.evaluate_stationary_points(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                               print_state="TPD")
    PlotTPD.ternary(f, tpd_results, dz=dz, min_z=min_z, max_z=max_z, state={"pressure": 60.1, "temperature": 273.15},
                    composition_to_plot=[0.8, 0.1, 0.1])

if 1:
    """ Plot Gibbs energy surfaces and flash results """
    composition = [0.1, 0.7]
    composition += [1. - sum(composition)]

    mixing_props = {"H": EoS.Property.ENTHALPY}
    results_1p = f.evaluate_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                          print_state="1P", properties_to_evaluate={}, mix_properties_to_evaluate=mixing_props)
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")
    results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions, state_variables=["pressure", "temperature"] + components,
                                          print_state="NP", flash_results=flash_results, mix_properties_to_evaluate=mixing_props)

    plot1p = True
    results = results_1p if plot1p else results_np
    PlotProps.ternary(f, state={"pressure": 60., "temperature": 273.15}, variable_comp_idx=[0, 1], dz=dz, min_z=min_z, max_z=max_z,
                      prop_name="H", plot_1p=plot1p, props=results, flash_results=flash_results, composition_to_plot=composition)
    PlotFlash.ternary(f, flash_results, dz=dz, min_z=min_z, max_z=max_z, state={"pressure": 60., "temperature": 273.15},
                      composition_to_plot=composition)
    PlotPhaseDiagram.ternary(f, flash_results, dz=dz*10, min_z=min_z, max_z=max_z,
                             state={"pressure": 60., "temperature": 273.15}, plot_tielines=0)

plt.show()
