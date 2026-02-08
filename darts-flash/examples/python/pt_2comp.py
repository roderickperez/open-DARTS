import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess
from dartsflash.libflash import Flash
from dartsflash.libflash import CubicEoS, AQEoS, Ballard

from dartsflash.dartsflash import DARTSFlash, CompData
from dartsflash.plot import *


if 1:
    components = ["H2O", "CO2"]

    comp_data = CompData(components)
    f = DARTSFlash(comp_data=comp_data)

    ceos = CubicEoS(comp_data, CubicEoS.PR)
    ceos.set_preferred_roots(i=0, x=0.75, root_flag=EoS.MAX)
    f.add_eos("CEOS", ceos, trial_comps=[0, 1],
              switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
    f.add_eos("AQ", AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                                          AQEoS.solute: AQEoS.Ziabakhsh2012}),
              switch_tol=1e-2, stability_tol=1e-16,
              trial_comps=[0], eos_range={0: [0.6, 1.]}, max_iter=10, use_gmix=True)

    if 0:
        f.add_eos("sI", Ballard(comp_data, "sI"),
                  trial_comps=[0], max_iter=20, switch_tol=1e2)
        eos_order = ["AQ", "sI", "CEOS"]
        np_max = 3
    else:
        eos_order = ["AQ", "CEOS"]
        np_max = 2

    f.flash_params.stability_variables = FlashParams.alpha
    f.flash_params.split_variables = FlashParams.nik
    f.flash_params.split_tol = 1e-20
    f.flash_params.split_switch_tol = 1e-1
    f.flash_params.tpd_close_to_boundary = 1e-2
    # f.flash_params.negative_
    f.flash_params.tpd_tol = 1e-11

elif 0:
    # components = ["H2S", "C1"]
    components = ["C1", "nC4"]
    mix = Mixture(components=components)
    f = DARTSFlash(mixture=mix)

    f.add_eos("CEOSl", CubicEoS(mix.comp_data, CubicEoS.SRK),
              trial_comps=[InitialGuess.Yi.Wilson],
              root_flag=EoSParams.MIN,
              )
    f.add_eos("CEOSv", CubicEoS(mix.comp_data, CubicEoS.SRK),
              trial_comps=[InitialGuess.Yi.Wilson],
              root_flag=EoSParams.MAX,
              )
    f.init_flash(stabilityflash=True)

    ceosl = f.eos["CEOSl"]
    ceosv = f.eos["CEOSv"]

f.flash_params.verbose = 0
f.init_flash(flash_type=DARTSFlash.FlashType.NegativeFlash, eos_order=eos_order, nf_initial_guess=[InitialGuess.Henry_AV])

""" Define state specifications and compositions """
if 0:
    # PT
    state_spec = {"pressure": np.arange(10, 300, 5),
                  "temperature": np.arange(273.15, 673.15, 5),
                  }
elif 0:
    # Px
    state_spec = {"pressure": np.arange(1, 300, 1),
                  "temperature": 473.15,
                  }
else:
    # Tx
    state_spec = {"temperature": np.arange(273.15, 573.15, 1),
                  "pressure": 100.,
                  }

dz = 0.01
min_z = [0.]
max_z = [1.]
compositions = {comp: np.arange(min_z[i], max_z[i]+0.1*dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.

x0 = compositions[components[0]]

if 1:
    """ Plot tpd """
    tpd_results = f.evaluate_stationary_points(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                               print_state="TPD")
    PlotTPD.binary(f, tpd_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z)

if 1:
    """ Plot Gibbs energy surfaces and flash results """
    composition = [0.5]
    composition += [1. - sum(composition)]

    props = {"G": EoS.Property.GIBBS}
    results_1p = f.evaluate_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                          print_state="1P", properties_to_evaluate={}, mix_properties_to_evaluate=props)
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")
    results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions, state_variables=['pressure', 'temperature'] + components,
                                          print_state="NP", flash_results=flash_results, mix_properties_to_evaluate=props)

    # PlotFlash.pt(f, flash_results, composition=composition)
    plot1p = True
    results = results_1p if plot1p else results_np
    PlotProps.binary(f, state={"pressure": 60., "temperature": 273.15}, variable_comp_idx=0, dz=dz, min_z=min_z, max_z=max_z,
                     prop_name="G_mix", plot_1p=plot1p, props=results, flash_results=flash_results,
                     composition_to_plot=composition
                     )
    PlotFlash.binary(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z, plot_phase_fractions=True)
    PlotPhaseDiagram.binary(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z)

plt.show()
