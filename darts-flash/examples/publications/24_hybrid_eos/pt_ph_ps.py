import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoS, CubicEoS, AQEoS, InitialGuess

from dartsflash.dartsflash import DARTSFlash
from dartsflash.mixtures import Mixture
from dartsflash.plot import *


""" DEFINE FLASH """
def define_flash(components: list, aqueous: bool, np_max: int, flash_type: DARTSFlash.FlashType = 0) -> DARTSFlash:
    mix = Mixture(components)
    f = DARTSFlash(mixture=mix)

    if not aqueous:
        ceos = CubicEoS(mix.comp_data, CubicEoS.PR)
        f.add_eos("CEOS", ceos, trial_comps=[i for i, comp in enumerate(components)],
                  switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
        eos_order = ["CEOS"]
    else:
        h2o_idx = components.index("H2O")
        ceos = CubicEoS(mix.comp_data, CubicEoS.PR)
        ceos.set_preferred_roots(i=h2o_idx, x=0.75, root_flag=EoS.MAX)
        f.add_eos("CEOS", ceos, trial_comps=[i for i, comp in enumerate(components)],
                  switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)

        f.add_eos("AQ", AQEoS(mix.comp_data, {AQEoS.water: AQEoS.Jager2003,
                                              AQEoS.solute: AQEoS.Ziabakhsh2012,
                                              AQEoS.ion: AQEoS.Jager2003}),
                  trial_comps=[h2o_idx], eos_range={h2o_idx: [0.6, 1.]}, max_iter=10, use_gmix=True)
        eos_order = ["AQ", "CEOS"]

    f.flash_params.stability_variables = FlashParams.alpha
    f.flash_params.split_variables = FlashParams.nik
    f.flash_params.split_tol = 1e-20
    # f.flash_params.split_switch_tol = 1e-1
    f.flash_params.tpd_close_to_boundary = 1e-2
    # f.flash_params.negative_
    f.flash_params.tpd_tol = 1e-11

    f.flash_params.verbose = 0
    f.init_flash(flash_type=flash_type, eos_order=eos_order, np_max=np_max, initial_guess=[InitialGuess.Henry_AV])
    return f


""" TOTAL ENTHALPY FOR NARROW BOILING MIXTURES """
if 1:
    if 0:
        components = ["CO2"]
        z = [1.]
        aqueous = False
        ref_p = np.arange(10., 100., 5e-1)
        ref_T = np.arange(253.15, 343.15, 5e-1)
        np_max = 2
    elif 1:
        components = ["CO2", "C1"]
        z = [0.9, 0.1]
        aqueous = False
        ref_p = np.arange(10., 100., 5e-1)
        ref_T = np.arange(253.15, 343.15, 5e-1)
        np_max = 2
    elif 1:
        components = ["CO2", "H2O"]
        z = [0.99, 0.01]
        aqueous = True
        ref_p = np.arange(10., 100., 1.)
        ref_T = np.arange(253.15, 343.15, 1)
        np_max = 2
    else:
        components = ["H2O", "CO2", "C1"]
        z = [0.025, 0.95, 0.025]
        aqueous = True
        ref_p = np.arange(10., 100., 1.)
        ref_T = np.arange(273.15, 323.15, 0.1)
        np_max = 3

    """ DEFINE PT STATE """
    f = define_flash(components, aqueous, np_max, flash_type=0)
    state_spec = {"temperature": ref_T,
                  "pressure": ref_p,
                  }
    compositions = {components[i]: zi for i, zi in enumerate(z)}

    """ Calculate flash and plot total enthalpy """
    props = ["H"]
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="PT-flash")
    results_pt = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions,
                                          state_variables=['pressure', 'temperature'] + components,
                                          flash_results=flash_results, total_properties_to_evaluate=props,
                                          print_state="PT props")

    plotPT = PlotEoS.surf(flash=f, props=results_pt, x_var='temperature', y_var='pressure', prop_names=["H_total"],
                          composition=z, title="Total enthalpy of " + f.mixture.name,
                          ax_labels=["temperature, K", "pressure, bar"])

    pressures = [20, 40, 60, 80]
    if 1:
        state = state_spec
        state["pressure"] = pressures
        plot = PlotEoS.plot(flash=f, props=results_pt, x_var='temperature', composition=z, state=state_spec,
                            prop_names=["H_total"], title="Total enthalpy of " + f.mixture.name,
                            ax_labels=["temperature, K", r"$H^m$/R"],
                            datalabels=["P = {} bar".format(p) for p in pressures]
                            )
        plot.add_attributes(legend=True, legend_loc='lower right')

    """ DEFINE PH STATE """
    """ Calculate PH flash along isenthalpic depletion starting from certain PT conditions """
    Prange = np.linspace(30., 95., 100)
    temps = [320., 330., 340.]
    state_spec = {"pressure": Prange,
                  "enthalpy": [results_pt.sel(pressure=Prange[-1], temperature=t, method="nearest").H_total.values * R
                               for t in temps]
                  }
    f = define_flash(components, aqueous, np_max, DARTSFlash.FlashType.PHFlash)
    results_ph = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                  print_state="PH-flash")

    if 0:
        plotPT.draw_point(X=[[temp] for temp in temps], Y=[[Prange[-1]] for i in range(3)], colours=plotPT.colours[3:],
                          markers='x', widths=100, datalabels=["T = {} K".format(temp) for temp in temps])
        plotPT.add_attributes(legend=True, legend_loc='lower right')
        if 1:
            plotPT.draw_line(xdata=results_ph.squeeze().transpose('enthalpy', 'pressure', ...).temp.values, ydata=Prange,
                             colours=plotPT.colours[3:], widths=3.5, datalabels=["T = {} K".format(temp) for temp in temps])
    elif 0:
        plotPT.draw_line(xdata=[[260, 320] for pres in pressures], ydata=[[pres, pres] for pres in pressures],
                         colours=plotPT.colours, widths=3.5, datalabels=["P = {} bar".format(pres) for pres in pressures])
        plotPT.add_attributes(legend=True, legend_loc='lower right')

plt.show()
