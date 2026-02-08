import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import EoS, InitialGuess, FlashParams, EoSParams
from dartsflash.libflash import CubicEoS, AQEoS, PureSolid, Ballard

from dartsflash.dartsflash import DARTSFlash, CompData, R
from dartsflash.mixtures import VL, VLAq, VLAqH
from dartsflash.plot import *


if 1:
    case = 4
    if case == 0:
        components = ["nC4", "nC10", "CO2"]
        pt = [(69.1, 344.15), (50., 450.)]
        P, T = pt[1]
        rich_phase_order = [2, -1]
    elif case == 1:
        components = ["C1", "CO2", "nC10"]
        pt = [(100., 250.)]
        P, T = pt[0]
        rich_phase_order = [1, -1]
    elif case == 2:
        components = ["C2", "CO2", "nC10"]
        pt = [(30., 300.)]
        P, T = pt[0]
        rich_phase_order = [1, -1]
    elif case == 3:
        components = ["CO2", "C1", "H2S"]
        pt = [(40., 190)]
        P, T = pt[0]
        rich_phase_order = [0, 2, -1]
    else:
        components = ["nC4", "CO2", "C1"]
        pt = [(10., 200.)]
        P, T = pt[0]
        rich_phase_order = [1, -1]

    comp_data = CompData(components, setprops=True)
    f = VL(comp_data=comp_data)
    f.set_vl_eos("PR", root_order=[EoS.MAX, EoS.MIN],
                 rich_phase_order=rich_phase_order,
                 trial_comps=[InitialGuess.Wilson, InitialGuess.Wilson13] + [i for i in range(3)],
                 switch_tol=1e-2
                 )
    eoslabels = ["V", "L"]

    f.flash_params.verbose = 0
    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash,
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                 split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                 # verbose=True
                 )

elif 0:
    """ Aq-V-L """
    components = ["H2O", "CO2", "C1"]
    comp_data = CompData(components, setprops=True)
    pt = [(60., 293.15)]
    temperature = np.linspace(273.15, 313.15, 20)
    P, T = pt[0]

    f = VLAq(comp_data=comp_data)
    f.set_vl_eos("PR", root_order=[EoS.MAX, EoS.MIN],
                 # rich_phase_order=[0, 2, -1],
                 trial_comps=[InitialGuess.Wilson] + [i for i in range(3)],
                 switch_tol=1e-2
                 )
    f.set_aq_eos("Aq", switch_tol=1e-2, stability_tol=1e-16,
                 max_iter=10, use_gmix=True)

    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                 split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                 # verbose=True
                 )

    eoslabels = ["Aq", "V", "L"]

    # State for GE plot
    P, T = 10., 387.78

    # Tx
    state_spec = {"temperature": temperature,
                  "pressure": P
                  }
    prefix = "T-x-"

else:
    """ Aq-V-L-H """
    components = ["H2O", "CO2", "C1"]
    comp_data = CompData(components, setprops=True)

    """ HYBRID-EOS """
    f = VLAqH(comp_data=comp_data, hybrid=True)

    # Add CubicEoS with preferred roots
    f.set_vl_eos("SRK", root_order=[EoS.RootFlag.MAX, EoS.RootFlag.MIN],
                 trial_comps=[0, 1, 2], switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)

    # Add Aq EoS
    f.set_aq_eos("Aq", switch_tol=1e-2, max_iter=10, use_gmix=True)

    # Add VdWP EoS for sI
    f.set_h_eos("sI", stability_tol=1e-16, switch_tol=2e2, max_iter=20, use_gmix=True)

    # Add VdWP EoS for sII
    # f.set_h_eos("sII", stability_tol=1e-20, switch_tol=1e2, max_iter=20, use_gmix=True)

    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL", "sI"],# "sII"],
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                 split_tol=1e-20, split_switch_tol=1e2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                 split_negative_flash_iter=10, split_negative_flash_tol=1e-2,
                 # verbose=True
                 )

    eoslabels = ["Aq", "V", "L", "sI"]#, "sII"]

    # State for GE plot
    P, T = 50., 280.

""" Define state specifications and compositions """
dz = 0.002
min_z = [0., 0.]
max_z = [1., 1.]
compositions = {comp: np.arange(min_z[i], max_z[i]+0.1*dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.

x0 = compositions[components[0]]
x1 = compositions[components[1]]

""" Plot ternary diagram, GE and tpd """
if 1:
    state_spec = {'pressure': P,
                  'temperature': T
                  }
    suffix = "-" + str(int(P)) + "-" + str(int(T))

    props = {"G": EoS.Property.GIBBS}
    results_1p = f.evaluate_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                          print_state="1P", properties_to_evaluate=props, mix=True)
    xref = [0.15, 0.35]
    # xref = [0.4, 0.5]
    ref_comp = xref + [1. - np.sum(xref)]

    compositions = {components[0]: [ref_comp[0]], components[1]: [ref_comp[1]], components[2]: 1.}
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")

    print(flash_results.sel({components[0]: xref[0], components[1]: xref[1]}, method='nearest').squeeze().nu.values)
    print(flash_results.sel({components[0]: xref[0], components[1]: xref[1]}, method='nearest').squeeze().X.values)

    if 0:
        sp_results = f.evaluate_stationary_points(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                                  print_state="TPD")
        suffix += "-tpd"
    else:
        sp_results = None

    plot = PlotProps.ternary(f, state=state_spec, variable_comp_idx=[0, 1], dz=dz, min_z=min_z, max_z=max_z,
                             prop_name="G", plot_1p=True, props=results_1p,
                             # title="Gibbs energy of mixing for " + mix.name,
                             flash_results=flash_results, composition_to_plot=ref_comp,
                             sp_results=sp_results
                             )

    for j, eoslabel in enumerate(eoslabels):
        plot.subplot_idx = j
        plot.add_attributes(title=eoslabel)

    plt.savefig("GE-" + f.filename + suffix + ".pdf")

    # plot = PlotFlash.ternary(f, flash_results, state=state_spec, dz=dz, min_z=min_z, max_z=max_z,
    #                          plot_phase_fractions=False, cmap="jet",
    #                          # composition_to_plot=[0.5, 0.4, 0.1],
    #                          )

    if 0:
        lnphi_properties = {'lnphi_' + eosname: eos.lnphi for eosname, eos in f.eos.items()}
        lnphi_1p = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                                  properties_to_evaluate=lnphi_properties, print_state="lnphi")
        sp_results = f.evaluate_stationary_points(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                                  print_state="TPD")
        PlotTPD.ternary_tpd(f, state=state_spec, ref_composition=ref_comp, variable_comp_idx=[0, 1], dz=dz, min_z=min_z, max_z=max_z,
                            title="TPD for " + mix.name, lnphi_1p=lnphi_1p, flash_results=flash_results, sp_results=sp_results,
                            )

if 0:
    """ Plot Px, Tx and Hx diagrams """
    if 0:
        # Px
        state_spec = {"pressure": np.arange(1, 300, 1),
                      "temperature": 473.15,
                      }
        prefix, suffix = "P-x-", str(int(state_spec["temperature"]))
    elif 1:
        # Tx
        state_spec = {"temperature": np.arange(273.15, 293.16, 1),
                      "pressure": 100.,
                      }
        prefix, suffix = "T-x-", str(int(state_spec["pressure"]))
    else:
        # Hx
        state_spec = {"pressure": P,
                      "enthalpy": np.linspace(-5000., -1000., 101) * R}
        f.init_flash(flash_type=DARTSFlash.FlashType.PHFlash, eos_order=eos_order,
                     split_tol=1e-22, split_switch_tol=1e1, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                     t_min=270., t_max=300.,
                     split_variables=FlashParams.lnK, stability_variables=FlashParams.alpha,
                     split_negative_flash_iter=10,
                     # verbose=True
                     )
        prefix, suffix = "H-x-", str(int(state_spec["pressure"]))

    dz = 0.005
    min_z = [0., 0.]
    max_z = [1., 1.]
    gas_composition = [0.9, 0.1]
    compositions = {components[1]: np.arange(min_z[0], max_z[0] + 0.1 * dz, dz),
                    components[0]: gas_composition[0],
                    components[2]: gas_composition[1]}

    variable_comp_idx = 0
    variable_comp = components[variable_comp_idx]
    variable_comp_name = f.comp_data.comp_labels[variable_comp_idx]

    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")

    if "enthalpy" in state_spec.keys():
        plot_hz = PlotFlash.hx(f, flash_results, variable_comp_idx=variable_comp_idx, state=state_spec, dz=dz,
                               min_z=min_z, max_z=max_z, plot_phase_fractions=False, min_temp=270, max_temp=300,
                               cmap="jet")
    else:
        plot_xz = PlotFlash.binary(f, flash_results, variable_comp_idx=variable_comp_idx, state=state_spec, dz=dz,
                                   min_z=min_z, max_z=max_z, plot_phase_fractions=False, cmap="jet")

    plt.savefig(prefix + f.filename + suffix + ".pdf")

if 0:
    """ Plot enthalpies along isobar """
    gas_composition = [0.9, 0.1]
    compositions = {components[1]: np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                    components[0]: gas_composition[0],
                    components[2]: gas_composition[1]}

    variable_comp_idx = 1
    variable_comp = components[variable_comp_idx]
    variable_comp_name = f.comp_data.comp_labels[variable_comp_idx]

    props = {"H": EoS.Property.ENTHALPY}
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")
    results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions,
                                          state_variables=['pressure', 'temperature'] + components,
                                          print_state="NP", flash_results=flash_results, total_properties_to_evaluate=props)

    pressures = state_spec["pressure"]
    state_spec["pressure"] = pressures

    plot = Diagram(figsize=(8, 5))
    comps = {variable_comp: compositions[variable_comp]}
    props_at_state = results_np.sel(comps, method='nearest').squeeze().transpose(..., 'temperature')
    x = props_at_state.coords['temperature'].values

    prop_array = props_at_state.H_total.values * R

    plot.draw_line(X=x, Y=prop_array,
                   datalabels=["z" + variable_comp_name + " = {:.2f}".format(xi) for xi in compositions[variable_comp]],
                   )
    plot.add_attributes(  # title="Total enthalpy of " + f.mixture_name,
        ax_labels=["temperature, K", "enthalpy, kJ/mol"],
        legend=True, legend_loc='lower right')

    plt.savefig("H-T-" + f.filename + ".pdf")

plt.show()
