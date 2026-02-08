import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess

from dartsflash.dartsflash import DARTSFlash, CompData
from dartsflash.mixtures import VL, VLAq, VLAqH
from dartsflash.plot import *


if 0:
    components = ["H2S", "C1"]
    # components = ["C1", "nC4"]
    f = VL(comp_data=CompData(components, setprops=True))

    f.set_vl_eos(vl_eos="SRK", root_order=[EoS.MAX, EoS.MIN],
                 rich_phase_order=[i for i in range(f.nc)],
                 trial_comps=[InitialGuess.Wilson] + [i for i in range(f.nc)],
                 stability_tol=1e-20, switch_tol=1e-2, max_iter=50)

    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash,
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                 split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2)
    eoslabels = ["CEOS-V", "CEOS-L"]
    eoslabels = ["V", "L"]

    # State for GE plot
    P, T = 45. * 1.01325, 190.
    # P, T = 40., 300.

    # Px
    state_spec = {"pressure": np.linspace(1, 80, 200),
                  "temperature": T
                  }
    prefix = "P-x-"

elif 0:
    """ Aq-V-L """
    components = ["H2O", "CO2"]
    # temperature = np.arange(273.15, 573.15, 1),
    components = ["H2O", "nC5"]
    temperature = np.arange(373.15, 473.15, 1),
    comp_data = CompData(components, setprops=True)

    """ HYBRID-EOS """
    f = VLAq(comp_data=comp_data, hybrid=True)

    # Add CubicEoS with preferred roots
    f.set_vl_eos("PR", root_order=[EoS.RootFlag.MAX, EoS.RootFlag.MIN],
                 trial_comps=[0, 1], switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)

    # Add Aq EoS
    f.set_aq_eos("Aq", switch_tol=1e-2, max_iter=10, use_gmix=True)

    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                 split_tol=1e-20, split_switch_tol=1e-2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2)

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
    components = ["H2O", "CO2"]
    comp_data = CompData(components, setprops=True)

    """ HYBRID-EOS """
    f = VLAqH(comp_data=comp_data, hybrid=True)

    # Add CubicEoS with preferred roots
    f.set_vl_eos("SRK", root_order=[EoS.RootFlag.MAX, EoS.RootFlag.MIN],
                 trial_comps=[0, 1], switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)

    # Add Aq EoS
    f.set_aq_eos("Aq", max_iter=10, use_gmix=True)

    # Add VdWP EoS for sI
    f.set_h_eos("sI", stability_tol=1e-20, switch_tol=1e2, max_iter=20, use_gmix=True)

    # Add VdWP EoS for sII
    # f.set_h_eos("sII", stability_tol=1e-20, switch_tol=1e2, max_iter=20, use_gmix=True)

    eos_order = ["Aq", "VL", "sI"]  #, "sII"]
    f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=eos_order,
                 stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                 split_tol=1e-20, split_switch_tol=1e2, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                 # split_negative_flash_iter=10, split_negative_flash_tol=1e-4,
                 # split_switch_diff=0.,
                 # split_max_iter=2,
                 # verbose=True
                 )

    eoslabels = ["Aq", "V", "L", "sI"]#, "sII"]

    # State for GE plot
    P, T = 79.8 * 1.01325, 280.
    # P, T = 100., 273.

    # Tx
    state_spec = {"temperature": np.linspace(273, 293, 1000),
                  "pressure": P,
                  }
    prefix = "T-x-"

""" Define compositions """
dz = 0.002
min_z = [0.]
max_z = [1.]
compositions = {}
compositions[components[0]] = np.arange(min_z[0], max_z[0]+0.1*dz, dz)
compositions[components[1]] = 1.

x0 = compositions[components[0]]

if 1:
    """ Plot Gibbs energy surfaces, TPD and flash results """
    state = {'pressure': P, 'temperature': T}

    props = {"G": EoS.GIBBS}
    ge1p_results = f.evaluate_properties_1p(state_spec=state, compositions=compositions, mole_fractions=True,
                                            print_state="GE 1P", properties_to_evaluate=props, mix=True)
    # flash_results = f.evaluate_flash(state_spec=state, compositions=compositions, mole_fractions=True,
    #                                  print_state="Flash")
    # genp_results = f.evaluate_properties_np(state_spec=state, compositions=compositions,
    #                                         state_variables=['pressure', 'temperature'] + components,
    #                                         flash_results=flash_results, properties_to_evaluate=props, mix=True,
    #                                         print_state="GE NP")

    xref = 0.35
    ref_comp = [xref, 1. - xref]
    ge_plot = PlotProps.binary(f, state=state, prop_name="G", variable_comp_idx=0, dz=dz, min_z=min_z, max_z=max_z,
                               # flash_results=flash_results, composition_to_plot=ref_comp,
                               props=ge1p_results, plot_1p=True,
                               datalabels=eoslabels, ax_label=r"G$^m$/R")
    # ge_plot.set_axes(ylim=[-10, 100.],
    #                  xlim=[0.9, 1.])

    if 0:
        yref = ge1p_results.sel({components[0]: xref}, method='nearest').squeeze().G.values[1]
        grad = (ge1p_results.sel({components[0]: xref + 0.001}, method='nearest').squeeze().G.values[1] - yref)/0.001
        y_tangent = [yref - xref * grad, yref + (1.-xref) * grad]

        ge_plot.draw_point(X=xref, Y=yref, colours='k', markers="*", widths=40)
        ge_plot.draw_line(X=[0., 1.], Y=y_tangent, colours='k', styles='dashed')

    plt.savefig("GE-" + f.filename + "-" + str(int(P)) + "-" + str(int(T)) + ".pdf")

    if 0:
        lnphi_properties = {'lnphi_' + eosname: eos.lnphi for eosname, eos in f.eos.items()}
        lnphi_1p = f.evaluate_phase_properties_1p(state_spec=state, compositions=compositions, mole_fractions=True,
                                                  properties_to_evaluate=lnphi_properties, print_state="lnphi")

        sp_results = f.evaluate_stationary_points(state_spec=state, compositions=compositions, mole_fractions=True,
                                                  print_state="SP")
        tpd_plot = PlotTPD.binary_tpd(f, state=state, ref_composition=ref_comp, variable_comp_idx=0, dz=dz, min_z=min_z, max_z=max_z,
                                      lnphi_1p=lnphi_1p, ax_label="TPD",
                                      # flash_results=flash_results,
                                      sp_results=sp_results,
                                      )
        tpd_plot.set_axes(ylim=[-0.4, None])
        plt.savefig("TPD-" + f.filename + "-" + str(int(P)) + "-" + str(int(T)) + ".pdf")

    # PlotProps.binary(f, state=state_spec, variable_comp_idx=0, dz=dz, min_z=min_z, max_z=max_z,
    #                  prop_name="G_mix", title="Gibbs energy of mixing for " + mix.name, plot_1p=True,
    #                  props=results_1p, flash_results=flash_results, sp_results=sp_results, composition_to_plot=ref_comp
    #                  )

if 0:
    """ Plot Px, Tx and Hx diagrams """
    if 0:
        # Px
        state_spec = {"pressure": np.arange(1, 300, 1),
                      "temperature": 473.15,
                      }
        prefix, suffix = "P-x-", "-" + str(int(state_spec["temperature"]))
    elif 1:
        # Tx
        state_spec = {"temperature": np.arange(273.15, 293.16, 1),
                      "pressure": P,
                      }
        prefix, suffix = "T-x-", "-" + str(int(state_spec["pressure"]))
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
        prefix, suffix = "H-x-", "-" + str(int(state_spec["pressure"]))

    variable_comp_idx = 0
    variable_comp = components[variable_comp_idx]
    variable_comp_name = f.comp_data.comp_labels[variable_comp_idx]
    xx = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")

    if "enthalpy" in state_spec.keys():
        plot_hz = PlotFlash.hx(f, flash_results, variable_comp_idx=variable_comp_idx, state=state_spec, dz=dz,
                               min_z=min_z, max_z=max_z, plot_phase_fractions=False, min_temp=270, max_temp=300,
                               cmap="jet")
    else:
        plot_xz = PlotFlash.binary(f, flash_results, variable_comp_idx=variable_comp_idx, state=state_spec, dz=dz,
                                   min_z=min_z, max_z=max_z, plot_phase_fractions=False, cmap="jet")
        plot_xz.draw_line(X=[[xi, xi] for xi in xx], Y=[[state_spec["temperature"][0], state_spec["temperature"][-1]] for xi in xx],
                          # colours=plot_xz.colours
                          )

    if 0:
        plot.add_text("Aq - V", xloc=0.55, yloc=0.4, fontsize=12, colours='red', box_colour='k')
        plot.add_text("V - L", xloc=0.02, yloc=0.3, fontsize=12, colours='red', box_colour='k')
        plot.add_text("Aq - L", xloc=0.554, yloc=0.08, fontsize=12, colours='red', box_colour='k')
        plot.add_text("V", xloc=0.18, yloc=0.75, fontsize=12, colours='red', box_colour='k')
        plot.add_text("L", xloc=0.015, yloc=0.05, fontsize=12, colours='red', box_colour='k')

    plt.savefig(prefix + f.filename + suffix + ".pdf")

if 0:
    """ Plot enthalpies along isobar """
    compositions = {components[0]: np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                    components[1]: 1.}

    variable_comp_idx = 0
    variable_comp = components[variable_comp_idx]
    variable_comp_name = f.comp_data.comp_labels[variable_comp_idx]

    props = {"H": EoS.Property.ENTHALPY}
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")
    results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions, state_variables=['pressure', 'temperature'] + components,
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
    plot.add_attributes(#title="Total enthalpy of " + f.mixture_name,
                        ax_labels=["temperature, K", "enthalpy, kJ/mol"],
                        legend=True, legend_loc='lower right')

    plt.savefig("H-T-" + f.filename + ".pdf")

plt.show()
