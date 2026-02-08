import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoSParams, EoS, InitialGuess
from dartsflash.libflash import Flash
from dartsflash.libflash import CubicEoS, AQEoS, Ballard

from dartsflash.dartsflash import DARTSFlash, CompData, R
from dartsflash.mixtures import VLAq
from dartsflash.plot import *


if 0:
    # NEGATIVE FLASH
    components = ["H2O", "CO2"]
    comp_data = CompData(components, setprops=True)
    flash_type = DARTSFlash.FlashType.NegativeFlash

    trial_comps = None
    root_order = [EoS.RootFlag.STABLE]
    nf_initial_K = [InitialGuess.Ki.Henry_AV]

else:
    # PT/PH/PS flash
    if 0:
        components = ["H2O", "CO2"]
        dz = 0.01
        min_z = [0.]
        max_z = [1.]
        P = 500
        Trange = [273.15, 473.15]
    elif 0:
        components = ["H2O", "nC5"]
        dz = 0.0005
        min_z = [0.]
        max_z = [1.]
        P = 10.
        Trange = [373.15, 453.15]
        nt = 200
    elif 1:
        # components = ["H2O", "CO2", "nC5"]
        components = ["H2O", "CO2", "C1"]
        dz = 0.05
        min_z = [0., 0.]
        max_z = [1., 1.]
        P = 10.
        Trange = [273.15, 283.15]
        P = 107.
        Trange = [375.6,]
        nt = 400
        Trange = [273., 313]
        # Trange = [302., 305.]
        P = 70
        ternary = False

    comp_data = CompData(components, setprops=True)
    flash_type = DARTSFlash.FlashType.PTFlash

    trial_comps = [i for i in range(comp_data.nc)]# + [InitialGuess.Wilson]
    root_order = [EoS.RootFlag.MAX, EoS.RootFlag.MIN]
    nf_initial_K = None

f = VLAq(comp_data=comp_data, hybrid=True)

f.set_vl_eos("PR", trial_comps=trial_comps,
             root_order=root_order,
             switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
f.set_aq_eos("Aq", switch_tol=1e-2, stability_tol=1e-16, max_iter=10, use_gmix=True)

eos_order = ["Aq", "VL"]

f.init_flash(flash_type=flash_type, eos_order=eos_order,
             split_tol=1e-20, split_switch_tol=1e-5, tpd_tol=1e-11, tpd_close_to_boundary=1e-3,
             split_variables=FlashParams.lnK, stability_variables=FlashParams.alpha,
             # pxflash_type=FlashParams.BRENT_NEWTON,
             nf_initial_guess=nf_initial_K,
             # verbose=True
             )

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
elif 0:
    # Tx
    state_spec = {"temperature": np.linspace(Trange[0], Trange[-1], nt),
                  "pressure": P,
                  }
elif 0:
    # Hx
    state_spec = {"pressure": 10.,
                  "enthalpy": np.linspace(-2200., 1500., 100) * R}
    f.init_flash(flash_type=DARTSFlash.FlashType.PHFlash, eos_order=eos_order,
                 split_tol=1e-22, split_switch_tol=1e-1, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                 t_min=270., t_max=500.,
                 split_variables=FlashParams.lnK, stability_variables=FlashParams.alpha,
                 nf_initial_guess=nf_initial_K,
                 # verbose=True
                 )
else:
    # Ternary
    ternary = True
    state_spec = {"temperature": 300.,
                  "pressure": P,
                  }

compositions = {comp: np.arange(min_z[i], max_z[i] + 0.1 * dz, dz) for i, comp in enumerate(components[:-1])}
compositions[components[-1]] = 1.
x0 = compositions[components[0]]

if 0:
    """ Plot tpd """
    tpd_results = f.evaluate_stationary_points(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                               print_state="TPD")
    plot_method = {2: PlotTPD.binary_tpd, 3: PlotTPD.ternary_tpd}
    plot_method[len(components)](flash=f, sp_results=tpd_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z)

if 1:
    """ Plot Gibbs energy surfaces and flash results """
    if len(components) == 2:
        composition = [0.5]
        composition += [1. - sum(composition)]

        props = {"G": EoS.Property.GIBBS}
        # results_1p = f.evaluate_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
        #                                       print_state="1P", properties_to_evaluate=props, mix=True)
        flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                         print_state="Flash")
        # results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions, state_variables=['pressure', 'temperature'] + components,
        #                                       print_state="NP", flash_results=flash_results, mix_properties_to_evaluate=props)

        # PlotFlash.pt(f, flash_results, composition=composition)
        plot1p = True
        # results = results_1p if plot1p else results_np

        # PlotProps.binary(f, state=state_spec, variable_comp_idx=0, dz=dz, min_z=min_z, max_z=max_z,
        #                  prop_name="G_mix", plot_1p=plot1p, props=results, flash_results=flash_results,
        #                  composition_to_plot=composition
        #                  )
        if 0:
            plot_hx = PlotFlash.hx(f, flash_results, variable_comp_idx=0, min_z=min_z, max_z=max_z, dz=dz, state=state_spec,
                                   plot_phase_fractions=False, min_temp=270, max_temp=470, cmap="jet")
            enths = np.array([[-15093.79733541, -20214.00569624, -25334.21405706, -30454.42241788, -35574.63077871],
                              [19739.29454626, 16619.20661265, 13368.37574256, 10003.77688854, 6532.39436907]]).transpose()
            comps = [[xi, xi] for xi in [0.1, 0.3, 0.5, 0.7, 0.90]]
            plot_hx.draw_line(X=comps, Y=enths)
            plt.savefig("H-x-" + f.filename + ".pdf")
        else:
            plot_xz = PlotFlash.binary(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z,
                                       plot_phase_fractions=False, cmap="jet")
            temps = [state_spec["temperature"][0], state_spec["temperature"][-1]]
            comps = [[xi, xi] for xi in [0.1, 0.3, 0.5, 0.7, 0.90]]
            plot_xz.draw_line(X=comps, Y=[temps for i in range(5)])

            plot_xz.add_text("Aq - V", xloc=0.55, yloc=0.4, fontsize=12, colours='red', box_colour='k')
            plot_xz.add_text("V - L", xloc=0.02, yloc=0.3, fontsize=12, colours='red', box_colour='k')
            plot_xz.add_text("Aq - L", xloc=0.554, yloc=0.08, fontsize=12, colours='red', box_colour='k')
            plot_xz.add_text("V", xloc=0.18, yloc=0.75, fontsize=12, colours='red', box_colour='k')
            plot_xz.add_text("L", xloc=0.015, yloc=0.05, fontsize=12, colours='red', box_colour='k')

            plt.savefig("T-x-" + f.filename + ".pdf")

        # PlotPhaseDiagram.binary(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z)

    elif len(components) == 3:
        dz = 0.005
        min_z = [0., 0.]
        max_z = [1., 1.]
        gas_composition = [0.9, 0.1]

        if ternary:
            """ PLOT TERNARY DIAGRAM """
            compositions = {comp: np.arange(min_z[i], max_z[i] + 0.1 * dz, dz) for i, comp in
                            enumerate(components[:-1])}
            compositions[components[-1]] = 1.

            x0 = compositions[components[0]]
            x1 = compositions[components[1]]

            flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                             print_state="Flash")

            plot = PlotFlash.ternary(f, flash_results, state=state_spec, dz=dz, min_z=min_z, max_z=max_z,
                                     plot_phase_fractions=False, cmap="jet",
                                     # composition_to_plot=[0.5, 0.4, 0.1],
                                     )
            xA = [0., 1., 0.]
            xV = [gas_composition[0], 0., gas_composition[1]]
            plot.draw_compositions(compositions=[xA, xV], colours=plot.colours[0], connect_compositions=True,
                                   linestyle=(0, (5, 10)), )  # loosely dotted

            plot.add_text("Aq - V", xloc=0.45, yloc=0.3, fontsize=12, colours='red', box_colour='k')
            plot.add_text("Aq - V - L", xloc=0.65, yloc=0.13, fontsize=12, colours='red', box_colour='k')
            plot.add_text("Aq - L", xloc=0.44, yloc=-0.025, fontsize=12, colours='red', box_colour='k')

            plt.savefig(f.filename + "-" + str(int(state_spec["pressure"])) + "-" + str(
                int(state_spec["temperature"])) + "_xCO2.pdf")
        else:
            compositions = {components[1]: np.arange(min_z[0], max_z[0] + 0.1 * dz, dz),
                            components[0]: gas_composition[0],
                            components[2]: gas_composition[1]}

            flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                             print_state="Flash")

            if 0:
                plot_hx = PlotFlash.hx(f, flash_results, variable_comp_idx=0, min_z=min_z, max_z=max_z, dz=dz, state=state_spec,
                                       plot_phase_fractions=False, min_temp=270, max_temp=470, cmap="jet")
                enths = np.array([[-15093.79733541, -20214.00569624, -25334.21405706, -30454.42241788, -35574.63077871],
                                  [19739.29454626, 16619.20661265, 13368.37574256, 10003.77688854, 6532.39436907]]).transpose()
                comps = [[xi, xi] for xi in [0.1, 0.3, 0.5, 0.7, 0.90]]
                plot_hx.draw_line(X=comps, Y=enths)
                plt.savefig("H-x-" + f.filename + ".pdf")

            else:
                plot_xz = PlotFlash.binary(f, flash_results, variable_comp_idx=1, dz=dz, state=state_spec, min_z=min_z, max_z=max_z,
                                           plot_phase_fractions=False, cmap="jet")
                temps = [state_spec["temperature"][0], state_spec["temperature"][-1]]
                comps = [[xi, xi] for xi in [0.1, 0.3, 0.5, 0.7, 0.90]]
                plot_xz.draw_line(X=comps, Y=[temps for i in range(5)])

                plot_xz.add_text("Aq - V", xloc=0.165, yloc=0.6, fontsize=12, colours='red', box_colour='k')
                plot_xz.add_text("Aq - V - L", xloc=0.335, yloc=0.3, fontsize=12, colours='red', box_colour='k')
                plot_xz.add_text("Aq - L", xloc=0.565, yloc=0.18, fontsize=12, colours='red', box_colour='k')

                plt.savefig("T-x-" + f.filename + ".pdf")
    else:
        pass

if 0:
    """ Plot enthalpies along isobar """
    if 1:
        if len(components) == 2:
            compositions = {components[0]: np.array([0.1, 0.3, 0.5, 0.7, 0.9]),
                            components[1]: 1.}
        else:
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
        plot.add_attributes(  # title="Total enthalpy of " + f.mixture_name,
            ax_labels=["temperature, K", "enthalpy, kJ/mol"],
            legend=True, legend_loc='lower right')

        plt.savefig("H-T-" + f.filename + ".pdf")

    if 0:
        """ Plot enthalpy-composition diagrams at constant pressure """
        # Hx
        state_spec = {"pressure": 10.,
                      "enthalpy": np.linspace(-3000., 0., 51) * R}
        # state_spec = {"pressure": np.linspace(1., 30., 30),
        #               "enthalpy": np.linspace(-1000., 1000., 51) * R}
        f.init_flash(flash_type=DARTSFlash.FlashType.PHFlash, eos_order=eos_order,
                     split_tol=1e-22, split_switch_tol=1e-1, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                     t_min=270., t_max=500.,
                     split_variables=FlashParams.lnK, stability_variables=FlashParams.alpha,
                     nf_initial_guess=nf_initial_K,
                     # verbose=True
                     )
        flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                         print_state="Flash")

        plot_xz = PlotFlash.binary(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec, min_z=min_z, max_z=max_z,
                                   plot_phase_fractions=False, cmap="jet")

        # plot_method = PlotFlash.ph if f.flash_type == DARTSFlash.FlashType.PHFlash else PlotFlash.ps
        # plot = plot_method(f, flash_results, composition=[0.5, 0.5],
        #                    # min_temp=T_min, max_temp=T_max,
        #                    min_val=0., max_val=1.,
        #                    plot_phase_fractions=True)

        plt.savefig(f.filename + "-ph.pdf")

plt.show()
