import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import FlashParams, EoS

from dartsflash.mixtures import VLAq, IAPWS
from dartsflash.dartsflash import DARTSFlash, CompData, R


components = ["H2O"]
compositions = {"H2O": 1.}
comp_data = CompData(components, setprops=True)

from dartsflash.libflash import IdealGas, CubicEoS, AQEoS, IAPWS95, IAPWSIce, PureSolid

if 1:
    f = DARTSFlash(comp_data=comp_data,)

    if 1:
        # f.add_eos("V", IdealGas(comp_data))
        f.add_eos("V", CubicEoS(comp_data, CubicEoS.PR), preferred_roots=[(0, 0.75, EoS.MAX)])
        f.add_eos("Aq", AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                                          AQEoS.solute: AQEoS.Ziabakhsh2012}))
        eos_order = ["V", "Aq"]
    else:
        f.add_eos("VL", CubicEoS(comp_data, CubicEoS.PR), root_order=[EoS.MAX, EoS.MIN])
        eos_order = ["VL"]

    f.add_eos("I", PureSolid(comp_data, "Ice"))
    eos_order += ["I"]
elif 0:
    f = VLAq(comp_data, hybrid=True)
    f.set_vl_eos("PR", )
    f.set_aq_eos("Aq", )

    eos_order = ["Aq", "VL"]
else:
    f = IAPWS(iapws_ideal=False, ice_phase=True)
    eos_order = None

f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=eos_order)

if 1:
    """ PLOT PT-DIAGRAM """
    if eos_order is None:
        f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash)
    else:
        f.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=eos_order,
                     # verbose=True
                     )

    state_spec = {"pressure": np.logspace(-4, 2.5, 100),
                  "temperature": np.linspace(250, 650, 100)}
    results_pt = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True, print_state="Flash")
    props = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions,
                                     state_variables=f.state_vars + components, flash_results=results_pt,
                                     total_properties_to_evaluate={"H": EoS.Property.ENTHALPY}, print_state="Properties")

    from dartsflash.plot import PlotFlash
    pt_plot = PlotFlash.pt(f, results_pt, composition=[1.], plot_phase_fractions=False, logP=True,
                           pt_props=props)
    plt.savefig(f.filename + "-PT.pdf")

if 0:
    """ PLOT PH-DIAGRAM """
    if eos_order is None:
        f.init_flash(flash_type=DARTSFlash.FlashType.PHFlash)
    else:
        f.init_flash(flash_type=DARTSFlash.FlashType.PHFlash, eos_order=eos_order,
                     # verbose=True
                     )

    Xrange = f.get_ranges(prange=[1e-7, 1e-3], trange=[200, 270], composition=[1.])
    state_spec = {"pressure": np.logspace(-7, -3, 100),
                  "enthalpy": np.linspace(Xrange[0], Xrange[1], 100)}
    results_ph = f.evaluate_flash_1c(state_spec=state_spec)

    from dartsflash.plot import PlotFlash
    ph_plot = PlotFlash.ph(f, results_ph, composition=[1.], plot_phase_fractions=False, logP=True)
    plt.savefig(f.filename + "-PH.pdf")

if 0:
    """ Plot phase boundaries in PT """
    A = f.eos["Aq"]
    V = f.eos["V"]
    I = f.eos["I"]

    # Calculate triple point (V-Aq-I)
    triplePT = np.array([6e-3, 273.15])
    tol = 1e-12
    max_iter = 500
    it = 0
    while True:
        # A.solve_PT(triplePT[0], triplePT[1], [1.], 0, True)
        gA = A.G(triplePT[0], triplePT[1], [1.], 0, pt=True) / triplePT[1]
        gA = A.lnphi(triplePT[0], triplePT[1], [1.])[0]
        dgA_dT = A.dlnphi_dT()
        dgA_dP = A.dlnphi_dP()

        # V.solve_PT(triplePT[0], triplePT[1], [1.], 0, True)
        gV = V.G(triplePT[0], triplePT[1], [1.], 0, pt=True) / triplePT[1]
        gV = V.lnphi(triplePT[0], triplePT[1], [1.])[0]
        dgV_dT = V.dlnphi_dT()
        dgV_dP = V.dlnphi_dP()

        # I.solve_PT(triplePT[0], triplePT[1], [1.], 0, True)
        gS = I.G(triplePT[0], triplePT[1], [1.], 0, pt=True) / triplePT[1]
        gS = I.lnphi(triplePT[0], triplePT[1], [1.])[0]
        dgS_dT = I.dlnphi_dT()
        dgS_dP = I.dlnphi_dP()

        res = np.array([gA - gV, gV - gS])
        Jac = np.array([[dgA_dP[0] - dgV_dP[0], dgA_dT[0] - dgV_dT[0]],
                        [dgV_dP[0] - dgS_dP[0], dgV_dT[0] - dgS_dT[0]]])
        triplePT -= np.linalg.solve(Jac, res)

        it += 1
        if np.linalg.norm(res) < tol:
            # print(gA, gV, gS)
            break
        elif it > max_iter:
            print("max iter")
            break
    print("triple point", triplePT)

    pres_SV = np.logspace(-4, np.log10(triplePT[0]), 100)
    pres_SV = np.array(list(reversed(pres_SV)))
    T_sub = np.empty(np.shape(pres_SV))
    T_sub0 = triplePT[1]

    for i, p in enumerate(pres_SV):
        # Calculate sublimation curve
        it = 0

        T_sub[i] = T_sub0
        while True:
            # V.solve_PT(p, T_sub[i], [1.])
            gV = V.G(p, T_sub[i], [1.], 0, pt=True) / T_sub[i]
            gV = V.lnphi(p, T_sub[i], [1.])[0]
            dgV_dT = V.dlnphi_dT()

            # I.solve_PT(p, T_sub[i], [1.])
            gS = I.G(p, T_sub[i], [1.], 0, pt=True) / T_sub[i]
            gS = I.lnphi(p, T_sub[i], [1.])[0]
            dgS_dT = I.dlnphi_dT()

            res = gV - gS
            dres_dT = dgV_dT[0] - dgS_dT[0]

            T_sub[i] -= res / dres_dT

            if np.isnan(T_sub[i]):
                print("S-V NANs", it)
                break
            if np.abs(res) < tol:
                T_sub0 = T_sub[i]
                break
            elif it > max_iter:
                print("S-V it >", max_iter, T_sub0, T_sub[i])
                T_sub0 = T_sub[i]
                break
            it += 1

    pres = np.logspace(np.log10(triplePT[0]), np.log10(500.), 100)
    arr_shape = np.shape(pres)
    T_sol, T_vap, T_vapPR = np.empty(arr_shape), np.empty(arr_shape), np.empty(arr_shape)
    T_sol0, T_vap0, T_vapPR0 = 273.15, 273.15, 273.15

    for i, p in enumerate(pres):
        # Calculate evaporation curve Aq-V
        it = 0

        if p <= 170.:
            T_vap[i] = T_vap0
            while True:
                gA = A.G(p, T_vap[i], [1.], 0, pt=True) / T_vap[i]
                gA = A.lnphi(p, T_vap[i], [1.])[0]
                dgA_dT = A.dlnphi_dT()

                gV = V.G(p, T_vap[i], [1.], 0, pt=True) / T_vap[i]
                gV = V.lnphi(p, T_vap[i], [1.])[0]
                dgV_dT = V.dlnphi_dT()

                res = gA - gV
                dres_dT = dgA_dT[0] - dgV_dT[0]

                T_vap[i] -= res/dres_dT

                if np.abs(res) < tol:
                    T_vap0 = T_vap[i]
                    break
                elif it > max_iter:
                    print("A-V it >", max_iter)
                    T_vap0 = T_vap[i]
                    break
                it += 1
        else:
            T_vap[i] = np.nan

        # Calculate evaporation curve L-V full PR
        it = 0

        T_vapPR[i] = np.nan
        # T_vapPR[i] = T_vapPR0
        # while True:
        #     gA = A.G(p, T_vapPR[i], [1.], 0, pt=True) / T_vapPR[i]
        #     gA = A.lnphi(p, T_vapPR[i], [1.])[0]
        #     dgA_dT = A.dlnphi_dT()
        #
        #     gV = V.G(p, T_vapPR[i], [1.], 0, pt=True) / T_vapPR[i]
        #     gV = V.lnphi(p, T_vapPR[i], [1.])[0]
        #     dgV_dT = V.dlnphi_dT()
        #
        #     res = gA - gV
        #     dres_dT = dgA_dT[0] - dgV_dT[0]
        #
        #     T_vapPR[i] -= res / dres_dT
        #
        #     if np.abs(res) < tol:
        #         T_vapPR0 = T_vapPR[i]
        #         break
        #     elif it > max_iter:
        #         print("PR-VL it >", max_iter)
        #         T_vapPR0 = T_vapPR[i]
        #         break
        #     it += 1

        # Calculate solid curve
        it = 0

        T_sol[i] = T_sol0
        while True:
            gA = A.G(p, T_sol[i], [1.], 0, pt=True) / T_sol[i]
            gA = A.lnphi(p, T_sol[i], [1.])[0]
            dgA_dT = A.dlnphi_dT()

            gS = I.G(p, T_sol[i], [1.], 0, pt=True) / T_sol[i]
            gS = I.lnphi(p, T_sol[i], [1.])[0]
            dgS_dT = I.dlnphi_dT()

            res = gA - gS
            dres_dT = dgA_dT[0] - dgS_dT[0]

            T_sol[i] -= res / dres_dT

            if np.abs(res) < tol:
                T_sol0 = T_sol[i]
                break
            elif it > max_iter:
                print("V-S it >", max_iter)
                T_sol0 = T_sol[i]
                break
            it += 1

    from dartsflash.diagram import Diagram
    plot = Diagram(figsize=(8, 5))
    plot.add_attributes(#suptitle=r"Phase diagram of H$_2$O",
                        ax_labels=["Temperature, K", "Pressure, bar"],
                        # grid=True
                        )

    plot.draw_line(X=[T_sub, T_sol, T_vap], Y=[pres_SV, pres, pres])
    plot.draw_line(X=T_vapPR, Y=pres, styles='dotted', colours=plot.colours[2])
    plot.draw_point(X=[triplePT[1]], Y=[triplePT[0]], colours=plot.colours[3], markers='^', widths=80)
    # plot.draw_point(X=[comp_data.Tc[0]], Y=[comp_data.Pc[0]], markers='o')

    trange = [225, 650]
    prange = [1e-4, 500]
    # pressures = [1e-3, 1e-1, 1e1]
    base, expon = [1, 1, 1], [-3, -1, 1]
    pressures = [base[i] * 10 ** expon[i] for i in range(3)]
    for i, p in enumerate(pressures):
        plot.draw_line(X=trange, Y=[p, p], styles='dashed', colours=plot.colours[i+3])
    plot.set_axes(logy=True, xlim=trange, ylim=prange)

    plt.savefig("H2O-PT.pdf")

    if 1:
        """ Plot enthalpies along isobars """
        state_spec = {"pressure": pressures,
                      "temperature": np.linspace(trange[0], trange[1], 500)}
        flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                         print_state="Flash")

        props = {"H": EoS.Property.ENTHALPY}
        results_np = f.evaluate_properties_np(state_spec=state_spec, compositions=compositions,
                                              state_variables=['pressure', 'temperature'] + components,
                                              print_state="NP", flash_results=flash_results,
                                              total_properties_to_evaluate=props)

        pressures = state_spec["pressure"]
        state_spec["pressure"] = pressures

        plot2 = Diagram(figsize=(8, 5))
        # comps = compositions
        # props_at_state = results_np.sel(comps, method='nearest').squeeze().transpose(..., 'temperature')
        x = results_np.coords['temperature'].values

        prop_array = results_np.H_total.values * R

        plot2.draw_line(X=x, Y=prop_array,
                        datalabels=[r"P = ${0:.1f}\cdot10^{{{1:d}}}$".format(base[i], expon[i]) for i, p in enumerate(pressures)],
                        colours=plot.colours[3:]
                        )
        plot2.add_attributes(  # title="Total enthalpy of " + f.mixture_name,
            ax_labels=["temperature, K", "enthalpy, kJ/mol"],
            legend=True, legend_loc='lower right')

        plt.savefig("H-T-" + f.filename + ".pdf")

plt.show()
