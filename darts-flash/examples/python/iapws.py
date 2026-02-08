import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.dartsflash import DARTSFlash, EoS, FlashParams
from dartsflash.plot import *


from dartsflash.mixtures import IAPWS
ice_phase = True
f = IAPWS(iapws_ideal=True, ice_phase=ice_phase)

components = f.components
z = [1.]
n = z
compositions = {comp: n[i] for i, comp in enumerate(components)}

iapws = f.eos["IAPWS"]
ice = f.eos["Ice"] if ice_phase else None

if 1:
    """ Compressibility factor and P-V diagram of Cubic EoS """
    temperature = np.array([600., 625., 650., 675.])

    """ Evaluate properties at PT """
    if 0:
        pressure = np.arange(1., 251.1, 25)

        state_spec = {"pressure": pressure,
                      "temperature": temperature
                      }
        properties = {"V": iapws.V,
                      "Z": iapws.Z,
                      }

        pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                                properties_to_evaluate=properties, mole_fractions=True)
    else:
        pprops = None

    """ Evaluate properties at d-T (reduced volume) """
    if 1:
        state_spec = {"temperature": temperature,
                      "volume": np.linspace(1e-8, 3., 200)
                      }
        properties = {"Z": iapws.Zd,
                      "P": iapws.Pd
                      }

        vprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                                properties_to_evaluate=properties, mole_fractions=True)
    else:
        vprops = None

    """ Plot PV and PZ diagrams """
    pv = PlotEoS.pressure_volume(f, temperatures=temperature, compositions=n,
                                 p_props=pprops, v_props=vprops,
                                 v_range=[0, 2.5],
                                 p_range=[1e1, 2e4],
                                 logy=True,
                                 )
    plt.savefig("P-d-H2O.pdf")

    pz = PlotEoS.compressibility(f, temperatures=temperature, compositions=n,
                                 p_props=pprops, v_props=vprops, z_range=[-0.1, 1.1], p_range=[0., 500.])
    plt.savefig("P-Z-" + f.filename + ".pdf")

if 0:
    """ PT-diagram of properties """
    """ Evaluate properties at PT """
    state_spec = {"pressure": np.linspace(1e-5, 1e2, 100),
                  "temperature": np.linspace(150, 375, 100)
                  }

    if 1:
        properties = {# "Volume": iapws.V,
                      # "Density": iapws.rho,
                      # "VolumeIterations": iapws.volume_iterations,
                      "Root": iapws.is_root_type,
                      # "CriticalT": iapws.is_critical,
                      }

        pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                                properties_to_evaluate=properties, mole_fractions=True)
        plot = PlotEoS.surf(f, props=pprops, x_var='temperature', y_var='pressure', prop_names=list(properties.keys()),
                            composition=n, logy=True)

    if 0:
        props = {
            # "H": EoS.Property.ENTHALPY,
            "S": EoS.Property.ENTROPY,
            # "G": EoS.Property.GIBBS
        }
        h = f.evaluate_properties_1p(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     properties_to_evaluate=props, )
        # cp = iapws.critical_point(n)

        plot2 = PlotEoS.surf(f, props=h, x_var='temperature', y_var='pressure', prop_names=list(props.keys()),
                             composition=n, logy=True)
        # for i in range(len(properties.keys())):
        #     plot.subplot_idx = i
        #     plot.draw_point(X=cp.Tc, Y=cp.Pc, colours='red')
        # plt.savefig("IAPWS-enthalpy.pdf")

if 0:
    # Water-steam conditions
    if 1:
        ice_phase = False
        logP = False

        Trange = [250., 700.]
        Prange = [1e-3, 1e0] if logP else [10., 250.]
        min_t = Trange[0] - 10.
        max_t = Trange[1] + 10.

        if 0:
            flash_type = DARTSFlash.FlashType.PTFlash
            Xrange = Trange
        elif 1:
            flash_type = DARTSFlash.FlashType.PHFlash
            Xrange = [iapws.H(Prange[1], Trange[i], n, 0, pt=True) * R for i in range(2)]
            # Xrange = [-40000., 10000.]
        else:
            flash_type = DARTSFlash.FlashType.PSFlash
            Xrange = [iapws.S(Prange[1], Trange[i], n, 0, pt=True) * R for i in range(2)]
    else:
        ice_phase = True
        logP = True

        Trange = [250., 275.]
        Prange = [1.e-3, 1.e-0]
        min_t = Trange[0] - 10.
        max_t = Trange[1] + 10.

        if 0:
            flash_type = DARTSFlash.FlashType.PTFlash
            Xrange = Trange
        elif 1:
            flash_type = DARTSFlash.FlashType.PHFlash
            X_L = [iapws.H(Prange[1], Trange[i], n, 0, pt=True) * R for i in range(2)]
            X_I = [ice.H(Prange[1], Trange[i], n, 0, pt=True) * R for i in range(2)]
            Xrange = [min(X_L[0], X_I[0]), max(X_L[1], X_I[1])]
            Xrange = X_I
            Xrange = [-6.5e6, 4.55e4]
        else:
            flash_type = DARTSFlash.FlashType.PSFlash
            X_L = [iapws.S(Prange[1], Trange[i], n, 0, pt=True) * R for i in range(2)]
            X_I = [ice.S(Prange[1], Trange[i], n, 0, pt=True) * R for i in range(2)]
            Xrange = [min(X_L, X_I), max(X_L, X_I)]

    f.init_flash(flash_type=flash_type,  # pxflash_type=FlashParams.BRENT_NEWTON,
                 t_min=Trange[0], t_max=Trange[1],)

    state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 200) if not logP
    else np.logspace(np.log10(Prange[0]), np.log10(Prange[1]), 10),
                  "temperature" if flash_type == DARTSFlash.FlashType.PTFlash
                      else "enthalpy" if flash_type == DARTSFlash.FlashType.PHFlash else "entropy":
                      np.linspace(Xrange[0], Xrange[1], 50),
                  }
    if len(components) == 1:
        flash_results = f.evaluate_flash_1c(state_spec=state_spec, print_state="Flash")
    else:
        flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                         print_state="Flash")

    plot_pt = False
    if plot_pt:
        state_pt = {"pressure": state_spec["pressure"],
                    "temperature": np.linspace(Trange[0], Trange[1], 100), }
        pt_props = f.evaluate_properties_1p(state_spec=state_pt, compositions=compositions, mole_fractions=True,
                                            properties_to_evaluate={"H": f.eos["CEOS"].H} if enthalpy
                                            else {"S": f.eos["CEOS"].S}
                                            )
    else:
        pt_props = None

    plot_method = PlotFlash.pt if flash_type == DARTSFlash.FlashType.PTFlash else (PlotFlash.ph if flash_type == DARTSFlash.FlashType.PHFlash else PlotFlash.ps)
    plot = plot_method(f, flash_results, composition=z,
                       # min_temp=Trange[0], max_temp=Trange[1],
                       min_val=0., max_val=1.,
                       plot_phase_fractions=True, pt_props=pt_props, logP=logP)

    plt.savefig(f.filename + "-" + "-".join(str(int(zi * 100)) for zi in z) + "-ph.pdf")

plt.show()
