import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import CubicEoS

from dartsflash.dartsflash import DARTSFlash, CompData
from dartsflash.mixtures import VL
from dartsflash.plot import *


vmin, vmax = None, None
comp_data = None
mixture_name = None
if 1:
    components = ["CO2"]
    n = [1.]
    pressure = np.arange(1., 75, 2)
    # pressure = np.array([60.])
    temperature = np.array([273.15, 293.15, 304.10, 313.15])#, 418, 500])
    # temperature = np.array([203.15])
    temperaturePT = np.arange(403.15, 434.15, 10)
    vmin = 3e-5
    vmax = 6e-3
    nvol = 1001
    prange = [-25, 200]
elif 0:
    components = ["H2O"]
    n = [1.]
    pressure = np.arange(1, 300, 0.1)
    temperature = np.array([373, 473, 573, 673])
    temperaturePT = np.arange(273.15, 743.15, 10)
    vmax = 6e-3
    nvol = 3001
    prange = [-25, 300]
elif 0:
    components = ["H2O", "CO2"]
    n = [0.9, 0.1]
    pressure = np.arange(1, 300, 1)
    temperature = np.array([423, 473, 523, 573, 623, 673])
    temperaturePT = np.arange(273.15, 643.15, 10)
    vmax = 6e-3
    nvol = 3001
    prange = [-25, 300]
elif 0:
    components = ["H2O", "H2S"]
    n = [0.9, 0.1]
    pressure = np.arange(1, 300, 1)
    temperature = np.array([423, 473, 523, 573, 623, 673])
    vmax = 6e-3
    nvol = 3001
    prange = [-25, 300]
elif 0:
    components = ["H2O", "CO2"]
    n = [np.linspace(0., 1., 11), 1.]
    pressure = np.arange(1, 300, 1)
    temperature = np.array([304.1, 320.5, 340.5, 365.0, 392.8, 425.6, 462.6, 503.1, 547.5, 595.2, 647.5])
    vmax = 6e-3
    nvol = 3001
    prange = [-25, 300]
elif 0:
    components = ["H2O", "H2S"]
    n = [np.linspace(0., 1., 11), 1.]
    pressure = np.arange(1, 300, 1)
    temperature = np.array([373.5, 391.2, 411.2, 433.5, 457.9, 485.4, 513.2, 543.9, 576.9, 611.3, 647.5])
    vmax = 6e-3
    nvol = 3001
    prange = [-25, 300]
else:
    mixture_name = "11c"
    components = ['CO2', 'C2', 'C3', 'C6', 'N2+C1', 'iC4+nC4', 'iC5+nC5', 'C7-C15', 'C16-C27', 'C28-C44', 'C45-C80']
    nc = len(components)

    comp_data = CompData(components, setprops=False)
    comp_data.Mw = [44.0098, 30.0704, 44.0968, 86.1759, 16.1696, 58.1232, 72.1517, 138.9024, 287.0269, 481.4092, 798.4030]
    comp_data.Pc = [73.7646, 48.8387, 42.4552, 29.6882, 45.7788, 37.5365, 33.7809, 24.2755, 16.0835, 14.7207, 14.7919]
    comp_data.Tc = [304.200, 305.400, 369.800, 507.400, 189.410, 420.020, 465.993, 607.025, 751.327, 894.601, 1094.780]
    comp_data.ac = [0.22500, 0.09800, 0.15200, 0.29600, 0.00859, 0.18785, 0.24159, 0.60567, 0.94451, 1.21589, 1.08541]

    comp_data.set_binary_coefficients(0, [0.00, 0.12, 0.12, 0.1184, 0.12, 0.12, 0.10, 0.10, 0.10, 0.10, 0.1])
    comp_data.set_binary_coefficients(1, [0.12, 0.00, 0.00, 0.0004, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(2, [0.12, 0.00, 0.00, 0.0008, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(3, [0.1184, 0.0004, 0.0008, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(4, [0.12, 0.00, 0.00, 0.00, 0.0008, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009])
    comp_data.set_binary_coefficients(5, [0.12, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(6, [0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(7, [0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(8, [0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    comp_data.set_binary_coefficients(9, [0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])
    # comp_data.set_binary_coefficients(10, [0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0])

    # n = [0.06641613561, 0.04806311519, 0.0331903831, 0.00998422608, 0.2891535984, 0.02140980071, 0.01306798678, 0.1141307836, 0.07294298073, 0.1004718332, 0.2311691567]
    # n = [0.1128686759, 0.0733462345, 0.0464768473, 0.0112070728, 0.5021318403, 0.02763601165, 0.01571531181, 0.1114846953, 0.04674849427, 0.02999235525, 0.02239246092]
    # n = [0.1128698358864567, 0.0733467151411221, 0.046477096937412546, 0.011207092984191671, 0.5021359274588869, 0.027636126412142296, 0.01571535905775327, 0.11148459417984317, 0.04674793947318594, 0.029990922319077573, 0.02238839014992785]
    n = [0.08747947066, 0.06260118661, 0.04391053469, 0.01335348325, 0.3602799875, 0.02844163953, 0.01746332637, 0.1638591177, 0.09204568264, 0.07208690825, 0.05847866284]
    # n = [0.1293136564, 0.08025179906, 0.04805633722, 0.009767204589, 0.5944599143, 0.02704002478, 0.014526791, 0.07666346092, 0.01686283727, 0.002664093183]

    pressure = np.arange(1, 300, 1)
    temperature = np.arange(600, 801, 100)
    temperature = np.array([437, 500, 530, 780])
    state_spec['pressure'][0] = 314.7143 / 10
    state_spec['temperature'] = 530.6122
    vmax = 6e-3
    nvol = 3001
    prange = [-25, 300]

comp_data = CompData(components, setprops=True) if comp_data is None else comp_data
f = VL(comp_data=comp_data, mixture_name=mixture_name)
f.set_vl_eos("PR")

ceos = f.eos["VL"]

if 0:
    """ Compressibility factor and P-V diagram of Cubic EoS """
    """ Evaluate properties at PT """
    state_spec = {"pressure": pressure,
                  "temperature": temperature
                  }
    compositions = {comp: n[i] for i, comp in enumerate(components)}
    properties = {"V": ceos.V,
                  "Z": ceos.Z,
                  }

    pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)

    """ Evaluate properties at VT """
    vmin = ceos.V(p=pressure[-1], T=temperature[0], n=n) if vmin is None else vmin
    vmax = ceos.V(p=pressure[0], T=temperature[-1], n=n) if vmax is None else vmax

    state_spec = {"temperature": temperature,
                  "volume": np.linspace(vmin, vmax, nvol)
                  }
    properties = {"Z": ceos.Z,
                  "P": ceos.P
                  }

    vprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)

    """ Plot PV and PZ diagrams """
    pv = PlotEoS.pressure_volume(f, temperatures=temperature, compositions=n,
                                 p_props=pprops, v_props=vprops, v_range=[0, vmax], p_range=[0, prange[1]])
    plt.savefig("P-V-" + f.filename + ".pdf")

    pz = PlotEoS.compressibility(f, temperatures=temperature, compositions=n,
                                 p_props=None, v_props=vprops, z_range=[-0.1, 1.1], p_range=prange)
    plt.savefig("P-Z-" + f.filename + ".pdf")

if 1:
    """ Cubic polynomial and phase identification """
    state_spec = {"pressure": np.array([20, 40, 60, 80]),
                  "temperature": 313.15
                  }
    # state_spec['pressure'][0] = 314.7143/10
    # state_spec['temperature'] = 530.6122
    compositions = {comp: n[i] for i, comp in enumerate(components)}
    properties = {"coeff": ceos.calc_coefficients,
                  }

    pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)

    """ Plot f-Z diagrams """
    temperature = np.array([state_spec['temperature']]) if np.isscalar(state_spec['temperature']) else state_spec['temperature']
    def F(Z):
        # Slice Dataset at composition
        comps = {comp: compositions[comp] for i, comp in enumerate(f.components[:-1]) if comp in pprops.dims}

        coeff = pprops.sel(comps, method='nearest').coeff.transpose('temperature', ...).values
        return np.transpose(np.array([z ** 3 + coeff[..., 0] * z ** 2 + coeff[..., 1] * z + coeff[..., 2] for z in Z]), axes=(1, 2, 0))

    # Initialize Plot object
    from dartsflash.diagram import Diagram
    plot = Diagram(figsize=(8, 5), nrows=1, ncols=len(temperature))
    # plot.add_attributes(suptitle="Cubic polynomial of " + f.mixture_name +
    #                              " at T = {} K".format(state_spec['temperature']), ax_labels=["Z", "f"])

    zmin, zmax = 0., 1.
    Z = np.linspace(zmin, zmax, 50)
    y = F(Z)
    for i, temp in enumerate(temperature):
        plot.subplot_idx = i
        plot.draw_line(X=Z, Y=y[i], styles="solid",
                       datalabels=['{} bar'.format(pres) for pres in state_spec['pressure']])
        plot.set_axes(#xlim=[0, 1],
                      ylim=[-0.1, 0.1],
                       )
        plot.add_attributes(legend=True, legend_loc='lower right')
        plot.draw_line(X=[zmin, zmax], Y=[0., 0.], colours='k', styles='dashed')

    plt.savefig("f-Z-" + f.filename + ".pdf")
    plt.show()

if 0:
    """ PT-diagram of properties """
    """ Evaluate properties at PT """
    state_spec = {"pressure": np.arange(10, 701, 1e1),
                  "temperature": np.arange(100, 700, 1e1)
                  }
    compositions = {comp: n[i] for i, comp in enumerate(components)}
    properties = {"Volume": ceos.V,
                  "Root": ceos.is_root_type,
                  "CriticalT": ceos.is_critical,
                  }

    pprops = f.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                            properties_to_evaluate=properties, mole_fractions=True)
    cp = ceos.critical_point(n)

    plot = PlotEoS.surf(f, props=pprops, x_var='temperature', y_var='pressure', prop_names=list(properties.keys()),
                        composition=n)
    for i in range(len(properties.keys())):
        plot.subplot_idx = i
        plot.draw_point(X=cp.Tc, Y=cp.Pc, colours='red')

plt.show()
