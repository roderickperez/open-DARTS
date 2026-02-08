import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from dartsflash.libflash import InitialGuess, FlashParams, EoS

from dartsflash.dartsflash import DARTSFlash, CompData, R
from dartsflash.plot import *


""" SPECIFY COMPONENTS AND COMPOSITIONS """
mixture_name = None
trial_comps = None
rich_phase_order = None
T_min, T_max, T_init = 100, 1000, 300

if 0:
    components = ["CO2"]
    comp_data = CompData(components, setprops=True)
    rich_phase_order = None  # order of L phases

    z = [1.]
    Trange = [250., 350.]
    Prange = [10., 100.]
    # T_min, T_max = 200., 400.
    T_min, T_max = 100, 1000

elif 0:
    components = ["CO2", "C1"]
    comp_data = CompData(components, setprops=True)
    trial_comps = [0, 1]
    rich_phase_order = [0, -1]  # order of L phases rich in component i

    z = [0.75, 0.25]  # for PT/PH/PS-diagram

    # For binary diagram
    dz = 0.01
    min_z = [z[0]]
    max_z = [z[0]]

    Trange = [260., 350.]
    Prange = [10., 100.]
    T_min, T_max = 100., 500.

elif 0:
    components = ["C1", "nC4"]
    z = [0.99, 0.01]  # for PT/PH/PS-diagram

    comp_data = CompData(components, setprops=True)
    comp_data.Pc = [46.0, 38.0]
    comp_data.Tc = [190.60, 425.20]
    comp_data.ac = [0.008, 0.193]
    comp_data.kij = np.zeros(4)
    comp_data.T0 = 273.15

    # For binary diagram
    dz = 0.01
    min_z = [0.]
    max_z = [1.]

    Trange = [100., 300.]
    Prange = [10., 100.]
    T_min, T_max = 100., 300.

elif 0:
    """ Seven-component gas mixture (Michelsen, 1982a fig. 2) """
    mixture_name = "M7"
    components = ["C1", "C2", "C3", "nC4", "nC5", "nC6", "N2"]
    z = [0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [45.99, 48.72, 42.48, 33.70, 27.40, 21.10, 34.00]
    comp_data.Tc = [190.56, 305.32, 369.83, 469.70, 540.20, 617.70, 126.20]
    comp_data.ac = [0.011, 0.099, 0.152, 0.252, 0.350, 0.490, 0.0377]
    comp_data.Mw = [16.043, 30.07, 44.097, 58.124, 72.151, 86.178, 28.013]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)

elif 0:
    """ Ternary mixture (Michelsen, 1982a fig. 4) """
    components = ["C1", "CO2", "H2S"]
    z = [0.50, 0.10, 0.40]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [46.04, 73.75, 89.63]
    comp_data.Tc = [190.58, 304.10, 373.53]
    comp_data.ac = [0.012, 0.239, 0.0942]
    comp_data.Mw = [16.043, 44.01, 34.1]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)

    # For ternary diagram
    dz = 0.01
    min_z = [0., 0.]
    max_z = [1., 1.]

elif 0:
    """ Y8 mixture """
    mixture_name = "Y8"
    components = ["C1", "C2", "C3", "nC5", "nC7", "nC10"]
    z = [0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [45.99, 48.72, 42.48, 33.70, 27.40, 21.10]
    comp_data.Tc = [190.56, 305.32, 369.83, 469.70, 540.20, 617.70]
    comp_data.ac = [0.011, 0.099, 0.152, 0.252, 0.350, 0.490]
    comp_data.Mw = [16.043, 30.07, 44.097, 72.151, 100.205, 142.2848]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)

elif 0:
    """ MY10 mixture """
    mixture_name = "MY10"
    components = ["C1", "C2", "C3", "nC4", "nC5", "nC6", "nC7", "nC8", "nC10", "nC14"]
    z = [0.35, 0.03, 0.04, 0.06, 0.04, 0.03, 0.05, 0.05, 0.30, 0.05]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [45.99, 48.72, 42.48, 37.96, 33.70, 30.25, 27.40, 24.9, 21.10, 15.7]
    comp_data.Tc = [190.56, 305.32, 369.83, 425.12, 469.70, 507.6, 540.20, 568.7, 617.70, 693.0]
    comp_data.ac = [0.011, 0.099, 0.152, 0.2, 0.252, 0.3, 0.350, 0.399, 0.490, 0.644]
    comp_data.Mw = [16.043, 30.07, 44.097, 58.124, 72.151, 86.178, 100.205, 114.231, 142.2848, 0.0]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., 0., 0., 0.02, 0.02, 0.025, 0.025, 0.035, 0.045, 0.045])

elif 0:
    """ Depleted oil field with injected CO2 mixture """
    components = ["C1", "nC10", "CO2"]

    comp_data.Pc = [46.0, 21.2, 73.75]
    comp_data.Tc = [190.58, 617.7, 304.10]
    comp_data.ac = [0.012, 0.489, 0.239]
    comp_data.Mw = [16.043, 142.2848, 44.01]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., 0.048388, 0.0936])
    comp_data.set_binary_coefficients(1, [0.048388, 0., 0.1])

elif 0:
    """ Oil B mixture (Shelton and Yarborough, 1977), data from (Li, 2012) """
    mixture_nam = "OilB"
    components = ["CO2", "N2", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "C6",
                  "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
    composition = [0.0011, 0.0048, 0.1630, 0.0403, 0.0297, 0.0036, 0.0329, 0.0158, 0.0215, 0.0332,
                   0.181326, 0.161389, 0.125314, 0.095409, 0.057910, 0.022752]
    comp_data.Pc = [73.819, 33.5, 45.4, 48.2, 41.9, 36., 37.5, 33.4, 33.3, 33.9, 25.3, 19.1, 14.2, 10.5, 7.5, 4.76]
    comp_data.Tc = [304.211, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 506.35, 566.55,
                                 647.06, 719.44, 784.93, 846.33, 919.39]
    comp_data.ac = [0.225, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.299, 0.3884, 0.5289,
                                 0.6911, 0.8782, 1.1009, 1.4478]
    comp_data.Mw = [44.01, 28.01, 16.04, 30.07, 44.1, 58.12, 58.12, 72.15, 72.15, 84., 112.8, 161.2, 223.2,
                                 304.4, 417.5, 636.8]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., -0.02, 0.075, 0.08, 0.08, 0.085, 0.085, 0.085, 0.085, 0.095, 0.095, 0.095,
                                          0.095, 0.095, 0.095, 0.095])
    comp_data.set_binary_coefficients(1, [-0.02, 0., 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05, 0.1, 0.12, 0.12,
                                             0.12, 0.12, 0.12])
    comp_data.set_binary_coefficients(2, [0.075, 0.08, 0., 0.003, 0.01, 0.018, 0.018, 0.025, 0.026, 0.036, 0.049, 0.073,
                                          0.098, 0.124, 0.149, 0.181])

elif 0:
    """ Maljamar reservoir mixture (Orr, 1981), data from (Li, 2012) """
    mixture_name = "MaljamarRes"
    components = ["CO2", "C1", "C2", "C3", "nC4", "C5-7", "C8-10", "C11-14", "C15-20", "C21-28", "C29+"]
    z = [0., 0.2939, 0.1019, 0.0835, 0.0331, 0.1204, 0.1581, 0.0823, 0.0528, 0.0276, 0.0464]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [73.819, 45.4, 48.2, 41.9, 37.5, 28.82, 23.743, 18.589, 14.8, 11.954, 8.523]
    comp_data.Tc = [304.211, 190.6, 305.4, 369.8, 425.2, 516.667, 590., 668.611, 745.778, 812.667, 914.889]
    comp_data.ac = [0.225, 0.008, 0.098, 0.152, 0.193, 0.2651, 0.3644, 0.4987, 0.6606, 0.8771, 1.2789]
    comp_data.Mw = [44.01, 16.043, 30.1, 44.1, 58.1, 89.9, 125.7, 174.4, 240.3, 336.1, 536.7]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115])
    comp_data.set_binary_coefficients(1, [0.115, 0., 0., 0., 0., 0.045, 0.055, 0.055, 0.06, 0.08, 0.28])

elif 0:
    """ Maljamar separator mixture (Orr, 1981), data from (Li, 2012) """
    mixture_name = "MaljamarSep"
    components = ["CO2", "C5-7", "C8-10", "C11-14", "C15-20", "C21-28", "C29+"]
    z = [0.0, 0.2354, 0.3295, 0.1713, 0.1099, 0.0574, 0.0965]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [73.9, 28.8, 23.7, 18.6, 14.8, 12.0, 8.5]
    comp_data.Tc = [304.2, 516.7, 590.0, 668.6, 745.8, 812.7, 914.9]
    comp_data.ac = [0.225, 0.265, 0.364, 0.499, 0.661, 0.877, 1.279]
    comp_data.Mw = [44.01, 89.9, 125.7, 174.4, 240.3, 336.1, 536.7]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0.0, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115])

elif 0:
    """ Sour gas mixture, data from (Li, 2012) """
    mixture_name = "SourGas"
    components=["CO2", "N2", "H2S", "C1", "C2", "C3"]
    z = [0.70592, 0.07026, 0.01966, 0.06860, 0.10559, 0.02967]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [73.819, 33.9, 89.4, 45.992, 48.718, 42.462]
    comp_data.Tc = [304.211, 126.2, 373.2, 190.564, 305.322, 369.825]
    comp_data.ac = [0.225, 0.039, 0.081, 0.01141, 0.10574, 0.15813]
    comp_data.Mw = [44.01, 28.013, 34.1, 16.043, 30.07, 44.097]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., -0.02, 0.12, 0.125, 0.135, 0.150])
    comp_data.set_binary_coefficients(1, [-0.02, 0., 0.2, 0.031, 0.042, 0.091])
    comp_data.set_binary_coefficients(2, [0.12, 0.2, 0., 0.1, 0.08, 0.08])

elif 0:
    """ Bob Slaughter Block mixture, data from (Li, 2012) """
    mixture_name="BSB"
    components = ["CO2", "C1", "PC1", "PC2"]
    z = [0.0337, 0.0861, 0.6478, 0.2324]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [73.77, 46., 27.32, 17.31]
    comp_data.Tc = [304.2, 160., 529.03, 795.33]
    comp_data.ac = [0.225, 0.008, 0.481, 1.042]
    comp_data.Mw = [44.01, 16.043, 98.45, 354.2]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., 0.055, 0.081, 0.105])

elif 0:
    """ NorthWardEstes mixture, data from (Li, 2012) """
    mixture_name="NWE"
    components = ["CO2", "C1", "PC1", "PC2", "PC3", "PC4", "PC5"]
    z = [0.0077, 0.2025, 0.118, 0.1484, 0.2863, 0.149, 0.0881]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [73.77, 46., 45.05, 33.51, 24.24, 18.03, 17.26]
    comp_data.Tc = [304.2, 190.6, 343.64, 466.41, 603.07, 733.79, 923.2]
    comp_data.ac = [0.225, 0.008, 0.13, 0.244, 0.6, 0.903, 1.229]
    comp_data.Mw = [44.01, 16.04, 38.4, 72.82, 135.82, 257.75, 479.95]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)
    comp_data.set_binary_coefficients(0, [0., 0.12, 0.12, 0.12, 0.09, 0.09, 0.09])

else:
    mixture_name = '11c'
    components = ['CO2', 'C2', 'C3', 'C6', 'N2+C1', 'iC4+nC4', 'iC5+nC5', 'C7-C15', 'C16-C27', 'C28-C44', 'C45-C80']
    nc = len(components)

    comp_data = CompData(components, setprops=False)
    comp_data.Mw = [44.0098, 30.0704, 44.0968, 86.1759, 16.1696, 58.1232, 72.1517, 138.9024, 287.0269, 481.4092, 798.4030]
    comp_data.Pc = [73.7646, 48.8387, 42.4552, 29.6882, 45.7788, 37.5365, 33.7809, 24.2755, 16.0835, 14.7207, 14.7919]
    comp_data.Tc = [304.200, 305.400, 369.800, 507.400, 189.410, 420.020, 465.993, 607.025, 751.327, 894.601, 1094.780]
    comp_data.ac = [0.22500, 0.09800, 0.15200, 0.29600, 0.00859, 0.18785, 0.24159, 0.60567, 0.94451, 1.21589, 1.08541]

    comp_data.set_binary_coefficients(0, [0.0000, 0.1200, 0.1200, 0.1200, 0.1184, 0.1200, 0.1200, 0.1000, 0.1000, 0.1000, 0.1000])
    comp_data.set_binary_coefficients(1, [0.1200, 0.0000, 0.0000, 0.0000, 0.0004, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(2, [0.1200, 0.0000, 0.0000, 0.0000, 0.0008, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(3, [0.1200, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(4, [0.1184, 0.0004, 0.0008, 0.0009, 0.0000, 0.0008, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009])
    comp_data.set_binary_coefficients(5, [0.1200, 0.0000, 0.0000, 0.0000, 0.0008, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(6, [0.1200, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(7, [0.1000, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(8, [0.1000, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(9, [0.1000, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
    comp_data.set_binary_coefficients(10, [0.1000, 0.0000, 0.0000, 0.0000, 0.0009, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

    # rich_phase_order = []  # order of L phases
    trial_comps = [InitialGuess.Yi.Wilson, InitialGuess.Yi.Wilson13, 0]

    # # f.mixture.name = mix.comp_data
    # f.flash_params.stability_variables = FlashParams.alpha
    # f.flash_params.split_variables = FlashParams.nik
    # # f.flash_params.split_tol = 1e-20
    # f.flash_params.split_switch_tol = 1e-5
    # # f.flash_params.tpd_tol = 1e-11
    # # f.flash_params.tpd_close_to_boundary = 1e-2

    ini_comp = [0.112600, 0.073200, 0.046400, 0.011200, 0.500900, 0.027600, 0.015700, 0.111500, 0.046900, 0.030400, 0.023600]
    inj_stream = [0.0920 + 0.1, 0.1015, 0.0578, 0.0011, 0.7177 - 0.1, 0.024, 0.0054, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

    # data = np.loadtxt("3comp_thermodynamic_data.txt", skiprows=1)  # Use `delimiter=','` or `delimiter='\t'` if needed
    # ref_data = [data[:, 0] + 273.15, data[:, 1]]  # temperature in K, pressure in bar
    N = 400
    Prange = [1., 550.]
    Trange = [200., 800.]
    # state_spec = {"pressure": np.linspace(min(ref_data[1]) - 10, max(ref_data[1]) + 10, N),
    #               "temperature": np.linspace(min(ref_data[0]) - 10, max(ref_data[0]) + 10, N),
    #               }
    z = ini_comp

    # critical_point = [337.02 + 273.15, 475.03]  # temperature, pressure

compositions = {comp: z[i] for i, comp in enumerate(components)}

""" CREATE INSTANCE OF V-L DARTSFLASH OBJECT """
from dartsflash.mixtures import VL
f = VL(comp_data=comp_data, mixture_name=mixture_name)

vl_eos_type = "PR"
f.set_vl_eos(vl_eos_type, trial_comps=trial_comps,
             rich_phase_order=rich_phase_order,
             root_order=[EoS.RootFlag.MAX, EoS.RootFlag.MIN],
             # stability_tol=1e-16,
             switch_tol=1e-3,
             )
vl_eos = f.eos["VL"]

flash_type = DARTSFlash.FlashType.PTFlash
f.init_flash(flash_type=flash_type,  # pxflash_type=FlashParams.BRENT_NEWTON,
             t_min=T_min, t_max=T_max, split_tol=1e-24, split_switch_tol=1e-1,
             # split_variables=FlashParams.lnK_chol,
             vl_eos_name = "VL", light_comp_idx=4,
             # verbose=True,
             )

""" SPECIFY STATES """
logy = False
Xrange = f.get_ranges(prange=Prange, trange=Trange, composition=z)
n_points = {"pressure": N, f.state_vars[1]: N}
state_spec = {"pressure": np.linspace(Prange[0], Prange[1], n_points["pressure"]) if not logy
                        else np.logspace(np.log10(Prange[0]), np.log10(Prange[1]), n_points["pressure"]),
              f.state_vars[1]: np.linspace(Xrange[0], Xrange[1], n_points[f.state_vars[1]]),
              }

if len(components) == 1:
    flash_results = f.evaluate_flash_1c(state_spec=state_spec, print_state="Flash")
else:
    flash_results = f.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                     print_state="Flash")

plot_pt = False
if flash_type > DARTSFlash.FlashType.PTFlash and plot_pt:
    state_pt = {"pressure": state_spec["pressure"],
                "temperature": np.linspace(Trange[0], Trange[1], 100),}
    pt_props = f.evaluate_properties_1p(state_spec=state_pt, compositions=compositions, mole_fractions=True,
                                        properties_to_evaluate={"H": vl_eos.H} if flash_type == DARTSFlash.FlashType.PHFlash
                                                          else {"S": vl_eos.S}
                                        )
else:
    pt_props = None

if 1:
    plot_method = {DARTSFlash.FlashType.PTFlash: PlotFlash.pt,
                   DARTSFlash.FlashType.PHFlash: PlotFlash.ph,
                   DARTSFlash.FlashType.PSFlash: PlotFlash.ps
                   }
    plot = plot_method[flash_type](f, flash_results, composition=z,
                                   # min_temp=250., max_temp=350.,
                                   min_val=0., max_val=1.,
                                   plot_phase_fractions=True, pt_props=pt_props)

    plt.savefig(f.filename + "-" + "-".join(str(int(zi*100)) for zi in z) + "-ph.pdf")

""" PLOT COMPOSITIONAL DIAGRAMS """
if 0 and len(components) > 1:
    compositions = {comp: np.arange(min_z[i], max_z[i] + 0.1 * dz, dz) for i, comp in enumerate(components[:-1])}
    compositions[components[-1]] = 1.

    plot = PlotFlash.binary_xz(f, flash_results, variable_comp_idx=0, dz=dz, state=state_spec,

                               )
    plt.savefig(f.filename + "-" + "-".join(str(int(zi*100)) for zi in z) + "-hx.pdf")

plt.show()
