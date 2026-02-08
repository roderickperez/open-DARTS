import numpy as np
import xarray as xr
import pytest

from dartsflash.components import CompData
from dartsflash.mixtures import VL
from dartsflash.libflash import InitialGuess
from .conftest import flash_types, suffix


@pytest.fixture()
def ref0():
    return "./tests/python/ref/flash_vl/"

# Test of PT/PH/PS diagram for CO2 V-L
@pytest.fixture()
def flash_obj_co2():
    comp_data = CompData(components=["CO2"], setprops=True)
    f = VL(comp_data=comp_data)

    f.set_vl_eos("PR")
    return f


@pytest.fixture()
def ref_co2(ref0):
    return ref0 + "co2"


def test_co2(flash_obj_co2, ref_co2):
    # Test PT, PH and PS flashes
    for i in range(3):
        flash_obj_co2.init_flash(flash_type=flash_types[i],
                                 t_tol=1e-1)

        Prange = [10., 100.]
        Trange = [260., 350.]
        z = [1.]
        Xrange = flash_obj_co2.get_ranges(prange=Prange, trange=Trange, composition=z)
        state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 20),
                      flash_obj_co2.state_vars[1]: np.linspace(Xrange[0], Xrange[1], 20)
                      }
        props = ["nu", "temp"]

        results = flash_obj_co2.evaluate_flash_1c(state_spec=state_spec, initialize_with_previous_results=True)

        # Plot results
        if 0:
            import os
            filedir = "./tests/python/fig/flash_vl/"
            os.makedirs(os.path.dirname(filedir), exist_ok=True)

            from dartsflash.plot import PlotFlash, plt
            plot_methods = [PlotFlash.pt, PlotFlash.ph, PlotFlash.ps]

            plot_res = plot_methods[i](flash_obj_co2, results, composition=z,
                                       # min_temp=Trange[0], max_temp=Trange[1],
                                       min_val=0., max_val=1.,
                                       plot_phase_fractions=True, pt_props=None, logP=False)

            plt.savefig(filedir + "co2" + suffix[i] + "-result.pdf")

        results = results[flash_obj_co2.state_vars + props]

        if 1:
            # Use assertions from xarray
            ref = xr.open_dataset(ref_co2 + suffix[i] + ".h5", engine='h5netcdf')
            ref = ref[flash_obj_co2.state_vars + props]

            try:
                xr.testing.assert_allclose(results, ref, rtol=1e-5)
            except AssertionError as e:
                for prop in props:
                    diff = eval("results." + prop + ".values") - eval("ref." + prop + ".values")
                    print(prop + ":", "Maximum absolute difference:", np.nanmax(diff),
                          "Number of different results (relative):",
                          np.count_nonzero(np.isclose(eval("results." + prop + ".values"),
                                                      eval("ref." + prop + ".values"),
                                                      rtol=1e-5, equal_nan=True)),
                          )
                raise e

        else:
            # Update reference file
            results.to_netcdf(ref_co2 + suffix[i] + ".h5", engine='h5netcdf')
            plt.savefig(filedir + "co2" + suffix[i] + "-ref.pdf")


# Test of PT/PH diagram for CO2-C1 V-L-L
@pytest.fixture()
def flash_obj_co2c1():
    f = VL(comp_data=CompData(components=["CO2", "C1"], setprops=True))

    f.set_vl_eos("PR", rich_phase_order=[0, -1], trial_comps=[InitialGuess.Wilson],
                 # trial_comps=[InitialGuess.Wilson, InitialGuess.Wilson13],
                 switch_tol=1e-3)
    return f


@pytest.fixture()
def ref_co2c1(ref0):
    return ref0 + "co2c1"


def test_co2c1(flash_obj_co2c1, ref_co2c1):
    # Test PT, PH and PS flashes
    for i in range(3):
        flash_obj_co2c1.init_flash(flash_type=flash_types[i], split_switch_tol=1e-2,
                                   t_tol=1e-1)

        Prange = [10., 100.]
        Trange = [260., 350.]
        z = [0.9, 0.1]
        Xrange = flash_obj_co2c1.get_ranges(prange=Prange, trange=Trange, composition=z)
        state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 10),
                      flash_obj_co2c1.state_vars[1]: np.linspace(Xrange[0], Xrange[1], 10)
                      }
        compositions = {comp: z[i] for i, comp in enumerate(flash_obj_co2c1.components)}
        props = ["nu", "X", "temp"]

        results = flash_obj_co2c1.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True, 
                                                 initialize_with_previous_results=True)

        # Plot results
        if 0:
            import os
            filedir = "./tests/python/fig/flash_vl/"
            os.makedirs(os.path.dirname(filedir), exist_ok=True)

            from dartsflash.plot import PlotFlash, plt
            plot_methods = [PlotFlash.pt, PlotFlash.ph, PlotFlash.ps]

            plot_res = plot_methods[i](flash_obj_co2c1, results, composition=z,
                                       # min_temp=Trange[0], max_temp=Trange[1],
                                       min_val=0., max_val=1.,
                                       plot_phase_fractions=True, pt_props=None, logP=False)

            plt.savefig(filedir + "co2c1" + suffix[i] + "-result.pdf")

        results = results[flash_obj_co2c1.state_vars + props]

        if 1:
            # Use assertions from xarray
            ref = xr.open_dataset(ref_co2c1 + suffix[i] + ".h5", engine='h5netcdf')
            ref = ref[flash_obj_co2c1.state_vars + props]

            try:
                xr.testing.assert_allclose(results, ref, rtol=1e-5)
            except AssertionError as e:
                for prop in props:
                    diff = eval("results." + prop + ".values") - eval("ref." + prop + ".values")
                    print(prop + ":", "Maximum absolute difference:", np.nanmax(diff),
                          "Number of different results (relative):",
                          np.count_nonzero(np.isclose(eval("results." + prop + ".values"),
                                                      eval("ref." + prop + ".values"),
                                                      rtol=1e-5, equal_nan=True)),
                          )
                raise e
        else:
            # Update reference file
            results.to_netcdf(ref_co2c1 + suffix[i] + ".h5", engine='h5netcdf')
            plt.savefig(filedir + "co2c1" + suffix[i] + "-ref.pdf")


# Test of ternary diagram for M3 V-L
""" Three-component gas mixture (Michelsen, 1982a fig. 4) """
@pytest.fixture()
def flash_obj_m3():
    components = ["C1", "CO2", "H2S"]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [46.04, 73.75, 89.63]
    comp_data.Tc = [190.58, 304.10, 373.53]
    comp_data.ac = [0.012, 0.239, 0.0942]
    comp_data.Mw = [16.043, 44.01, 34.1]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)

    f = VL(comp_data=comp_data)

    f.set_vl_eos("PR",
                 trial_comps=[InitialGuess.Wilson],
                 # trial_comps=[InitialGuess.Wilson, InitialGuess.Wilson13],
                 switch_tol=1e-3)
    return f


@pytest.fixture()
def ref_m3(ref0):
    return ref0 + "m3"


def test_m3(flash_obj_m3, ref_m3):
    # Test PT, PH and PS flashes
    for i in range(0):
        flash_obj_m3.init_flash(flash_type=flash_types[i], split_switch_tol=1e-2,
                                t_tol=1e-1)

        Prange = [10., 100.]
        Trange = [150., 300.]
        Xrange = flash_obj_m3.get_ranges(prange=Prange, trange=Trange, composition=z)
        state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 20),
                      flash_obj_m3.state_vars[1]: np.linspace(Xrange[0], Xrange[1], 20)
                      }
        compositions = {comp: z[i] for i, comp in enumerate(flash_obj_m3.components)}
        props = ["nu", "X", "temp"]

        results = flash_obj_m3.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True, 
                                              initialize_with_previous_results=True)

        # Plot results
        if 0:
            import os
            filedir = "./tests/python/fig/flash_vl/"
            os.makedirs(os.path.dirname(filedir), exist_ok=True)

            from dartsflash.plot import PlotFlash, plt
            plot_methods = [PlotFlash.pt, PlotFlash.ph, PlotFlash.ps]

            plot_res = plot_methods[i](flash_obj_m7, results, composition=z,
                                       # min_temp=Trange[0], max_temp=Trange[1],
                                       min_val=0., max_val=1.,
                                       plot_phase_fractions=True, logP=False)

            plt.savefig(filedir + "m3" + suffix[i] + "-result.pdf")

        results = results[flash_obj_m3.state_vars + props]

        if 1:
            # Use assertions from xarray
            ref = xr.open_dataset(ref_m3 + suffix[i] + ".h5", engine='h5netcdf')
            ref = ref[flash_obj_m3.state_vars + props]

            try:
                xr.testing.assert_allclose(results, ref, rtol=1e-5)
            except AssertionError as e:
                for prop in props:
                    diff = eval("results." + prop + ".values") - eval("ref." + prop + ".values")
                    print(prop + ":", "Maximum absolute difference:", np.nanmax(diff),
                          "Number of different results (relative):",
                          np.count_nonzero(np.isclose(eval("results." + prop + ".values"),
                                                      eval("ref." + prop + ".values"),
                                                      rtol=1e-5, equal_nan=True)),
                          )
                raise e
        else:
            # Update reference file
            results.to_netcdf(ref_m3 + suffix[i] + ".h5", engine='h5netcdf')
            plt.savefig(filedir + "m3" + suffix[i] + "-ref.pdf")


# Test of PT diagram for M7 V-L
""" Seven-component gas mixture (Michelsen, 1982a fig. 2) """
@pytest.fixture()
def flash_obj_m7():
    components = ["C1", "C2", "C3", "nC4", "nC5", "nC6", "N2"]

    comp_data = CompData(components, setprops=False)
    comp_data.Pc = [45.99, 48.72, 42.48, 33.70, 27.40, 21.10, 34.00]
    comp_data.Tc = [190.56, 305.32, 369.83, 469.70, 540.20, 617.70, 126.20]
    comp_data.ac = [0.011, 0.099, 0.152, 0.252, 0.350, 0.490, 0.0377]
    comp_data.Mw = [16.043, 30.07, 44.097, 58.124, 72.151, 86.178, 28.013]
    comp_data.kij = np.zeros(comp_data.nc * comp_data.nc)

    f = VL(comp_data=comp_data)

    f.set_vl_eos("PR",
                 trial_comps=[InitialGuess.Wilson],
                 # trial_comps=[InitialGuess.Wilson, InitialGuess.Wilson13],
                 switch_tol=1e-3)
    return f


@pytest.fixture()
def ref_m7(ref0):
    return ref0 + "m7"


def test_m7(flash_obj_m7, ref_m7):
    # Test PT, PH and PS flashes
    for i in range(0):
        flash_obj_m7.init_flash(flash_type=flash_types[i], split_switch_tol=1e-2,
                                t_tol=1e-1)

        Prange = [10., 100.]
        Trange = [150., 300.]
        z = [0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140]
        Xrange = flash_obj_m7.get_ranges(prange=Prange, trange=Trange, composition=z)
        state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 20),
                      flash_obj_m7.state_vars[1]: np.linspace(Xrange[0], Xrange[1], 20)
                      }
        compositions = {comp: z[i] for i, comp in enumerate(flash_obj_m7.components)}
        props = ["nu", "X", "temp"]

        results = flash_obj_m7.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True, 
                                              initialize_with_previous_results=True)

        # Plot results
        if 0:
            import os
            filedir = "./tests/python/fig/flash_vl/"
            os.makedirs(os.path.dirname(filedir), exist_ok=True)

            from dartsflash.plot import PlotFlash, plt
            plot_methods = [PlotFlash.pt, PlotFlash.ph, PlotFlash.ps]

            plot_res = plot_methods[i](flash_obj_m7, results, composition=z,
                                       # min_temp=Trange[0], max_temp=Trange[1],
                                       min_val=0., max_val=1.,
                                       plot_phase_fractions=True, logP=False)

            plt.savefig(filedir + "m7" + suffix[i] + "-result.pdf")

        results = results[flash_obj_m7.state_vars + props]

        if 1:
            # Use assertions from xarray
            ref = xr.open_dataset(ref_m7 + suffix[i] + ".h5", engine='h5netcdf')
            ref = ref[flash_obj_m7.state_vars + props]

            try:
                xr.testing.assert_allclose(results, ref, rtol=1e-5)
            except AssertionError as e:
                for prop in props:
                    diff = eval("results." + prop + ".values") - eval("ref." + prop + ".values")
                    print(prop + ":", "Maximum absolute difference:", np.nanmax(diff),
                          "Number of different results (relative):",
                          np.count_nonzero(np.isclose(eval("results." + prop + ".values"),
                                                      eval("ref." + prop + ".values"),
                                                      rtol=1e-5, equal_nan=True)),
                          )
                raise e
        else:
            # Update reference file
            results.to_netcdf(ref_m7 + suffix[i] + ".h5", engine='h5netcdf')
            plt.savefig(filedir + "m7" + suffix[i] + "-ref.pdf")
