import numpy as np
import xarray as xr
import pytest

from dartsflash.mixtures import IAPWS
from .conftest import flash_types, suffix


# Test of IAPWS and VLAq implementations of 1-component H2O-flash in PT-PH-PS
@pytest.fixture()
def flash_obj_iapws():
    return IAPWS(iapws_ideal=False, ice_phase=False)


@pytest.fixture()
def ref_iapws():
    return "./tests/python/ref/flash_iapws/iapws"


def test_iapws(flash_obj_iapws, ref_iapws):
    # Test PT, PH and PS flashes
    for i in range(3):
        flash_obj_iapws.init_flash(flash_type=flash_types[i],
                                   t_min=250., t_max=900., t_init=500.,
                                   t_tol=1e-1)

        Prange = [10., 200.]
        Trange = [270., 700.]
        z = [1.]
        Xrange = flash_obj_iapws.get_ranges(prange=Prange, trange=Trange, composition=z)
        state_spec = {"pressure": np.linspace(Prange[0], Prange[1], 10),
                      flash_obj_iapws.state_vars[1]: np.linspace(Xrange[0], Xrange[1], 10)
                      }
        props = ["nu", "temp"]

        results = flash_obj_iapws.evaluate_flash_1c(state_spec=state_spec, initialize_with_previous_results=True)

        # Plot results
        if 0:
            import os
            filedir = "./tests/python/fig/flash_iapws/"
            os.makedirs(os.path.dirname(filedir), exist_ok=True)

            from dartsflash.plot import PlotFlash, plt
            plot_methods = [PlotFlash.pt, PlotFlash.ph, PlotFlash.ps]

            plot_res = plot_methods[i](flash_obj_iapws, results, composition=z,
                                       # min_temp=Trange[0], max_temp=Trange[1],
                                       min_val=0., max_val=1.,
                                       plot_phase_fractions=True, pt_props=None, logP=False)

            plt.savefig(filedir + "iapws" + suffix[i] + "-result.pdf")

        results = results[flash_obj_iapws.state_vars + props]

        if 1:
            # Use assertions from xarray
            ref = xr.open_dataset(ref_iapws + suffix[i] + ".h5", engine='h5netcdf')
            ref = ref[flash_obj_iapws.state_vars + props]

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
            results.to_netcdf(ref_iapws + suffix[i] + ".h5", engine='h5netcdf')
            plt.savefig(filedir + "iapws" + suffix[i] + "-ref.pdf")
