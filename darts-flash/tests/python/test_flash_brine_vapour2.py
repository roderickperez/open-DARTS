import numpy as np
import xarray as xr
import pytest

from dartsflash.libflash import InitialGuess, CubicEoS, AQEoS
from dartsflash.dartsflash import DARTSFlash
from .conftest import compdata


# Test of full compositional space for H2O-CO2-C1 mixture
@pytest.fixture()
def flash_obj(compdata):
    f = DARTSFlash(comp_data=compdata)

    f.add_eos("PR", CubicEoS(compdata, CubicEoS.PR),
              trial_comps=[InitialGuess.Yi.Wilson])
    f.add_eos("AQ", AQEoS(compdata, {AQEoS.water: AQEoS.Jager2003,
                                          AQEoS.solute: AQEoS.Ziabakhsh2012}))
    eos_order = ["AQ", "PR"]
    f.init_flash(eos_order=eos_order, flash_type=DARTSFlash.FlashType.NegativeFlash,
                 nf_initial_guess=[InitialGuess.Henry_AV])
    return f


@pytest.fixture()
def ref_brine_vapour2():
    return "./tests/python/ref/ref_flash_brine_vapour2.h5"


def test_brine_vapour2(flash_obj, ref_brine_vapour2):
    state_spec = {"pressure": np.array([1., 2., 5., 10., 50., 100., 200., 400.]),
                  "temperature": np.arange(273.15, 373.15, 10),
                  }
    zrange = np.concatenate([np.array([1e-12, 1e-10, 1e-8]),
                             np.linspace(1e-6, 1. - 1e-6, 10),
                             np.array([1. - 1e-8, 1. - 1e-10, 1. - 1e-12])])
    compositions = {"H2O": zrange,
                    "CO2": zrange,
                    "C1": 1.,
                    }

    props = ["nu", "X"]
    results = flash_obj.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True)
    results = results[["pressure", "temperature", "H2O", "CO2"] + props]

    if 1:
        # Use assertions from xarray
        ref = xr.open_dataset(ref_brine_vapour2, engine='h5netcdf')
        ref = ref[["pressure", "temperature", "H2O", "CO2"] + props]

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
        results.to_netcdf(ref_brine_vapour2, engine='h5netcdf')
