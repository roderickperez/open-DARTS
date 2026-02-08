import numpy as np
import xarray as xr
import pytest

from dartsflash.libflash import CubicEoS, AQEoS, Ballard, InitialGuess, EoS
from dartsflash.dartsflash import DARTSFlash
from .conftest import compdata


# Test of full compositional space for H2O-CO2-C1 mixture
@pytest.fixture()
def flash_obj_cubic(compdata):
    f = DARTSFlash(comp_data=compdata)

    f.add_eos("PR", CubicEoS(compdata, CubicEoS.PR),
              trial_comps=[InitialGuess.Yi.Wilson])
    return f


@pytest.fixture()
def flash_obj_hybrid(compdata):
    f = DARTSFlash(comp_data=compdata)

    f.add_eos("PR", CubicEoS(compdata, CubicEoS.PR),
              trial_comps=[InitialGuess.Yi.Wilson])
    f.add_eos("AQ", AQEoS(compdata, {AQEoS.water: AQEoS.Jager2003,
                                          AQEoS.solute: AQEoS.Ziabakhsh2012}))
    f.add_eos("sI", Ballard(compdata, "sI"))
    return f


@pytest.fixture()
def ref0():
    return "./tests/python/ref/eos/"

@pytest.fixture()
def ref_properties_pt_cubic(ref0):
    return ref0 + "pt_cubic.h5"


@pytest.fixture()
def ref_properties_pt_hybrid(ref0):
    return ref0 + "pt_hybrid.h5"


@pytest.fixture()
def ref_properties_vt(ref0):
    return ref0 + "vt.h5"


def test_properties_pt_cubic(flash_obj_cubic, ref_properties_pt_cubic):
    state_spec = {"pressure": np.array([1., 2., 5., 10., 50., 100., 200., 400.]),
                  "temperature": np.arange(273.15, 373.15, 10)
                  }
    zrange = np.concatenate([np.array([1e-12, 1e-10, 1e-8]),
                             np.linspace(1e-6, 1. - 1e-6, 10),
                             np.array([1. - 1e-8, 1. - 1e-10, 1. - 1e-12])])
    compositions = {"H2O": zrange,
                    "CO2": zrange,
                    "C1": 1.
                    }
    eos = flash_obj_cubic.eos["PR"]
    properties = {"V": eos.V,
                  "Cv": eos.Cv,
                  "Cp": eos.Cp,
                  "JT": eos.JT,
                  "vs": eos.vs,
                  }

    results = flash_obj_cubic.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                                           properties_to_evaluate=properties, mole_fractions=True)
    results = results[["pressure", "temperature", "H2O", "CO2"] + list(properties.keys())]

    if 1:
        # Use assertions from xarray
        ref = xr.open_dataset(ref_properties_pt_cubic, engine='h5netcdf')
        ref = ref[["pressure", "temperature", "H2O", "CO2"] + list(properties.keys())]

        try:
            xr.testing.assert_allclose(results, ref, rtol=1e-5)
        except AssertionError as e:
            for prop in properties.keys():
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
        results.to_netcdf(ref_properties_pt_cubic, engine='h5netcdf')


def test_properties_pt_hybrid(flash_obj_hybrid, ref_properties_pt_hybrid):
    state_spec = {"pressure": np.array([1., 2., 5., 10., 50., 100., 200., 400.]),
                  "temperature": np.arange(273.15, 373.15, 10)
                  }
    zrange = np.concatenate([np.array([1e-12, 1e-10, 1e-8]),
                             np.linspace(1e-6, 1. - 1e-6, 10),
                             np.array([1. - 1e-8, 1. - 1e-10, 1. - 1e-12])])
    compositions = {"H2O": zrange,
                    "CO2": zrange,
                    "C1": 1.
                    }
    properties = {"S": EoS.Property.ENTROPY,
                  "G": EoS.Property.GIBBS,
                  "H": EoS.Property.ENTHALPY,
                  }

    results = flash_obj_hybrid.evaluate_properties_1p(state_spec=state_spec, compositions=compositions,
                                                      properties_to_evaluate=properties, mix=True,
                                                      mole_fractions=True)
    results = results[["pressure", "temperature", "H2O", "CO2"] + list(properties.keys())]

    if 1:
        # Use assertions from xarray
        ref = xr.open_dataset(ref_properties_pt_hybrid, engine='h5netcdf')
        ref = ref[["pressure", "temperature", "H2O", "CO2"] + list(properties.keys())]

        try:
            xr.testing.assert_allclose(results, ref, atol=1e-5)
        except AssertionError as e:
            for prop in properties.keys():
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
        results.to_netcdf(ref_properties_pt_hybrid, engine='h5netcdf')


def test_properties_vt(flash_obj_cubic, ref_properties_vt):
    pr = flash_obj_cubic.eos["PR"]
    vmax = pr.V(p=1., T=273.15, n=[1e-12, 1e-12, 1.-2e-12])
    vmin = pr.V(p=100., T=273.15, n=[1.-2e-12, 1e-12, 1e-12])
    state_spec = {"volume": np.linspace(vmin, vmax, 10),
                  "temperature": np.arange(273.15, 373.15, 10)
                  }
    zrange = np.concatenate([np.array([1e-12, 1e-10, 1e-8]),
                             np.linspace(1e-6, 1. - 1e-6, 10),
                             np.array([1. - 1e-8, 1. - 1e-10, 1. - 1e-12])])
    compositions = {"H2O": zrange,
                    "CO2": zrange,
                    "C1": 1.
                    }

    eos = flash_obj_cubic.eos["PR"]
    properties = {"P": eos.P}

    results = flash_obj_cubic.evaluate_phase_properties_1p(state_spec=state_spec, compositions=compositions,
                                                           properties_to_evaluate=properties, mole_fractions=True)
    results = results[["volume", "temperature", "H2O", "CO2"] + list(properties.keys())]

    if 1:
        # Use assertions from xarray
        ref = xr.open_dataset(ref_properties_vt, engine='h5netcdf')
        ref = ref[["volume", "temperature", "H2O", "CO2"] + list(properties.keys())]

        try:
            xr.testing.assert_allclose(results, ref, rtol=1e-5)
        except AssertionError as e:
            for prop in properties.keys():
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
        results.to_netcdf(ref_properties_vt, engine='h5netcdf')
