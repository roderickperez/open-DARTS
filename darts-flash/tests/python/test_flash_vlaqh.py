import numpy as np
import xarray as xr
import pytest

from dartsflash.components import CompData
from dartsflash.dartsflash import DARTSFlash, R, EoS, FlashParams
from dartsflash.mixtures import VLAqH
from .conftest import flash_types, suffix


@pytest.fixture
def ref0():
    return "./tests/python/ref/flash_vlaqh/"


# Test of VLAq implementations of 1-component H2O-flash in PT-PH-PS
@pytest.fixture()
def flash_obj_h2o_co2():
    f = VLAqH(comp_data=CompData(components=["H2O", "CO2"], setprops=True), hybrid=True)

    trial_comps = [i for i in range(f.comp_data.nc)]
    root_order = [EoS.RootFlag.MAX, EoS.RootFlag.MIN]
    f.set_vl_eos("SRK", trial_comps=trial_comps, root_order=root_order,
                 switch_tol=1e-1, stability_tol=1e-20, max_iter=50, use_gmix=False)
    f.set_aq_eos("Aq", switch_tol=1e-2, stability_tol=1e-16, max_iter=10, use_gmix=True)
    f.set_h_eos("sI", stability_tol=1e-20, switch_tol=1e2, max_iter=20, use_gmix=True)

    return f


@pytest.fixture()
def ref_h2o_co2(ref0):
    return ref0 + "h2o_co2"


def test_vlaqh_h2o_co2(flash_obj_h2o_co2, ref_h2o_co2):
    # Test PT and PH flashes
    dz = 0.1
    min_z = [0.]
    max_z = [1.]

    Xrange = np.array([[273.15, 293.15], [-4000. * R, -2000. * R]])
    for i in range(2):
        flash_obj_h2o_co2.init_flash(flash_type=flash_types[i], eos_order=["Aq", "VL", "sI"],
                                     stability_variables=FlashParams.alpha, split_variables=FlashParams.nik,
                                     split_tol=1e-20, split_switch_tol=1e1, tpd_tol=1e-11, tpd_close_to_boundary=1e-2,
                                     t_min=270., t_max=300., t_tol=1e-1
                                     )

        # Tx
        state_spec = {"pressure": 100.,
                      flash_obj_h2o_co2.state_vars[1]: np.linspace(Xrange[i, 0], Xrange[i, 1], 10)
                      }
        compositions = {comp: np.arange(min_z[ii], max_z[ii] + 0.1 * dz, dz)
                        for ii, comp in enumerate(flash_obj_h2o_co2.components[:-1])}
        compositions[flash_obj_h2o_co2.components[-1]] = 1.
        props = ["nu", "X", "temp"]

        results = flash_obj_h2o_co2.evaluate_flash(state_spec=state_spec, compositions=compositions, mole_fractions=True,
                                                   initialize_with_previous_results=False)

        # Plot results
        if 0:
            import os
            filedir = "./tests/python/fig/flash_vlaqh/"
            os.makedirs(os.path.dirname(filedir), exist_ok=True)

            from dartsflash.plot import PlotFlash, plt
            y_vars = ["temperature", "enthalpy", "entropy"]

            plot_res = PlotFlash.compositional(flash_obj_h2o_co2, results, y_var=y_vars[i], state=state_spec,
                                               variable_comp_idx=0, dz=dz,
                                               # min_temp=Trange[0], max_temp=Trange[1],
                                               min_val=0., max_val=1.,
                                               plot_phase_fractions=False,  # logy=False
                                               )

            plt.savefig(filedir + "h2o_co2" + suffix[i] + "-result.pdf")

        results = results[flash_obj_h2o_co2.state_vars + ["H2O"] + props]

        if 1:
            # Use assertions from xarray
            ref = xr.open_dataset(ref_h2o_co2 + suffix[i] + ".h5", engine='h5netcdf')
            ref = ref[flash_obj_h2o_co2.state_vars + ["H2O"] + props]

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
            results.to_netcdf(ref_h2o_co2 + suffix[i] + ".h5", engine='h5netcdf')
            plt.savefig(filedir + "h2o_co2" + suffix[i] + "-ref.pdf")
