import numpy as np
import pytest

import dartsflash.libflash
from dartsflash.dartsflash import DARTSFlash
from .conftest import compdata, compdata_ions, flash_params, flash_obj, cubic_eos


# Tests of class CompData
def test_compdata(compdata, compdata_ions):
    assert isinstance(compdata, dartsflash.libflash.CompData)
    assert compdata.nc == 3
    assert compdata.ni == 0
    assert compdata.ns == 3

    assert isinstance(compdata_ions, dartsflash.libflash.CompData)
    assert compdata_ions.nc == 3
    assert compdata_ions.ni == 2
    assert compdata_ions.ns == 5


def test_flashparams_default(flash_params, flash_obj, cubic_eos):
    assert isinstance(flash_params, dartsflash.libflash.FlashParams)

    # Check if DARTSFlash.add_eos() default parameters are the same as EoSParams defaults
    flash_obj.add_eos("CEOS", cubic_eos)
    eos_params1 = flash_obj.flash_params.eos_params["CEOS"]
    flash_params.add_eos("CEOS", cubic_eos)
    eos_params2 = flash_params.eos_params["CEOS"]

    assert eos_params1.stability_tol == pytest.approx(eos_params2.stability_tol)
    assert eos_params1.stability_switch_tol == pytest.approx(eos_params2.stability_switch_tol)
    assert eos_params1.stability_switch_diff == pytest.approx(eos_params2.stability_switch_diff)
    assert eos_params1.stability_line_tol == pytest.approx(eos_params2.stability_line_tol)
    assert eos_params1.stability_max_iter == pytest.approx(eos_params2.stability_max_iter)
    assert eos_params1.stability_line_iter == pytest.approx(eos_params2.stability_line_iter)
    assert eos_params1.use_gmix == eos_params2.use_gmix

    # Check if DARTSFlash.init_flash() default parameters are the same as FlashParams defaults
    flash_obj.init_flash(eos_order=["CEOS"], flash_type=DARTSFlash.FlashType.PTFlash)
    assert flash_obj.flash_params.min_z == pytest.approx(flash_params.min_z)
    assert flash_obj.flash_params.y_pure == pytest.approx(flash_params.y_pure)
    assert flash_obj.flash_params.tpd_tol == pytest.approx(flash_params.tpd_tol)
    assert flash_obj.flash_params.tpd_1p_tol == pytest.approx(flash_params.tpd_1p_tol)
    assert flash_obj.flash_params.tpd_close_to_boundary == pytest.approx(flash_params.tpd_close_to_boundary)
    assert flash_obj.flash_params.comp_tol == pytest.approx(flash_params.comp_tol)

    assert flash_obj.flash_params.rr2_tol == pytest.approx(flash_params.rr2_tol)
    assert flash_obj.flash_params.rrn_tol == pytest.approx(flash_params.rrn_tol)
    assert flash_obj.flash_params.rr_max_iter == flash_params.rr_max_iter
    # assert flash_obj.flash_params.rr_line_iter == flash_params.rr_line_iter

    assert flash_obj.flash_params.split_tol == pytest.approx(flash_params.split_tol)
    assert flash_obj.flash_params.split_switch_tol == pytest.approx(flash_params.split_switch_tol)
    assert flash_obj.flash_params.split_switch_diff == pytest.approx(flash_params.split_switch_diff)
    assert flash_obj.flash_params.split_line_tol == pytest.approx(flash_params.split_line_tol)
    assert flash_obj.flash_params.split_max_iter == flash_params.split_max_iter
    assert flash_obj.flash_params.split_line_iter == flash_params.split_line_iter
    assert flash_obj.flash_params.split_negative_flash_iter == flash_params.split_negative_flash_iter
    # assert flash_obj.flash_params.split_negative_flash_tol == pytest.approx(flash_params.split_negative_flash_tol)

    assert flash_obj.flash_params.split_variables == flash_params.split_variables
    assert flash_obj.flash_params.modChol_split == flash_params.modChol_split
    assert flash_obj.flash_params.stability_variables == flash_params.stability_variables
    assert flash_obj.flash_params.modChol_stability == flash_params.modChol_stability

    assert flash_obj.flash_params.pxflash_type == flash_params.pxflash_type
    assert flash_obj.flash_params.pxflash_Ftol == pytest.approx(flash_params.pxflash_Ftol)
    assert flash_obj.flash_params.pxflash_Ttol == pytest.approx(flash_params.pxflash_Ttol)
    assert flash_obj.flash_params.phase_boundary_Gtol == pytest.approx(flash_params.phase_boundary_Gtol)
    assert flash_obj.flash_params.phase_boundary_Ttol == pytest.approx(flash_params.phase_boundary_Ttol)

    assert flash_obj.flash_params.T_min == pytest.approx(flash_params.T_min)
    assert flash_obj.flash_params.T_max == pytest.approx(flash_params.T_max)
    assert flash_obj.flash_params.T_init == pytest.approx(flash_params.T_init)

    assert flash_obj.flash_params.save_performance_data == pytest.approx(flash_params.save_performance_data)
    assert flash_obj.flash_params.verbose == pytest.approx(flash_params.verbose)
    