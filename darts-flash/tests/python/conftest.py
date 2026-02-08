import pytest
from dartsflash.libflash import FlashParams, CubicEoS

from dartsflash.components import CompData
from dartsflash.dartsflash import DARTSFlash


# Define pytest.fixtures: Mocking objects only necessary in the test of this files.
# If one of this is needed in other files, consider defining it in conftest.py

# If you have a function that generate this data and is used in multiple files, you can also put it in conftest.py.
# In that case (function): you may have to import the file as from tests.python.conftest import function_in_conftest

suffix = ["_pt", "_ph", "_ps"]
flash_types = [DARTSFlash.FlashType.PTFlash, DARTSFlash.FlashType.PHFlash, DARTSFlash.FlashType.PSFlash]


@pytest.fixture()
def compdata() -> CompData:
    return CompData(components=["H2O", "CO2", "C1"], setprops=True)


@pytest.fixture()
def compdata_ions() -> CompData:
    return CompData(components=["H2O", "CO2", "C1"], ions=["Na+", "Cl-"], setprops=True)


@pytest.fixture()
def flash_params(compdata) -> FlashParams:
    return FlashParams(compdata)


@pytest.fixture()
def flash_obj(compdata) -> DARTSFlash:
    return DARTSFlash(compdata)


@pytest.fixture()
def cubic_eos(compdata) -> CubicEoS:
    return CubicEoS(compdata, CubicEoS.PR)
