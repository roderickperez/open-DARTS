# DARTS-flash

[![Documentation Status](https://readthedocs.org/projects/darts-flash/badge/?version=latest)](https://darts-flash.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12686020.svg)](https://doi.org/10.5281/zenodo.12686020)
[![Latest Release](https://gitlab.com/open-darts/darts-flash/-/badges/release.svg)](https://gitlab.com/open-darts/darts-flash/-/releases)
[![pipeline status](https://gitlab.com/open-darts/darts-flash/badges/development/pipeline.svg)](https://gitlab.com/open-darts/darts-flash/-/commits/development)
[![pypi](https://img.shields.io/pypi/v/open-darts-flash.svg?colorB=blue)](https://pypi.python.org/project/open-darts-flash/)
<!-- [![RSD](https://img.shields.io/badge/rsd-openDARTSflash-00a3e3.svg)](https://research-software-directory.org/software/opendartsflash) -->

DARTS-flash is a standalone library for performing multiphase equilibrium and thermodynamic properties calculations from a range of thermodynamic models. DARTS-flash has interfaces in C++ and Python and it depends on `Eigen` and `Pybind11` libraries.

## Features
DARTS-flash has been developed primarily for simulation of flow and transport in CO2-sequestration and geothermal-related subsurface applications. The robustness and accuracy of thermodynamic modelling routines determine the robustness of the compositional simulation.

- Thermodynamic models
    - Helmholtz-form EoS: Cubic, CPA, IAPWS-95 EoS
    - Activity models for aqueous phase
    - Van der Waals-Platteeuw hydrate EoS
    - Solid phase EoS
- Stability test and multiphase split
    - Hybrid-EoS implementation
    - Newton methods for second-order convergence
    - Choice of variables, line search procedures, modified Cholesky decomposition
    - (coming soon) (Augmented) free-water flash methods
- Solution strategies for multiphase equilibrium
    - Two-phase negative flash
    - N-phase stability-flash
    - Multiphase flash at PT/PH/PS state specification
- C++ interface
    - Partial derivatives for simulation
- Python interface
    - Phase diagrams
    - EoS property evaluation
    - Gibbs energy and tangent plane analysis
    - Hydrate equilibrium curves
    - (coming soon) PVT experiments

## Installation

### Via pip

DARTS-flash is available for Python 3.8 to 3.14.

```shell
pip install open-darts-flash
```

### Building dartsflash
The package can be built and installed by executing the build scripts (see [building darts-flash](https://gitlab.com/open-darts/darts-flash/-/wikis/Build-instructions)). 

For Linux and macOS:
```shell
./helper_scripts/build.sh
```
and for Windows
```shell
./helper_scripts/build.bat
```
Call the build script with the `-h` option to display the help menu.

## For developers

Check our [wiki](https://gitlab.com/open-darts/darts-flash/-/wikis/home) and the section on [how to contribute](https://gitlab.com/open-darts/darts-flash/-/wikis/Contributing).

## Citing DARTS-flash

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12686020.svg)](https://doi.org/10.5281/zenodo.12686020)
