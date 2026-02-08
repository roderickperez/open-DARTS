# DWell

- DWell is a standalone simulator for multi-phase multi-component non-isothermal fluid flow in pipes. 
- DWell is written in the Python programming language. The fluid model and thermodynamic calculations can be done using [DARTS-Flash](https://pypi.org/project/open-darts-flash/), the core of which is written in C++. 
- DWell can be coupled with reservoir simulators to have a coupled wellbore-reservoir model. It is already coupled with the [DARTS](https://gitlab.com/open-darts/open-darts/-/tree/sajjad/dfm_ms_well?ref_type=heads) reservoir simulator in a fully coupled way.


## Installation
- Clone the repository
- Create and activate a conda environment 
- Change the directory of the terminal to the directory of the DWell Python project
- Execute the following command to install DWell and its dependencies:
```shell
pip install .
```
- Enjoy running the examples!

## Notes
- In DWell, indexing of wellbore segments is from bottom (bottom-hole) to top (wellhead), while in DARTS-well, indexing is from top to bottom.
- In DWell, array of primary variables are stored property-wise (e.g., [p_0, p_1, p_2, z_0, z_1, z_2, T_0, T_1, T_2]), while in DARTS-well, it is stored block-wise (e.g., [p_0, z_0, T_0, p_1, z_1, T_1, p_2, z_2, T_2]).
- In DWell, strict SI units are used, while DARTS-well uses a different set of units.
- open-darts-flash==0.10.2 is used for the CI tests.

## Documentation
- Indexing of well segments starts from the well bottom (0) and ends at the wellhead.