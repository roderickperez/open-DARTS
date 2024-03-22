---
title: 'openDARTS: open Delft Advanced Research Terra Simulator'
tags:
  - energy transition
  - geothermal
  - CO2 sequestration
  - multi-physics
  - geo-engineering
authors:
  - name: Denis Voskov
    orcid: 0000-0002-5399-1755
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Xiaocong Lyu
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Stephan de Hoop
    affiliation: 1
  - name: Mark Khait
    equal-contrib: true
    affiliation: 1
  - name: Aleks Novikov
    affiliation: 1
  - name: Michiel Wapperom
    affiliation: 1
    orcid: 0000-0003-3432-4233
    equal-contrib: true
  - name: Ilshat Saifullin
    orcid: 0009-0001-0089-8629 
    affiliation: 1
  - name: Xiaoming Tian
    affiliation: "5, 1"
    orcid: 0000-0003-0642-6064
    equal-contrib: true
  - name: Luisa Orozco
    orcid: 0000-0002-9153-650X
    affiliation: 3
    corresponding: true # (This is how to denote the corresponding author)
  - name: Artur Palha
    orcid: 0000-0002-3217-0747
    affiliation: 4
affiliations:
 - name: Department of Geoscience and Engineering, TU Delft, Delft, Netherlands
   index: 1
 - name: Energy Science and Engineering Department, Stanford, CA, USA
   index: 2
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 3
 - name: Delft Institute of Applied Mathematics, Delft University of Technology
   index: 4
 - name: Guangzhou Institute of Energy Conversion, Chinese Academy of Sciences, Guangzhou, China
   index: 5
date: 1 April 2024
bibliography: paper.bib
---
> Hard maximum for entire paper: 1000 words
# Summary

> Describe the high-level functionality and purpose of the software for a diverse, non-specialist audience.

Open Delft Advanced Research Terra Simulator [@openDARTS_2023] is a simulation framework for forward and inverse modelling and 
uncertainty quantification of multi-physics processes in geo-engineering applications such as geothermal, CO2 sequestration, 
water pumping, and hydrogen storage. To efficiently achieve high levels of accuracy on complex geometries, it utilizes advanced numerical methods such as fully implicit thermo-hydro-mechanical-chemical formulation, a highly flexible finite-volume spatial approximation, operator-based linearization for nonlinear terms, and efficient physics-based preconditioners. openDARTS goals are computational efficiency, expandability, and easiness of use. For this reason, openDARTS is based on a hybrid design with an efficient core C++ implementation wrapped around a highly customisable and easy to use Python code. 


# Statement of need

> Illustrates the research purpose of the software and places it in the context of related work (other software packages doing similar simulations with references and how is DARTS special. Also list here other software packages that are pertinent to DARTS, for example dependencies, darts-flash?).

The openDARTS framework is fully validated and benchmarked for geothermal, CO2 sequestration, gas storage, hydrocarbon production and induced seismicity applications. 
The framework design and parallel implementations for CPU and GPU architectures provides an exceptional level of flexibility and performance. Furthermore, advanced inverse capabilities based on adjoint gradients allows openDARTS to effectively address data assimilation, risk analysis and uncertainty quantification for energy transition applications.

openDARTS is designed to use Python as its user interface, which makes it widely used in educational and research institutions for both introductory and advanced programming. 
It is a reservoir simulator with advanced capabilities that is not reliant on proprietary software, 
reducing significantly the entry barrier for researchers and students interested in energy transition applications for the subsurface.

# Key features

## Unified thermal-compositional PDE formulation

openDARTS has a generic PDE formulation for thermal compositional flow in porous media.
This makes possible to adjust terms in PDE to account for various physical effects.
Darcy flow, gravity and capillary effects give rise to convective fluxes and 
diffusive/conductive fluxes are driven by thermodynamic potentials between grid cells.  ## (chemical potential/entropy)
In addition, a source/sink term can account for chemistry and kinetic reactions.

Observing how the conservation equations for mass and energy contain similar terms, 
one can formulate the conservation of each quantity in a control volume in a uniformly integral way.
The nonlinear equations are discretized using a Finite Volume Method in space and with a backward Euler approximation in time.

## Operator-Based Linearization
One of the most computationally expensive parts is a calculation of derivatives to construct the Jacobian. 
openDARTS exploits Operator-Based Linearization (OBL), where the terms in the PDEs are separated into space-dependent terms $\xi$ and thermodynamic state-dependent operators $\omega$.
The $\omega$ operators can be parameterized with respect to the nonlinear unknowns in multidimensional tables under different resolutions.
The values and derivatives required for assembly of the linear system can be approximated through multi-linear interpolation in the primary variables space using calculated values at the nodes.

Using adaptive parametrization, derivative computation is performed at nodes of the structured grid in the primary variables space around the required point.
Re-using computed values at nodal points can significantly reduce the Jacobian construction stage, especially in case of ensemble-based simulations.

## Discretization

Different grid types supported by openDARTS are useful for different applications: 
- structured grid - for teaching
- radial grid?
- corner-point geometry - for industry-related applications
- unstructured grid - for modelling of flow with complex geometries, discrete fracture networks and core scale laboratory experiments

openDARTS uses Finite Volume Method for space and Fully Implicit Method for time discretization.
There are two-point and multi-point flux approximations implemented in openDARTS.

## HPC

The most computationally expensive part of openDARTS is written in C++.
This allows the user to use parallelization using openMP for multi-core systems and GPU acceleration using NVIDIA CUDA.

## Python

openDARTS can be installed as a Python module and it has a Python-based interface. 
There are several benefits of that compared to a fully C++ code.

- Easy installation via pip and PyPI.
- No need to install compilers to work with openDARTS.
- Flexible implementation of simulation framework, physical modelling and grids.
- Easy data visualization, including internal arrays and vtk.
- Use popular python modules within openDARTS and user's model for data processing and input/output.
- Coupling with other Python-based numerical modeling software.
- The time-step loop is written in Python, so one can adjust it to specific needs.

This makes openDARTS attractive for teaching and for users unfamiliar with C++ language.

## Inverse modelling
Inverse modeling can be notably time-consuming, particularly when employing gradient-based methods. 
The implementation of the adjoint method in openDARTS significantly enhances its efficiency in computing the required gradients for inverse modeling or history matching processes. 
Moreover, the inverse modeling module of openDARTS accommodates various types of observation data. 
These observations includes well rates, well temperatures, BHP, time-lapse temperature distributions, and any custom outputs definable in the form of operators within openDARTS.

# Contributions ?


# Acknowledgements

The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.OEC.2021.026.

# References
