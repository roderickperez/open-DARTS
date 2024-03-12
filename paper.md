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
  - name: Ilshat Saifullin
    orcid: 0009-0001-0089-8629 
    affiliation: 1
  - name: Xiaoming Tian
    affiliation: 1
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
date: 1 April 2024
bibliography: paper.bib
---
> Hard maximum for entire paper: 1000 words
# Summary

> Describe the high-level functionality and purpose of the software for a diverse, non-specialist audience.

Open Delft Advanced Research Terra Simulator [@openDARTS_2023] is a simulation framework for forward and inverse modelling and uncertainty quantification of multi-physics processes in geo-engineering applications such as geothermal, CO2 sequestration, water pumping, and hydrogen storage. openDARTS has a hybrid design combining C++ and Python code. It utilizes advanced numerical methods such as fully implicit thermo-hydro-mechanical-chemical formulation, highly flexible finite-volume spatial approximation, operator-based linearization for nonlinear and physics-based preconditioning for linear solutions.

# Statement of need

> Illustrates the research purpose of the software and places it in the context of related work (other software packages doing similar simulations with references and how is DARTS special. Also list here other software packages that are pertinent to DARTS, for example dependencies, darts-flash?).

The openDARTS framework is fully validated and benchmarked for geothermal, CO2 sequestration, gas storage, hydrocarbon production and induced seismicity applications. The framework design and parallel implementations for CPU and GPU architectures provides an exceptional level of flexibility and performance: up to two orders of magnitude faster than the best academic and commercial software analogues.
> @Luisa: Should we add some references here?

Advanced inverse capabilities based on adjoint gradients allowed to effectively address data assimilation, risk analysis and uncertainty quantification for energy transition applications.

openDARTS is designed to use Python as its user interface, which makes it widely used in educational and research institutions for both introductory and advanced programming. It is a reservoir simulators with advanced capabilities that is not reliant on proprietary software, reducing significantly the entry barrier for researchers and students interested in energy transition applications for the subsurface.

# Key features

## Super-engine

openDARTS has a generic PDE formulation for thermal compositional flow in porous media. This makes possible to adjust terms in PDE to account for various physical effects.

## OBL
One of the most computationally expensive parts is a calculation of derivatives to construct the Jacobian. openDARTS exploits Operator-Based Linearization (OBL), where PDE's terms can be approximated through the interpolation in a primary variables space. Using adaptive parametrization, derivative computation is performed at nodes of the structured grid in the primary variables space around the required point. The derivatives at this point are computed via multi-linear interpolation using calculated values at the nodes. Re-using computed values at nodal points can significantly reduce the Jacobian construction stage, especially in case of ensemble-based simulations.

## Discretization

Different grid types supported by openDARTS are useful for different applications: structured grid - for teaching, corner-point geometry - for industry-related applications, and unstructured grid - for flow modeling in discrete fracture networks and core scale laboratory experiments modeling.
openDARTS uses Finite Volume Method for space and Fully Implicit Method for time discretization.
There are two-point and multi-point flux approximations implemented in openDARTS.

## HPC

The most computationally expensive part of openDARTS is written in C++. This allows the user to use parallelization using openMP for multi-core systems and GPU acceleration using NVIDIA CUDA.

## Python

openDARTS can be installed as a python module and it has a python-based interface. There are several benefits of that compared to a fully C++ code.

- Easy installation via pip and PyPI.
- No need to install compilers to work with openDARTS.
- Easy data visualization, including internal arrays and vtk.
- Coupling with other Python-based numerical modeling software.
- Use popular python modules within openDARTS and user's model for data processing and input/output.
- One might re-define relationships between variables using exclusively Python code.
- The time-step loop is written in Python, so one can adjust it fto specific needs.

This makes openDARTS attractive for teaching and for users unfamiliar with C++ language.

## Inverse modelling


# Acknowledgements

The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.OEC.2021.026.

# References
