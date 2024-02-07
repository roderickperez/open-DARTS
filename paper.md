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
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1
  - name: Mark Khait
    affiliation: 1
  - name: Aleks Novikov
    affiliation: 1
  - name: Michiel Wapperom
    affiliation: 1
  - name: Ilshat Saifullin
    affiliation: 1
  - name: Xiaoming Tian
    affiliation: 1
  - name: Luisa Orozco
    orcid: 0000-0002-9153-650X
    affiliation: 3
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
date: 1 February 2024
bibliography: paper.bib
---
> Hard maximum for entire paper: 1000 words
# Summary

> Describe the high-level functionality and purpose of the software for a diverse, non-specialist audience.

Open Delft Advanced Reserach Terra Simulator [@openDARTS_2023] is a simulation framework for forward and inverse modelling and uncertainty quantification of multi-physics processes in geo-engineering applications such as geothermal, CO2 sequestration, water pumping, and hydrogen storage. OpenDARTS has a hybrid design combining C++ and Python code. It utilizes advanced numerical methods such as fully implicit thermo-hydro-mechanical-chemical formulation, highly flexible finite-volume spatial approximation, operator-based linearization for nonlinear and physics-based preconditioning for linear solutions. 


# Statement of need

> Illustrates the research purpose of the software and places it in the context of related work (other software packages doing similar simulations with references and how is DARTS special. Also list here other software packages that are pertinent to DARTS, for example dependencies, darts-flash?).

The openDARTS framework is fully validated and benchmarked for geothermal, CO2 sequestration, gas storage, hydrocarbon production and induced seismicity applications. The framework design and parallel implementations for CPU and GPU architectures provides an exceptional level of flexibility and performance: up to two orders of magnitude faster than the best academic and commercial software analogues. Advanced inverse capabilities based on adjoint gradients allowed to effectively address data assimilation, risk analisys and uncertainty quantification for energy transition applications.

# Key features


# Acknowledgements

The authors would like to acknowledge the Netherlands eScience Center for the funding provided under grant number NLESC.OEC.2021.026.

# References
