# What is DARTS?

**Delft:** 
    we belong to Civil Engineering and Geoscience (CEG) Department at Civil Engineering Faculty of TU Delft. The 
    development team directly linked to the new GeoEnergy program which connects Geology, Geophysics and Petroleum 
    Engineering sections of the department.

**Advanced:** 
    the simulation framework is based on the recently proposed Operator-Based Linearization (OBL) approach, 
    which helps to decouple the complex nonlinear physics and advanced unstructured discretization from the 
    core simulation engine. The framework is targeting the solution of forward and inverse problems.

**Research:**
    the development team includes five PhD students from the CEG department and multiple MSc students 
    working on their thesis project on DARTS platform. The simulation platform is developed within 
    Delft Advanced Reservoir Simulation (DARSim) program and linked to multiple research in the area of 
    reservoir simulation, inverse modeling and uncertainty quantification.

**Terra:**
    the developed framework is utilized for forward and inverse problems in petroleum engineering, 
    low- and high-enthalpy geothermal applications, subsurface storage and subsurface integrity. 
    The primary focus and developed capabilities are currently cover low-enthalpy geothermal operations which 
    include multicomponent multiphase flow of mass and heat with complex chemical interactions. 
    Another focus is thermal-compositional processes for Enhanced Oil Recovery.

**Simulator:**
    the main simulation kernel implemented on multi-core CPU and many-core GPU architectures. 
    Advance multiscale nonlinear formulation improves the performance of forward and inverse models. 
    Additional afford invested in representative proxy models for complex subsurface processes including
    multiphase multicomponent flow.



DARTS is constructed within the Operator-based Linearization (OBL) framework for general-purpose reservoir simulations. The 'super-engine' in DARTS contains the governing equations characterizing the general thermal-compositional-reactive system should be expressed in form of operators, which are the functions of the thermodynamic state. These operators are then utilized to construct the residual and Jacobian for the solution of a nonlinear system using interpolation in thermodynamic parameter space. The values and derivatives of these operators at the supporting points will be evaluated adaptively during the simulation or pre-calculated in form of tables  through the physics implemented in either C++ or Python.

The assembly of resulting Jacobian matrix in C++ is generalized and improved by the OBL approach, which greatly facilitates code development. The constructed linear system is passed to the dedicated linear solver for a solution. More details about the DARTS architecture are described in [1].

To enable the convenient usage of DARTS, the functionalities in C++ are exposed to users via a Python interface. At the same time, the Python interface provides the possibility to integrate different external modules (e.g., packages for physical property calculation) into DARTS. 


\section{Conservation Equations}

Mass and heat transfer involves a thermal multiphase flow system, which requires a set of equations to depict the flow dynamics. In this section, the governing equations and detailed spatial and temporal discretization and linearization procedures are introduced.

\subsection{Governing Equations}

For the investigated domain with volume $\Omega$, bounded by surface $\Gamma$, the mass and energy conservation can be expressed in a uniformly integral way, as
%
\begin{equation}\indent
\frac{\partial}{\partial{t}} \int_{\Omega}{M^c}d{\Omega} + \int_{\Gamma}{\bm{F}^c\bm{\cdot}{\bm{n}}}d{\Gamma} = \int_{\Omega}{Q^c}d{\Omega}.
\end{equation}
%
Here, $M^c$ denotes the accumulation term for the $c^{\mathrm{th}}$ component ($c = 1, \ldots, n_c$, indexing for the mass components, [e.g., water, $\mathrm{CO_2}$] and $c = n_c + 1$ for the energy quantity); $\bm{F}_c$ refers to the flux term of the $c^{\mathrm{th}}$ component; ${\bm{n}}$ refers to the unit normal pointing outward to the domain boundary;
$Q_c$ denotes the source/sink term of the $c^{\mathrm{th}}$ component.

The mass accumulation term collects each component distribution over $n_p$ fluid phases in a summation form, 
%
\begin{equation} \indent
    \begin{aligned}
        M^c = \phi\sum\limits^{n_p}_{j=1}x_{cj}\rho_js_j + (1-\phi), \quad c = 1, \ldots, n_c,
    \end{aligned}
\end{equation}
%
where $\phi$ is porosity, $s_j$ is phase saturation, $\rho_j$ is phase density {$[\mathrm{kmol/m^3}]$} and $x_{cj}$ is molar fraction of $c$ component in $j$ phase.

The energy accumulation term contains the internal energy of fluid and rock,
%
\begin{equation} \indent
    \begin{aligned}
        M^{n_c+1} = \phi\sum\limits^{n_p}_{j=1}\rho_js_jU_j + (1 - \phi)U_r,
    \end{aligned}
\end{equation}
%
where $U_j$ is phase internal energy {$[\mathrm{kJ}]$} and $U_r$ is rock internal energy {$[\mathrm{kJ}]$}.
%
The rock is assumed compressible and represented by the change of porosity through:
%
\begin{equation} \indent
    \phi = \phi_0 \big(1 + c_r (p - p_\mathrm{ref}) \big),
\end{equation}
%
where $\phi_0$ is the initial porosity, $c_r$ is the rock compressibility {[1/bar]} and $p_\mathrm{ref}$ is the reference pressure {[bars]}.

The mass flux of each component is represented by the summation over $n_p$ fluid phases,
%
\begin{equation} \indent
    \begin{aligned}
        \bm{F}^c = \sum\limits_{j=1}^{n_p}x_{cj}\rho_j \bm{u_j} + s_{j}\rho_{j} \textbf{J}_{cj}, \quad c = 1, \ldots, n_c.
    \end{aligned}
\end{equation}
%
Here the velocity $\bm{u_j}$ follows the extension of Darcy's law to multiphase flow,
%
\begin{equation} \indent
    \small
    \bm{u_j} = \mathbf{K}\frac{k_{rj}}{\mu_j}(\nabla{p_j}-\bm{\gamma_j}\nabla{z}),
\end{equation}
%
where $\mathbf{K}$ is the permeability tensor {$[\mathrm{mD}]$}, $k_{rj}$ is the relative permeability of phase $j$, $\mu_j$ is the viscosity of phase $j$ {$[\mathrm{mPa\cdot s}]$}, $p_j$ is the pressure of phase $j$ {[bars]}, $\bm{\gamma_j}=\rho_j\bm{g}$ is the specific weight {$[\mathrm{N/m^3}]$} and $z$ is the depth vector {[m]}.
%
The $\textbf{J}_{cj}$ is the diffusion flux of component $c$ in phase $j$, which is described by Fick's law as
\begin{equation}
\textbf{J}_{cj} = - \phi \textbf{D}_{cj} \nabla x_{cj},
\label{eq: diffusion equation}
\end{equation}
where $\textbf{D}_{cj}$ is the diffusion coefficient {[m$^2$/day]}.
 

The energy flux includes the thermal convection and conduction terms, \useshortskip
%
\begin{equation} \indent
    \begin{aligned}
        \bm{F}^{n_c+1} = \sum\limits^{n_p}_{j=1}h_j\rho_j \bm{u_j} + \kappa\nabla{T},
    \end{aligned}
\end{equation}
%
where $h_j$ is phase enthalpy {$[\mathrm{kJ/kg}]$} and $\kappa$ is effective thermal conductivity {$[\mathrm{kJ/m/day/K}]$}.
%
% The saturation constraint is required to close the system:
% %
% \begin{equation} \indent
%     \small
%     \sum\limits^{n_p}_{p=1}s_p=1.
% \end{equation}

Finally, the source term in mass conservation equations can be present in the following form
%
\begin{equation} \indent
    \begin{aligned}
        {Q}^{c} = \sum\limits_{k=1}^{n_k}v_{ck}r_k, \quad c = 1, \ldots, n_c,
    \end{aligned}
\end{equation}
%
where $q_j$ is the phase source/sink term from the well, $v_{ck}$ is the stoichiometric coefficient associated with chemical reaction $k$ for the component $c$ and $r_{k}$ is the rate for the reaction. %Here we assume that equilibrium reactions are absent. 
Similarly, the source term in the energy balance equation can be written as
%
\begin{equation} \indent
    \begin{aligned}
        {Q}^{n_c+1} = \sum\limits_{k=1}^{n_k}v_{ek}r_{ek}.
    \end{aligned}
\end{equation}
%
Here $v_{ek}$ is the stoichiometric coefficient associated with kinetic reaction $k$ for the energy and $r_{ek}$ is the energy rate for kinetic reaction.

The nonlinear equations are discretized with the finite volume method using the multi-point flux approximation on general unstructured mesh in space and with the backward Euler approximation in time. For the $i^{\mathrm{th}}$ reservoir block, the governing equation in discretized residual form reads:
%
\begin{equation} \indent
    \begin{aligned}
        R^c_i = V_i \Big(M^{c}_i(\omega_i) - M^{c}_i(\omega^n_{i}) \Big) - 
        \Delta{t} \Big(\sum_l{A_{l}F^{c}_{l}(\omega)} + V_iQ^{c}_{i}(\omega) \Big) = 0, \quad c = 1, \ldots, n_c+1.
    \end{aligned}
\end{equation}
%
Here $V_i$ is the volume of the $i^{th}$ grid block, $\omega_{i}$ refers to state variables at the current time step, $\omega^{n}_i$ refers to state variables at previous time step, $A_l$ is the contact area between neighboring grids.
