import numpy as np
from darts.physics.super.physics import Compositional, timer_node
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, ConstFunc
from darts.physics.properties.flash import ConstantK, IonFlash
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import Flash, PXFlash, NegativeFlash, FlashParams, EoS, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData


class BrineVapour(Compositional):
    def __init__(self, components: list, phases: list, swc: float, vl_phases: bool, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_t: float = None, max_t: float = None,
                 ions: bool = False, zero: float = 1e-12, temperature: float = None, cache: bool = False):
        # Fluid components, ions and solid
        h2o_idx = components.index("H2O")
        if vl_phases:
            phases += ["LCO2"]

        if ions:
            ions = ["Na+", "Cl-"]
            comp_data = CompData(components, ions, setprops=True)
            nc, ni = comp_data.nc, comp_data.ni

            # Combine ions into single component and create species MW list
            ion_comp = ["NaCl"]
            combined_ions = np.array([1, 1])
            species = components + ion_comp
            species_Mw = np.append(comp_data.Mw[:nc],
                                   comp_data.Mw[nc] + comp_data.Mw[nc + 1])  # components + combined ions
        else:
            ions = None
            comp_data = CompData(components, ions, setprops=True)
            nc, ni = comp_data.nc, comp_data.ni

            # Combine ions into single component and create species MW list
            combined_ions = None
            species = components
            species_Mw = comp_data.Mw

        """ Activate physics """
        if temperature is None:
            ph = False  # switch for PH-simulation
            state_spec = Compositional.StateSpecification.PH if ph else Compositional.StateSpecification.PT
        else:
            ph = False  # isothermal case should run PT flash
            state_spec = Compositional.StateSpecification.P

        super().__init__(species, phases, timer, n_points=n_points, min_p=min_p, max_p=max_p, min_z=zero / 10,
                         max_z=1 - zero / 10, min_t=min_t, max_t=max_t, state_spec=state_spec, cache=cache)

        """ Initialize flash """
        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)
        pr.set_preferred_roots(h2o_idx, 0.75, EoS.MAX)
        aq = AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                               AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                               AQEoS.CompType.ion: AQEoS.Jager2003
                               })
        aq.set_eos_range(h2o_idx, [0.6, 1.])

        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]
        flash_params.eos_params["PR"].root_order = [EoS.MAX, EoS.MIN] if vl_phases else [EoS.STABLE]

        # Define initial guesses for stability + flash
        params = flash_params.eos_params["PR"]
        params.initial_guesses = [i for i in range(comp_data.nc)]
        params.stability_tol = 1e-20
        params.stability_switch_tol = 1e-2
        params.stability_max_iter = 50
        params.use_gmix = False

        params = flash_params.eos_params["AQ"]
        params.initial_guesses = [h2o_idx]
        params.stability_max_iter = 10
        params.use_gmix = True

        # Flash-related parameters
        # flash_params.split_switch_tol = 1e-3
        # flash_params.split_tol = 1e-14
        flash_params.comp_tol = 1e-2
        # flash_params.verbose = True

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases, species, Mw=species_Mw, min_z=zero, temperature=temperature)

        # flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
        flash_ev = PXFlash(flash_params, PXFlash.ENTHALPY) if ph else Flash(flash_params)
        if ions is not None:
            property_container.flash_ev = IonFlash(flash_ev, nph=2, nc=nc, ni=ni, combined_ions=combined_ions)
        else:
            property_container.flash_ev = flash_ev

        property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=species_Mw)),
                                              ('LCO2', EoSDensity(eos=pr, Mw=species_Mw)),
                                              ('Aq', Garcia2001(components, ions, combined_ions)), ])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('LCO2', Fenghour1998()),
                                                ('Aq', Islam2012(components, ions, combined_ions)), ])

        diff = 8.64e-6
        property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(len(species)) * diff)),
                                                ('LCO2', ConstFunc(np.ones(len(species)) * diff)),
                                                ('Aq', ConstFunc(np.ones(len(species)) * diff * 1e-3))])

        if self.thermal:
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('LCO2', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])

            property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                       ('LCO2', ConstFunc(10.)),
                                                       ('Aq', ConstFunc(180.)), ])

        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('LCO2', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ('Aq', PhaseRelPerm("wat", swc=swc, sgr=swc, n=4))])

        """ Add property region """
        self.add_property_region(property_container)
        property_container.output_props = {}

        property_container.output_props['temperature'] = lambda: property_container.temperature
        property_container.output_props['rho_g'] = lambda: np.sum(property_container.sat[1:] * property_container.dens[1:])
        for j, ph in enumerate(phases):
            property_container.output_props['sat' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['rhom' + ph] = lambda jj=j: property_container.dens_m[jj]
            property_container.output_props['enth' + ph] = lambda jj=j: property_container.enthalpy[jj]
            for i, comp in enumerate(species):
                property_container.output_props[comp + '_' + ph] = lambda jj=j, ii=i: property_container.x[jj, ii]


class VapourLiquidCO2(Compositional):
    def __init__(self, components: list, swc: float, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_t: float = None, max_t: float = None,
                 zero: float = 1e-12, temperature: float = None, cache: bool = False):
        # Fluid components
        comp_data = CompData(components, setprops=True)
        nc = comp_data.nc
        phases = ["V", "LCO2", "L"]

        """ Activate physics """
        if temperature is None:
            ph = False  # switch for PH-simulation
            state_spec = Compositional.StateSpecification.PH if ph else Compositional.StateSpecification.PT
        else:
            ph = False  # isothermal case should run PT flash
            state_spec = Compositional.StateSpecification.P
        super().__init__(components, phases, timer, n_points=n_points, min_p=min_p, max_p=max_p, min_z=zero / 10,
                         max_z=1 - zero / 10, min_t=min_t, max_t=max_t, state_spec=state_spec, cache=cache)

        """ Initialize flash """
        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)

        flash_params.add_eos("PR", pr)
        flash_params.eos_order = ["PR"]
        flash_params.eos_params["PR"].root_order = [EoS.MAX, EoS.MIN]
        flash_params.eos_params["PR"].rich_phase_order = [components.index("CO2"), -1]

        # Define initial guesses for stability + flash
        params = flash_params.eos_params["PR"]
        params.initial_guesses = [i for i in range(comp_data.nc)]
        params.stability_tol = 1e-20
        params.stability_switch_tol = 1e-2
        params.stability_max_iter = 50
        # params.use_gmix = True

        # Flash-related parameters
        flash_params.split_variables = FlashParams.SplitVars.lnK
        flash_params.stability_variables = FlashParams.StabilityVars.alpha
        flash_params.split_switch_tol = 1e-1
        flash_params.split_tol = 1e-14
        flash_params.comp_tol = 1e-2
        # flash_params.verbose = True

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases, components, Mw=comp_data.Mw, min_z=zero, temperature=temperature)

        flash_ev = PXFlash(flash_params, PXFlash.ENTHALPY) if ph else Flash(flash_params)
        property_container.flash_ev = flash_ev

        property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('L', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('LCO2', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('L', ConstFunc(0.8)),
                                                ('LCO2', Fenghour1998()),
                                                ])

        diff = 8.64e-6
        property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff)),
                                                ('L', ConstFunc(np.ones(nc) * diff * 1e-3)),
                                                ('LCO2', ConstFunc(np.ones(nc) * diff * 1e-3)),
                                                ])

        if self.thermal:
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('L', EoSEnthalpy(eos=pr)),
                                                   ('LCO2', EoSEnthalpy(eos=pr)),
                                                   ])

            property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                       ('L', ConstFunc(180.)),
                                                       ('LCO2', ConstFunc(10.)),
                                                       ])

        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('L', PhaseRelPerm("wat", swc=swc, sgr=swc, n=4)),
                                               ('LCO2', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ])

        """ Add property region """
        self.add_property_region(property_container)
        property_container.output_props = {}

        property_container.output_props['rhoT'] = lambda: np.sum(property_container.sat * property_container.dens)
        for j, ph in enumerate(phases):
            property_container.output_props['sat' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['rhom' + ph] = lambda jj=j: property_container.dens_m[jj]
            property_container.output_props['enth' + ph] = lambda jj=j: property_container.enthalpy[jj]
            for i, comp in enumerate(components):
                property_container.output_props[comp + ' in ' + ph] = lambda jj=j, ii=i: property_container.x[jj, ii]


class BrineVapourLiquidCO2(Compositional):
    def __init__(self, components: list, swc: float, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_t: float = None, max_t: float = None,
                 zero: float = 1e-12, temperature: float = None, cache: bool = False):
        # Fluid components
        comp_data = CompData(components, setprops=True)
        nc = comp_data.nc
        h2o_idx = components.index("H2O")
        phases = ["Aq", "V", "LCO2", "L"]

        """ Activate physics """
        if temperature is None:
            ph = False  # switch for PH-simulation
            state_spec = Compositional.StateSpecification.PH if ph else Compositional.StateSpecification.PT
        else:
            ph = False  # isothermal case should run PT flash
            state_spec = Compositional.StateSpecification.P
        super().__init__(components, phases, timer, n_points=n_points, min_p=min_p, max_p=max_p, min_z=zero / 10,
                         max_z=1 - zero / 10, min_t=min_t, max_t=max_t, state_spec=state_spec, cache=cache)

        """ Initialize flash """
        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        pr = CubicEoS(comp_data, CubicEoS.PR)
        pr.set_preferred_roots(h2o_idx, 0.75, EoS.MAX)
        aq = AQEoS(comp_data, {AQEoS.CompType.water: AQEoS.Jager2003,
                               AQEoS.CompType.solute: AQEoS.Ziabakhsh2012,
                               AQEoS.CompType.ion: AQEoS.Jager2003
                               })
        aq.set_eos_range(h2o_idx, [0.6, 1.])

        flash_params.add_eos("PR", pr)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "PR"]
        flash_params.eos_params["PR"].root_order = [EoS.MAX, EoS.MIN]
        flash_params.eos_params["PR"].rich_phase_order = [components.index("CO2"), -1]

        # Define initial guesses for stability + flash
        params = flash_params.eos_params["PR"]
        params.initial_guesses = [i for i in range(comp_data.nc)]
        params.stability_tol = 1e-20
        params.stability_switch_tol = 1e-2
        params.stability_max_iter = 50
        params.use_gmix = False

        params = flash_params.eos_params["AQ"]
        params.initial_guesses = [h2o_idx]
        params.stability_max_iter = 10
        params.use_gmix = True

        # Flash-related parameters
        flash_params.split_variables = FlashParams.SplitVars.lnK
        flash_params.stability_variables = FlashParams.StabilityVars.alpha
        # flash_params.split_switch_tol = 1e-1
        flash_params.split_tol = 1e-14
        flash_params.comp_tol = 1e-2
        # flash_params.verbose = True

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases, components, Mw=comp_data.Mw, min_z=zero, temperature=temperature)

        # flash_ev = NegativeFlash(flash_params, ["AQ", "PR"], [InitialGuess.Henry_AV])
        flash_ev = PXFlash(flash_params, PXFlash.ENTHALPY) if ph else Flash(flash_params)
        ions = None
        if ions is not None:
            property_container.flash_ev = IonFlash(flash_ev, nph=2, nc=nc, ni=ni, combined_ions=combined_ions)
        else:
            property_container.flash_ev = flash_ev

        property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('L', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('LCO2', EoSDensity(eos=pr, Mw=comp_data.Mw)),
                                              ('Aq', Garcia2001(components)),
                                              ])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('L', ConstFunc(0.8)),
                                                ('LCO2', Fenghour1998()),
                                                ('Aq', Islam2012(components)),
                                                ])

        diff = 8.64e-6
        property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(nc) * diff)),
                                                ('L', ConstFunc(np.ones(nc) * diff * 1e-3)),
                                                ('LCO2', ConstFunc(np.ones(nc) * diff * 1e-3)),
                                                ('Aq', ConstFunc(np.ones(nc) * diff * 1e-3)),
                                                ])

        if self.thermal:
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('L', EoSEnthalpy(eos=pr)),
                                                   ('LCO2', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)),
                                                   ])

            property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                       ('L', ConstFunc(180.)),
                                                       ('LCO2', ConstFunc(10.)),
                                                       ('Aq', ConstFunc(180.)),
                                                       ])

        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=swc, sgr=swc, n=1.5)),
                                               ('L', PhaseRelPerm("wat", swc=swc, sgr=swc, n=4)),
                                               ('LCO2', PhaseRelPerm("oil", swc=swc, sgr=swc, n=1.5)),
                                               ('Aq', PhaseRelPerm("wat", swc=swc, sgr=swc, n=4)),
                                               ])

        """ Add property region """
        self.add_property_region(property_container)
        property_container.output_props = {}

        property_container.output_props['rhoT'] = lambda: np.sum(property_container.sat[1:] * property_container.dens[1:])
        for j, ph in enumerate(phases):
            property_container.output_props['sat' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['rhom' + ph] = lambda jj=j: property_container.dens_m[jj]
            property_container.output_props['enth' + ph] = lambda jj=j: property_container.enthalpy[jj]
            # for i, comp in enumerate(components):
            #     property_container.output_props[comp + ' in ' + ph] = lambda jj=j, ii=i: property_container.x[jj, ii]
