import numpy as np
from darts.physics.super.physics import Compositional, timer_node
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import PhaseRelPerm, CapillaryPressure, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.flash import IonFlash
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash, FlashParams, InitialGuess
from dartsflash.libflash import CubicEoS, AQEoS
from dartsflash.components import CompData


class BrineVapour(Compositional):
    def __init__(self, components: list, phases: list, swc: float, timer: timer_node, n_points: int,
                 min_p: float, max_p: float, min_t: float = None, max_t: float = None,
                 ions: bool = False, zero: float = 1e-12, temperature: float = None, cache: bool = False):
        # Fluid components, ions and solid
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
        epsilon_z = zero / 10
        state_spec = Compositional.StateSpecification.PT if temperature is None else Compositional.StateSpecification.P
        super().__init__(species, phases, timer, n_points=n_points, min_p=min_p, max_p=max_p, min_z=0., max_z=1.,
                         epsilon_z=epsilon_z, extrapolation_flag=True, min_t=min_t, max_t=max_t, state_spec=state_spec,
                         cache=cache)

        """ Initialize flash """
        from dartsflash.libflash import EoS
        from dartsflash.mixtures import DARTSFlash, VLAq
        # Fluid components, ions and solid
        vl_phases = False
        phases = ["Aq", "V", "L"] if vl_phases else ["Aq", "V"]
        nc = len(components)

        flash_ev = VLAq(comp_data, hybrid=True)
        flash_ev.set_vl_eos("PR", root_order=[EoS.MAX, EoS.MIN] if vl_phases else [EoS.STABLE],
                            trial_comps=[InitialGuess.Yi.Wilson, components.index("CO2")],
                            stability_tol=1e-20, switch_tol=1e-2, max_iter=50, use_gmix=False
                            )
        flash_ev.set_aq_eos("Aq", stability_tol=1e-20, max_iter=10, use_gmix=True)
        pr = flash_ev.eos["VL"]
        aq = flash_ev.eos["Aq"]

        flash_ev.init_flash(flash_type=DARTSFlash.FlashType.PTFlash, eos_order=["Aq", "VL"],
                            # nf_initial_guess=[InitialGuess.Henry_AV],
                            # t_min=270., t_max=500., t_init=300., t_tol=1e-3,
                            verbose=False
                            )

        """ PropertyContainer object and correlations """
        property_container = PropertyContainer(phases, species, Mw=species_Mw, eps_z=epsilon_z, temperature=temperature)

        # flash_ev = ConstantK(nc=2, ki=[0.001, 100])
        if ions is not None:
            property_container.flash_ev = IonFlash(flash_ev, nph=2, nc=nc, ni=ni, combined_ions=combined_ions)
        else:
            property_container.flash_ev = flash_ev

        property_container.density_ev = dict([('V', EoSDensity(eos=pr, Mw=species_Mw)),
                                              ('Aq', Garcia2001(components, ions, combined_ions)), ])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(components, ions, combined_ions)), ])

        diff = 8.64e-6
        property_container.diffusion_ev = dict([('V', ConstFunc(np.ones(len(species)) * diff * 1e3)),
                                                ('Aq', ConstFunc(np.ones(len(species)) * diff))])

        if self.thermal:
            property_container.enthalpy_ev = dict([('V', EoSEnthalpy(eos=pr)),
                                                   ('Aq', EoSEnthalpy(eos=aq)), ])

            property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                       ('Aq', ConstFunc(180.)), ])

        property_container.rel_perm_ev = dict([('Aq', PhaseRelPerm("wat", sgr=swc, swc=swc)),
                                               ('V', PhaseRelPerm("gas", sgr=swc, swc=swc))])

        """ Activate physics """
        self.add_property_region(property_container)
        property_container.output_props = {}
        for j, ph in enumerate(phases):
            property_container.output_props['sat' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['rho' + ph] = lambda jj=j: property_container.dens[jj]
            property_container.output_props['rhom' + ph] = lambda jj=j: property_container.dens_m[jj]
            property_container.output_props['enth' + ph] = lambda jj=j: property_container.enthalpy[jj]

        for i, comp in enumerate(components):
            property_container.output_props['x' + comp] = lambda ii=i: property_container.x[0, ii]
            property_container.output_props['y' + comp] = lambda ii=i: property_container.x[1, ii]

        property_container.output_props.update({"grad00": lambda: 0.0001728 * property_container.dens_m[0] * property_container.x[0, 0],  # check Lyu and Voskov (2023) section 3.2
                                                "grad01": lambda: 0.0001728 * property_container.dens_m[0] * property_container.x[0, 1],
                                                "grad10": lambda: 0.0001728 * property_container.dens_m[1] * property_container.x[1, 0],
                                                "grad11": lambda: 0.0001728 * property_container.dens_m[1] * property_container.x[1, 1],
                                                })
