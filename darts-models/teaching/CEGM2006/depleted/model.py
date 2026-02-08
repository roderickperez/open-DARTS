import numpy as np
import os

from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.base.operators_base import PropertyOperators
from darts.engines import well_control_iface
from darts.physics.properties.basic import PhaseRelPerm, ConstFunc
from darts.physics.properties.density import Garcia2001
from darts.physics.properties.viscosity import Fenghour1998, Islam2012
from darts.physics.properties.eos_properties import EoSDensity, EoSEnthalpy

from dartsflash.libflash import NegativeFlash
from dartsflash.libflash import CubicEoS, AQEoS, FlashParams, InitialGuess
from dartsflash.components import CompData


class Model(DartsModel):
    def set_reservoir_radial(self, nr, dr, nz, dz, poro, perm):
        self.seg_ratio = 1
        if 1:
            from depleted.nearwellbore import RadialStruct
            self.reservoir = RadialStruct(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, permr=perm, permz=perm/10, poro=poro,
                                          hcap=2200, rcond=100, R1=1000, logspace=True, boundary_volume=4.3e10)

        else:
            from depleted.nearwellbore import RadialUnstruct
            self.reservoir = RadialUnstruct()

        self.reservoir.boundary_volumes['xy_plus'] = 1e8
        self.reservoir.boundary_volumes['xy_minus'] = 1e8

        return


    def set_reservoir(self, nx, nz, poro, perm):
        L = 600
        H = 50
        ny = 1
        nb = nx * ny * nz

        x_axes = np.logspace(0, np.log10(L), nx+1)
        dx_slice = x_axes[1:] - x_axes[:-1]
        dx = np.tile(dx_slice, nz)

        dz = H / nz
        depth = np.zeros(nb)
        n_layer = nx*ny
        for k in range(nz):
            depth[k*n_layer:(k+1)*n_layer] = 0 + k * dz

        self.reservoir = StructReservoir(self.timer, nx, ny, nz, dx=dx, dy=10, dz=dz, permx=perm, permy=perm,
                                         permz=perm/10, hcap=2200, rcond=100, poro=poro, depth=depth)
        self.reservoir.boundary_volumes['yz_plus'] = 1e8

    def set_initial_conditions(self):
        self.physics.set_initial_conditions_from_array(mesh=self.reservoir.mesh, input_distribution=self.initial_values)

    def set_wells(self):
        from darts.reservoirs.reservoir_base import ReservoirBase
        if 0:#type(self.reservoir).set_wells is not ReservoirBase.set_wells:
            # If the function has not been overloaded, pass
            self.reservoir.set_wells()
        else:
            self.reservoir.add_well("I1")
            for k in range(0, self.reservoir.nz):
                self.reservoir.add_perforation("I1", res_cell_idx=(1, 1, k+1), well_diameter=0.4, well_indexD=0,
                                               verbose=True, ms_epm=True)

    def set_physics(self, zero, n_points, components, temperature=None, temp_inj=350.):
        """Physical properties"""
        # Fluid components, ions and solid
        phases = ["Aq", "V"]
        comp_data = CompData(components, setprops=True)

        ceos = CubicEoS(comp_data, CubicEoS.PR)
        aq = AQEoS(comp_data, {AQEoS.water: AQEoS.Jager2003,
                               AQEoS.solute: AQEoS.Ziabakhsh2012,
                               })

        flash_params = FlashParams(comp_data)

        # EoS-related parameters
        flash_params.add_eos("CEOS", ceos)
        flash_params.add_eos("AQ", aq)
        flash_params.eos_order = ["AQ", "CEOS"]

        # Flash-related parameters
        flash_params.split_tol = 1e-14

        """ properties correlations """
        property_container = PropertyContainer(phases_name=phases, components_name=components, Mw=comp_data.Mw,
                                               temperature=temperature, rock_comp=0, min_z=zero / 10)

        property_container.flash_ev = NegativeFlash(flash_params, ["AQ", "CEOS"], [InitialGuess.Henry_AV])
        # K_val = np.array([110, 0.016, 0.0015])
        # property_container.flash_ev = ConstantK(len(components), K_val, zero)
        property_container.density_ev = dict([('V', EoSDensity(ceos, comp_data.Mw)),
                                              ('Aq', Garcia2001(components))])
        property_container.viscosity_ev = dict([('V', Fenghour1998()),
                                                ('Aq', Islam2012(components))])
        property_container.rel_perm_ev = dict([('V', PhaseRelPerm("gas", swc=self.swc, sgr=0.0, n=1.5)),
                                               ('Aq', PhaseRelPerm("oil", swc=self.swc, sgr=0.0, n=4))])

        property_container.enthalpy_ev = dict([('V', EoSEnthalpy(ceos)),
                                               ('Aq', EoSEnthalpy(aq))])
        property_container.conductivity_ev = dict([('V', ConstFunc(10.)),
                                                   ('Aq', ConstFunc(180.)), ])


        property_container.output_props = {}
        for j, ph in enumerate(phases):
            property_container.output_props['sat' + ph] = lambda jj=j: property_container.sat[jj]
            property_container.output_props['dens' + ph] = lambda jj=j: property_container.dens[jj]

        for i, comp in enumerate(components):
            property_container.output_props['x' + comp] = lambda ii=i: property_container.x[0, ii]
            property_container.output_props['y' + comp] = lambda ii=i: property_container.x[1, ii]

        state_spec = Compositional.StateSpecification.PT if temperature is None else Compositional.StateSpecification.P  # if None, then thermal
        self.physics = Compositional(components, phases, self.timer, n_points, min_p=1, max_p=400, min_z=zero / 10,
                                     max_z=1 - zero / 10, min_t=263.15, max_t=393.15, state_spec=state_spec, cache=False)
        self.physics.add_property_region(property_container)

    def set_well_controls(self):


        # define all wells as closed
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=True,
                    target=self.p_inj,
                    inj_composition = self.inj_stream[:-1],
                    inj_temp=self.inj_stream[-1]
                )

                print(f'setting well {w.name} controls bhp to {self.p_inj} bar at {self.inj_stream[-1]} K')

            else:
                self.physics.set_well_controls(
                    wctrl=w.control,
                    control_type=well_control_iface.BHP,
                    is_inj=False,
                    target=self.p_init
                )

                print(f'setting well {w.name} controls bhp to ', self.p_init)

    def plot(self, output_properties: list, fig=None, lims: dict = None, i: int = -1):
        output_data = self.output_properties()

        tot_props = self.physics.vars + self.physics.property_operators[0].props_name

        # loop for adding units to the titles of axes
        output_idxs = {}
        new_lims = {}
        for prop in output_properties:
            if prop == 'pressure':
                axis_name = 'pressure, bar'
            elif prop == 'temperature':
                axis_name = 'temperature, K'
            else:
                axis_name = prop
            output_idxs[axis_name] = tot_props.index(prop)
            new_lims[axis_name] = lims[prop]

        return self.reservoir.plot(output_idxs, output_data, fig=fig, lims=new_lims)

    def populate_data_for_radial_vtk_output(self, data):
        new_data = {}
        n_cells = self.reservoir.mesh.n_res_blocks
        for prop, val in data.items():
            # populate r-z data to all angles
            new_data[prop] = np.tile(val, self.reservoir.nphi)

        return new_data

    def output_to_vtk(self, sol_filepath : str = None, ith_step: int = None, output_directory: str = None,
                      output_properties: list = None, engine : bool = False):

        # Set default output directory
        if output_directory is None:
            output_directory = os.path.join(self.output_folder, 'vtk_files')
        os.makedirs(output_directory, exist_ok=True)

        timestep, output_data = self.output.output_properties(self.sol_filepath if sol_filepath is None else sol_filepath,
                                                              output_properties,
                                                               ith_step,
                                                                engine)
        data = self.populate_data_for_radial_vtk_output(output_data)

        self.reservoir.output_to_vtk(output_directory=output_directory,
                                     data=data,
                                     ith_step=ith_step,
                                     t=timestep,
                                     prop_names=list(output_data.keys()))

    def get_unknowns_for_radial_vtk_output(self):
        X = np.array(self.physics.engine.X, copy=False)

        # prepare data
        data = {}
        n_cells = self.reservoir.mesh.n_res_blocks
        for i, var in enumerate(self.physics.vars):
            # write r-z data
            data[var] = X[i:self.physics.n_vars * n_cells:self.physics.n_vars]
            # populate r-z data to all angles
            data[var] = np.tile(data[var], self.reservoir.nphi)

        return data

