import numpy as np
from darts.models.darts_model import DartsModel
from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm

from physics import BrineVapour, VapourLiquidCO2, BrineVapourLiquidCO2
import os


class Model(DartsModel):
    def set_reservoir(self, nr, dr, nz, dz, poro, perm):
        self.seg_ratio = 1
        if 1:
            from nearwellbore import RadialStruct
            self.reservoir = RadialStruct(self.timer, nr=nr, nz=nz, dr=dr, dz=dz, permr=perm, permz=perm/10, poro=poro,
                                          hcap=2200, rcond=100, boundary_volume=1e8)

        else:
            from nearwellbore import RadialUnstruct
            self.reservoir = RadialUnstruct()

        return

    def set_physics_VAq(self, components: list, phases: list, salinity: float = 0., temperature: float = None,
                        vl_phases: bool = False, zero: float = 1e-12, n_points: int = 1001, swc: float = 0.2,
                        impurity: float = 0.):
        self.components = components
        self.salinity = salinity
        self.zero = zero
        self.swc = swc
        self.impurity = impurity if impurity >= zero else zero
        self.vl_phases = vl_phases

        self.physics = BrineVapour(components=components, phases=phases, ions=(salinity > 0.), swc=swc, vl_phases=vl_phases,
                                   timer=self.timer, n_points=n_points, min_p=1, max_p=300, min_t=253.15, max_t=423.15,
                                   zero=zero, temperature=temperature, cache=False)

    def set_wells(self):
        from darts.reservoirs.reservoir_base import ReservoirBase
        if 0:#type(self.reservoir).set_wells is not ReservoirBase.set_wells:
            # If the function has not been overloaded, pass
            self.reservoir.set_wells()
        else:
            self.reservoir.add_well("I1")
            for k in range(25, 75):
                self.reservoir.add_perforation("I1", cell_index=(1, 1, k+1), well_index=1e3, well_indexD=0,
                                               verbose=True, multi_segment=self.ms_well_flag)

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        nb = 100
        depths = np.linspace(min_depth, max_depth, nb)

        from darts.physics.super.initialize import Initialize
        init = Initialize(self.physics, aq_idx=0, h2o_idx=self.components.index("H2O"))

        # Gas-water contact
        known_idx = int(nb / 2)
        mid_depth = depths[known_idx]

        nc = len(self.components)
        primary_specs = {comp: np.ones(nb) * np.nan for comp in self.components}
        secondary_specs = {}
        bound_specs = {}

        if nc == 2:
            # H2O-CO2, initially pure brine
            # need 1 specification: H2O = 1-zero
            primary_specs["CO2"][:] = self.zero
            #secondary_specs.update({'satAq': np.ones(nb)*(1-self.zero)})
            #bound_specs.update({'satAq': self.swc})
            # if self.components[0] == "H2O":
            #     primary_specs["H2O"][:] = 1.-self.zero
            # else:
            #     primary_specs["CO2"][:] = self.zero
        else:
            # H2O-CO2-C1, initially residual water saturation and pure C1 gas
            # need 2 specifications: CO2 = zero, sAq = swc
            primary_specs["CO2"][:] = self.zero
            secondary_specs.update({'satAq': np.ones(nb) * self.swc})
            bound_specs.update({'satAq': self.swc})
        if self.salinity:
            # + ions, need extra specification for ion molality
            # H2O cannot be specified because of salinity, CO2 instead
            mol = np.ones(nb) * self.salinity
            secondary_specs.update({'m' + str(nc): mol})
            bound_specs.update({'m' + str(nc): mol[known_idx]})
            primary_specs["H2O"][:] = None
            primary_specs["CO2"][:] = self.zero

        # Solve boundary state
        X0 = ([self.p_init, 0.05] +  # pressure, CO2
              ([0.9] if nc > 2 else []) +  # H2O
              ([1. - 0.9 - mol[known_idx] * 0.9 / 55.509] if self.salinity else []) +  # last component if ions
              ([self.t_init] if init.thermal else []))  # temperature
        X0 = init.solve_state(X0, primary_specs={'pressure': self.p_init,
                                                 'temperature': self.t_init if init.thermal else None} |
                                                {comp: primary_specs[comp][known_idx] for comp in self.components},
                              secondary_specs=bound_specs)
        boundary_state = {v: X0[i] for v, i in init.var_idxs.items()}

        # Solve vertical equilibrium
        X = init.solve(depth_bottom=max_depth, depth_top=min_depth, depth_known=mid_depth, nb=nb,
                       primary_specs=primary_specs, secondary_specs=secondary_specs,
                       boundary_state=boundary_state, dTdh=0.03).reshape((nb, self.physics.n_vars))

        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, input_depth=init.depths,
                                                             input_distribution={v: X[:, i] for v, i in
                                                                                 init.var_idxs.items()})
        return

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if 'I' in w.name:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=True, target=self.p_inj, inj_composition=self.inj_stream, inj_temp=self.t_inj)
            else:
                self.physics.set_well_controls(wctrl=w.control, control_type=well_control_iface.BHP,
                                               is_inj=False, target=self.p_prod)

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

