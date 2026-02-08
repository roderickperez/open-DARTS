import numpy as np
from darts.models.darts_model import DartsModel
from darts.reservoirs.struct_reservoir import StructReservoir
from physics import BrineVapour


class Model(DartsModel):
    def set_reservoir(self, nx, nz, dx=0.2, dz=0.2, poro=0.25, perm=100):
        """Reservoir"""
        nb = nx * nz

        depth = np.zeros(nb)
        for k in range(nz):
            depth[k * nx:(k + 1) * nx] = 0.5 * dz + k * dz

        self.reservoir = StructReservoir(self.timer, nx=nx, ny=1, nz=nz, dx=dx, dy=1, dz=dz,
                                         permx=perm, permy=perm, permz=perm, poro=poro, depth=depth)

    def set_physics(self, components: list, phases: list, salinity: float = 0., temperature: float = None,
                    n_points: int = 1001, zero: float = 1e-12, swc: float = 0.2, impurity: float = 0.):
        self.components = components
        self.salinity = salinity
        self.zero = zero
        self.swc = swc
        self.impurity = impurity

        self.physics = BrineVapour(components=components, phases=phases, ions=(salinity > 0.), swc=swc, timer=self.timer,
                                   n_points=n_points, min_p=50, max_p=300, min_t=273.15, max_t=373.15, zero=zero,
                                   temperature=temperature, cache=False)

    def set_initial_conditions(self):
        depths = np.asarray(self.reservoir.mesh.depth)
        min_depth = np.min(depths)
        max_depth = np.max(depths)
        nb = int(self.reservoir.nz)
        depths = np.linspace(min_depth, max_depth, nb)

        from darts.physics.super.initialize import Initialize
        init = Initialize(self.physics, aq_idx=0, h2o_idx=0)

        # Gas-water contact
        p_init, T_init = 100., 300.
        known_idx = int(nb / 2)
        mid_depth = depths[known_idx]

        nc = len(self.physics.components)
        # Above the GWC, we have residual water saturation, gas concentration of impurity, salinity
        bound_specs = {"satAq": self.swc}
        specs_above = {"H2O": None, "CO2": None, "satAq": self.swc}
        # Below the GWC, we have zero CO2, impurity, and pure water or salinity
        specs_below = {"H2O": 1. - (nc-1) * self.zero}
        specs_below.update({comp: self.zero for comp in self.physics.components[1:-1]})

        if nc > 2:
            bound_specs["x2V"] = self.impurity
            specs_above["x2V"] = self.impurity
        if self.salinity:
            for specs in [specs_above, specs_below, bound_specs]:
                specs["m_" + self.physics.components[-1]] = self.salinity
                specs["H2O"] = None

        # Solve boundary state
        bound_specs.update({'pressure': p_init, 'temperature': T_init if init.thermal else None})
        X0 = ([p_init, 0.5] +  # pressure, H2O
              ([0.5 * (1.-self.impurity)] if nc > 2 else []) +  # CO2
              ([1. - 0.5 - 0.5 * (1.-self.impurity) - self.salinity * 0.5 / 55.509] if self.salinity else []) +  # last component if ions
              ([T_init] if init.thermal else []))  # temperature
        X0 = init.solve_state(X0, specs=bound_specs)

        # Initialize depth table
        X, bc_idx = init.init_depth_table(depth_bottom=max_depth,
                                          depth_top=min_depth,
                                          depth_known=mid_depth,
                                          X0=X0,
                                          nb=nb,
                                          dTdh=0.03
                                          )

        # Solve vertical equilibrium
        X = init.solve(X=X, bc_idx=bc_idx, specs=specs_above, downward=False)  # solve above
        X = init.solve(X=X, bc_idx=bc_idx, specs=specs_below, downward=True)  # solve below

        # assign initial condition with evaluated initialized properties
        self.physics.set_initial_conditions_from_depth_table(mesh=self.reservoir.mesh, input_depth=init.depths,
                                                             input_distribution={var: X[:, i] for i, var in
                                                                                 enumerate(self.physics.vars)})
        return
