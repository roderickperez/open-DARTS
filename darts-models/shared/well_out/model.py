from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
from darts.engines import value_vector, sim_params
from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer

import numpy as np
import os

class Model(CICDModel):
    def __init__(self, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics(n_points)

        self.set_sim_params(first_ts=1e-5, mult_ts=8, max_ts=31, runtime=365, tol_newton=1e-4, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

        T_init = 450.
        state_init = value_vector([200., 0.])
        enth_init = self.physics.property_containers[0].enthalpy_ev['total'](T_init).evaluate(state_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init
                               }

    def find_conn_id_for_perforation(self, perfs):
        res_cell_ids = [perf[1] for perf in perfs]
        block_m = np.array(self.reservoir.mesh.block_m, copy=False)
        block_p = np.array(self.reservoir.mesh.block_p, copy=False)
        perf_conn_ids = np.nonzero(np.logical_and(np.isin(block_p, res_cell_ids), block_m >= self.reservoir.mesh.n_res_blocks))[0]
        assert (len(perf_conn_ids) == len(perfs) and (block_m[perf_conn_ids] > self.reservoir.mesh.n_res_blocks).all())
        return perf_conn_ids

    def set_reservoir(self):

        (nx, ny, nz) = (100, 1, 10)
        nb = nx * ny * nz
        perm = np.ones(nb) * 1000

        poro = np.ones(nb) * 0.2
        dx = 1000. / nx
        dy = 10
        dz = 100 / nz

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=500)
        # self.reservoir.boundary_volumes['xy_minus'] = 1e8
        # self.reservoir.boundary_volumes['xy_plus'] = 1e8
        # self.reservoir.boundary_volumes['yz_minus'] = 1e8
        # self.reservoir.boundary_volumes['yz_plus'] = 1e8
        # self.reservoir.boundary_volumes['xz_minus'] = 1e8
        # self.reservoir.boundary_volumes['xz_plus'] = 1e8

        return

    def set_wells(self):
        # add well's start locations
        iw = [1, self.reservoir.nx]

        n = self.reservoir.nz

        well_radius = 0.3

        # add well
        self.reservoir.add_well("INJ")
        for k in range(n):
            self.reservoir.add_perforation("INJ", cell_index=(iw[0], 1, k + 1), well_radius=well_radius,
                                           multi_segment=True)

        self.reservoir.add_well("PRD")
        for j in range(n):
            self.reservoir.add_perforation("PRD", cell_index=(iw[1], 1, j + 1), well_radius=well_radius,
                                           multi_segment=True)

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=50000, cache=False)
        self.physics.add_property_region(property_container)

        return

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(well=w, is_control=True, is_inj=True, control_type=well_control_iface.BHP,
                                               target=205., inj_composition=[], inj_temp=300.)
            else:
                self.physics.set_well_controls(well=w, is_control=True, is_inj=False, control_type=well_control_iface.BHP,
                                               target=195.)

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp
