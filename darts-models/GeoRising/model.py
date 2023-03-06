from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.physics.geothermal import Geothermal
from darts.models.darts_model import DartsModel, sim_params
from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector


class Model(DartsModel):
    def __init__(self, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        #perm = np.ones(60*60*7) * 3000

        self.reservoir = StructReservoir(self.timer, nx=60, ny=60, nz=7, dx=30, dy=30, dz=12, permx=perm,
                                         permy=perm, permz=perm*0.1, poro=0.2, depth=2000)

        self.reservoir.set_boundary_volume(xy_minus=30*30*400, xy_plus=30*30*400)
        self.reservoir.add_well("I1")
        n_perf = self.reservoir.nz
        for n in range(n_perf):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=30, j=14, k=n+1, well_radius=0.1,
                                           well_index=10, multi_segment=False, verbose=True)

        self.reservoir.add_well("P1")
        for n in range(n_perf):
            self.reservoir.add_perforation(self.reservoir.wells[-1], 30, 46, n+1, well_radius=0.1, well_index=10,
                                           multi_segment=False, verbose=True)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)
        rcond.fill(181.44)


        self.physics = Geothermal(self.timer, n_points, 1, 351, 1000, 10000, cache=False)

        self.params.first_ts = 1e-3
        self.params.mult_ts = 8
        self.params.max_ts = 365

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = 1e-2
        self.params.tolerance_linear = 1e-6
        self.params.max_i_newton = 20
        self.params.max_i_linear = 40

        self.params.newton_type = sim_params.newton_global_chop
        self.params.newton_params = value_vector([1])

        self.runtime = 365
        # self.physics.engine.silent_mode = 0
        self.timer.node["initialization"].stop()

    def set_initial_conditions(self):
        self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
                                                    uniform_temperature=348.15)

    def set_boundary_conditions(self):
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_rate_water_inj(8000, 298.15)
                # w.control = self.physics.new_bhp_water_inj(230, 308.15)
            else:
                w.control = self.physics.new_rate_water_prod(8000)
                # w.control = self.physics.new_bhp_prod(180)

    def compute_temperature(self, X):
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_itor_well]
        op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        op_num[self.reservoir.mesh.n_res_blocks:] = 1


    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.physics.engine.X, copy=False)
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)