from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.cicd_model import CICDModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer


class Model(CICDModel):
    def __init__(self, resolution=10, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir(resolution)
        self.set_physics(n_points)

        self.set_sim_params(first_ts=1e-5, mult_ts=8, max_ts=31, runtime=365, tol_newton=1e-4, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

        T_init = 450.
        state_init = value_vector([200., 0.])
        enth_init = self.physics.property_containers[0].total_enthalpy(T_init).evaluate(state_init)
        self.initial_values = {self.physics.vars[0]: state_init[0],
                               self.physics.vars[1]: enth_init
                               }

    def set_reservoir(self, resolution):
        y_scale = 3
        (nx, ny, nz) = (resolution, y_scale * resolution, resolution)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000

        poro = np.ones(nb) * 0.2
        dx = 20. / resolution
        dy = 20. / resolution
        dz = 20. / resolution

        # discretize structured reservoir
        reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                    permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                    # hcap=2200, rcond=500
                                    )

        # add well's start locations
        iw = [resolution // 2, resolution // 2]
        jw = [4, ny - 4]

        n = nz // 2
        j_mid = ny // 2

        well_radius = 0.3

        # add well
        perf_list = [(iw[0], j, n+1) for j in range(jw[0], j_mid + 1)]
        reservoir.add_well("INJ", perf_list=perf_list,
                           well_radius=well_radius, segment_direction='y_axis', well_index=0)
        # add perforations with well_index=0 (closed pipe, only thermal losses)
        # for j in range(jw[0], j_mid + 1):
        #     reservoir.add_perforation(well=reservoir.wells[-1], i=iw[0], j=j, k=n + 1,
        #                               well_radius=well_radius, segment_direction='y_axis', well_index=0)
        perf_list = [(iw[1], j, n+1) for j in range(jw[1], j_mid, -1)]
        reservoir.add_well("PRD", perf_list=perf_list,
                           well_radius=well_radius, segment_direction='y_axis', well_index=0)
        # # add perforations with well_index=0 (closed pipe, only thermal losses)
        # for j in range(jw[1], j_mid, -1):
        #     reservoir.add_perforation(well=reservoir.wells[-1], i=iw[1], j=j, k=n + 1,
        #                               well_radius=well_radius, segment_direction='y_axis', well_index=0)

        return super().set_reservoir(reservoir)

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=50000, cache=False)
        physics.add_property_region(property_container)

        return super().set_physics(physics)

    # def set_initial_conditions(self):
    #     self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200,
    #                                                 uniform_temperature=450)

    def set_boundary_conditions(self):
        # connect the last two perforations of two wells
        well_1 = self.reservoir.wells[0]
        well_2 = self.reservoir.wells[1]
        perf_1 = len(well_1.perforations)  # last segment is n_perf+1
        perf_2 = len(well_2.perforations)
        # dictionary: key is a pair of 2 well names; value is a list of well perforation indices to connect
        self.reservoir.connected_well_segments = {(well_1.name, well_2.name): [(perf_1, perf_2)]}

        # Set boundary volumes
        self.reservoir.set_boundary_volume(self.mesh, xz_minus=1e8, xz_plus=1e8, yz_minus=1e8, yz_plus=1e8,
                                           xy_minus=1e8, xy_plus=1e8)

        return

    def set_well_controls(self):
        # rock heat capacity and rock thermal conduction
        hcap = np.array(self.mesh.heat_capacity, copy=False)
        rcond = np.array(self.mesh.rock_cond, copy=False)
        hcap.fill(2200)
        rcond.fill(500)

        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                w.control = self.physics.new_bhp_water_inj(205, 300)
            else:
                w.control = self.physics.new_bhp_prod(195)

    def compute_temperature(self, X):
        nb = self.mesh.n_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]
        op_num = np.array(self.mesh.op_num, copy=False)
        op_num[self.mesh.n_res_blocks:] = 1

    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.engine.X, copy=False)
        nb = self.mesh.n_res_blocks
        print(self.compute_temperature(X))
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        print(temp)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)
