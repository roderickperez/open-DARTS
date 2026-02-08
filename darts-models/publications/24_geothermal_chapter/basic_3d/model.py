from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.physics.properties.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
from darts.engines import value_vector, sim_params

from darts.physics.geothermal.physics import Geothermal
from darts.physics.geothermal.property_container import PropertyContainer


class Model(DartsModel):
    def __init__(self, n_points=128):
        # call base class constructor
        super().__init__()

        self.timer.node["initialization"].start()

        self.set_reservoir()
        self.set_physics(n_points)

        self.set_sim_params(first_ts=1e-3, mult_ts=8, max_ts=365, runtime=3650, tol_newton=1e-2, tol_linear=1e-6,
                            it_newton=20, it_linear=40, newton_type=sim_params.newton_global_chop,
                            newton_params=value_vector([1]))

        self.timer.node["initialization"].stop()

        state_pt_init = value_vector([200., 350.])
        enth_init = self.physics.property_containers[0].compute_total_enthalpy(state_pt_init)
        self.initial_values = {self.physics.vars[0]: state_pt_init[0],
                               self.physics.vars[1]: enth_init
                               }

    def set_reservoir(self):
        (nx, ny, nz) = (60, 60, 3)
        nb = nx * ny * nz
        perm = np.ones(nb) * 2000
        perm = load_single_keyword('perm.in', 'PERMX')
        perm = perm[:nb]

        poro = np.ones(nb) * 0.2
        dx = 30
        dy = 30
        dz = np.ones(nb) * 30
        rcond = np.ones(nb) * 500

        poro[:nx*ny] = 1e-6
        perm[:nx*ny] = 1e-6
        dz[:nx*ny] = 30
        #rcond[:nx*ny] = 0

        # discretize structured reservoir
        self.reservoir = StructReservoir(self.timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
                                         permx=perm, permy=perm, permz=perm * 0.1, poro=poro, depth=2000,
                                         hcap=2200, rcond=rcond)

        self.reservoir.boundary_volumes['yz_minus'] = 1e8
        self.reservoir.boundary_volumes['yz_plus'] = 1e8
        self.reservoir.boundary_volumes['xz_minus'] = 1e8
        self.reservoir.boundary_volumes['xz_plus'] = 1e8

        self.reservoir.boundary_volumes['xy_minus'] = dx * dy * 300

    def set_wells(self):
        # add well's locations
        iw = [30, 30]
        jw = [14, 46]

        # add well
        self.reservoir.add_well("INJ")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("INJ", res_cell_idx=(iw[0], jw[0], k + 1),
                                           well_diameter=0.32, ms_epm=True)

        # add well
        self.reservoir.add_well("PRD")
        for k in range(1, self.reservoir.nz):
            self.reservoir.add_perforation("PRD", res_cell_idx=(iw[1], jw[1], k + 1),
                                           well_diameter=0.32, ms_epm=True)

    def set_physics(self, n_points):
        # create pre-defined physics for geothermal
        property_container = PropertyContainer()
        self.physics = Geothermal(self.timer, n_points=n_points, min_p=1, max_p=351, min_e=1000, max_e=10000, cache=False)
        self.physics.add_property_region(property_container)

    def set_initial_conditions(self):
        self.physics.set_initial_conditions_from_array(self.reservoir.mesh, self.initial_values)
        return

    def set_well_controls(self):
        from darts.engines import well_control_iface
        for i, w in enumerate(self.reservoir.wells):
            if i == 0:
                self.physics.set_well_controls(wctrl=w.control, is_inj=True, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               target=5000., phase_name='water', inj_composition=[], inj_temp=300.)
            else:
                self.physics.set_well_controls(wctrl=w.control, is_inj=False, control_type=well_control_iface.VOLUMETRIC_RATE,
                                               target=5000., phase_name='water')

    def compute_temperature(self, X):
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        return temp

    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor[0], self.physics.acc_flux_w_itor]
        self.op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        self.op_num[self.reservoir.mesh.n_res_blocks:] = 1

    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.physics.engine.X, copy=False)
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'].flatten()
                           }

        self.export_vtk(file_name, local_cell_data=local_cell_data)
