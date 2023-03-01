from darts.models.reservoirs.struct_reservoir import StructReservoir
from geothermal import Geothermal
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
from darts.models.physics.iapws.custom_rock_property import *
from property_container import *
from darts.engines import value_vector

from darts.models.darts_model import DartsModel, sim_params
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np


class Model(DartsModel):
    def __init__(self, n_points=1000):
        # call base class constructor
        super().__init__()
        self.n_points = n_points
        # measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.init_grid()
        self.init_wells()
        self.init_physics()

        self.params.first_ts = 0.0001
        self.params.mult_ts = 16
        self.params.max_ts = 365

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = 1e-1
        self.params.tolerance_linear = 1e-5
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50

        self.params.newton_type = sim_params.newton_global_chop
        self.params.newton_params = value_vector([0.2])

        self.runtime = 1000
        self.timer.node["initialization"].stop()


    def init_grid(self):
        perm = load_single_keyword('permXVanEssen.in', 'PERMX')
        #perm = np.ones(60*60*7) * 3000

        self.reservoir = StructReservoir(self.timer, nx=60, ny=60, nz=7, dx=30, dy=30, dz=12, permx=perm,
                                         permy=perm, permz=perm*0.1, poro=0.2, depth=2000)

        self.reservoir.set_boundary_volume(xy_minus=30*30*400, xy_plus=30*30*400)

        hcap = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        rcond = np.array(self.reservoir.mesh.rock_cond, copy=False)

        hcap.fill(2200)
        rcond.fill(181.44)

    def init_wells(self):
        n_perf = self.reservoir.nz

        self.reservoir.add_well("I1")
        for n in range(n_perf):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=30, j=14, k=n+1, well_radius=0.1,
                                           well_index=10, multi_segment=False, verbose=True)

        self.reservoir.add_well("P1")
        for n in range(n_perf):
            self.reservoir.add_perforation(self.reservoir.wells[-1], 30, 46, n+1, well_radius=0.1, well_index=10,
                                           multi_segment=False, verbose=True)

    def init_physics(self):
        # Create property containers:
        self.property_container = model_properties(phases_name=['water', 'steam', 'temperature', 'energy'],
                                                   components_name=['H2O'])

        # Define properties in property_container (IAPWS is the default property package for Geothermal in DARTS)
        # Users can define their custom properties in custom_properties.py; several property examples are defined there.
        self.rock = [value_vector([1, 0, 273.15])]
        self.property_container.temp_ev = iapws_temperature_evaluator()
        self.property_container.enthalpy_ev = dict([('water', iapws_water_enthalpy_evaluator()),
                                                    ('steam', iapws_steam_enthalpy_evaluator())])
        self.property_container.saturation_ev = dict([('water', iapws_water_saturation_evaluator()),
                                                    ('steam', iapws_steam_saturation_evaluator())])
        self.property_container.rel_perm_ev = dict([('water', iapws_water_relperm_evaluator()),
                                                    ('steam', iapws_steam_relperm_evaluator())])
        self.property_container.density_ev = dict([('water', iapws_water_density_evaluator()),
                                                   ('steam', iapws_steam_density_evaluator())])
        self.property_container.viscosity_ev = dict([('water', iapws_water_viscosity_evaluator()),
                                                     ('steam', iapws_steam_viscosity_evaluator())])
        self.property_container.saturation_ev = dict([('water', iapws_water_saturation_evaluator()),
                                                      ('steam', iapws_steam_saturation_evaluator())])

        self.property_container.rock_compaction_ev = custom_rock_compaction_evaluator(self.rock)
        self.property_container.rock_energy_ev = custom_rock_energy_evaluator(self.rock)

        self.physics = Geothermal(property_container=self.property_container, timer=self.timer, n_points=128, min_p=0.1,
                                  max_p=451, min_e=0, max_e=50000, grav=True)

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

    def export_info_vtk(self):
        X = np.array(self.physics.engine.X, copy=False)
        from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
        nb = self.reservoir.mesh.n_res_blocks
        T = _Backward1_T_Ph_vec(X[0:nb * 2:2] / 10, X[1:nb * 2:2] / 18.015)
        local_cell_data = {'temperature': T,
                           'kx': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}
        return local_cell_data

    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.physics.engine.X, copy=False)
        nb = self.reservoir.mesh.n_res_blocks
        temp = _Backward1_T_Ph_vec(X[0:2 * nb:2] / 10, X[1:2 * nb:2] / 18.015)
        local_cell_data = {'Temperature': temp,
                           'Perm': self.reservoir.global_data['permx'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)

class model_properties(property_container):
    def __init__(self, phases_name, components_name):
        # Call base class constructor
        self.nph = len(phases_name)
        Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw)

        # remove the virtual phase from the parent class
        self.dens = np.zeros(self.nph-2)
        self.sat = np.zeros(self.nph-2)
        self.mu = np.zeros(self.nph-2)
        self.kr = np.zeros(self.nph-2)
        self.enthalpy = np.zeros(self.nph-2)

    def evaluate(self, state):
        vec_state_as_np = np.asarray(state)

        for j in range(self.nph-2):
            self.enthalpy[j] = self.enthalpy_ev[self.phases_name[j]].evaluate(state)
            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(state)
            self.sat[j] = self.saturation_ev[self.phases_name[j]].evaluate(state)
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(state)
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(state)

        self.temp = self.temp_ev.evaluate(state)
        self.rock_compaction = self.rock_compaction_ev.evaluate(state)
        self.rock_int_energy = self.rock_energy_ev.evaluate(state)

        return self.enthalpy, self.dens, self.sat, self.kr, self.mu, self.temp, self.rock_compaction, self.rock_int_energy
