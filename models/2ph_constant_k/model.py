from darts.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
from darts.engines import sim_params, value_vector, index_vector
from darts.tools.keyword_file_tools import load_single_keyword
import numpy as np
import os

from darts.physics.super.physics import Compositional
from darts.physics.super.property_container import PropertyContainer
from darts.physics.properties.flash import ConstantK
from darts.physics.properties.basic import ConstFunc, PhaseRelPerm
from darts.physics.properties.density import DensityBasic


class Model(DartsModel):
    def __init__(self, obl_points, reservoir_type, nx: int = None, components: list = []):
        # Call base class constructor
        super().__init__()

        self.nx = nx
        self.obl_points = obl_points
        self.discr_type = 'tpfa'
        self.reservoir_type = reservoir_type
        self.components = components

        # Measure time spend on reading/initialization
        self.timer.node["initialization"].start()

        self.set_reservoir()
        #self.set_wells()
        self.set_physics()

        self.set_sim_params(first_ts=0.05, mult_ts=2, max_ts=5.00, runtime=1000, tol_newton=1e-2, tol_linear=1e-3,
                            it_newton=10, it_linear=50, newton_type=sim_params.newton_local_chop)
        self.params.linear_type = sim_params.cpu_superlu

        self.timer.node["initialization"].stop()

        self.initial_values = {
                self.physics.vars[0]: self.p_init,
                **{self.physics.vars[i + 1]: self.ini_comp[i] for i in range(len(self.physics.vars) - 1)} }
        
        self.inj_stream = self.inj_comp[:self.physics.nc-1]
        self.physics.components = self.components
        # for i in range(self.physics.nc - 1):
        #     self.initial_values[self.components[i]][0] = self.inj_stream[i]

    def set_reservoir(self):
        if self.reservoir_type == '1D':
            self.p_init = 100.
            self.ny = self.nz = 1
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=1, dy=1, dz=1, permx=100, permy=100, permz=100,
                                        poro=0.3, depth=1000)
            self.well_cell_id = [[1, 1], [self.nx, 1]]
        elif self.reservoir_type == '2D':
            self.p_init = 100.
            self.ny = self.nx
            self.nz = 1
            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz, dx=1, dy=1, dz=1, permx=100, permy=100, permz=100,
                                        poro=0.3, depth=1000)
            self.well_cell_id = [[1, 1], [self.nx, self.nx]]
        else: # SPE10
            # read properties
            self.nx, self.ny, self.nz = [int(n) for n in self.reservoir_type.split('_')[1:]]
            input_folder = os.path.join('input', self.reservoir_type.split('_')[0], '_'.join(self.reservoir_type.split('_')[1:]))
            porosity = np.flip(np.swapaxes(load_single_keyword(os.path.join(input_folder, 'poro.txt'), 'PORO', cache=0).
                                     reshape(self.nz, self.ny, self.nx), 0, 2), axis=2)
            permeability = np.flip(np.swapaxes(load_single_keyword(os.path.join(input_folder, 'perm.txt'), 'PERM', cache=0).
                                     reshape(self.nz, self.ny, self.nx, 3), 0, 2), axis=2)
            self.p_init = load_single_keyword(os.path.join(input_folder, 'ref_pres.txt'), 'REF_PRESSURE', cache=0)
            # SPE10 reservoir geometry
            foot2meter = 0.3048
            Lx, Ly, Lz = np.array([960., 2080., 160.]) * foot2meter # feet to meters
            dx, dy, dz = Lx / self.nx, Ly / self.ny, Lz / self.nz
            depth = -12000 * foot2meter + Lz

            self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=self.nz,
                                                         dx=dx, dy=dy, dz=dz,
                                                         permx=permeability[:,:,:,0],
                                                         permy=permeability[:,:,:,1],
                                                         permz=permeability[:,:,:,2],
                                                         poro=porosity, depth=depth)


            # find well cells
            self.well_cell_id = [[int(self.nx / 2) + 1, int((Ly / 2 + 200.) / dx) + 1],
                                 [int(self.nx / 2) + 1, int((Ly / 2 - 200.) / dx) + 1]]

            # extend initial pressure array for well bodies/heads
            n_wells = len(self.well_cell_id)
            self.p_init = np.append(self.p_init, 2 * n_wells * [depth])

        return

    def set_wells(self):
        self.reservoir.add_well("I1")
        self.reservoir.add_well("P1")
        for k in range(1, self.nz + 1):
            self.reservoir.add_perforation("I1", cell_index=(self.well_cell_id[0][0], self.well_cell_id[0][1], k))
            self.reservoir.add_perforation("P1", cell_index=(self.well_cell_id[1][0], self.well_cell_id[1][1], k))
        return

    def set_physics(self):
        """Physical properties"""
        self.zero = 1e-8
        # Create property containers:

        n_comps = len(self.components)
        self.inj_comp = [1. - (n_comps - 1) * 0.001] + (n_comps - 1) * [0.001]
        if n_comps == 3:
            self.ini_comp = [0.005, 0.550, 0.445]
        elif n_comps == 4:
            self.ini_comp = [0.005, 0.500, 0.300, 0.195]
        elif n_comps == 6:
            self.ini_comp = [0.005, 0.350, 0.250, 0.195, 0.125, 0.075]
        elif n_comps == 8:
            self.ini_comp = [0.005, 0.350, 0.170, 0.150, 0.125, 0.100, 0.075, 0.025]
        elif n_comps == 10:
            self.ini_comp = [0.005, 0.300, 0.170, 0.150, 0.100, 0.100, 0.075, 0.050, 0.025, 0.025]
        elif n_comps == 12:
            self.ini_comp = [0.005, 0.300, 0.150, 0.100, 0.075, 0.075, 0.065, 0.065, 0.060, 0.050, 0.035, 0.020]
        elif n_comps == 20:
            self.ini_comp = [0.005, 0.240, 0.120, 0.090, 0.070, 0.070, 0.060, 0.060, 0.050, 0.045,
                             0.040, 0.035, 0.030, 0.025, 0.020, 0.015, 0.010, 0.007, 0.005, 0.003]
        assert (sum(self.ini_comp) == 1.0)

        Mw_comps = {
            'CO2': 44.010,
            'C1': 16.043,
            'C2': 30.069,
            'C3': 44.096,
            'C4': 58.123,
            'nC4': 58.123,
            'C5': 72.150,
            'nC5': 72.150,
            'C6': 86.178,
            'C7': 100.205,
            'C8': 114.232,
            'C9': 128.259,
            'C10': 142.286,
            'C11': 156.313,
            'C12': 170.340,
            'C13': 184.367,
            'C14': 198.394,
            'C15': 212.421,
            'C16': 226.448,
            'C17': 240.475,
            'C18': 254.502,
            'C19': 268.529,
            'C20': 282.556
        }
        K_comps = {
            'CO2': 1.5,
            'C1': 2.5,
            'C2': 2.0,
            'C3': 1.0,
            'C4': 0.7,
            'nC4': 0.5,
            'C5': 0.4,
            'nC5': 0.3,
            'C6': 0.2,
            'C7': 0.15,
            'C8': 0.12,
            'C9': 0.10,
            'C10': 0.05,
            'C11': 0.04,
            'C12': 0.035,
            'C13': 0.03,
            'C14': 0.025,
            'C15': 0.02,
            'C16': 0.01,
            'C17': 0.01,
            'C18': 0.01,
            'C19': 0.01,
            'C20': 0.01
        }


        Mw = [Mw_comps[c] for c in self.components]
        K = [K_comps[c] for c in self.components]

        phases = ['gas', 'oil']
        thermal = 0

        property_container = ModelProperties(phases_name=phases, components_name=self.components,
                                               Mw=Mw, min_z=self.zero / 10, temperature=1.)

        """ properties correlations """
        property_container.flash_ev = ConstantK(len(self.components), K, self.zero)
        property_container.density_ev = dict([('gas', DensityBasic(compr=1e-3, dens0=200)),
                                              ('oil', DensityBasic(compr=1e-5, dens0=600))])
        property_container.viscosity_ev = dict([('gas', ConstFunc(0.1)),
                                                ('oil', ConstFunc(1.0))])
        property_container.rel_perm_ev = dict([('gas', PhaseRelPerm("gas")),
                                               ('oil', PhaseRelPerm("oil"))])

        """ Activate physics """
        self.physics = Compositional(self.components, phases, self.timer, n_points=self.obl_points,
                                     min_p=1, max_p=300, min_z=self.zero/10, max_z=1-self.zero/10, cache=False)
        self.physics.add_property_region(property_container)
        
        return

    def set_well_controls(self):
        injector = self.reservoir.get_well('I1')
        producer = self.reservoir.get_well('P1')

        zero = self.physics.axes_min[1]
        if self.reservoir_type == '1D':
            injector.control = self.physics.new_rate_inj(0.5, self.inj_stream, 0)
            producer.control = self.physics.new_bhp_prod(50.)
        elif self.reservoir_type == '2D':
            injector.control = self.physics.new_rate_inj(20., self.inj_stream, 0)
            producer.control = self.physics.new_bhp_prod(50.)
        else:
            injector.control = self.physics.new_rate_inj(0., self.inj_stream, 0)
            producer.control = self.physics.new_rate_prod(0., 0)

    def set_spe10_well_controls_initialized(self):
        injector = self.reservoir.get_well('I1')
        producer = self.reservoir.get_well('P1')
        injector.control = self.physics.new_rate_inj(20., self.inj_stream, 0)
        producer.control = self.physics.new_bhp_prod(np.min(self.p_init) - 50.)

class ModelProperties(PropertyContainer):
    def __init__(self, phases_name, components_name, Mw, min_z=1e-11, temperature = 1.):
        # Call base class constructor
        self.nph = len(phases_name)
        # Mw = np.ones(self.nph)
        super().__init__(phases_name, components_name, Mw, min_z, temperature=1.)

    def evaluate(self, state: value_vector):
        """
        Class methods which evaluates the state operators for the element based physics

        :param state: state variables [pres, comp_0, ..., comp_N-1, temperature (optional)]
        :type state: value_vector

        :return: updated value for operators, stored in values
        """
        # Composition vector and pressure from state:
        pressure, temperature, zc = self.get_state(state)

        self.clean_arrays()

        self.ph = self.run_flash(pressure, temperature, zc)

        for j in self.ph:
            M = np.sum(self.Mw * self.x[j][:])

            self.dens[j] = self.density_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :])  # output in [kg/m3]
            
            ########################################################## 
            self.dens_m[j] = self.dens[j] / M  # molar density [kg/m3]/[kg/kmol]=[kmol/m3]
            ##########################################################
            
            self.mu[j] = self.viscosity_ev[self.phases_name[j]].evaluate(pressure, temperature, self.x[j, :], self.dens[j])  # output in [cp]
        self.compute_saturation(self.ph)

        self.pc = self.capillary_pressure_ev.evaluate(self.sat)

        for j in self.ph:
            self.kr[j] = self.rel_perm_ev[self.phases_name[j]].evaluate(self.sat[j])
            self.pc = [0, 0]

        mass_source = self.evaluate_mass_source(pressure, temperature, zc)

        return self.ph, self.sat, self.x, self.dens, self.dens_m, self.mu, self.kr, self.pc, mass_source