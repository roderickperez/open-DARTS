from darts.models.reservoirs.struct_reservoir import StructReservoir
from darts.models.darts_model import DartsModel
import numpy as np
from darts.tools.keyword_file_tools import load_single_keyword
from darts.hdata.geothermal import Geothermal
from darts.models.physics.iapws.iapws_property import *
from darts.models.physics.iapws.custom_rock_property import *
from darts.hdata.property_container import *
from darts.engines import value_vector, conn_mesh, sim_params, timer_node
from darts.models.physics.iapws.iapws_property_vec import _Backward1_T_Ph_vec
import sys
import matplotlib.pyplot as plt
from scipy import interpolate
from darts.engines import value_vector, redirect_darts_output, sim_params

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

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

        self.params.first_ts = 1e-6
        self.params.mult_ts = 16
        self.params.max_ts = 365

        # Newton tolerance is relatively high because of L2-norm for residual and well segments
        self.params.tolerance_newton = 1e-1
        self.params.tolerance_linear = 1e-5
        self.params.max_i_newton = 10
        self.params.max_i_linear = 50

        self.params.newton_type = sim_params.newton_global_chop
        self.params.newton_params = value_vector([1])

        self.runtime = 1000
        self.timer.node["initialization"].stop()

        redirect_darts_output('log.txt')

    def init_grid(self):
        self.nx = 230
        self.ny = 206
        nz = 30
        nb = self.nx * self.ny * nz
        id = 1
        all_active = True

        dx = np.zeros(nb)
        dy = np.zeros(nb)
        dz = np.zeros(nb)
        permx = np.zeros(nb)
        permy = np.zeros(nb)
        permz = np.zeros(nb)
        poro = np.zeros(nb)
        depth = np.zeros(nb)
        actnum = np.zeros(nb)

        # Grid size assignment
        dx[0::] = load_single_keyword('dap_model.GRDECL', 'DX', cache=1)
        dy[0::] = load_single_keyword('dap_model.GRDECL', 'DY', cache=1)
        dz[0::] = load_single_keyword('dap_model.GRDECL', 'DZ', cache=1)
        permx[0::] = load_single_keyword('dap_model.GRDECL', 'PERMX', cache=1)
        permy[0::] = permx
        permz[0::] = permx / 10
        poro[0::] = load_single_keyword('dap_model.GRDECL', 'PORO', cache=1)
        depth[0::] = load_single_keyword('dap_model.GRDECL', 'DEPTH', cache=1)
        self.coord = load_single_keyword('CASE_GRID.GRDECL', 'COORD', cache=1)
        zcorn = load_single_keyword('CASE_GRID.GRDECL', 'ZCORN', cache=1)

        actnum[::] = 1

        permx[permx <= 1e-5] = 1e-5
        permy[permx <= 1e-5] = 1e-5
        permz[permx <= 1e-5] = 1e-5
        poro[poro <= 1e-5] = 1e-5

        self.reservoir = StructReservoir(self.timer, nx=self.nx, ny=self.ny, nz=nz, dx=dx, dy=dy, dz=dz, permx=permx,
                                         permy=permy, permz=permz, poro=poro, depth=depth,
                                         actnum=actnum, coord=self.coord, zcorn=zcorn)

        self.reservoir.set_boundary_volume(yz_minus=1e15, yz_plus=1e15, xz_minus=1e15,
                                           xz_plus=1e15, xy_minus=1e15, xy_plus=1e15)

        # get wrapper around local array (length equal to active blocks number)
        cond_mesh = np.array(self.reservoir.mesh.rock_cond, copy=False)
        hcap_mesh = np.array(self.reservoir.mesh.heat_capacity, copy=False)
        cond_mesh[self.reservoir.mesh.poro == 1e-5] = 2.2*86.4
        cond_mesh[self.reservoir.mesh.poro != 1e-5] = 3*86.4
        hcap_mesh[self.reservoir.mesh.poro == 1e-5] = 400*2.5
        hcap_mesh[self.reservoir.mesh.poro != 1e-5] = 400*2.5

        # compute temperature gradient correlated with x,y coordinate
        self.anomaly = np.zeros(nb)

        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(nz):
                    self.anomaly[i + j * self.nx + k * self.nx * self.ny] = \
                        float(self.nx - i) / self.nx * 2000 - float(self.ny - j) / self.ny * 1000

        self.anomaly = self.anomaly[self.reservoir.discretizer.local_to_global]


    def init_wells(self):

        well_list = ['I01', 'P01', 'I02', 'P02', 'I03', 'P03', 'E01', 'E02', 'E03', 'E04', 'E05', 'E06']
        well_type = ['INJ', 'PRD', 'INJ', 'PRD', 'INJ', 'PRD', 'EXP', 'EXP', 'EXP', 'EXP', 'EXP', 'EXP']
        well_x    = [  190,   164,   136,   162,   109,   134,   195,   145,    46,    52,    72,   170]
        well_y    = [  100,   110,    75,    87,   113,   115,    70,    41,    46,   100,   152,   170]
        well_x_coord = []
        well_y_coord = []

        n_wells = len(well_list)

        for i in range(n_wells):
            well_x_coord.append(self.coord[6 * (well_x[i] + well_y[i] * (self.nx + 1))])
            well_y_coord.append(self.coord[6 * (well_x[i] + well_y[i] * (self.nx + 1)) + 1])

        def str2file(fp, name_in, list_in):
            fp.write("%s = [" % name_in)
            for item in list_in:
                fp.write("\'%s\', " % item)
            fp.write("]\n")

        def num2file(fp, name_in, list_in):
            fp.write("%s = [" % name_in)
            for item in list_in:
                fp.write("%d, " % int(item))
            fp.write("]\n")

        f = open('well_gen.txt', 'w')
        str2file(f, 'well_list', well_list)
        str2file(f, 'well_type', well_type)
        num2file(f, 'well_x', well_x_coord)
        num2file(f, 'well_y', well_y_coord)
        f.close()

        for i in range(n_wells):
            self.add_well(well_list[i], well_x[i], well_y[i])


    def add_well(self, name, i_coord, j_coord):

        for w in self.reservoir.wells:
            if name == w.name:
                print("Warning! Well %s already exist." % name)

        self.reservoir.add_well(name)
        for n in range(5, 25):
            self.reservoir.add_perforation(well=self.reservoir.wells[-1], i=i_coord, j=j_coord, k=n+1,
                                           well_index=-1, multi_segment=False, verbose=False)

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

        self.physics = Geothermal(property_container=self.property_container, timer=self.timer, n_points=32, min_p=0.1,
                                  max_p=450, min_e=1, max_e=10000, grav=False)

    def set_initial_conditions(self):
        # self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=200, uniform_temperature=273.15+75)

        self.timer.node["initialization"].start()
        self.timer.node["initialization"].node["initial conditions"] = timer_node()
        self.timer.node["initialization"].node["initial conditions"].start()

        # self.physics.set_uniform_initial_conditions(self.reservoir.mesh, uniform_pressure=115,
        #                                             uniform_temperature=316.65)

        self.physics.set_nonuniform_initial_conditions(self.reservoir.mesh, 100, 30, self.anomaly)

        self.timer.node["initialization"].node["initial conditions"].stop()
        self.timer.node["initialization"].stop()

    def set_boundary_conditions(self):
        # define all wells as closed
        for i, w in enumerate(self.reservoir.wells):
            pres_lim = 0.130 * self.reservoir.mesh.depth[w.perforations[0][1]]  # 30% from natural gradient
            w.constraint = self.physics.new_bhp_water_inj(pres_lim, 273.15 + 25)
            if 'I' in w.name or 'N01' in w.name:
                w.control = self.physics.new_rate_water_inj(0, 273.15 + 25)
            else:
                w.control = self.physics.new_rate_water_prod(0)



    def set_op_list(self):
        self.op_list = [self.physics.acc_flux_itor, self.physics.acc_flux_itor_well]
        op_num = np.array(self.reservoir.mesh.op_num, copy=False)
        op_num[self.reservoir.mesh.n_res_blocks:] = 1

    def export_pro_vtk(self, file_name='Results'):
        X = np.array(self.physics.engine.X, copy=False)
        nb = self.reservoir.mesh.n_res_blocks
        T = _Backward1_T_Ph_vec(X[0:nb*2:2] / 10, X[1:nb*2:2] / 18.015)
        local_cell_data = {'Temperature': T,
                           'Poro': self.reservoir.global_data['poro'][self.reservoir.discretizer.local_to_global]}

        self.export_vtk(file_name, local_cell_data=local_cell_data)

    def run_welltests(self, prev_time=0.0):
        drawdown_time = 10
        buildup_time = 10
        num_of_steps = 1
        prev_id = 0
        self.params.first_ts = 1e-6

        for w in self.reservoir.wells:
            self.set_well_control(w, q_wat=200.0)
            dt = 1.E-6
            self.params.mult_ts = 1.3
            self.params.max_ts = 5

            for i in range(num_of_steps):
                self.run_python(drawdown_time / num_of_steps, restart_dt=dt)
            self.set_well_control(w, q_wat=0.0)
            dt = 1e-6
            self.params.mult_ts = 1.3
            self.params.max_ts = 5
            for i in range(num_of_steps):
                self.run_python(buildup_time / num_of_steps, restart_dt=dt)
            # Save
            time_data = self.physics.engine.time_data
            time = np.array(time_data['time'], copy=True)  # from days to seconds
            pressure = np.array(time_data[w.name + ' : BHP (bar)'])
            st = -1 #np.argwhere(time == prev_time + drawdown_time)[0, 0] + 1
            water_rate = np.array(time_data[w.name + ' : water rate (m3/day)'])
            self.print_report(w, time[prev_id:st] - prev_time, pressure[prev_id:st],
                                water_rate[prev_id:st])
            prev_id = time.size
            prev_time = time[prev_id - 1]

    def print_report(self, well, time, pressure, water_rate):
        name = 'welltests/' + well.name + '_welltest.txt'
        with open(name, 'w+') as f:
            f.write("@@@ Well-test report, powered by DARTS (https://darts.citg.tudelft.nl/) @@@\n")
            f.write("@@@           Special edition for SPE GeoEnergy Hackathon 2021          @@@\n")
            f.write("Well name: %s\n" % well.name)
            pu = well.perforations[0]
            pl = well.perforations[-1]
            f.write("Perforation interval: %d to %d\n" % (self.reservoir.mesh.depth[pu[1]], self.reservoir.mesh.depth[pl[1]]))
            f.write("Fluid in place: %s\n" % 'brine')

            X = np.array(self.physics.engine.X, copy=False)
            P = X[pu[1]*2]
            T = _Backward1_T_Ph_vec(P / 10, X[pu[1]*2+1] / 18.015)

            f.write("Initial pressure: %12.3f\n" % P)
            f.write("Temperature: %12.3f\n" % T)
            f.write('time(day)\t water_rate(m3/d)\t BHP(bars)\n')
            for i in range(len(time)):
                f.write("%8.6f\t%12.3f\t%12.6f\n"
                        % (time[i], -water_rate[i], pressure[i]))
        f.close()


    def generate_logs(self, well):
        name = 'welllogs/' + well.name + '_welllogs.txt'
        n_logs = 500
        dtf = 580  # acoustic transit time in brine (micron/sec)
        dtr = 170  # acoustic transit time in quartz (micron/sec)

        with open(name, 'w+') as f:
            f.write("@@@ Acoustic log report, powered by DARTS (https://darts.citg.tudelft.nl/) @@@\n")
            f.write("@@@            Special edition for SPE GeoEnergy Hackathon 2021            @@@\n")
            f.write("Well name: %s\n" % well.name)

            poro_perf = np.zeros(len(well.perforations))
            depth_perf = np.zeros(len(well.perforations))
            for i, p in enumerate(well.perforations):
                poro_perf[i] = self.reservoir.mesh.poro[p[1]]
                depth_perf[i] = self.reservoir.mesh.depth[p[1]]

            depth = np.linspace(depth_perf[0], depth_perf[-1], n_logs)
            interp = interpolate.interp1d(depth_perf, poro_perf, kind="nearest")
            noise = np.random.normal(0, 0.01, size=n_logs)
            poro = interp(depth) + noise
            acoustic = poro * (dtf - dtr) + dtr

            f.write('depth(m)\t transit time (micron/sec)\n')
            for i in range(n_logs):
                f.write("%8.4f\t%12.6f\n" % (depth[i], acoustic[i]))

        # plt.plot(acoustic, depth)
        # plt.grid()
        # plt.show()
        f.close()


    def set_well_control(self, w, q_wat, p_bhp_min=50, p_bhp_max=450):
        if q_wat < 0.0:
            w.control = self.physics.new_rate_water_inj(-q_wat, [0.99])
            w.constraint = self.physics.new_bhp_water_inj(p_bhp_max, [0.99])
        else:
            w.control = self.physics.new_rate_water_prod(q_wat)
            w.constraint = self.physics.new_bhp_prod(p_bhp_min)



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


