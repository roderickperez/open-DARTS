import numpy as np
import csv
import pickle
import os
from darts.reservoirs.mesh.geometry.map_mesh import MapMesh, _translate_curvature


class CSVResults:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.volume = []
        self.centroids = []
        self.conn_0 = []
        self.conn_1 = []
        self.poro = []
        self.op_num = []
        self.corey = {}
        self.seal = []
        self.regions = []
        self.M_CO2 = 44.01  # kg/kmol

        # SPATIAL MAP
        self.x = np.arange(5E-3, 2.86, 1E-2)
        self.y = np.array([0.])
        self.z = np.arange(5E-3, 1.23, 1E-2)
        self.spatial_map_size = (len(self.x), len(self.y), len(self.z))
        self.spatial_map_cells = np.zeros(self.spatial_map_size)

        # BOX C SPATIAL MAP
        self.xC = np.arange(1.13, 2.63, 1e-2)
        self.yC = np.array([0.])
        self.zC = np.arange(0.13, 0.43, 1e-2)
        self.boxC_map_size = (len(self.xC), len(self.yC), len(self.zC))
        self.boxC_map_cells = np.zeros(self.boxC_map_size)

        # self.timestep_sm = 1  # 1 day
        self.ith_step_sm = 1

        # TIME SERIES & SPARSE DATA
        self.sensors = []
        self.sensor_loc = [[1.53, 0., 0.53], [1.73, 0., 1.13]]
        self.boxes = []
        self.boxes_range = [[[1.13, 2.83], [0., 0.025], [0.03, 0.63]],
                            [[0.03, 1.13], [0., 0.025], [0.63, 1.23]],
                            [[1.13, 2.63], [0., 0.025], [0.13, 0.43]]]
        self.connections = []
        self.max_p1 = 0
        self.max_p2 = 0
        self.max_mobA = 0

        self.time_series = []
        self.time_series_boxC = []
        self.timestep_ts = 10 / (24 * 60)  # 10 min
        self.ith_step_ts = 0

        self.sparse_data_label = ['1a', '1b', '2', '3a', '3b', '3c', '3d', '4a', '4b', '4c', '4d', '5', '6']
        self.sparse_data = np.zeros(len(self.sparse_data_label))

    def find_output_cells(self, curvature=0, cache=0):
        """
        Function to find indices of cells needed for reporting:
        - Spatial map: xy-grid of 1 cm x 1 cm cells
        - Time series & sparse data:
            -- P1, P2
            -- boxes A, B and C
            -- connections in box C
        """
        read_from_cache = False

        if cache:
            if os.path.isfile(self.output_dir + '/outputCells.cache'):
                with open(self.output_dir + '/outputCells.cache', 'rb') as handle:
                    self.outputCells = pickle.load(handle)

                self.spatial_map_cells = self.outputCells['spatial_map_cells']
                self.boxC_map_cells = self.outputCells['boxC_map_cells']
                self.sensors = self.outputCells['sensors']
                self.boxes = self.outputCells['boxes']
                # self.connections = self.outputCells['connections']
                read_from_cache = True
                print("Read output cells from cache.")

        if not read_from_cache:
            if curvature:
                self.centroids = _translate_curvature(self.centroids)

            mapmesh = MapMesh(self.centroids)
            self.spatial_map_cells = mapmesh.map_to_structured(self.x, self.y, self.z).astype(int)
            self.boxC_map_cells = mapmesh.map_to_structured(self.xC, self.yC, self.zC).astype(int)
            self.sensors = mapmesh.find_nearest_cells(self.sensor_loc).astype(int)
            self.boxes = []
            for box_range in self.boxes_range:
                self.boxes.append(mapmesh.find_cells_in_domain(box_range[0][:], box_range[1][:], box_range[2][:]))
            # self.conns = mapmesh.find_connections(self.boxes[2][:], self.conn_0, self.conn_1)

        if cache and not read_from_cache:
            self.outputCells = {}
            self.outputCells['spatial_map_cells'] = self.spatial_map_cells
            self.outputCells['boxC_map_cells'] = self.boxC_map_cells
            self.outputCells['sensors'] = self.sensors
            self.outputCells['boxes'] = self.boxes
            # self.outputCells['connections'] = self.connections

            with open(self.output_dir + '/outputCells.cache', 'wb') as handle:
                pickle.dump(self.outputCells, handle, protocol=4)
            print("Files have been read and cached.")

        return

    def update_time_series(self, data):
        """
        Function to update time series, reported every 10 minutes:
        - [s] t
        - [N/m2] p_1, p_2
        - [kg] mobile free phase CO2, immobile free phase CO2, dissolved CO2 in Aq, CO2 in seal (box A and B)
        - [-] convective mixing: M = (box C)
        """
        # Time
        t = round(self.ith_step_ts * self.timestep_ts * 86400, 1)
        row = [t]

        # Pressure at sensors 1 and 2
        p1 = data["pressure"][0, self.sensors[0]] * 1E5
        p2 = data["pressure"][0, self.sensors[1]] * 1E5
        row.extend([p1, p2])

        # Phase amounts [kg] in boxes A and B
        for box in self.boxes[:-1]:
            mob = 0
            imm = 0
            diss = 0
            seal = 0
            for cell in box:
                V = self.volume[cell]
                phi = self.poro[cell]
                sgc = self.corey[self.regions[self.op_num[cell]]].sgc

                sg = data["satV"][0, cell]
                xCO2 = data["xCO2"][0, cell]
                rho_v = data["rhoV"][0, cell]
                rho_m_Aq = data["rho_mA"][0, cell]

                # Find total mass of CO2 present in box
                # in reservoir layer
                if sg > sgc:
                    mob += V * phi * (sg - sgc) * rho_v
                    imm += V * phi * sgc * rho_v
                else:
                    imm += V * phi * sg * rho_v
                diss += V * phi * (1 - sg) * xCO2 * rho_m_Aq * self.M_CO2
                # mass in sealing layer
                if cell in self.seal:
                    seal += V * phi * sg * rho_v
                    seal += V * phi * (1 - sg) * xCO2 * rho_m_Aq * self.M_CO2
            row.extend([mob, imm, diss, seal])

        # Convective mixing in box C
        M_C = 0
        # xCO2_max = 0.00085
        # for i, conns in enumerate(self.connections):
        #     cell0 = self.boxes[2][i]
        #     vol = self.volume[cell0]
        #     dz = 0.022  # guess
        #     grad = 0
        #     for cell1 in conns:
        #         xCO2_0 = data[cell0, self.xCO2_idx]
        #         xCO2_1 = data[cell1, self.xCO2_idx]
        #         dist = np.sqrt((self.centroids[cell0][0]-self.centroids[cell1][0])**2 +
        #                        (self.centroids[cell0][1]-self.centroids[cell1][1])**2 +
        #                        (self.centroids[cell0][2]-self.centroids[cell1][2])**2)
        #         grad += 1/3*(xCO2_0-xCO2_1)/xCO2_max/dist
        #     M_C += np.abs(grad*vol/dz)
        row.extend([M_C])

        # Total mass of CO2 in kg inside the computational domain
        mass_CO2 = 0.
        for i, centroid in enumerate(self.centroids):
            V = self.volume[i]
            phi = self.poro[i]
            sg = data["satV"][0, i]
            xCO2 = data["xCO2"][0, i]
            rho_v = data["rhoV"][0, i]
            rho_m_Aq = data["rho_mA"][0, i]  # kmol/m3

            mass_CO2 += V * phi * sg * rho_v
            mass_CO2 += V * phi * (1 - sg) * xCO2 * rho_m_Aq * self.M_CO2  # m3*kmol/m3*kg/kmol = [kg]
        row.extend([mass_CO2])

        self.time_series.append(row)
        print("time series", row)

        # Record concentration in box C
        row = [t]
        for k, z in enumerate(self.zC):
            for j, y in enumerate(self.yC):
                for i, x in enumerate(self.xC):
                    cell = self.boxC_map_cells[i, j, k]
                    xCO2 = data["xCO2"][0, cell]
                    rho_m_Aq = data["rho_mA"][0, cell]
                    row.extend([xCO2 * rho_m_Aq * self.M_CO2])

        self.time_series_boxC.append(row)

        return

    def update_sparse_data(self, data):
        """
        Function to update sparse data for a single run
        - max pressure at sensor P1 and P2
        - time of maximum mobile free phase in box A
        - mob_A, imm_A, diss_A, seal_A 72 hours after injection starts
        - mob_B, imm_B, diss_B, seal_B 72 hours after injection starts
        - time at which M_C first exceeds 110%
        - seal_A at final time
        """
        t = round(self.ith_step_ts * self.timestep_ts * 86400, 1)  # Current time

        # Maximum pressure at sensors 1 and 2
        p1 = data["pressure"][0, self.sensors[0]] * 1E5
        p2 = data["pressure"][0, self.sensors[1]] * 1E5
        if p1 > self.max_p1:
            self.max_p1 = p1
            self.sparse_data[0] = p1
        if p2 > self.max_p2:
            self.max_p2 = p2
            self.sparse_data[1] = p2

        # Time of maximum mobile free phase in box A
        mob = 0
        for cell in self.boxes[0]:
            V = self.volume[cell]
            phi = self.poro[cell]
            sgc = self.corey[self.op_num[cell]].sgc

            sg = data["satV"][0, cell]
            rho_v = data["rhoV"][0, cell]

            if sg > sgc:
                mob += V * phi * (sg - sgc) * rho_v

        if mob > self.max_mobA:
            self.max_mobA = mob
            self.sparse_data[2] = t

        # Phase amounts [kg] in boxes A and B at t = 72 hours
        if self.ith_step_ts == 72*6:  #
            for i, box in enumerate(self.boxes[:-1]):
                mob = 0
                imm = 0
                diss = 0
                seal = 0
                for cell in box:
                    V = self.volume[cell]
                    phi = self.poro[cell]
                    sgc = self.corey[self.op_num[cell]].sgc

                    sg = data["satV"][0, cell]
                    xCO2 = data["xCO2"][0, cell]
                    rho_v = data["rhoV"][0, cell]
                    rho_m_Aq = data["rho_mA"][0, cell]

                    # Find total mass of CO2 present in box
                    # in reservoir layer
                    if sg > sgc:
                        mob += V * phi * (sg - sgc) * rho_v
                        imm += V * phi * sgc * rho_v
                    else:
                        imm += V * phi * sg * rho_v
                    diss += V * phi * (1 - sg) * xCO2 * rho_m_Aq * self.M_CO2
                    # mass in sealing layer
                    if cell in self.seal:
                        seal += V * phi * sg * rho_v
                        seal += V * phi * (1 - sg) * xCO2 * rho_m_Aq * self.M_CO2
                self.sparse_data[3+4*i:7+4*i] = [mob, imm, diss, seal]

        # Time at which M_C first exceeds 110% of boxC width (1.13-2.63m)
        # if M_C > 1.1*1.5 and self.sparse_data[11] == None:
        #     self.sparse_data[11] = t

        # Total CO2 in seal_A [kg] at final time
        if self.ith_step_ts == 5*24*6:
            seal = 0
            for cell in self.boxes[0]:
                V = self.volume[cell]
                phi = self.poro[cell]

                sg = data["satV"][0, cell]
                xCO2 = data["xCO2"][0, cell]
                rho_v = data["rhoV"][0, cell]
                rho_m_Aq = data["rho_mA"][0, cell]

                # Find total mass of CO2 present in box
                if cell in self.seal:  # mass in sealing layer
                    seal += V * phi * sg * rho_v
                    seal += V * phi * (1 - sg) * xCO2 * rho_m_Aq * self.M_CO2

            self.sparse_data[12] = seal

        return

    def write_spatial_map(self, data, t):
        sg = np.zeros(self.spatial_map_size)  # gas saturation
        cCO2 = np.zeros(self.spatial_map_size)  # CO2 concentration

        for i, x in enumerate(self.x):
            for j, y in enumerate(self.y):
                for k, z in enumerate(self.z):
                    cell = self.spatial_map_cells[i, j, k]
                    sg[i, j, k] = data["satV"][0, cell]
                    xCO2 = data["xCO2"][0, cell]
                    rho_m_Aq = data["rho_mA"][0, cell]
                    cCO2[i, j, k] = xCO2 * rho_m_Aq * self.M_CO2

        # Write spatial_map_<X>h.csv
        with open('{:s}/spatial_map_{:s}.csv'.format(self.output_dir, t), 'w', newline='') as csvfile:
            ts_writer = csv.writer(csvfile, delimiter=',')
            ts_writer.writerow(['# x', ' y', ' gas saturation [-]', ' CO2 concentration in water [kg/m3]'])
            for j, z in enumerate(self.z):
                for i, x in enumerate(self.x):
                    ts_writer.writerow(['{:f}'.format(x), '{:f}'.format(z), '{:f}'.format(sg[i, 0, j]), '{:f}'.format(cCO2[i, 0, j])])

        return

    def write_time_series(self):
        # Write time_series.csv
        with open('{:s}/time_series.csv'.format(self.output_dir), 'w', newline='') as csvfile:
            ts_writer = csv.writer(csvfile, delimiter=',')
            ts_writer.writerow(['# t',' p_1',' p_2',' mob_A',' imm_A',' diss_A',' seal_A',' mob_B',' imm_B',' diss_B',' seal_B',' M_C', ' total_CO2_mass'])
            for ith_step, data in enumerate(self.time_series):
                row = []
                for item in data:
                    row.append('{:f}'.format(item))
                ts_writer.writerow(row)

        # Write time_series_boxC.csv
        with open('{:s}/time_series_boxC.csv'.format(self.output_dir), 'w', newline='') as csvfile:
            ts_writer = csv.writer(csvfile, delimiter=',')
            for ith_step, data in enumerate(self.time_series_boxC):
                row = []
                for item in data:
                    row.append('{:f}'.format(item))
                ts_writer.writerow(row)

        return

    def write_sparse_data(self):
        # Write sparse_data_X.csv
        with open('{:s}/sparse_data_{:d}.csv'.format(self.output_dir, self.X), 'w', newline='') as csvfile:
            sd_writer = csv.writer(csvfile, delimiter=',')
            for i, label in enumerate(self.sparse_data_label):
                row = [label, self.sparse_data[i]]
                sd_writer.writerow(row)

        # # Write sparse_data.csv
        # with open('sparse_data.csv', 'w', newline='') as csvfile:
        #     sd_writer = csv.writer(csvfile, delimiter=',')
        #     sd_writer.writerow(['# idx',' p10_mean',' p50_mean',' p90_mean',' p10_dev',' p50_dev',' p90_dev'])
        #     for ith_line, data in enumerate(self.sparse_data):
        #         row = [self.sparse_data_label[ith_line]]
        #         for item in data:
        #             row.append('{:f}'.format(item))
        #         sd_writer.writerow(row)
        return

    def write_realization(self, layers, poro, perm, corey):
        # Write realization_X.csv
        with open('{:s}/realization_{:d}.csv'.format(self.output_dir, self.X), 'w', newline='') as csvfile:
            r_writer = csv.writer(csvfile, delimiter=',')

            # Layer porperm
            r_writer.writerow(['# Layers poro perm'])
            r_writer.writerow(layers)
            r_writer.writerow(poro)
            r_writer.writerow(perm)
            r_writer.writerow(' ')

            # Corey regions
            r_writer.writerow(['# Corey regions nw ng swc sgc krwe krge labda p_entry'])
            for ith_line, corey_region in enumerate(corey):
                row = []
                for field in corey_region.__dataclass_fields__:
                    if field != "sigma":
                        value = getattr(corey_region, field)
                        row.append(value)
                r_writer.writerow(row)

        return

    def generate_sparse_from_time_series(self):
        """
        Function to generate sparse data from time series from multiple realizations
        - max pressure at sensor P1 and P2 - sparse[0], sparse[1]
        - time of maximum mobile free phase in box A - sparse[2]
        - mob_A, imm_A, diss_A, seal_A 72 hours after injection starts - sparse[3]-sparse[6]
        - mob_B, imm_B, diss_B, seal_B 72 hours after injection starts - sparse[7]-sparse[10]
        - time at which M_C first exceeds 110% - sparse[11]
        - seal_A at final time - sparse[12]
        """
        csv_filenames = []
        for file in os.listdir('csv_dir00'):
            if 'time_series' in file and not '.png' in file:
                csv_filenames.append(file)
        csv_filenames.sort()

        data = np.zeros((len(self.sparse_data_label), len(csv_filenames)))
        for x, filename in enumerate(csv_filenames):
            csvData = np.genfromtxt('csv_dir00/' + filename, delimiter=',', skip_header=1)

            max_p1 = 0
            max_p2 = 0
            max_mobA = 0
            mc = False

            for line in csvData:
                if line[1] > max_p1:
                    max_p1 = line[1]
                    data[0, x] = line[1]
                if line[2] > max_p2:
                    max_p2 = line[2]
                    data[1, x] = line[2]
                if line[3] > max_mobA:
                    max_mobA = line[3]
                    data[2, x] = line[0]

                if line[11] > 1.5 and not mc:
                    data[11, x] = line[0]
                    mc = True

                if line[0] == 72*60*60:
                    data[3, x] = line[3]  # mobA at 72 hrs
                    data[4, x] = line[4]  # immA
                    data[5, x] = line[5]  # dissA
                    data[6, x] = line[6]  # sealA

                    data[7, x] = line[7]  # mobB
                    data[8, x] = line[8]  # immB
                    data[9, x] = line[9]  # dissB
                    data[10, x] = line[10]  # sealB

                if line[0] == 120*60*60:
                    data[12, x] = line[6]  # sealA at final time

        sparse = np.zeros((len(self.sparse_data_label), 6))

        for i, arr in enumerate(data):
            sparse[i, 0] = np.percentile(arr, 10)  # p10
            sparse[i, 1] = np.percentile(arr, 50)  # p50
            sparse[i, 2] = np.percentile(arr, 90)  # p90
            dev = np.std(arr)
            sparse[i, 3] = dev  # stddev
            sparse[i, 4] = dev
            sparse[i, 5] = dev

        with open('csv_dir00/sparse_data.csv', 'w', newline='') as csvfile:
            sd_writer = csv.writer(csvfile, delimiter=',')
            sd_writer.writerow(['# idx',' p10_mean',' p50_mean',' p90_mean',' p10_dev',' p50_dev',' p90_dev'])
            for ith_line, line in enumerate(sparse):
                row = [self.sparse_data_label[ith_line]]
                for item in line:
                    row.append('{:f}'.format(item))
                sd_writer.writerow(row)
