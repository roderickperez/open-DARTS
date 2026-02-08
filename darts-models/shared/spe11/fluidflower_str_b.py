import numpy as np
from darts.reservoirs.struct_reservoir import StructReservoir

from darts.reservoirs.mesh.geometry.structured import Structured
from darts.reservoirs.mesh.geometry.fluidflower import FluidFlower


class FluidFlowerStruct(StructReservoir):
    def __init__(self, timer, layer_properties, layers_to_regions, model_specs, well_centers):
        str_mesh = Structured(dim=2, axs=[0, 2])

        spe11b = FluidFlower(lc=[0.1])
        spe11b.convert_to_spe11b()
        str_mesh.add_shape(spe11b)
        # str_mesh.add_shape(FluidFlower(lc=[0.1]))


        nx, ny, nz = model_specs['nx'], 1, model_specs['nz']
        x, y, z, cell_to_layer, actnum = str_mesh.generate_mesh2(nx=nx, ny=ny, nz=nz)

        dx, dy, dz = x[1] - x[0], 1.0, z[1] - z[0]

        # Define depth, porosity and permeability for each cell
        grid_size = nx * ny * nz
        ACTNUM, depth, poro, permx, permy, permz, hcap, rcond, op_num = (
            np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size),
            np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size), np.zeros(grid_size, dtype=int)
        )

        n_local = np.sum(actnum)
        self.centroids = np.zeros((n_local, 3))
        self.seal = []

        local_idx = 0
        water_column_depth = 1200.0
        self.min_depth = water_column_depth - nz * dz
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    global_idx = k * nx * ny + j * nx + i
                    depth[global_idx] = water_column_depth - k * dz
                    if actnum[i, j, k]:
                        tag = cell_to_layer[i, j, k] - 1 + 900001
                        ACTNUM[global_idx] = 1
                        poro[global_idx] = layer_properties[tag].poro
                        perm = layer_properties[tag].perm
                        anisotropy = layer_properties[tag].anisotropy
                        permx[global_idx] = perm * anisotropy[0]
                        permy[global_idx] = perm * anisotropy[1]
                        permz[global_idx] = perm * anisotropy[2]
                        hcap[global_idx] = layer_properties[tag].hcap
                        rcond[global_idx] = layer_properties[tag].rcond
                        op_num[global_idx] = layers_to_regions[layer_properties[tag].type]

                        self.centroids[local_idx] = np.array([x[i] + 0.5 * dx, y[j] + 0.5 * dy, z[k] + 0.5 * dz])
                        # poro[local_idx] = poro[global_idx]
                        # op_num[local_idx] = op_num[global_idx]
                        if layer_properties[tag].type == "ESF":
                            self.seal.append(local_idx)
                        local_idx += 1

        super().__init__(timer, nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, permx=permx, permy=permy, permz=permz,
                         poro=poro, rcond=rcond, hcap=hcap, op_num=op_num, depth=depth, actnum=actnum)

        self.well_centers = well_centers

    def set_layer_properties(self):
        pass

    def set_wells(self, verbose: bool = False):
        for name, center in self.well_centers.items():
            # cell_index = self.find_cell_index(center)
            dx, dy, dz = self.global_data['dx'], self.global_data['dy'], self.global_data['dz']
            cell_index = (int(center[0] / dx[0, 0, 0])+1, int(center[1] / dy[0, 0, 0])+1, int(center[2] / dz[0, 0, 0])+1)
            self.add_well(name, wellbore_diameter=0.3)
            self.add_perforation(name, cell_index, well_radius=0.15, well_indexD=0, segment_direction='y_axis', verbose=verbose)

        # self.add_well('P1')
        # for n in range(self.nx):
        #     self.add_perforation("P1", cell_index=(n + 1, 1, self.nz), well_index=100, well_indexD=0, verbose=verbose)
        return
