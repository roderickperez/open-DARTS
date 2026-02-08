import numpy as np
from math import inf, pi, asin

from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector

from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer

from darts.reservoirs.mesh.geometry.unstructured import Unstructured
from darts.reservoirs.mesh.geometry.fluidflower import FluidFlower
from darts.reservoirs.mesh.geometry.wells import CircularWell, WellCell

import os


class FluidFlowerUnstruct(UnstructReservoir):
    def __init__(self, timer, layer_properties):
        """
        Class constructor for UnstructReservoir class
        :param permx: Matrix permeability in the x-direction (scalar or vector)
        :param permy: Matrix permeability in the y-direction (scalar or vector)
        :param permz: Matrix permeability in the z-direction (scalar or vector)
        :param frac_aper: Aperture of the fracture (scalar or vector)
        :param mesh_file: Name and relative path to the mesh-file (string)
        :param poro: Matrix (and fracture?) porosity (scalar or vector)
        :param bound_cond: switch which determines the type of boundary conditions used (string)
        """
        #filename = 'malla_1'

        fname = 'malla_1.msh'
        mesh_file = os.path.join('meshes', fname)
        
        self.layer_properties = layer_properties

        self.physical_tags['matrix'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 ]  # 
        #self.physical_tags['boundary'] = [1,2,30,31,32,40,41,42,50,51,52,60,61,62]  # order: Z- (bottom); Z+ (top) ; Y-; X+; Y+; X-


        #     permx = 5.0 # [mD]
        #     permy = 5.0  # [mD]
        #     permz = 5.0  # [mD]
        poro = 0.01 #    = 1%
        hcap = 2470.0  # [kJ/m3/K]
        rcond = 172.8   # [kJ/m/day/K]

        # initialize reservoir

        super().__init__(timer=timer, mesh_file=mesh_file ,
                         permx=1, permy=1, permz=1, poro=poro, hcap=hcap, rcond=rcond)

    def discretize(self, cache: bool = False, verbose: bool = False):
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags, verbose=verbose)

        self.discretizer.n_dim = 3

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0, cache=cache)

        # # Calculate cell information of each geometric element in the .msh file:
        # self.discretizer.calc_cell_information(cache=cache)

        start_z= 1500.

        for ith_cell in self.discretizer.mat_cell_info_dict:
            self.discretizer.mat_cell_info_dict[ith_cell].depth += start_z
            
            # Sumar al último elemento de centroid (z en un vector 3x1)
            # MODIFICAR EL CENTROIDE CAMBIA COMPLETAMENTE LA SIMULACION, EVITARLO  !!!
            #self.discretizer.mat_cell_info_dict[ith_cell].centroid[-1] += start_z

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroid_all_cells()

        # Assign layer properties
        self.set_layer_properties()

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_connections_all_cells(cache=cache)

        # Initialize mesh with all four parameters (cell_m, cell_p, trans, trans_D):
        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        self.mesh = conn_mesh()
        self.mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(self.mesh.poro, copy=False)[:] = self.poro
        np.array(self.mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(self.mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(self.mesh.op_num, copy=False)[:] = self.op_num


        n_elements=self.discretizer.mat_cells_tot + self.discretizer.frac_cells_tot
        np.array(self.mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells[:n_elements]
        np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells[:n_elements]

        # número de celdas
        n_cells = len(self.discretizer.mat_cell_info_dict)

        # arreglo vacío 1D
        self.permx = np.zeros(n_cells)
        self.permy = np.zeros(n_cells)
        self.permz = np.zeros(n_cells)

        # llenar con kx de cada celda
        for i, ith_cell in enumerate(self.discretizer.mat_cell_info_dict):
            self.permx[i] = self.discretizer.mat_cell_info_dict[ith_cell].permeability[0]  
            self.permy[i] = self.discretizer.mat_cell_info_dict[ith_cell].permeability[1] 
            self.permz[i] = self.discretizer.mat_cell_info_dict[ith_cell].permeability[2] 

        
        # print(self.discretizer.mat_cell_info_dict[1810].permeability )
        # print(self.discretizer.mat_cell_info_dict[4072].permeability )
        # print(self.discretizer.mat_cell_info_dict[6334].permeability )

        # print(self.discretizer.mat_cell_info_dict[1810].depth )
        # print(self.discretizer.mat_cell_info_dict[4072].depth )
        # print(self.discretizer.mat_cell_info_dict[6334].depth )

        # print(self.discretizer.mat_cell_info_dict[1810].centroid )
        # print(self.discretizer.mat_cell_info_dict[4072].centroid )
        # print(self.discretizer.mat_cell_info_dict[6334].centroid )


        #np.array(self.permx, copy=False)[:]= 10.
        #np.array(self.mesh.permx, copy=False)[:] = self.water_column_height  # - self.discretizer.depth_all_cells  # inverse height for depth

        
        # self.water_column_height = 0       #  
        # self.min_depth = self.water_column_height - max(self.discretizer.depth_all_cells)
        # np.array(self.mesh.depth, copy=False)[:] = self.water_column_height  # - self.discretizer.depth_all_cells  # inverse height for depth
        # np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells



        return self.mesh

    def set_layer_properties(self):
        # Extract layer type of each cell
        self.layers = {}
        self.seal = []
        for tag in self.discretizer.physical_tags['matrix']:
            self.layers[tag] = []

        for geometry, tags in sorted(list(self.discretizer.mesh_data.cell_data_dict['gmsh:physical'].items()),
                                     key=lambda x: -x[1][0]):
            # Main loop over different existing geometries
            for ith_cell, nodes_to_cell in enumerate(self.discretizer.mesh_data.cells_dict[geometry]):
                tag = tags[ith_cell]
                if tag in self.discretizer.physical_tags['matrix']:
                    self.layers[tags[ith_cell]].append(ith_cell)
                

        # Assign properties to layers
        n_layers = len(self.layer_properties)   #   31
        layer_op_num = np.zeros(n_layers)
        layer_poro = np.zeros(n_layers)
        layer_hcap = np.zeros(n_layers)
        layer_rcond = np.zeros(n_layers)
        layer_perm = np.zeros((n_layers, 3))
        for i, (layer, porperm) in enumerate(self.layer_properties.items()):
            layer_op_num[i] = 0                #          revisar!!
            layer_poro[i] = porperm.poro
            layer_hcap[i] = porperm.hcap
            layer_rcond[i] = porperm.rcond
            for j in range(3):
                layer_perm[i, j] = porperm.perm * porperm.anisotropy[j]

        self.poro = np.zeros(self.discretizer.volume_all_cells.size)
        self.hcap = np.zeros(self.discretizer.volume_all_cells.size)
        self.rcond = np.zeros(self.discretizer.volume_all_cells.size)
        self.op_num = np.zeros(self.discretizer.volume_all_cells.size)

        ith_layer = -1
        for tag, layer in self.layers.items():
            ith_layer += 1
            for ith_cell in layer:
                self.discretizer.mat_cell_info_dict[ith_cell].permeability = layer_perm[ith_layer, :]
                self.poro[ith_cell] = layer_poro[ith_layer]
                self.hcap[ith_cell] = layer_hcap[ith_layer]
                self.rcond[ith_cell] = layer_rcond[ith_layer]
                self.op_num[ith_cell] = layer_op_num[ith_layer]

        return
