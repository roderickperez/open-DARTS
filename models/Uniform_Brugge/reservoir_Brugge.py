from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, timer_node
import numpy as np
from math import pi
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
import sys
from calculate_WI import calc_equivalent_WI


class UnstructReservoir(ReservoirBase):
    def __init__(self, timer: timer_node, permx, permy, permz, frac_aper, mesh_file, poro, thickness,
                 rcond=0, hcap=0, op_num=0, calc_equiv_WI=True, cache: bool = False):
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
        super().__init__(timer, cache)

        self.mesh_file = mesh_file

        self.calc_equiv_WI = calc_equiv_WI
        self.thickness = thickness

        self.permx = permx
        self.permy = permy
        self.permz = permz
        self.poro = poro
        self.frac_aper = frac_aper
        self.rcond = rcond
        self.hcap = hcap
        self.op_num = op_num

        # Create empty list of wells:
        self.wells = []

    def discretize(self):
        # Construct instance of Unstructured Discretization class:
        self.discretizer = UnstructDiscretizer(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=self.frac_aper,
                                               mesh_file=self.mesh_file)

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh()

        # Calculate cell information of each geometric element in the .msh file:
        self.discretizer.calc_cell_information()

        # Store volumes and depth to single numpy arrays:
        self.discretizer.store_volume_all_cells()
        self.discretizer.store_depth_all_cells()
        self.discretizer.store_centroid_all_cells()
        self.cac = self.discretizer.centroid_all_cells

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_connections_all_cells()

        # # Write to files (in case someone needs this for Eclipse or other simulator):
        # self.discretizer.write_conn2p_to_file(cell_m, cell_p, tran, file_name='conn2p.dat')
        # self.discretizer.write_conn2p_therm_to_file(cell_m, cell_p, tran, tran_thermal, file_name='conn2p.dat.connsn')
        # self.discretizer.write_volume_to_file(file_name='vol.dat')
        # self.discretizer.write_depth_to_file(file_name='depth.dat')

        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        mesh = conn_mesh()

        # Initialize mesh with all four parameters (cell_m, cell_p, trans, trans_D):
        mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))
        # self.mesh.init('conn2p.dat.connsn')

        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.discretizer.volume_all_cells.size
        self.num_frac = self.discretizer.fracture_cell_count
        self.num_mat = self.discretizer.matrix_cell_count

        self.permx = self.permx * np.ones(self.nb)
        self.permy = self.permy * np.ones(self.nb)

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        return mesh

    def add_well(self, name, wellbore_diameter=0.15):
        """
        Class method which adds wells heads to the reservoir (Note: well head is not equal to a perforation!)
        :param name:
        :param wellbore_diameter:
        :return:
        """
        well = ms_well()
        well.name = name
        # so far put only area here,
        # to be multiplied by segment length later

        well.segment_volume = pi * wellbore_diameter ** 2 / 4

        # also to be filled up when the first perforation is made
        well.well_head_depth = 0
        well.well_body_depth = 0
        well.segment_depth_increment = 0
        self.wells.append(well)
        return well

    def add_perforation(self, well, res_block, well_index, well_indexD=0):
        """
        Class method which ads perforation to each (existing!) well
        :param well: data object which contains data of the particular well
        :param res_block: reservoir block in which the well has a perforation
        :param well_index: well index (productivity index)
        :return:
        """
        # well_block = 0
        # well.perforations = well.perforations + [(well_block, res_block, well_index)]

        well_block = len(well.perforations)
        # add completion
        well.perforations = well.perforations + [(well_block, res_block, well_index, well_indexD)]
        return 0

    def init_wells(self, mesh: conn_mesh):
        """
        Class method which initializes the wells (adding wells and their perforations to the reservoir)
        :return:
        """
        well_coord = np.genfromtxt('Brugge_struct/well_coord_Brugge.txt')
        self.index_cell = []

        for i, wc in enumerate(well_coord):
            distance = []
            for j, centroid in enumerate(self.cac):
                distance.append(np.linalg.norm(wc - centroid))
            min_dis = np.min(distance)
            self.index_cell.append(distance.index(min_dis) + 1)

        try:
            assert (len(set(self.index_cell)) == len(well_coord))
        except AssertionError:
            print('There are at least 2 wells locating in the same grid block!!! The mesh file should be modified!')
            sys.exit()

        self.well_index_list = []
        if self.calc_equiv_WI:
            well_coord_list = well_coord
            for wc in well_coord_list:
                WI = calc_equivalent_WI(mesh_file=self.mesh_file, well_coord=wc, centroid_info=self.cac,
                                        kx_list=self.permx, ky_list=self.permy, dz=self.thickness, skin=0)
                self.well_index_list.append(WI)
        else:
            self.well_index_list = [296.65303668, 69.71905642, 27.14929434, 27.58575654, 53.01869826,
                                    135.80457602, 345.34715322, 80.69146768, 74.07293499, 243.34286931,
                                    482.48726884, 592.37938461, 307.72095815, 542.44279506, 63.58456305,
                                    499.53586907, 213.09805386, 303.97957677, 78.80966839, 791.4220401,
                                    829.44984229, 794.13401768, 761.08006179, 62.37906275, 616.74501491,
                                    475.63754963, 397.50862698, 478.21742722, 504.06328513, 655.32259614]


        n_injector = 10  # the first 10 wells are injectors
        n_wells = 30  # the number of wells

        for i in range(np.size(self.well_index_list)):
            if i < n_injector:
                self.add_well("I" + str(i + 1))
                self.add_perforation(well=self.wells[-1], res_block=int(self.index_cell[i]),
                                     well_index=self.well_index_list[i])
            else:
                self.add_well("P" + str(i + 1 - n_injector))
                self.add_perforation(well=self.wells[-1], res_block=int(self.index_cell[i]),
                                     well_index=self.well_index_list[i])

        # Add wells to the DARTS mesh object and sort connection (DARTS related):
        mesh.add_wells(ms_well_vector(self.wells))
        mesh.reverse_and_sort()
        mesh.init_grav_coef()
        return self.wells



