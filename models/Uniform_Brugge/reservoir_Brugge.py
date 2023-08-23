from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, timer_node
import numpy as np
from math import pi
from darts.reservoirs.reservoir_base import ReservoirBase
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
import sys
from calculate_WI import calc_equivalent_WI


class UnstructReservoir(ReservoirBase):
    def __init__(self, timer: timer_node, mesh_file: str, permx, permy, permz, frac_aper, poro, thickness,
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
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file,
                                               permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=self.frac_aper)

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

        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        mesh = conn_mesh()
        mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.op_num, copy=False)[:] = self.op_num
        np.array(mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        # Calculate well_index (very primitive way....):
        self.well_index = np.mean(tran) * 1
        # self.well_index = 10

        return mesh

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
