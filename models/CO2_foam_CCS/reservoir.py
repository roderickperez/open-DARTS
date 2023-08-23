from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector, timer_node
import numpy as np
from math import pi
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.reservoirs.reservoir_base import ReservoirBase


# Definitions for the unstructured reservoir class:
class UnstructReservoir(ReservoirBase):
    def __init__(self, timer: timer_node, mesh_file: str, permx, permy, permz, frac_aper, poro,
                 rcond=0, hcap=0, op_num=0, cache: bool = False):
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

    def discretize(self) -> conn_mesh:
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

        # Perform discretization:
        cell_m, cell_p, tran, tran_thermal = self.discretizer.calc_connections_all_cells()

        # Create mesh object (C++ object used by DARTS for all mesh related quantities):
        mesh = conn_mesh()
        # Initialize mesh using built connection list
        mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.discretizer.volume_all_cells.size
        self.num_frac = self.discretizer.fracture_cell_count
        self.num_mat = self.discretizer.matrix_cell_count

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        # rock thermal properties
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond

        # Calculate well_index (very primitive way....):
        self.well_index = np.mean(tran) * 1
        # self.well_index = 10

        return mesh

    def calc_boundary_cells(self):
        # Set-up dictionary with data for boundary cells:
        boundary_data = dict()  # Dictionary containing boundary condition data (coordinate and value of boundary):
        boundary_data['first_boundary_dir'] = 'X'  # Indicates the boundary is located at constant X (in this case!)
        # Constant X-coordinate value at which the boundary is located (used to be 3.40885):
        boundary_data['first_boundary_val'] = np.min(self.discretizer.mesh_data.points[:, 0])

        # Same as above but for the second boundary condition!
        boundary_data['second_boundary_dir'] = 'X'
        # Constant X-coordinate value at which the boundary is located (used to be 13.0014):
        boundary_data['second_boundary_val'] = np.max(self.discretizer.mesh_data.points[:, 0])

        # Calculate boundary cells using the calc_boundary_cells method:
        self.left_boundary_cells, self.right_boundary_cells = self.discretizer.calc_boundary_cells(boundary_data)

        # Calc maximum size of well cells (used to have more homogeneous injection conditions by scaling the WI):
        dummy_vol = np.array(self.volume, copy=True)
        self.max_well_vol = np.max([np.max(dummy_vol[self.left_boundary_cells]),
                                    np.max(dummy_vol[self.right_boundary_cells])])

        self.volume[self.right_boundary_cells] = self.volume[self.right_boundary_cells]*1e8

        # Create empty list of wells:
        self.wells = []

    def add_perforations(self, well, res_block, well_index, well_indexD=0.):
        """
        Class method which ads perforation to each (existing!) well
        :param well: data object which contains data of the particular well
        :param res_block: reservoir block in which the well has a perforation
        :param well_index: well index (productivity index)
        :return:
        """
        well_block = 0
        well.perforations = well.perforations + [(well_block, res_block, well_index, well_indexD)]
        return
