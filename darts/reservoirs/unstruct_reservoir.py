from darts.reservoirs.reservoir_base import ReservoirBase
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer
from darts.engines import timer_node, ms_well, conn_mesh, value_vector, index_vector
import numpy as np
import os
import meshio

from dataclasses import dataclass


class UnstructReservoir(ReservoirBase):
    def __init__(self, timer: timer_node, mesh_file: str,
                 permx, permy, permz, poro, frac_aper=0, rcond=0, hcap=0, op_num=0, cache: bool = False):
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
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file,
                                               permx=self.permx, permy=self.permy, permz=self.permz)

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

        # Initialize mesh using built connection list
        mesh = conn_mesh()
        mesh.init(index_vector(cell_m), index_vector(cell_p), value_vector(tran), value_vector(tran_thermal))

        # Store number of control volumes (NOTE: in case of fractures, this includes both matrix and fractures):
        self.nb = self.discretizer.volume_all_cells.size
        self.num_frac = self.discretizer.fracture_cell_count
        self.num_mat = self.discretizer.matrix_cell_count

        # Create numpy arrays wrapped around mesh data (no copying, this will severely slow down the process!)
        np.array(mesh.poro, copy=False)[:] = self.poro
        np.array(mesh.rock_cond, copy=False)[:] = self.rcond
        np.array(mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(mesh.op_num, copy=False)[:] = self.op_num

        # Since we use copy==False above, we have to store the values by using the Python slicing option, if we don't
        # do this we will overwrite the variable, e.g. self.poro = poro --> overwrite self.poro with the variable poro
        # instead of storing the variable poro in self.mesh.poro (therefore "numpy array wrapped around mesh data!!!):
        np.array(mesh.depth, copy=False)[:] = self.discretizer.depth_all_cells
        np.array(mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        # Calculate well_index (very primitive way....):
        self.well_index = np.mean(tran) * 1
        # self.well_index = 10

        return mesh

    def set_boundary_volume(self, mesh: conn_mesh):
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

        self.volume[self.right_boundary_cells] = self.volume[self.right_boundary_cells] * 1e8
        return

    def calc_boundary_cells(self, boundary_data):
        """
        Class method which calculates constant boundary values at a specif constant x,y,z-coordinate

        :param boundary_data: dictionary with the boundary location (X,Y,Z, and location)
        :return:
        """
        # Specify boundary cells, simply set specify the single coordinate which is not-changing and its value:
        # First boundary:
        index = []  # Dynamic list containing indices of the nodes (points) which lay on the boundary:
        if boundary_data['first_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 0] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 1] == boundary_data['first_boundary_val']
        elif boundary_data['first_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 2] == boundary_data['first_boundary_val']

        # Convert dynamic list to numpy array:
        left_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Second boundary (same as above):
        index = []
        if boundary_data['second_boundary_dir'] == 'X':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 0] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Y':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 1] == boundary_data['second_boundary_val']
        elif boundary_data['second_boundary_dir'] == 'Z':
            # Check if first coordinate of points is on the boundary:
            index = self.discretizer.mesh_data.points[:, 2] == boundary_data['second_boundary_val']

        right_boundary_points = np.array(list(compress(range(len(index)), index)))

        # Find cells containing boundary cells, for wedges or hexahedrons, the boundary cells must contain,
        # on the X or Y boundary four nodes exactly!
        #     0------0          0
        #    /     / |         /  \
        #  0------0  0        0----0
        #  |      | /         |    |
        #  0------0           0----0
        # Hexahedron       Wedge (prism)
        # Create loop over all matrix cells which are of the geometry 'matrix_cell_type'
        left_count = 0  # Counter for number of left matrix cells on the boundary
        left_boundary_cells = {}  # Dictionary with matrix cells on the left boundary
        for geometry in self.discretizer.geometries_in_mesh_file:
            if geometry in self.discretizer.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the left_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.discretizer.mesh_data.cells_dict[geometry]):

                    if len(set.intersection(set(ith_row), set(left_boundary_points))) == 4:
                        # Store cell since it is on the left boundary:
                        left_boundary_cells[left_count] = ith_cell
                        left_count += 1

        right_count = 0
        right_boundary_cells = {}
        for geometry in self.discretizer.geometries_in_mesh_file:
            if geometry in self.discretizer.available_matrix_geometries:
                # Matrix geometry found, check if any matrix control volume has exactly 4 nodes which intersect with
                # the right_boundary_points list:
                for ith_cell, ith_row in enumerate(
                        self.discretizer.mesh_data.cells_dict[geometry]):
                    if len(set.intersection(set(ith_row), set(right_boundary_points))) == 4:
                        # Store cell since it is on the left boundary:
                        right_boundary_cells[right_count] = ith_cell
                        right_count += 1

        self.left_boundary_cells = np.array(list(left_boundary_cells.values()), dtype=int) + \
                                   self.discretizer.fracture_cell_count
        self.right_boundary_cells = np.array(list(right_boundary_cells.values()), dtype=int) + \
                                    self.discretizer.fracture_cell_count
        return 0

    def add_perforations(self, mesh, verbose: bool = False):
        """
        Function to add perforations to wells.

        :param mesh: :class:`Mesh` object
        :param verbose: Switch for verbose level
        """
        for perf in self.perforations:
            well = self.get_well(perf.well_name)

            # calculate well index and get local index of reservoir block
            i, j, k = perf.cell_index
            res_block_local, wi, wid = self.discretizer.calc_well_index(i, j, k, well_radius=perf.well_radius,
                                                                        segment_direction=perf.segment_direction,
                                                                        skin=perf.skin)

            if perf.well_index is None:
                perf.well_index = wi

            if perf.well_indexD is None:
                perf.well_indexD = wid

            # set well segment index (well block) equal to index of perforation layer
            if perf.multi_segment:
                well_block = len(well.perforations)
            else:
                well_block = 0

            # add completion only if target block is active
            if res_block_local > -1:
                if len(well.perforations) == 0:
                    well.well_head_depth = np.array(mesh.depth, copy=False)[res_block_local]
                    well.well_body_depth = well.well_head_depth
                    perf.well_indexD *= np.array(mesh.rock_cond, copy=False)[res_block_local]  # assume perforation condution = rock conduction
                    if self.is_cpg:
                        dx, dy, dz = self.discretizer.calc_cell_dimensions(i-1, j-1, k-1)
                        # TODO: need segment_depth_increment and segment_length logic
                        if perf.segment_direction == 'z_axis':
                            well.segment_depth_increment = dz
                        elif perf.segment_direction == 'x_axis':
                            well.segment_depth_increment = dx
                        else:
                            well.segment_depth_increment = dy
                    else:
                        well.segment_depth_increment = self.discretizer.len_cell_zdir[i-1, j-1, k-1]

                    well.segment_volume *= well.segment_depth_increment
                for p in well.perforations:
                    if p[0] == well_block and p[1] == res_block_local:
                        print('Neglected duplicate perforation for well %s to block [%d, %d, %d]' %
                              (well.name, i, j, k))
                        return
                well.perforations = well.perforations + [(well_block, res_block_local, perf.well_index, perf.well_indexD)]
                if verbose:
                    print('Added perforation for well %s to block %d [%d, %d, %d] with WI=%f and WID=%f' %
                          (well.name, res_block_local, i, j, k, perf.well_index, perf.well_indexD))
            else:
                if verbose:
                    print('Neglected perforation for well %s to block [%d, %d, %d] (inactive block)' %
                          (well.name, i, j, k))
        return

    def output_to_vtk(self, output_directory: str, output_filename: str, property_data: dict, ith_step: int):
        """
        Class method which writes output of unstructured grid to VTK format

        :param output_directory: directory of output files
        :param property_array: np.array containing all cell properties (N_cells x N_prop)
        :param cell_property: list with property names (visible in ParaView (format strings)
        :param ith_step: integer containing the output step
        :return:
        """
        # First check if output directory already exists:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        # Allocate empty new cell_data_dict dictionary:
        cell_data_dict = dict()

        for ith_prop in range(len(cell_property)):
            cell_data_dict[cell_property[ith_prop]] = []
            left_bound = 0
            right_bound = 0
            for ith_geometry in self.mesh_data.cells_dict:
                left_bound = right_bound
                right_bound = right_bound + self.mesh_data.cells_dict[ith_geometry].shape[0]
                cell_data_dict[cell_property[ith_prop]].append(list(property_array[left_bound:right_bound, ith_prop]))

        cell_data_dict['matrix_cell_bool'] = []
        left_bound = 0
        right_bound = 0
        for ith_geometry in self.mesh_data.cells_dict:
            left_bound = right_bound
            right_bound = right_bound + self.mesh_data.cells_dict[ith_geometry].shape[0]

            if (ith_geometry in self.available_fracture_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.zeros(((right_bound - left_bound),))))

            elif (ith_geometry in self.available_matrix_geometries) and (right_bound - left_bound) > 0:
                cell_data_dict['matrix_cell_bool'].append(list(np.ones(((right_bound - left_bound),))))

        # Temporarily store mesh_data in copy:
        # Mesh = meshio.read(self.mesh_file)

        mesh = meshio.Mesh(
            # Mesh.points,
            # Mesh.cells_dict.items(),
            self.mesh_data.points,  # list of point coordinates
            self.mesh_data.cells_dict.items(),  # list of
            # Each item in cell data must match the cells array
            cell_data=cell_data_dict)

        print('Writing data to VTK file for {:d}-th reporting step'.format(ith_step))
        meshio.write("{:s}/solution{:d}.vtk".format(output_directory, ith_step), mesh)
        return 0
