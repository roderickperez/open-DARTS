import numpy as np
from math import inf, pi, asin

from darts.engines import conn_mesh, ms_well, ms_well_vector, index_vector, value_vector

from darts.reservoirs.unstruct_reservoir import UnstructReservoir
from darts.reservoirs.mesh.unstruct_discretizer import UnstructDiscretizer

from darts.reservoirs.mesh.geometry.unstructured import Unstructured
from darts.reservoirs.mesh.geometry.fluidflower import FluidFlower
from darts.reservoirs.mesh.geometry.wells import CircularWell, WellCell


class FluidFlowerUnstruct(UnstructReservoir):
    def __init__(self, timer, layer_properties, layers_to_regions, model_specs, well_centers):
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
        filename = 'fluidflower'
        m = Unstructured(dim=2, axs=[0, 2])
        m.add_shape(FluidFlower(lc=[model_specs['lc'][0]]))

        self.well_centers = well_centers
        for i, (name, center) in enumerate(well_centers.items()):
            in_surfaces = m.find_surface(center)
            m.refine_around_point(center, radius=0.015, lc=len(m.lc))

            w = WellCell(center, lc=model_specs['lc'][1], orientation='xz', in_surfaces=in_surfaces)
            # w = CircularWell(center, re=0.01, lc_well=1, axs=[0, 2], in_surfaces=in_surfaces)
            m.add_shape(w)

        m.extrude_mesh(length=0.025, layers=1, axis=1, recombine=True)

        m.write_geo(filename)
        m.generate_msh(filename)

        self.layer_properties = layer_properties
        self.layers_to_regions = layers_to_regions
        self.physical_tags['matrix'] = [900001, 900002, 900003, 900004, 900005, 900006, 900007, 900008,
                                        900009, 900010, 900011, 900012, 900013, 900014, 900015, 900016,
                                        900017, 900018, 900019, 900020, 900021, 900022, 900023, 900024,
                                        900025, 900026, 900027, 900028, 900029, 900030, 900031,
                                        ]  # C, D, E, ESF, F, Fault-1, Fault-2, G, W(ater)

        self.thickness = model_specs['thickness']
        self.curvature = model_specs['curvature']

        super().__init__(timer=timer, mesh_file=filename + '.msh',
                         permx=1, permy=1, permz=1, poro=0.2, hcap=2200, rcond=181.44)

    def discretize(self, cache: bool = False, verbose: bool = False):
        self.discretizer = UnstructDiscretizer(mesh_file=self.mesh_file, physical_tags=self.physical_tags, verbose=verbose)

        self.discretizer.n_dim = 3

        # Use class method load_mesh to load the GMSH file specified above:
        self.discretizer.load_mesh(permx=self.permx, permy=self.permy, permz=self.permz, frac_aper=0, cache=cache)

        # Make curvature and thickness correction to all cells using spline interpolation of thickness map
        self.curvature_thickness_correction()

        # # Calculate cell information of each geometric element in the .msh file:
        # self.discretizer.calc_cell_information(cache=cache)

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
        self.water_column_height = 1.5
        self.min_depth = self.water_column_height - max(self.discretizer.depth_all_cells)
        np.array(self.mesh.depth, copy=False)[:] = self.water_column_height - self.discretizer.depth_all_cells  # inverse height for depth
        np.array(self.mesh.volume, copy=False)[:] = self.discretizer.volume_all_cells

        np.array(self.mesh.op_num, copy=False)[:] = self.op_num
        np.array(self.mesh.poro, copy=False)[:] = self.poro
        np.array(self.mesh.heat_capacity, copy=False)[:] = self.hcap
        np.array(self.mesh.rock_cond, copy=False)[:] = self.rcond

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
                if self.layer_properties[tag] == "ESF":
                    self.seal.append(ith_cell)

        # Assign properties to layers
        n_layers = len(self.layer_properties)
        layer_op_num = np.zeros(n_layers)
        layer_poro = np.zeros(n_layers)
        layer_hcap = np.zeros(n_layers)
        layer_rcond = np.zeros(n_layers)
        layer_perm = np.zeros((n_layers, 3))
        for i, (layer, porperm) in enumerate(self.layer_properties.items()):
            layer_op_num[i] = self.layers_to_regions[self.layer_properties[layer].type]
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

    def set_wells(self, verbose: bool = False):
        water_column_depth = 1.5
        z_max = max(self.discretizer.mesh_data.points[:, 2])  # for finding top cells
        self.min_depth = water_column_depth - z_max

        for name, center in self.well_centers.items():
            well_cell = self.find_cell_index(center)
            # well_depth = self.water_column_height - well_center[0][2]
            self.add_well(name)

            # self.reservoir.depth[ith_cell] = well_depth
            # well_index = self.reservoir.calculate_well_index(r_w=0.001, res_block=ith_cell)
            well_index = 2e0
            well_indexD = 0
            self.add_perforation(name, cell_index=well_cell, well_index=well_index, well_indexD=well_indexD, verbose=verbose)

        # Find top cells and Add top layer production well for boundary condition
        top_cells = []
        for ith_cell, nodes_to_cell in enumerate(self.discretizer.mesh_data.cells_dict['wedge']):
            if self.discretizer.centroid_all_cells[ith_cell][2] > 1.25:
                coord_nodes_in_cell = self.discretizer.mat_cell_info_dict[ith_cell].coord_nodes_to_cell
                if sum(coord_nodes_in_cell[:, 2] == z_max) >= 4:
                    top_cells.append(ith_cell)

        self.add_well("P1")
        # well_depth = self.min_depth
        for nth_cell in top_cells:
            well_index = 1E2
            well_indexD = 0
            self.add_perforation("P1", cell_index=nth_cell, well_index=well_index, well_indexD=well_indexD, verbose=verbose)
        return

    def find_cell_index(self, well_center) -> int:
        dist0, idx = None, None

        for l, centroid in enumerate(self.discretizer.centroid_all_cells):
            if self.curvature:
                r = np.sqrt(centroid[0] ** 2 + centroid[1] ** 2)  # radius of circle at centroid
                theta = pi / 8 + asin(centroid[0] / r)  # angle from left end of domain

                X = r * theta  # arc length L == x
            else:
                X = centroid[0]
            Z = centroid[2]
            dist1 = np.sqrt((X - well_center[0]) ** 2 + (Z - well_center[2]) ** 2)
            if dist0 is None or dist1 < dist0:
                idx = l
                dist0 = dist1

        return idx

    def curvature_thickness_correction(self):
        """
        Correct for thickness variation throughout domain using spline interpolation
        """
        from scipy.interpolate import RectBivariateSpline
        from math import sin, cos

        Lx = 2.83
        Lz = 1.23
        x = np.array([25, 589, 1153, 1718, 2282, 2847, 3411, 3976, 4540, 5105, 5669, 6234, 6798, 7363, 7927, ])/(7927+25) * Lx
        z = np.array([25, 589, 1153, 1718, 2282, 2847, 3411, ])/(3411+25) * Lz
        data = np.array([[19.22, 18.93, 19.4, 19.2, 19.2, 19.2, 19.2],
                         [19.2, 21.6, 22.3, 23.0, 22.1, 20.9, 20.0],
                         [19.4, 22.9, 25.6, 26.5, 25.5, 23.8, 22.7],
                         [19.7, 24.3, 27.4, 27.8, 27.3, 26.3, 24.8],
                         [19.9, 24.2, 26.4, 27.5, 28.1, 27.8, 25.7],
                         [19.8, 24.3, 26.9, 28.1, 28.3, 27.7, 25.9],
                         [19.8, 24.1, 26.3, 27.6, 27.0, 26.9, 26.0],
                         [19.0, 23.6, 25.6, 26.4, 26.2, 25.5, 24.5],
                         [18.6, 23.5, 25.8, 27.1, 27.2, 26.5, 25.3],
                         [19.6, 24.1, 25.9, 26.9, 27.4, 26.9, 26.2],
                         [19.3, 23.8, 26.1, 26.6, 26.0, 25.3, 25.3],
                         [18.5, 22.3, 24.1, 25.4, 26.0, 24.5, 23.7],
                         [17.9, 20.2, 21.2, 21.7, 21.2, 20.9, 20.9],
                         [17.5, 18.2, 19.0, 19.4, 19.2, 18.4, 18.4],
                         [18.5, 18.7, 18.5, 18.2, 18.5, 18.5, 18.5],])/1000
        if not self.thickness:
            data = np.ones((len(x), len(z))) * 0.025

        spline = RectBivariateSpline(x, z, data)

        if self.curvature:
            r = 3.6  # domain is curved; 1/8 circle with radius 3.6
            theta_mid = pi/8  # pi/8 is half way the curve
            for node in self.discretizer.mesh_data.points:
                theta = node[0]/r  # length of circle arc = r*theta
                theta -= theta_mid
                if node[1] == 0.:  # front
                    r1 = r - spline.ev(node[0], node[2])  # radius - thickness
                    node[0] = r1 * sin(theta)
                    node[1] = r1 * cos(theta)
                else:  # back panel
                    node[0] = r * sin(theta)
                    node[1] = r * cos(theta)
        else:
            for node in self.discretizer.mesh_data.points:
                if node[1] != 0.:
                    node[1] = spline.ev(node[0], node[2])

        return
