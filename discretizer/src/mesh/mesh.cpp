#include <iostream>
#include <fstream>
#include <chrono>
#include <numeric>
#include <unordered_set>

#include "mesh.h"
#include "mesh/mshio/mshio.h"
#include "linalg/matrix.h"
#include "linalg/vector3.h"
#include "utils.h"

using namespace mesh;
using linalg::Vector3;
using std::unordered_map;
using std::unordered_set;
using std::string;
using std::vector;
using std::set;
using std::pair;
using std::distance;
using std::chrono::steady_clock;
using std::chrono::duration_cast;
using std::cout;
using std::endl;

Mesh::Mesh()
{
	elem_type_map[LINE] = vector<index_t>();
	elem_type_map[TRI] = vector<index_t>();
	elem_type_map[QUAD] = vector<index_t>();
	elem_type_map[TETRA] = vector<index_t>();
	elem_type_map[HEX] = vector<index_t>();
	elem_type_map[PRISM] = vector<index_t>();
	elem_type_map[PYRAMID] = vector<index_t>();
}

Mesh::~Mesh()
{
}

void Mesh::gmsh_mesh_processing(string filename, const PhysicalTags& tags)
{
	gmsh_mesh_reading(filename, tags);
	gmsh_mesh_construct_connections(tags);
	generate_adjacency_matrix();
}

// fills:
// nodes, elems, elems_of_node, elem_nodes, elem_nodes_sorted, 
// elem_type_map, region_ranges, region_elems_num, 
// volumes, centroids
void Mesh::gmsh_mesh_reading(string filename, const PhysicalTags& tags)
{
	steady_clock::time_point t1, t2;
	mesh_type = GMSH;
	// mesh loading
	t1 = steady_clock::now();
	// load mesh file, which is a list of points in 3D
	mshio::MshSpec spec = mshio::load_msh(filename);
	t2 = steady_clock::now();
	cout << "Reading of " + filename + ":\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;

	t1 = steady_clock::now();

	// count all the points
	num_of_nodes = 0;
	for (const auto& block : spec.nodes.entity_blocks)
		num_of_nodes += static_cast<index_t>(block.num_nodes_in_block);

	// reserve memory for points
	nodes.reserve(num_of_nodes);

	// insert all the points
	for (const auto& block : spec.nodes.entity_blocks)
	{
		for (index_t i = 0; i < block.num_nodes_in_block; i++)
			nodes.emplace_back(block.data[ND * i], block.data[ND * i + 1], block.data[ND * i + 2]);
	}

	// count all the elements
	num_of_elements = 0;
	for (const auto& block : spec.elements.entity_blocks)
	{
		num_of_elements += static_cast<index_t>(block.num_elements_in_block);
	}

	// reserve memory for arrays
	elems.reserve(num_of_elements);
	volumes.resize(num_of_elements);
	centroids.resize(num_of_elements); 
	element_tags.resize(num_of_elements);
	elem_nodes.reserve(MAX_PTS_PER_3D_ELEM * num_of_elements);
	elem_nodes_sorted.reserve(MAX_PTS_PER_3D_ELEM * num_of_elements);
	std::vector<index_t> num_elems_of_node(num_of_nodes, 0);

	// create all the elements
	uint8_t stride;
	index_t counter = 0, offset = 0;
	for (const auto& region : elem_order)
	{
		region_ranges[region].first = counter;
		for (const auto& block : spec.elements.entity_blocks)
		{
			const auto& loc_tags = tags.at(region);
			if (loc_tags.find(block.entity_tag) == loc_tags.end()) continue;

			stride = Etype_PTS.at(static_cast<ElemType>(block.element_type)) + 1;

			for (index_t i = 0; i < block.num_elements_in_block; i++)
			{
				// element
				Elem el;
				el.loc = region;
				el.type = static_cast<ElemType>(block.element_type);
				el.elem_id = counter++;
				el.n_pts = Etype_PTS.at(el.type);
				element_tags[el.elem_id] = block.entity_tag;
				el.pts_offset = offset;

				offset += el.n_pts;

				// TODO: evaluate centroid and volume (surface for 2D, length for 1D) of the element
				// nodes, call Elem methods for el
				for (index_t j = stride * i + 1; j < stride * (i + 1); j++)
				{
					elem_nodes.push_back(static_cast<index_t>(block.data[j] - 1));
					elem_nodes_sorted.push_back(static_cast<index_t>(block.data[j] - 1));
					//if (j > stride * i + 1)
					//	assert(elem_nodes.back() > elem_nodes[elem_nodes.size() - 2]);
				}

				std::sort(elem_nodes_sorted.end() - (stride - 1), elem_nodes_sorted.end());

				// count for every node number of elements it belongs to
				for (index_t j = el.pts_offset; j < el.pts_offset + el.n_pts; j++)
					num_elems_of_node[elem_nodes[j]]++;
				// calculate the volume and centroid for the element

				el.calculate_volume_and_centroid(nodes, elem_nodes, volumes[el.elem_id], centroids[el.elem_id]);

#ifdef DEBUG_TRANS
				cout << "CPP GMSH ID=" << el.elem_id << ", L= " << el.loc << ", T= " << el.type << ", V= " << volumes[el.elem_id] << ", C= " << centroids[el.elem_id] << "\n";
#endif // DEBUG_TRANS
				//std::cout << "volume: " << el.volume << std::endl;
				//std::cout << "centroid: " << el.c << "\n" << std::endl;
			

				elems.push_back(el);
			}
		}
		region_ranges[region].second = counter;
		region_elems_num[region] = region_ranges[region].second - region_ranges[region].first;
	}

	n_cells = region_ranges[FRACTURE].second;
	assert(*std::max_element(elem_nodes.begin(), elem_nodes.end()) < num_of_nodes);

	// reserve elem types map
	for (auto& type : elem_type_map) 
	{
		type.second.reserve(elems.size());
	}

	// creating array of elem ids per node
	elems_of_node_offset.resize(num_of_nodes + 8);
	elems_of_node_offset[0] = 0;
	std::partial_sum(num_elems_of_node.begin(), num_elems_of_node.end(), elems_of_node_offset.begin() + 1);
	std::fill(num_elems_of_node.begin(), num_elems_of_node.end(), 0);
	
	elems_of_node.resize(offset);
	index_t node_id, pos;
	for (index_t i = 0; i < num_of_elements; i++)
	{
		const auto& el = elems[i];
		if (i < n_cells) elem_type_map[el.type].push_back(el.elem_id);
		for (index_t j = el.pts_offset; j < el.pts_offset + el.n_pts; j++)
		{
			node_id = elem_nodes[j];
			pos = elems_of_node_offset[node_id] + num_elems_of_node[node_id];
			elems_of_node[pos] = i;
			num_elems_of_node[node_id]++;
		}
	}

	// erase zeros in elem types map
	for (auto it = std::begin(elem_type_map); it != std::end(elem_type_map);)
	{
		if (it->second.size() == 0) it = elem_type_map.erase(it);
		else ++it;
	}

	t2 = steady_clock::now();
	cout << "Processing " + std::to_string(num_of_nodes) + " nodes, " +
		std::to_string(num_of_elements) + " elements:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

// uses: elems, elems_of_node_offset, elem_nodes, elems_of_node
// fills: conns
void Mesh::gmsh_mesh_construct_connections(const PhysicalTags& tags)
{
	steady_clock::time_point t1, t2;
	t1 = steady_clock::now();

	index_t nebr_id, counter = 0, offset = 0, node_id;
	size_t len;
	
	// set of connections to check if particular one was already created
	unordered_set<std::pair<index_t, index_t>, pair_xor_hash, one_way_connection_comparator> conn_set;
	conn_set.reserve(ND * num_of_elements);
	conns.reserve(ND * num_of_elements);
	// vector to store the result of intersections
	vector<index_t> intersect;
	intersect.reserve(4 * ND * num_of_elements);
	conn_nodes.reserve(4 * MAX_CONNS_PER_ELEM_GMSH * num_of_elements);

	unordered_set<std::pair<index_t,index_t>>::const_iterator it;
	pair<index_t, index_t> ids;
	ElemConnectionTable::const_iterator conn_type_it;

	// fault nodes
	unordered_set<set<index_t>, integer_set_hash> fault_nodes;
	fault_nodes.reserve(region_ranges[FRACTURE].second - region_ranges[FRACTURE].first);

	// loop through all elements
	for (index_t i = 0; i < num_of_elements; i++)
	{
		//cout << "\n==============\n i " << i << "\n";
		const auto& el1 = elems[i];
		// loop through all nodes belonging to particular element
		for (index_t l = el1.pts_offset; l < el1.pts_offset + el1.n_pts; l++)
		{
			node_id = elem_nodes[l];
			// loop through all elements that share this node 
			for (index_t k = elems_of_node_offset[node_id]; k < elems_of_node_offset[node_id + 1]; k++)
			{
				nebr_id = elems_of_node[k];

				// skip the same element
				if (nebr_id == i) continue;
				//cout << "node_id=" << node_id << ", nebr_id=" << nebr_id << "\n";
				const auto& el2 = elems[nebr_id];

				ids.first = el1.elem_id;		
				ids.second = el2.elem_id;
				it = conn_set.find(ids);
				if (it == conn_set.end())
				{
#if 0 // debug
						int *e1 = elem_nodes_sorted.data();
						cout << "E1: " << el1.pts_offset << " , " << int(el1.n_pts) << " : ";
						for (auto i1 = el1.pts_offset; i1 < el1.pts_offset + el1.n_pts; i1++)
						{
							cout << e1[i1] << " ";
						}
						cout << "\n";

						int *e2 = elem_nodes_sorted.data();
						cout << "E2: "<< el2.pts_offset << " , " << int(el2.n_pts) << " : ";
						for (auto i1 = el2.pts_offset; i1 < el2.pts_offset + el2.n_pts; i1++)
						{
							cout << e2[i1] << " ";
						}
						cout << "\n\n";
#endif //debug

					len = intersect.size();
					std::set_intersection(elem_nodes_sorted.data() + el1.pts_offset, elem_nodes_sorted.data() + el1.pts_offset + el1.n_pts,
						elem_nodes_sorted.data() + el2.pts_offset, elem_nodes_sorted.data() + el2.pts_offset + el2.n_pts,
						std::back_inserter(intersect));
					len = intersect.size() - len;
					assert(len > 0); // intersection of nodes sets for two elements which have common node, should be nonzero 
					if (len > 2)
					{
						conn_type_it = CONN_TYPE_TABLE.find({ el1.loc, el2.loc });
						if (conn_type_it == CONN_TYPE_TABLE.end())
							conn_type_it = CONN_TYPE_TABLE.find({ el2.loc, el1.loc });
						if (conn_type_it != CONN_TYPE_TABLE.end())
						{
							Connection conn;
							conn.conn_id = counter;
							conn.pts_offset = offset;
							conn.n_pts = static_cast<uint8_t>(len);
							conn_nodes.insert(conn_nodes.end(), intersect.end() - len, intersect.end());
							conn.elem_id1 = el1.elem_id;	
							conn.elem_id2 = el2.elem_id;
							conn.calculate_centroid(nodes, conn_nodes);
							conn.calculate_area(nodes, conn_nodes);
							if (conn.area == 0.0)
							{
								conn.n.values[0] = conn.n.values[1] = conn.n.values[2] = 0.0;
							}
							else
							{
								conn.calculate_normal(nodes, conn_nodes, elems, elem_nodes);
							}
							if (conn_type_it->second == MAT_FRAC)
							{
								conn.c -= init_apertures[el2.elem_id - region_ranges[FRACTURE].first] * conn.n / 2.0;
								fault_nodes.insert(set<index_t>(intersect.end() - len, intersect.end()));
							}
							else if (conn_type_it->second == FRAC_MAT)
							{
								conn.c += init_apertures[el1.elem_id - region_ranges[FRACTURE].first] * conn.n / 2.0;
								fault_nodes.insert(set<index_t>(intersect.end() - len, intersect.end()));
							}

							conn_set.insert(ids);
							offset += conn.n_pts;

							conn.type = conn_type_it->second;
							conns.push_back(conn);
							counter++;
						}
					}
					else if (len == 2) 
					{
						if (el1.loc == FRACTURE && el2.loc == FRACTURE)
						{
							Connection conn;
							conn.conn_id = counter;
							conn.pts_offset = offset;
							conn.n_pts = static_cast<uint8_t>(len);
							conn_nodes.insert(conn_nodes.end(), intersect.end() - len, intersect.end());
							conn.elem_id1 = el1.elem_id;	
							conn.elem_id2 = el2.elem_id;
							conn.calculate_centroid(nodes, conn_nodes);
							conn.calculate_area(nodes, conn_nodes);
							conn.area *= ( init_apertures[el1.elem_id - region_ranges[FRACTURE].first] + init_apertures[el2.elem_id - region_ranges[FRACTURE].first] ) / 2.0;
							conn.calculate_normal(nodes, conn_nodes, elems, elem_nodes);
							conn_set.insert(ids);
							offset += conn.n_pts;

							conn.type = CONN_TYPE_TABLE.at( { el1.loc, el2.loc } );
							conns.push_back(conn);
							counter++;
						}
						//else if(el2.loc == FRACTURE || el2.loc == FRACTURE_BOUNDARY)
					}
				}
			}
		}
	}

	elem_nodes_sorted.clear();

	//// erase matrix-matrix connections across faults
	// find the indices of remaining connections 
	vector<index_t> conns_to_remain;
	conns_to_remain.reserve(region_ranges[FRACTURE].second - region_ranges[FRACTURE].first);
	unordered_set<set<index_t>>::const_iterator it_faults;
	for (const auto& conn : conns)
	{
		if (conn.type == MAT_MAT)
		{
			set<index_t> cur(conn_nodes.begin() + conn.pts_offset, conn_nodes.begin() + conn.pts_offset + conn.n_pts);
			it_faults = fault_nodes.find(cur);
			if (it_faults == fault_nodes.end())
				conns_to_remain.push_back(conn.conn_id);
		}
		else 
			conns_to_remain.push_back(conn.conn_id);
	}
	// copy what remains
	std::vector<Connection> new_conns;
	new_conns.reserve(conns_to_remain.size());
	std::vector<index_t> new_conn_nodes;
	new_conn_nodes.reserve(4 * MAX_CONNS_PER_ELEM_GMSH * num_of_elements);
	counter = offset = 0;
	for (const auto& conn_id: conns_to_remain)
	{
		auto& old_conn = conns[conn_id];
		// nodes
		new_conn_nodes.insert(new_conn_nodes.end(), conn_nodes.begin() + old_conn.pts_offset, conn_nodes.begin() + old_conn.pts_offset + old_conn.n_pts);
		// connection
		old_conn.conn_id = counter++;
		old_conn.pts_offset = offset;
		new_conns.push_back(old_conn);

		offset += old_conn.n_pts;
	}

	conns = new_conns;
	conn_nodes = new_conn_nodes;

	t2 = steady_clock::now();
	cout << conns.size() << " connections:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl;
}

// uses: conns, num_of_elements
// fills: adj_matrix, adj_matrix_cols, adj_matrix_offset
void Mesh::generate_adjacency_matrix() 
{
	steady_clock::time_point t1, t2;

	t1 = steady_clock::now();
	
	// Temporary arrays to identify all connections per element
	std::vector<std::vector<index_t>> adj_2d (num_of_elements, std::vector<index_t> (MAX_CONNS_PER_ELEM));
	std::vector<size_t> conn_per_element(num_of_elements, 0);
	std::vector<std::vector<bool>> conn_signs(num_of_elements, std::vector<bool>(MAX_CONNS_PER_ELEM, true));

	// Append connections per element to 2D array
	for (auto& conn : conns) 
	{
		auto& conn_size1 = conn_per_element[conn.elem_id1];
		adj_2d[conn.elem_id1][conn_size1++] = conn.conn_id;

		if (conn.type != MAT_BOUND && conn.type != FRAC_BOUND)
		{
			auto& conn_size2 = conn_per_element[conn.elem_id2];
			adj_2d[conn.elem_id2][conn_size2] = conn.conn_id;
			conn_signs[conn.elem_id2][conn_size2++] = false;
		}
	}

	// Reserve space 
	adj_matrix.reserve(2 * conns.size());
	adj_matrix_cols.reserve(2 * conns.size());
	adj_matrix_offset.reserve(num_of_elements + 1);

	// First starts from zero
	index_t offset = 0;
	adj_matrix_offset.push_back(offset);

	// Flatten to 1D array and calculate offsets
	for (index_t i = 0; i < num_of_elements; i++)
	{
		offset += static_cast<index_t>(conn_per_element[i]);
		adj_matrix_offset.push_back(offset);
		const auto& elem_conns = adj_2d[i];
		const auto& elem_conn_signs = conn_signs[i];
		for (uint8_t j = 0; j < conn_per_element[i]; j++)
		{
			adj_matrix.push_back(elem_conns[j]);
			if (elem_conn_signs[j])
				adj_matrix_cols.push_back(conns[elem_conns[j]].elem_id2);
			else
				adj_matrix_cols.push_back(conns[elem_conns[j]].elem_id1);
		}
	}

	t2 = steady_clock::now();
	cout << "Adjacency matrix:\t" << duration_cast<std::chrono::milliseconds>(t2 - t1).count() << "\t[ms]" << endl; 
}

// fills:
// nodes, elems, elems_of_node, elem_nodes, elem_nodes_sorted, 
// elem_type_map, region_ranges, region_elems_num, //TODO
// volumes, centroids
// nx, ny, nz, bnd_faces_num2

// print internal arrays to screen (for debugging) 
void Mesh::print_elems_nodes()
{
	cout << "Elements:\n";

	// loop through all elements
	for (index_t i = 0; i < num_of_elements; i++)
	{
		const auto& el1 = elems[i];

		cout << "\t id=" << el1.elem_id << " " << get_ijk_as_str(el1.elem_id, false) << " n_pts=" << int(el1.n_pts) << " pts_offset=" << el1.pts_offset << "\n";
		// loop through all nodes belonging to particular element
		for (index_t l = el1.pts_offset; l < el1.pts_offset + el1.n_pts; l++)
		{
			index_t node_id = elem_nodes[l];
			cout << "\t Node=" << node_id << " Elems:\t";
			// loop through all elements that share this node 
			for (index_t k = elems_of_node_offset[node_id]; k < elems_of_node_offset[node_id + 1]; k++)
			{
				index_t nebr_id = elems_of_node[k];
				cout << nebr_id << " ";
			}
			cout << "\n";
		}
	}
}

//i,j,k are 0-based
void Mesh::calc_cell_nodes(const int i, const int j, const int k,
	std::array<double, 8>& X,
	std::array<double, 8>& Y,
	std::array<double, 8>& Z) const
{

	std::array<int, 8> zind;
	std::array<int, 4> pind;

	// calculate indices for grid pillars in COORD arrray
	const size_t p_offset = j * (nx + 1) * 6 + i * 6;

	pind[0] = p_offset;
	pind[1] = p_offset + 6;
	pind[2] = p_offset + (nx + 1) * 6;
	pind[3] = pind[2] + 6;

	// pind[0]-----pind[1]---> I-axis
	//   |           |
	//   |           |
	// pind[2]-----pind[3]
	//   |
	//   V  
	// J-axis

	// get depths from zcorn array in ZCORN array
	const size_t z_offset = k * nx * ny * 8 + j * nx * 4 + i * 2;
	// top
	zind[0] = z_offset;
	zind[1] = z_offset + 1;
	zind[2] = z_offset + nx * 2;
	zind[3] = zind[2] + 1;
	// bottom
	for (int n = 0; n < 4; n++)
		zind[n + 4] = zind[n] + nx * ny * 4;

	for (int n = 0; n < 8; n++)
		Z[n] = zcorn[zind[n]];

	for (int n = 0; n < 4; n++) {
		// top
		double xt = coord[pind[n]];
		double yt = coord[pind[n] + 1];
		double zt = coord[pind[n] + 2];
		// bottom
		double xb = coord[pind[n] + 3];
		double yb = coord[pind[n] + 4];
		double zb = coord[pind[n] + 5];

		if (zt == zb) {
			X[n] = xt;
			X[n + 4] = xt;

			Y[n] = yt;
			Y[n + 4] = yt;
		}
		else {
			double t = (xb - xt) / (zt - zb);
			X[n]     = xt + t * (zt - Z[n]);
			X[n + 4] = xt + t * (zt - Z[n + 4]);

			t =  (yb - yt) / (zt - zb);
			Y[n]		 = yt + t * (zt - Z[n]);
			Y[n + 4] = yt + t * (zt - Z[n + 4]);
		}
	}
}

//i,j,k are 0-based
std::array<value_t, 3> 
Mesh::calc_cell_sizes(const int i, const int j, const int k) const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	calc_cell_nodes(i, j, k, X, Y, Z);

	//DEBUG
	//for (int ii = 0; ii < 8; ii++){
	//  cout << "(" << X[ii] << ", " << Y[ii] <<  ", " << Z[ii] << ")\n";
	//}
	double x1 = X[0] + X[2] + X[6] + X[4];
	double x2 = X[1] + X[3] + X[7] + X[5];
	double dx = fabs(x2 - x1) / 4.;

	double y1 = Y[2] + Y[3] + Y[7] + Y[6];
	double y2 = Y[0] + Y[1] + Y[5] + Y[4];
	double dy = fabs(y2 - y1) / 4.;

	double z1 = Z[0] + Z[1] + Z[2] + Z[3];
	double z2 = Z[4] + Z[5] + Z[6] + Z[7];
	double dz = fabs(z2 - z1) / 4.;

	std::array<value_t, 3> d{ dx, dy, dz };

	//DEBUG
	//cout << "cell (" << i << ", " << j << ", " << k << ") dx=" << d[0] << " dy=" << d[1] <<  " dz=" << d[2] << "\n";

	return d;
}

//
std::array<value_t, 3>
Mesh::calc_cell_center(const int i, const int j, const int k) const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	calc_cell_nodes(i, j, k, X, Y, Z);

	double x1 = X[0] + X[2] + X[6] + X[4];
	double x2 = X[1] + X[3] + X[7] + X[5];
	double x = (x2 + x1) / 8.;

	double y1 = Y[2] + Y[3] + Y[7] + Y[6];
	double y2 = Y[0] + Y[1] + Y[5] + Y[4];
	double y = (y2 + y1) / 8.;

	double z1 = Z[0] + Z[1] + Z[2] + Z[3];
	double z2 = Z[4] + Z[5] + Z[6] + Z[7];
	double z = (z2 + z1) / 8.;

	std::array<value_t, 3> d{ x, y, z };

	//DEBUG
	//cout << "cell (" << i << ", " << j << ", " << k << ") Cx=" << d[0] << " Cy=" << d[1] <<  " Cz=" << d[2] << "\n";

	return d;
}

//
void Mesh::construct_local_global()
{
  index_t n_all = get_n_cells_total();

	local_to_global.resize(n_cells);
	global_to_local.resize(n_all);
#if 0
	index_t j = 0;
	for (index_t k1 = 0; k1 < nz; k1++)
		for (index_t j1 = 0; j1 < ny; j1++)
			for (index_t i1 = 0; i1 < nx; i1++)
			{
				index_t i = get_global_index(i1, j1, k1);
#else
	for (index_t i = 0, j = 0; i < n_all; i++) {
#endif
		if (actnum[i] > 0) {
			local_to_global[j] = i;
			global_to_local[i] = j;
			j++;
		}
		else {
			global_to_local[i] = -1;
		}
	}
}

// print centroids, volumes and cell sizes
void Mesh::print_arrays() const
{
	for (index_t idx = 0, counter = 0; idx < get_n_cells_total(); idx++) {
		if (!actnum[idx])
			continue;

		int i1, j1, k1;
		std::string ijk_str = get_ijk_as_str(idx, true);

		get_ijk(idx, i1, j1, k1, true);
		std::array<value_t, 3> d = calc_cell_sizes(i1, j1, k1);

		//if (i < 500 )
		cout<<idx<<" "<<ijk_str<<" C="<<centroids[counter]<<"\tV="<<volumes[counter]<<"\tDX="<<d[0]<<"\tDY="<<d[1]<<"\tDZ="<<d[2]<<"\n";
		
		counter++;
	}//loop by cells
}

// 
void Mesh::write_cell_sizes(std::string fname) const
{
	std::vector<value_t> dx, dy, dz;
	for (size_t idx = 0; idx < get_n_cells_total(); ++idx) {
		int i1, j1, k1;
		get_ijk(idx, i1, j1, k1, true);
		std::array<value_t, 3> d = calc_cell_sizes(i1, j1, k1);
		dx[idx] = d[0];
		dy[idx] = d[1];
		dz[idx] = d[2];
	}
	std::vector<int> actnum;
	write_array_to_file(fname, "DX", dx, actnum, get_n_cells_total(), 1.0, true);
	write_array_to_file(fname, "DY", dy, actnum, get_n_cells_total(), 1.0, true);
	write_array_to_file(fname, "DZ", dz, actnum, get_n_cells_total(), 1.0, true);
}


// order: Y1, Y2, X1, X2, Z2, Z1
std::vector<value_t>
Mesh::get_prisms() const
{
	std::array<double, 8> X;
	std::array<double, 8> Y;
	std::array<double, 8> Z;

	std::vector<value_t> prisms;
	prisms.resize(6 * n_cells);
	int idx_cell = 0, idx = 0;

  for (int k = 0; k < nz; k++)
		for (int j = 0; j < ny; j++)
			for (int i = 0; i < nx; i++) {
				if (global_to_local[idx_cell++] < 0) // cell is inactive
					continue;
				calc_cell_nodes(i, j, k, X, Y, Z);

				//DEBUG
				//for (int ii = 0; ii < 8; ii++){
				//  cout << "(" << X[ii] << ", " << Y[ii] <<  ", " << Z[ii] << ")\n";
				//}
				double x1 = X[0] + X[2] + X[6] + X[4];
				double x2 = X[1] + X[3] + X[7] + X[5];
				double dx = fabs(x2 - x1) / 4.;

				double y1 = Y[2] + Y[3] + Y[7] + Y[6];
				double y2 = Y[0] + Y[1] + Y[5] + Y[4];
				double dy = fabs(y2 - y1) / 4.;

				double z1 = Z[0] + Z[1] + Z[2] + Z[3];
				double z2 = Z[4] + Z[5] + Z[6] + Z[7];
				double dz = fabs(z2 - z1) / 4.;

				prisms[idx++] = Y[0];
				prisms[idx++] = Y[0] + dy;
				prisms[idx++] = X[0];
				prisms[idx++] = X[0] + dx;
				prisms[idx++] = Z[0] + dz;
				prisms[idx++] = Z[0];
			}

	//cout <<"centers: " << idx_cell << " " << idx / 6 << '\n';
	return prisms;
}

// prder: YXZ
std::vector<value_t>
Mesh::get_centers() const
{
	std::vector<value_t> centers_vec;
	centers_vec.resize(3 * n_cells);

	int idx = 0, idx_cell = 0;
	for (auto c : centroids) {
		if (idx_cell >= n_cells) //skip bnd faces in the end of centroids
			break;
		centers_vec[idx++] = c.y;
		centers_vec[idx++] = c.x;
		centers_vec[idx++] = c.z;
		idx_cell++;
	}

	return centers_vec;
}
