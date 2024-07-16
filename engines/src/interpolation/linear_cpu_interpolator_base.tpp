#include <numeric>
#include <algorithm>
#include <fstream>
#include "linear_cpu_interpolator_base.hpp"
#include "mech/matrix.h"

template <typename index_t, int N_DIMS, int N_OPS>
linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::linear_cpu_interpolator_base(operator_set_evaluator_iface *supporting_point_evaluator,
                                                                                   const std::vector<int> &axes_points,
                                                                                   const std::vector<double> &axes_min, const std::vector<double> &axes_max)
    : interpolator_base(supporting_point_evaluator, axes_points, axes_min, axes_max)
{

    axes_mult[N_DIMS - 1] = 1;
    for (int dim{N_DIMS - 2}; dim >= 0; dim--)
        axes_mult[dim] = axes_mult[dim + 1] * axes_points[dim + 1];

    // initialize the values with 0
    standard_simplex = {};
    // and then set some to 1
    for (int vertex_i = 0; vertex_i < N_DIMS; vertex_i++)
        for (int dim_i = vertex_i; dim_i < N_DIMS; dim_i++)
            standard_simplex[vertex_i][dim_i] = 1;

    double int64_max;
    if constexpr (std::is_same_v<index_t, __uint128_t>)
    {
#ifdef _MSC_VER
      const auto int128_max = std::numeric_limits<__uint128_t>::max();
      int64_max = std::ldexp(static_cast<double>(int128_max._Word[1]), 64) + static_cast<double>(int128_max._Word[0]);
#elif defined(__GNUC__)
      int64_max = std::numeric_limits<__uint128_t>::max();
#endif
    }
    else 
      int64_max = static_cast<double>(std::numeric_limits<index_t>::max());

    if (n_points_total_fp > int64_max)
    {
        std::string error = "Error: The total requested amount of points (" + std::to_string(n_points_total_fp) +
                            ") exceeds the limit in index type (" + std::to_string(std::numeric_limits<index_t>::max()) + ")\n";
        throw std::range_error(error);
    }

    transform_last_axis = 1;

    this->use_barycentric_interpolation = true;
    if (this->use_barycentric_interpolation)
    {
      load_delaunay_triangulation("c:\\work\\packages\\open-darts\\engines\\src\\interpolation\\hypercube_delaunay_7_new.bin");

      delaunay_map_axes_mult[N_DIMS - 1] = 1;
      for (int dim = N_DIMS - 2 ; dim >= 0; dim--)
        delaunay_map_axes_mult[dim] = delaunay_map_axes_mult[dim + 1] * delaunay_spatial_map_n_points;
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::load_delaunay_triangulation(const std::string filename)
{
  std::ifstream file(filename, std::ios::binary);

  if (!file)
  {
    printf("Unable to open file\n");
    exit(-1);
  }

  int n_simplices, n_simplex_size, n_dim, n_mappings, n_mapping_size;
  //int n_located_simplices;
  while (file.peek() != EOF)
  {
    file.read(reinterpret_cast<char*>(&n_simplices), sizeof(n_simplices));
    file.read(reinterpret_cast<char*>(&n_simplex_size), sizeof(n_simplex_size));
    file.read(reinterpret_cast<char*>(&n_mappings), sizeof(n_mappings));
    file.read(reinterpret_cast<char*>(&n_mapping_size), sizeof(n_mapping_size));
    n_dim = n_simplex_size - 1;

    Delaunay cur_tri(n_simplices, n_simplex_size, n_mappings, n_mapping_size);

    // read simplices
    for (auto& simplex : cur_tri.simplices)
    {
      file.read(reinterpret_cast<char*>(simplex.data()), n_simplex_size * sizeof(int));
    }

    // read points
    for (auto& pt : cur_tri.points)
    {
      file.read(reinterpret_cast<char*>(pt.data()), (n_simplex_size - 1) * sizeof(int));
    }

    // read barycentric transformation
    for (auto& mat : cur_tri.barycentric_matrices)
    {
      file.read(reinterpret_cast<char*>(mat.data()), mat.size() * sizeof(double));
    }

    // read spatial maps
    //n_located_simplices = 0;
    for (auto& map : cur_tri.spatial_map)
    {
      file.read(reinterpret_cast<char*>(map.data()), map.size() * sizeof(int));
      auto pos = std::find_if(map.rbegin(), map.rend(), [](int value) { return value != -1; }).base();
      map.erase(pos, map.end());
      //n_located_simplices += map.size();
    }
    //printf("Ndim = %d\tAvg .= %2.3f\n", n_dim, (double)n_located_simplices / (double)cur_tri.spatial_map.size());

    delaunay_spatial_map_n_points = std::round(std::pow(cur_tri.spatial_map.size(), 1.0 / (double)n_dim));
    delaunay_spatial_map_step = 1.0 / (double)delaunay_spatial_map_n_points;
    tri_info.emplace(std::make_pair(n_dim, std::move(cur_tri)));
  }

  file.close();
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_hypercube(const std::vector<double> &points,
                                                                          std::array<int, N_DIMS> &hypercube,
                                                                          std::array<double, N_DIMS> &scaled_point,
                                                                          const int point_index)
{

    for (int i = 0; i < N_DIMS; i++)
    {
        double point = points[point_index + i];
        if (transform_last_axis && i == (N_DIMS - 1))
        {
            point = axes_max[i] - (point - axes_min[i]);
        }
        scaled_point[i] = (point - axes_min[i]) * axes_step_inv[i];
        hypercube[i] = (int)scaled_point[i];
        scaled_point[i] -= hypercube[i];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::find_simplex(const std::array<int, N_DIMS> &hypercube,
                                                                        const std::array<double, N_DIMS> &scaled_point,
                                                                        std::array<int, N_DIMS> &tri_order,
                                                                        std::array<std::array<int, N_DIMS>, N_DIMS + 1> &simplex)
{
    std::iota(tri_order.begin(), tri_order.end(), 0);
    std::sort(tri_order.begin(), tri_order.end(),
              [&scaled_point](int i1, int i2) { return scaled_point[i1] < scaled_point[i2]; });

    for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
    {
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            simplex[vertex_i][dim_i] = hypercube[dim_i];
        for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            simplex[vertex_i][tri_order[dim_i]] += standard_simplex[vertex_i][dim_i];
    }
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate(const std::vector<value_t> &point, std::vector<value_t> &values)
{
    std::array<int, N_DIMS> hypercube;
    std::array<double, N_DIMS> scaled_point;
    find_hypercube(point, hypercube, scaled_point);
    std::array<std::array<int, N_DIMS>, N_DIMS + 1> simplex;
    std::array<int, N_DIMS> tri_order;
    find_simplex(hypercube, scaled_point, tri_order, simplex);
    std::array<double, N_DIMS + 1> weights;

    if (use_barycentric_interpolation)
    {
      printf("Not supported!\n");
    }
    else
    {
      weights[0] = scaled_point[tri_order[0]];
      weights[N_DIMS] = 1 - scaled_point[tri_order[N_DIMS - 1]];

      for (int dim_i = 1; dim_i < N_DIMS; dim_i++)
        weights[dim_i] = scaled_point[tri_order[dim_i]] - scaled_point[tri_order[dim_i - 1]];
      values.assign(N_OPS, 0);
      for (int dim_i = 0; dim_i <= N_DIMS; dim_i++)
      {
        std::array<double, N_OPS> supp_values;
        get_supporting_point(simplex[dim_i], supp_values);
        for (int op = 0; op < N_OPS; op++)
        {
          values[op] += weights[dim_i] * supp_values[op];
        }
      }
    }
    return 0;
}

template <typename index_t, int N_DIMS, int N_OPS>
int linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::interpolate_with_derivatives(const std::vector<double> &points,
                                                                                       const std::vector<int> &points_idxs,
                                                                                       std::vector<double> &values,
                                                                                       std::vector<double> &derivatives)
{
    for (std::size_t point_i = 0; point_i < points_idxs.size(); point_i++)
    {
        int point_offset = points_idxs[point_i];

        std::array<int, N_DIMS> hypercube;
        std::array<double, N_DIMS> scaled_point;
        find_hypercube(points, hypercube, scaled_point, point_offset * N_DIMS);

        if (use_barycentric_interpolation)
        {
          // size of n-simplex
          constexpr int n_size = N_DIMS + 1;
          bool next_simplex;
          int s;

          // find index of cell in Delaunay spatial map the scaled_point belong to
          index_t index = 0;
          for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            index += int(scaled_point[dim_i] / delaunay_spatial_map_step) * delaunay_map_axes_mult[dim_i];
          
          // indentify possible simplices
          const auto& simplices_for_cells = tri_info[N_DIMS].spatial_map[index];

          // barycentric coordinates
          std::array<double, N_DIMS + 1> weights;

          // loop over these simplices
          for (int sim_i = 0; sim_i < simplices_for_cells.size(); sim_i++)
          {
            // simplex
            s = simplices_for_cells[sim_i];

            // barycentric transformation
            const auto& mat = tri_info[N_DIMS].barycentric_matrices[s];

            // calculate barycentric coordinates
            next_simplex = false;
            for (int i = 0; i < n_size; i++)
            {
              weights[i] = 0.0;
              for (int j = 0; j < N_DIMS ; j++)
                weights[i] += mat[i * n_size + j] * scaled_point[j];

              weights[i] += mat[i * n_size + N_DIMS];

              // check consistency
              if (weights[i] < 0.0 || weights[i] > 1.0)
              {
                next_simplex = true;
                break;
              }
            }

            if (!next_simplex)
              break;
          }

          if (next_simplex)
          {
            printf("Not found correct barycentric coordinates!\n");
            exit(-1);
          }

          // simplex is located
          const auto& sim_pts = tri_info[N_DIMS].simplices[s];
          const auto& mat = tri_info[N_DIMS].barycentric_matrices[s];

          // estimate points comprising the simplex
          std::array<std::array<int, N_DIMS>, N_DIMS + 1> simplex;
          const auto& vertices = tri_info[N_DIMS].points;
          for (int vertex_i = 0; vertex_i <= N_DIMS; vertex_i++)
          {
            for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            {
              simplex[vertex_i][dim_i] = hypercube[dim_i] + vertices[sim_pts[vertex_i]][dim_i];
            }
          }

          // retrieve values at supporting points
          std::array<std::array<double, N_OPS>, N_DIMS + 1> simplex_vertex_values;
          for (int i = 0; i <= N_DIMS; i++)
            get_supporting_point(simplex[i], simplex_vertex_values[i]);

          int idx;
          for (int op_i = 0; op_i < N_OPS; op_i++)
          {
            values[point_offset * N_OPS + op_i] = 0.0;
            // calculate values
            for (int pt_i = 0; pt_i <= N_DIMS; pt_i++)
              values[point_offset * N_OPS + op_i] += weights[pt_i] * simplex_vertex_values[pt_i][op_i];


            for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
            {
              idx = (point_offset * N_OPS + op_i) * N_DIMS + dim_i;
              derivatives[idx] = 0.0;
              // calculate derivatives
              for (int pt_i = 0; pt_i <= N_DIMS; pt_i++)
                derivatives[idx] += mat[pt_i * n_size + dim_i] * simplex_vertex_values[pt_i][op_i];

              if (transform_last_axis && dim_i == (N_DIMS - 1))
                derivatives[idx] *= -axes_step_inv[dim_i];
              else
                derivatives[idx] *= axes_step_inv[dim_i];
            }
          }
        }
        else
        {
          std::array<std::array<int, N_DIMS>, N_DIMS + 1> simplex;
          std::array<int, N_DIMS> tri_order;
          find_simplex(hypercube, scaled_point, tri_order, simplex);

          std::array<std::array<double, N_OPS>, N_DIMS + 1> simplex_vertex_values;
          for (int i = 0; i <= N_DIMS; i++)
            get_supporting_point(simplex[i], simplex_vertex_values[i]);

          for (int i = 0; i < N_OPS; i++)
          {
            values[point_offset * N_OPS + i] = simplex_vertex_values[0][i];
          }
          double simplex_vertex_values_gap;
          for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
          {
            for (int op_i = 0; op_i < N_OPS; op_i++)
            {
              simplex_vertex_values_gap = simplex_vertex_values[dim_i + 1][op_i] - simplex_vertex_values[dim_i][op_i];
              values[point_offset * N_OPS + op_i] += (1 - scaled_point[tri_order[dim_i]]) * simplex_vertex_values_gap;
              if (transform_last_axis && tri_order[dim_i] == (N_DIMS - 1))
                simplex_vertex_values_gap *= -1;
              derivatives[(point_offset * N_OPS + op_i) * N_DIMS + tri_order[dim_i]] = -simplex_vertex_values_gap * axes_step_inv[tri_order[dim_i]];
            }
          }
        }
    }
    return 0;
}

template <typename index_t, int N_DIMS, int N_OPS>
index_t linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::get_index_from_vertex(const std::array<int, N_DIMS> &vertex)
{
    index_t index = 0;
    for (int dim_i = 0; dim_i < N_DIMS; dim_i++)
        index += vertex[dim_i] * this->axes_mult[dim_i];
    return index;
}

template <typename index_t, int N_DIMS, int N_OPS>
void linear_cpu_interpolator_base<index_t, N_DIMS, N_OPS>::get_point_from_vertex(const std::array<int, N_DIMS> &vertex,
                                                                                 std::vector<double> &point)
{
    for (int i = 0; i < N_DIMS; i++)
        point[i] = vertex[i] * axes_step[i] + axes_min[i];
    if (transform_last_axis)
        point[N_DIMS - 1] = axes_max[N_DIMS - 1] - (point[N_DIMS - 1] - axes_min[N_DIMS - 1]);
}