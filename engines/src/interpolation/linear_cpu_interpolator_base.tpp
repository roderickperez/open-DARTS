#include <numeric>
#include <algorithm>
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
      int64_max = std::numeric_limits<index_t>::max();

    if (n_points_total_fp > int64_max)
    {
        std::string error = "Error: The total requested amount of points (" + std::to_string(n_points_total_fp) +
                            ") exceeds the limit in index type (" + std::to_string(std::numeric_limits<index_t>::max()) + ")\n";
        throw std::range_error(error);
    }

    transform_last_axis = 1;
    use_barycentric_interpolation = false;
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

        std::array<std::array<int, N_DIMS>, N_DIMS + 1> simplex;
        std::array<int, N_DIMS> tri_order;
        find_simplex(hypercube, scaled_point, tri_order, simplex);

        std::array<std::array<double, N_OPS>, N_DIMS + 1> simplex_vertex_values;
        for (int i = 0; i <= N_DIMS; i++)
            get_supporting_point(simplex[i], simplex_vertex_values[i]);

        if (use_barycentric_interpolation)
        {
          //// find barycentric coordinates (weights)
          constexpr int n_size = N_DIMS + 1;
          linalg::Matrix<double> mat(n_size, n_size);
          std::array<double, n_size> rhs;
          std::array<double, N_DIMS + 1> weights;

          // fill matrix
          for (int j = 0; j < n_size; j++)
          {
            for (int i = 0; i < N_DIMS; i++)
              mat(i, j) = simplex[j][i] - hypercube[i];

            mat(N_DIMS, j) = 1.0;
          }
          // fill rhs
          for (int i = 0; i < N_DIMS; i++)
            rhs[i] = scaled_point[i];
          rhs[N_DIMS] = 1.0;
          // invert the system
          bool res = mat.inv();
          if (!res)
          {
            printf("Inversion failed!\n");
            exit(-1);
          }
          // find coordinates
          for (int i = 0; i < n_size; i++)
          {
            weights[i] = 0.0;
            for (int j = 0; j < n_size; j++)
              weights[i] += mat(i, j) * rhs[j];
          }

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
                derivatives[idx] += mat(pt_i, dim_i) * simplex_vertex_values[pt_i][op_i];

              if (transform_last_axis && dim_i == (N_DIMS - 1))
                derivatives[idx] *= -axes_step_inv[dim_i];
              else
                derivatives[idx] *= axes_step_inv[dim_i];
            }
          }
        }
        else
        {
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