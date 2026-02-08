#include <cmath>
#include <vector>
#include <complex>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/maths/geometry.hpp"
#include "dartsflash/maths/maths.hpp"
#include <Eigen/Dense>

bool is_in_simplex(std::vector<double>& point, std::vector<std::vector<double>>& coords)
{
    // "Triangle test" to calculate whether a point is located within an N-simplex (barycentric coordinates)
    // If point is located inside, the sum of subvolumes is equal to the simplex volume
    // Else, the sum of subvolumes will be larger
    size_t nd = coords.size();
    coords.push_back(point);
    std::vector<int> simplex_idxs(nd);
    std::iota(simplex_idxs.begin(), simplex_idxs.end(), 0);

    // Calculate simplex volume
    double simplex_volume = std::fabs(calc_simplex_volume(coords, simplex_idxs));

    // Calculate volumes of "subsimplices"
    double sum_of_simplex_volumes = 0.;
    Combinations subsimplex_idxs(nd, nd-1);
    for (std::vector<int>& idxs: subsimplex_idxs.combinations)
    {
        idxs.insert(idxs.begin(), static_cast<int>(nd));
        double volume = std::fabs(calc_simplex_volume(coords, idxs));
        sum_of_simplex_volumes += volume;
    }
    coords.pop_back();

    return (sum_of_simplex_volumes > simplex_volume + 1e-15) ? false : true;
}

double calc_simplex_volume(std::vector<std::vector<double>>& coords, std::vector<int>& idxs)
{
    // Calculate volume of N-simplex in R^n
    // Cayley-Menger Determinant? https://mathworld.wolfram.com/Cayley-MengerDeterminant.html
    size_t nd = idxs.size();
    Eigen::MatrixXd mat(nd, nd);
    for (size_t i = 0; i < idxs.size(); i++)
    {
        for (size_t j = 0; j < nd-1; j++)
        {
            mat(i, j) = coords[idxs[i]][j];
        }
        mat(i, nd-1) = 1.;
    }
    double det = mat.determinant();

    return 1./maths::factorial[nd-1] * det;
}

std::vector<double> find_midpoint(std::vector<std::vector<double>>& coords)
{
    size_t ncoords = coords.size();
    size_t ndims = coords[0].size();
    
    double weight = 1./ncoords;
    std::vector<double> midpoint(ndims, 0.);
    for (size_t i = 0; i < ndims; i++)
    {
        for (size_t j = 0; j < ncoords; j++)
        {
            midpoint[i] += weight * coords[j][i];
        }
    }
    return midpoint;
}
