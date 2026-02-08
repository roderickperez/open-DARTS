//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_MATHS_GEOMETRY_H
#define OPENDARTS_FLASH_MATHS_GEOMETRY_H
//--------------------------------------------------------------------------

#include <vector>
#include <complex>

// template <int ND>
bool is_in_simplex(std::vector<double>& point, std::vector<std::vector<double>>& coords);
double calc_simplex_volume(std::vector<std::vector<double>>& coords, std::vector<int>& idxs);
std::vector<double> find_midpoint(std::vector<std::vector<double>>& coords);

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_MATHS_GEOMETRY_H
//--------------------------------------------------------------------------
