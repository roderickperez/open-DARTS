#include <vector>
#include <cmath>

#include "dartsflash/global/global.hpp"

bool compare_compositions(std::vector<double>& X0, std::vector<double>& X1, double tol)
{
    for (size_t i = 0; i < X0.size(); i++)
    {
        double lnx0 = std::log(X0[i]);
        double x_diff = lnx0 < 0. ? std::fabs(X1[i]-X0[i]) : lnx0 - std::log(std::fabs(X1[i]) + 1e-15);
        if (x_diff > tol) // composition is different
        {
            return false;
        }
    }
    // composition is the same
    return true;
}
