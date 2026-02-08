//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_MATHS_ROOTFINDING_H
#define OPENDARTS_FLASH_MATHS_ROOTFINDING_H
//--------------------------------------------------------------------------

#include <vector>
#include <complex>
#include <functional>

// template <int ND>
class RootFinding
{
private:
    double x;

public:
    RootFinding();

    int bisection(std::function<double(double)> obj_fun, 
                  double& x_, double& a, double& b, double tol_f, double tol_t);
    int bisection_newton(std::function<double(double)> obj_fun, std::function<double(double)> gradient, 
                         double& x_, double& a, double& b, double tol_f, double tol_t);
    int brent(std::function<double(double)> obj_fun, 
              double& x_, double& a, double& b, double tol_f, double tol_t);
    int brent_newton(std::function<double(double)> obj_fun, std::function<double(double)> gradient, 
                     double& x_, double& a, double& b, double tol_f, double tol_t);

    double& getx() { return this->x; }
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_MATHS_ROOTFINDING_H
//--------------------------------------------------------------------------
