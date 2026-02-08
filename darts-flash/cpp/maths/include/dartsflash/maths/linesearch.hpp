//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_MATHS_LINESEARCH_H
#define OPENDARTS_FLASH_MATHS_LINESEARCH_H
//--------------------------------------------------------------------------

#include <vector>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "dartsflash/global/global.hpp"

#include <Eigen/Dense>

class LineSearch {
private:

    double ALF, TOLX;
    double a, alam2, alamin, b, disc, f2;
    double rhs1, rhs2, slope, sum, temp, test, tmplam;
    int  n;

    Eigen::VectorXd g, p, xold, x;
    double fold, f, stpmax;
    double alam;

public:
    LineSearch(int n_vars);
    ~LineSearch() = default;
    
    bool init(double alam_, Eigen::VectorXd& xold_, double fold_, Eigen::VectorXd& g_, Eigen::VectorXd& p_, const double stpmax_);
    bool process(Eigen::VectorXd& xnew_, double fnew_);

    double &get_alam() { return this->alam; }
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_MATHS_LINESEARCH_H
//--------------------------------------------------------------------------
