#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cassert>

#include "dartsflash/maths/linesearch.hpp"

LineSearch::LineSearch(int n_vars) 
{
    n = n_vars;

    xold = Eigen::VectorXd(n_vars);
    x = Eigen::VectorXd(n_vars);

    g = Eigen::VectorXd(n_vars);
    p = Eigen::VectorXd(n_vars);
}

bool LineSearch::init(double alam_, Eigen::VectorXd& xold_, double fold_, Eigen::VectorXd& g_, Eigen::VectorXd& p_, const double stpmax_) 
{
    // lambda
    alam = alam_;
    // Newton step
    p = p_;
    // Independent variable
    xold = xold_;
    x = xold_;
    // Objective function   
    fold = fold_;
    f = fold_;
    // Gradient
    g = g_;
    // Maximum step *problem dependent
    stpmax = stpmax_;

    ALF = 1.0e-4;
    TOLX = std::numeric_limits<double>::epsilon();
    alam2 = f2 = slope = sum = 0.0;

    // Scale Newton step if too big
    sum = p.norm();
    if (sum > stpmax)
    {
        for (int i = 0; i < n; i++)
        {
            p(i) *= stpmax/sum;
        }
    }

    // Slope of the objective function
    for (int i = 0; i < n; i++)
    {
        slope -= g(i)*p(i);
    }

    if (slope >= 0.0) 
    {
        return false;
    }

    test=0.0;
    // Calculate minimum step length
    for ( int i = 0; i < n; i++)
    {
        temp = std::fabs(p(i)) / std::max(std::fabs(xold(i)), 1.0);
        if (temp > test) { test = temp; }
    }

    alamin = TOLX/test;

    return true;
}

bool LineSearch::process(Eigen::VectorXd& xnew_, double fnew_)
{
    x = xnew_;
    f = fnew_;

    if (alam < alamin) 
    {
        for (int i = 0; i < n; i++) 
        {
            x[i] = xold[i];
        }
        return false;
    } 
    else if (f <= fold+ALF*alam*slope) {return false;}
    else 
    {
        if (alam == 1.0)
        {
            tmplam = -slope/(2.0*(f-fold-slope));
        }
        else 
        {
            rhs1 = f-fold-alam*slope;
            rhs2 = f2-fold-alam2*slope;
            a = (rhs1/(alam*alam)-rhs2/(alam2*alam2))/(alam-alam2);
            b = (-alam2*rhs1/(alam*alam)+alam*rhs2/(alam2*alam2))/(alam-alam2);
            if (a == 0.0) tmplam = -slope/(2.0*b);
            else 
            {
                disc = b*b-3.0*a*slope;
                if (disc < 0.0) tmplam=0.5*alam;
                else if (b <= 0.0) tmplam=(-b+sqrt(disc))/(3.0*a);
                else tmplam=-slope/(b+sqrt(disc));
            }
            if (tmplam > 0.5*alam)
                tmplam = 0.5*alam;
        }
    }

    alam2 = alam;
    f2 = f;
    alam = std::max(tmplam,0.1*alam);
    if(std::isnan(alam)) {return false;}

    return true;
}
