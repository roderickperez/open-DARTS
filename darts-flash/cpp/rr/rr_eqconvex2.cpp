#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "dartsflash/rr/rr.hpp"
#include "dartsflash/global/global.hpp"

RR_EqConvex2::RR_EqConvex2(FlashParams& flashparams, int nc_) : RR(flashparams, nc_, 2)
{
    ci.resize(nc);
    di.resize(nc);

    k_idxs.resize(nc);
    std::iota(k_idxs.begin(), k_idxs.end(), 0);
}

int RR_EqConvex2::solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_) {
    // Solve two-phase Rachford-Rice equation using convex transformations - Nichita and Leibovici (2013)
    this->init(z_, K_, nonzero_comp_);

    // Find ci values
    for (int i = 0; i < nc; i++)
    {
        ci[i] = 1./(1.-K[i]);
    }

    // Find order of K values and calculate di values
    k_idxs = this->sort_idxs(K);

    // If two components, solution is readily obtained
    if (nc == 2)
    {
        double a = this->aL(z[k_idxs[0]]);  // aL(z1)
        // double a = this->aR(z[k_idxs[nc-1]]);  // aR(zn)
        double v = this->V(a);
        nu = {1-v, v};
        return this->output(0, a);
    }
    
    for (int i = 0; i < nc; i++)
    {
        di[k_idxs[i]] = (ci[k_idxs[0]] - ci[k_idxs[i]]) / (ci[k_idxs[nc-1]] - ci[k_idxs[0]]);
    }

    return this->solve_fgh();
}

std::vector<int> RR_EqConvex2::sort_idxs(std::vector<double> ki) {
    // Sort idxs of K-values in descending order
    
    // sort indexes based on comparing values in ki
    // using std::stable_sort instead of std::sort
    // to avoid unnecessary index re-orderings
    // when ki contains elements of equal values 
    
    std::sort(k_idxs.begin(), k_idxs.end(),
       [&ki](int i1, int i2) {return ki[i1] > ki[i2];});

    return k_idxs;
}

int RR_EqConvex2::solve_gh() {
    // Calculate solution window aL and aR
    double a_L = this->aL(z[k_idxs[0]]);  // aL(z1)
    double a_R = this->aR(z[k_idxs[nc-1]]);  // aR(zn)

    // Calculate F(aL) and F(aR)
    double FaL = this->F(a_L);
    double FaR = this->F(a_R);

    // Calculate initial guess a0
    double m = (a_R - a_L) / (FaR - FaL); // inverse of slope
    double a = a_L - FaL * m;  // a0 = aL - F(aL)/slope

    // Calculate G(a)
    double Fa = this->F(a);
    double Ga = this->G(a, Fa);

    // If G(a0) > 0, solve G(a) = 0
    int iter = 0;
    if (Ga > 0)
    {
        iter++;
        double ak;
        while (iter <= max_iter) {
            ak = a - Ga/this->dG(a);
            norm = std::fabs(ak-a);
            if (norm < rr2_tol)
            {
                return this->output(0, ak);
            }
            else
            {
                Ga = this->G(ak);
                a = ak;
            }
        }
    }
    // Otherwise, solve H(a) = 0
    else
    {
        iter++;
        double ak;
        double Ha = -this->H(a, Fa);
        while (iter <= max_iter) {
            ak = a - Ha/this->dH(a);
            norm = std::fabs(ak-a);
            if (norm < rr2_tol)
            {
                return this->output(0, ak);
            }
            else
            {
                Ha = this->H(ak);
                a = ak;
            }
        }
    }

    return this->output(1, a);
}

int RR_EqConvex2::solve_fgh() {
    // Calculate solution window aL and aR
    double a_L = this->aL(z[k_idxs[0]]);  // aL(z1)
    double a_R = this->aR(z[k_idxs[nc-1]]);  // aR(zn)
    if (std::fabs(a_L-a_R) < rr2_tol)
    {
        return this->output(0, a_L);
    }

    // Calculate F(aL) and F(aR)
    double FaL = this->F(a_L);
    double FaR = this->F(a_R);

    // Calculate initial guess a0
    double m = (a_R - a_L) / (FaR - FaL); // inverse of slope
    double a = a_L - FaL * m;  // a0 = aL - F(aL)/slope

    // Convergence loop
    double Fa, dFa, ak;
    int iter = 0;
    while (iter <= max_iter)
    {
        iter++;

        // Calculate F(a) and F'(a)
        Fa = this->F(a);
        dFa = this->dF(a);
        norm = std::fabs(Fa);
        bool accept = false;

        // If F'(a) <= 0, perform Newton step: a_k = a - F(a)/F'(a)
        if (dFa <= 0.)
        {
            ak = a - Fa/dFa;

            // Check if ak has converged, if so return v
            norm = std::fabs(this->F(ak));
            if (norm < rr2_tol)
            {
                return this->output(0, ak);
            }

            // Check if ak is within range (aL, aR)
            if (a_L <= ak && ak <= a_R)
            {
                accept = true;
                a = ak;
            }
        }

        if (!accept)
        {
            // If F'(a) > 0 or ak not within bounds, don't accept Newton step and update a with either G or H
            if (Fa > 0)
            {
                // If F(a) > 0, update with G: a_k = a - G(a)/G'(a)
                a -= this->G(a, Fa)/this->dG(a);
            }
            else
            {
                // Else, update with H: a_k = a - H(a)/H'(a)
                a -= this->H(a, Fa)/this->dH(a);
            }
        }
    }
    return this->output(1, a);
}

int RR_EqConvex2::output(int error, double a)
{
    if (error == 1 && this->verbose)
    {
        print("MAX RR Iterations", max_iter);
        print("Norm", norm);
    }

    double v = this->V(a);
    nu = {1-v, v};
    return error;
}

std::vector<double> RR_EqConvex2::getx()
{
    std::vector<double> x(2*nc);

    // Find whether L or V is 'general' phase fraction
    double phi, phi_min, phi_max;
    int u_idx, v_idx;

    if (nu[0] > nu[1])
    {
        // In L > V, phi = V; take ui = xi and vi = yi
        u_idx = 0;
        v_idx = 1;
        phi = nu[v_idx];
        phi_min = ci[k_idxs[0]];
        phi_max = ci[k_idxs[nc-1]];
    }
    else
    {
        // Else, phi = L; take ui = yi and vi = xi
        for (int i = 0; i < nc; i++)
        {
            K[i] = 1./K[i];
            ci[i] = 1./(1.-K[i]);
        }
        u_idx = 1;
        v_idx = 0;
        phi = nu[v_idx];
        phi_min = ci[k_idxs[nc-1]];
        phi_max = ci[k_idxs[0]];
    }

    // Calculate ui
    double constexpr tol = 1e-16;
    double const a = (phi - phi_min) / (phi_max - phi);
    /* double const b = 1. / (phi - phi_min); */
    if (phi - phi_min < tol || phi_max - phi < tol) 
    { 
        // If phi is exactly on one of the bounds, use regular expression, avoiding division by zero
        for (int i = 0; i < nc; i++)
        {
            x[u_idx*nc + i] = z[i] * ci[i] / (ci[i] * (1.-phi));
            x[v_idx*nc + i] = x[u_idx*nc + i] * K[i];
        }
    }
    else
    {
        // Else, use Nielsen and Mia (2023) expression, avoiding division by zero
        for (int i = 0; i < nc; i++)
        {
            x[u_idx*nc + i] = -z[i] * ci[i] / (a * (phi_max - phi) - (ci[i] - phi_min));
            x[v_idx*nc + i] = x[u_idx*nc + i] * K[i];
        }
    }

	return x;
}

double RR_EqConvex2::aL(double z1) {
    return z1 / (1.-z1);
}

double RR_EqConvex2::aR(double zn) {
    return (1.-zn) / zn;
}

double RR_EqConvex2::V(double a) {
    return (ci[k_idxs[0]] + a * ci[k_idxs[nc-1]]) / (1. + a);
}

double RR_EqConvex2::F(double a) {
    // F(a) = z1 + sum_i [zi*a / (di + a(1+di))] - zn*a
    double z1 = z[k_idxs[0]];
    double zn = z[k_idxs[nc-1]];

    double sum_i = 0.;
    for (int i = 1; i < nc-1; i++)
    {
        sum_i += z[k_idxs[i]] * a / (di[k_idxs[i]] + a * (1. + di[k_idxs[i]]));
    }
    return z1 + sum_i - zn * a;
}

double RR_EqConvex2::dF(double a) {
    // F'(a) = sum_i [zi*di / (di + a(1+di))^2] - zn
    double zn = z[k_idxs[nc-1]];

    double sum_i = 0.;
    for (int i = 1; i < nc-1; i++)
    {
        int idx = k_idxs[i];
        sum_i += z[idx] * di[idx] / std::pow(di[idx] + a * (1. + di[idx]), 2);
    }
    return sum_i - zn;
}

double RR_EqConvex2::G(double a) {
    // G(a) = z1*(1+a)/a + sum_i [zi*(1+a) / (di + a*(1+di))] - zn * (1 + a)
    double z1 = z[k_idxs[0]];
    double zn = z[k_idxs[nc-1]];

    double sum_i = 0.;
    for (int i = 1; i < nc-1; i++)
    {
        int idx = k_idxs[i];
        sum_i += z[idx] * (1.+a) / (di[idx] + a*(1.+di[idx]));
    }
    return z1*(1.+a)/a + sum_i - zn*(1.+a);
}

double RR_EqConvex2::G(double a, double f) {
    // G(a) = (a+1)/a * F(a)
    return (a + 1.)/a * f;
}

double RR_EqConvex2::dG(double a) {
    // G'(a) = -z1/a^2 - sum_i [zi / (di + a*(1+di))^2] - zn
    double z1 = z[k_idxs[0]];
    double zn = z[k_idxs[nc-1]];

    double sum_i = 0.;
    for (int i = 1; i < nc-1; i++)
    {
        int idx = k_idxs[i];
        sum_i += z[idx] / std::pow(di[idx] + a*(1.+di[idx]), 2);
    }
    return -z1 / std::pow(a, 2) - sum_i - zn;
}

double RR_EqConvex2::H(double a) {
    // H(a) = -z1 * (1+a) - sum_i [zi*a*(1+a) / (di + a*(1+di))^2] + zn*a*(1+a)
    double z1 = z[k_idxs[0]];
    double zn = z[k_idxs[nc-1]];

    double sum_i = 0.;
    for (int i = 1; i < nc-1; i++)
    {
        int idx = k_idxs[i];
        sum_i += z[idx] * a * (1.+a) / (di[idx] + a*(1.+di[idx]));
    }
    return -z1*(1.+a) - sum_i + zn*a*(1.+a);
}

double RR_EqConvex2::H(double a, double f) {
    // H(a) = -a * G(a)
    return -a * this->G(a, f);
}

double RR_EqConvex2::dH(double a) {
    // H'(a) = -z1 - sum_i [zi * [di * (1+a)^2 + a^2] / (di + a*(1+di))^2] + zn*(1+2a)
    double z1 = z[k_idxs[0]];
    double zn = z[k_idxs[nc-1]];

    double sum_i = 0.;
    for (int i = 1; i < nc-1; i++)
    {
        int idx = k_idxs[i];
        sum_i += z[idx] * (di[idx] * std::pow(1+a, 2) + std::pow(a, 2)) / std::pow(di[idx] + a*(1.+di[idx]), 2);
    }
    return -z1 - sum_i + zn * (1.+2*a);
}
