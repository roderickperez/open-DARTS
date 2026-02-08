#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

#include "dartsflash/rr/rr.hpp"
#include "dartsflash/global/global.hpp"
#include <Eigen/Dense>

RR_Min::RR_Min(FlashParams& flash_params, int nc_, int np_) : RR(flash_params, nc_, np_) { }

int RR_Min::solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_)
{
    // Minimization of MRR with negative flash [Yan, 2012] formulated with K-values (fugacity coefficients in reference paper)
    Eigen::VectorXd nu_vec(np-1), nu_new = Eigen::VectorXd::Constant(np-1, 1./np);
    this->init(z_, K_, nonzero_comp_);

    // Calculate objective function
    double Q, Q_new = this->calc_obj(nu_new);
    if (std::isnan(Q_new))
    {
        if (this->verbose) { std::cout << "NaN value in RR Q!\n"; print("z", z); print("K", K_); }
        return 1;
    }
    Eigen::MatrixXd H(np-1, np-1);
    Eigen::VectorXd dnu(np-1);
    g = Eigen::VectorXd(np-1);

    int iter = 1, iter_loose = 0;
    while (iter < max_iter)
    {
        nu_vec = nu_new;
        nu[0] = 1.;
        for (int j = 0; j < np-1; j++)
        {
            nu[j+1] = nu_vec(j);
            nu[0] -= nu_vec(j);
        }
        Q = Q_new;

        // Calculate inverse of Ei for use in gradient vector and Hessian calculation
        std::vector<double> Ei_inv(nc);
        std::transform(Ei.begin(), Ei.end(), Ei_inv.begin(), [](double ei) { return 1./ei; });

        // Calculate gradient vector and norm
        this->update_g(Ei_inv);
        norm = this->l2norm();

        // If norm below tolerance, return 0
        if (norm < rrn_tol)
        {
            return this->output(0);
        }
        else if (norm < loose_tol)
        {
            iter_loose++;
            if (iter_loose == loose_iter)
            {
                return this->output(0);
            }
        }

        // Calculate Hessian H = U z U^T
        for (int j = 0; j < np-1; j++)
        {
            for (int k = j; k < np-1; k++)
            {
                double Hjk = 0.;
                for (int i = 0; i < nc; i++)
                {
                    Hjk += z[i]*std::pow(Ei_inv[i], 2) * (K[j*nc + i] - 1.) * (K[k*nc + i] - 1.);
                }

                H(j, k) = Hjk;
                H(k, j) = Hjk;
            }
        }
        dnu = H.llt().solve(g);

        // Update nu with approximate line search
        LineSearch linesearch{ np-1 };
        double lambda = 1.;
        int line_iter = 0;
        bool check = linesearch.init(lambda, nu_vec, Q, g, dnu, 1.0);

        while (line_iter < max_iter && check)
        {
            // Update nu_new
            nu_new = nu_vec - lambda * dnu;
            Q_new = this->calc_obj(nu_new);

            if (std::isnan(Q_new))
            {
                // Q_new is NaN, reduce step size
                lambda *= 0.5;
            }
            else if (Q_new < Q)
            {
                // Requires x_ij >= 0, which corresponds to Ei >= 0
                // If not, reduce step size
                if (std::count_if(Ei.begin(), Ei.end(), [](double ei) { return (ei < 0.); }) > 0.)
                {
                    check = linesearch.process(nu_new, Q_new);
                    lambda = linesearch.get_alam();
                    line_iter++;
                }
                else
                {
                    break;
                }
            }
            // else if (norm < rrn_tol)
            // {
            //     std::transform(Ei.begin(), Ei.end(), Ei_inv.begin(), [](double ei) { return 1./ei; });
            //     this->update_g(Ei_inv);
            //     double norm_new = this->l2norm();
            //     if (std::fabs(norm_new-norm) < norm)
            //     {
            //         nu = nu_new;
            //         norm = norm_new;
            //         return output(0);
            //     }
            //     else
            //     {
            //         lambda *= 0.5;
            //     }
            // }
            else
            {
                // Objective function not reduced, reduce step size
                check = linesearch.process(nu_new, Q_new);
                lambda = linesearch.get_alam();
                line_iter++;
            }

            line_iter++;
            if (line_iter == max_iter)
            {
                // Maximum number of line search iterations
                if (this->verbose) { std::cout << "Line search in RR_Min failed!\n"; }
                return 1;
            }
        }

        iter++;
    }

    return this->output(1);
}

double RR_Min::calc_obj(Eigen::VectorXd& nu_)
{
    // Calculate objective function and Ei
    Ei = std::vector<double>(nc, 1.);
    std::vector<double> lnE(nc);
    for (int i = 0; i < nc; i++)
    {
        for (int j = 1; j < np; j++)
        {
            Ei[i] += nu_(j-1) * (K[(j-1)*nc + i] - 1.);
        }
        lnE[i] = std::log(Ei[i]);
    }

    // Calculate updated objective function Q' = -Σ zi ln(Ei)
    return -std::inner_product(z.begin(), z.end(), lnE.begin(), 0.);
}

void RR_Min::update_g(std::vector<double>& Ei_inv)
{
    // Construct gradient vector
    for (int j = 0; j < np-1; j++)
    {
        g(j) = 0.;
        for (int i = 0; i < nc; i++)
        {
            g(j) -= z[i]*Ei_inv[i] * (K[j*nc + i] - 1.);
        }
    }
    return;
}

std::vector<double> RR_Min::getx()
{
    // Calculation of x from Yan (2012) minimization
    std::vector<double> x(np*nc);

    for (int i = 0; i < nc; i++)
    {
        x[i] = z[i] / Ei[i];
        for (int j = 1; j < np; j++)
        {
            x[j*nc + i] = K[(j-1)*nc + i] * x[i];
        }
    }

    return x;
}
