#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/timer.hpp"
#include "dartsflash/eos/eos.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/trial_phase.hpp"
#include "dartsflash/stability/stability.hpp"
#include <Eigen/Dense>

Stability::Stability(FlashParams& flashparams)
{
    this->flash_params = flashparams;

    this->nc = flash_params.nc;
    this->ni = flash_params.ni;
    this->ns = nc + ni;

    this->h.resize(nc);
    this->lnphi.resize(nc);
    this->lnY.resize(nc);
    this->alpha.resize(nc);
    this->dlnphidn.resize(nc*nc);
    this->g = Eigen::VectorXd(nc);
}

void Stability::init(std::vector<TrialPhase>& ref_comps)
{
    // Evaluate ln(phi(x)) of reference phase
    flash_params.timer.start(Timer::timer::STABILITY);
    this->ref_compositions = ref_comps;

    // Calculate h: ln(x) + ln(phi(x))
    bool second_order = false;
    this->update_fugacities(ref_compositions[0], second_order);
    for (int i = 0; i < nc; i++)
    {
        h[i] = std::log(ref_compositions[0].Y[i]) + lnphi[i];
    }
    flash_params.timer.stop(Timer::timer::STABILITY);

    if (flash_params.verbose)
    {
        std::cout << "Running stability with reference phase(s):\n";
        for (TrialPhase ref: ref_compositions)
        {
            ref.print_point();
        }
    }
    return;
}

void Stability::init_gmix(TrialPhase& trial_comp, std::vector<double>& lnphi0)
{
    // Evaluate ln(phi(x)) of reference phase
    flash_params.timer.start(Timer::timer::STABILITY);
    this->ref_compositions = {};
    trial_comp.Y = trial_comp.y;

    // Calculate h: ln(x) + ln(phi(x))
    for (int i = 0; i < ns; i++)
    {
        h[i] = lnphi0[i];
    }
    flash_params.timer.stop(Timer::timer::STABILITY);

    return;
}

int Stability::run(TrialPhase& trial_comp, bool gmix_min)
{
    // Run stability test for trial phase with composition Y
    flash_params.timer.start(Timer::timer::STABILITY);
    this->trial_composition = trial_comp;
    std::shared_ptr<EoSParams> params{ flash_params.eos_params[trial_comp.eos_name] };
    ssi_iter = newton_iter = switch_back_iter = 0;

    // If reference and trial phase are the same but reference EoS only allows a single minimum, return ref_comp
    if (!params->eos->has_multiple_minima())
    {
        for (TrialPhase ref: this->ref_compositions)
        {
            if (trial_composition.eos_name == ref.eos_name)
            {
                ref.set_stationary_point(ref.Y.begin(), 0.);
                trial_comp = ref;
                flash_params.timer.stop(Timer::timer::STABILITY);
                if (flash_params.verbose)
                {
                    print("Reference EoS only allows a single minimum", trial_comp.eos_name);
                    print("Y", trial_comp.Y);
                }
                return 0;
            }
        }
    }

    // Find nonzero compositions in trial phase
    this->nonzero_comp = {};
    for (int i = 0; i < ns; i++)
    {
        if (trial_comp.Y[i] > flash_params.min_z)
        {
            nonzero_comp.push_back(i);
        }
    }
    this->ns_nonzero = static_cast<int>(nonzero_comp.size());

    // // If minimum of gmix has been calculated and gmix is used, start from there
    // if (this->params.use_gmix && trial_comp.gmin_converged)
    // {
    //     print("starting from ymin", trial_composition.ymin);
    //     trial_composition.Y = trial_composition.ymin;
    // }

    // Perform successive substitution steps
    for (int i = 0; i < ns; i++)
    {
        lnY[i] = std::log(trial_composition.Y[i]);
        alpha[i] = 2.*std::sqrt(trial_composition.Y[i]);
    }

    // Update fugacities and gradient vector
    bool second_order = false;
    this->update_fugacities(trial_composition, second_order);
    this->update_g();
    tpd = this->calc_modtpd();
    double norm = this->l2norm();
    double old_norm = norm;

    if (flash_params.verbose)
    {
        trial_composition.print_point("Running stability with TrialPhase");
        print("norm", norm);
    }

    // Perform successive substitution steps
    while (ssi_iter < params->stability_max_iter)
    {
        if (error != 0 || std::isinf(norm) || std::isnan(norm))
        {
            // If some error occur during SSI procedure, return 1
            if (flash_params.verbose)
            {
                print("Error in stability, norm", norm);
            }
            flash_params.timer.stop(Timer::timer::STABILITY);
            return 1;
        }
        else if (norm < params->stability_tol ||
                (norm < params->stability_switch_tol && std::log(old_norm) - std::log(norm) < params->stability_switch_diff))
        {
            // Check if within tolerance, or else if SSI step is still effective
            break;
        }
        else if (!trial_composition.is_in_range)
        {
            trial_composition.set_stationary_point(trial_composition.Y.begin(), tpd);
            trial_comp = this->trial_composition;
            flash_params.timer.stop(Timer::timer::STABILITY);
            if (flash_params.verbose)
            {
                print("Stability test out of range for EoS, norm", norm);
                trial_comp.print_point();
            }
            return 0;
        }

        // Perform SSI step
        this->perform_ssi();
        old_norm = norm;
        norm = this->l2norm();
    }

    if (norm >= params->stability_tol && ssi_iter < params->stability_max_iter)
    {
        // Perform Newton steps
        second_order = true;
        this->update_fugacities(trial_composition, second_order);

        switch (flash_params.stability_variables)
        {
            case FlashParams::Y:
            {
                while (newton_iter + switch_back_iter < params->stability_max_iter)
                {
                    this->perform_Y();
                    norm = this->l2norm();
                    if (norm < params->stability_tol)
                    {
                        break;
                    }
                }
                break;
            }
            case FlashParams::lnY:
            {
                while (newton_iter + switch_back_iter < params->stability_max_iter)
                {
                    this->perform_lnY();
                    norm = this->l2norm();
                    if (norm < params->stability_tol)
                    {
                        break;
                    }
                }
                break;
            }
            case FlashParams::alpha:
            {
                while (newton_iter + switch_back_iter < params->stability_max_iter)
                {
                    this->perform_alpha();
                    norm = this->l2norm();
                    if (norm < params->stability_tol)
                    {
                        break;
                    }
                }
                break;
            }
            default:
            {
                std::cout << "Invalid stability variables defined\n";
                flash_params.timer.stop(Timer::timer::STABILITY);
                exit(1);
            }
        }
    }

    if (gmix_min)
    {
        trial_composition.set_gmix_min(trial_composition.Y.begin(), tpd);
    }
    else
    {
        trial_composition.set_stationary_point(trial_composition.Y.begin(), tpd);
    }
    // trial_composition.root = flash_params.eos_params[trial_composition.eos_name].eos->is_root_type();
    trial_comp = this->trial_composition;
    flash_params.timer.stop(Timer::timer::STABILITY);

    if (flash_params.verbose)
    {
        print("Stability", "===============");
        print("ssi iterations", ssi_iter);
        print("of which switch back from Newton", switch_back_iter);
        print("Newton iterations", newton_iter);
        print("norm", norm);
        trial_composition.print_point("Stationary point");
        print("============", "==============");
    }

    if (this->l2norm() > params->stability_tol)
    {
        trial_comp.has_converged = false;
        if (flash_params.verbose)
        {
            print("Stability test not converged, norm", norm);
            trial_comp.print_point();
        }
    }
    return 0;
}

void Stability::perform_ssi()
{
    // Perform successive substitution step
    // dlnY = -g
    // lnY + dlnY = lnY - g = lnY - lnY
    std::shared_ptr<EoSParams> params{ flash_params.eos_params[trial_composition.eos_name] };

    // Update lnY and Y
    std::vector<double> lnY_new(ns);
    double beta = 1.;
    int it = 0;
    while (true)
    {
        for (int i = 0; i < ns_nonzero; i++)
        {
            // lnY[i] -= g(i);
            int ii = nonzero_comp[i];
            lnY_new[ii] = lnY[ii] - beta*g(i);
            trial_composition.Y[ii] = std::exp(lnY_new[ii]);
            alpha[ii] = 2. * std::sqrt(trial_composition.Y[ii]);
        }

        if (params->eos->eos_in_range(trial_composition.Y.begin()))
        {
            trial_composition.is_in_range = true;
            lnY = lnY_new;
            break;
        }
        else if (it > params->stability_line_iter)
        {
            for (int i = 0; i < ns_nonzero; i++)
            {
                // lnY[i] -= g(i);
                int ii = nonzero_comp[i];
                trial_composition.Y[ii] = std::exp(lnY[ii]);
                alpha[ii] = 2. * std::sqrt(trial_composition.Y[ii]);
            }
            break;
        }

        // Reduce step size such that Y remains inside EoS range
        trial_composition.is_in_range = false;
        beta *= 0.5;
        it++;
    }

    // Update fugacities, gradient vector and TPD
    bool second_order = false;
    this->update_fugacities(trial_composition, second_order);
    this->update_g();
    tpd = this->calc_modtpd();

    ssi_iter++;
    return;
}

void Stability::perform_Y()
{
    // Perform Newton step with Yi as primary variables
    // Newton iteration: dY = -H^-1 g
    //                      = -L^-T L^-1 g
    // Gradient vector g_i = dTPD/dY_i = lnY_i + lnphi_i(Y) - d_i
    // Hessian matrix H_ij = dg_i/dY_j = dij/Y_i + dlnphi_i(Y)/dY_j
    //                     = U + PHI
    std::shared_ptr<EoSParams> params{ flash_params.eos_params[trial_composition.eos_name] };

    // Construct Hessian
    Eigen::MatrixXd U = this->construct_U();
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd H = this->construct_H(U, PHI);

    // Perform LL^T Cholesky decomposition of H
    Eigen::LLT<Eigen::MatrixXd> lltOfH(H);
    Eigen::VectorXd Y(ns_nonzero), dY(ns_nonzero);
    for (int i = 0; i < ns_nonzero; i++)
    {
        Y(i) = trial_composition.Y[nonzero_comp[i]];
    }

    // When H is not positive definite, there will be a problem in performing Cholesky decomposition
    bool sec_ord = true;
    if (lltOfH.info() == Eigen::NumericalIssue)
    {
        // Modified Cholesky applied
        int error_modchol = 0;
        if (flash_params.modChol_stability)
        {
            ModifiedCholeskyS99 mod_chol;
            error_modchol += mod_chol.initialize(H, 1);
            error_modchol += mod_chol.solve(g, dY);
        }
        // Switch back to SSI applied
        if (!flash_params.modChol_stability || error_modchol)
        {
            this->perform_ssi();
            switch_back_iter++;
            sec_ord = false;
        }
    }
    else
    {
        // Calculate Newton step: dY = -H^-1 g = -L^-T L^-1 g
        dY = lltOfH.solve(g);
    }
    
    if (sec_ord)
    {
        // Integrate line search:
        // When the solution pass is not minimizing the objective function TPD, perform line-search Halfing
        // Store previous value of Y and TPD
        // std::vector<double> Y_old = trial_composition.Y;
        Eigen::VectorXd Y_old = Y;
        double tpd_old = tpd;

        LineSearch linesearch{ns_nonzero};
        double lamb = 1.;
        int line_iter = 0;
        bool check = linesearch.init(lamb, Y_old, tpd_old, g, dY, 1.0);

        bool accepted = false;
        while (line_iter < params->stability_line_iter && check)
        {
            // Update vector of Y's
            bool negative_X = false;
            Y = Y_old - lamb * dY;
            for (int i = 0; i < ns_nonzero; i++)
            {
                if (Y(i) < 0)
                {
                    negative_X = true;
                    break;
                }
                int ii = nonzero_comp[i];
                trial_composition.Y[ii] = Y(i);
                lnY[ii] = std::log(Y(i));
            }
            
            // Cut step to remain within each EoS range
            bool in_range = true;
            if (!params->eos->eos_in_range(trial_composition.Y.begin()))
            {
                in_range = false;
            }

            if(!negative_X && in_range)
            {
                // Update fugacities and TPD
                bool second_order = false;
                this->update_fugacities(trial_composition, second_order);
                tpd = this->calc_modtpd();

                if (tpd - tpd_old < params->stability_line_tol)
                {
                    accepted = true;
                    break;
                }
                else
                {
                    check = linesearch.process(Y, tpd);
                    lamb = linesearch.get_alam();
                    line_iter++;
                }
            }
            else
            {
                lamb *= 0.5;
                line_iter++;
            }
        }
        if (!accepted)
        {
            // for (int i = 0; i < ns_nonzero; i++)
            // {
            //     int ii = nonzero_comp[i];
            //     trial_composition.Y[ii] = Y_old(i);
            //     lnY[ii] = std::log(Y_old(i));
            // }
            // error++;
            // return;
        }
        newton_iter++;
    }

    // Update 2nd order fugacities and gradient vector
    bool second_order = true;
    this->update_fugacities(trial_composition, second_order);
    this->update_g();

    return;
}

void Stability::perform_lnY()
{
    // Perform Newton step with ln Yi as primary variables

    // Newton iteration: dlnY = -J^-1 g, where J = H U^-1 and dlnY = U dY
    //                        = -U L^-T L^-1 g

    // Gradient vector g_i = dTPD/dY_i = lnY_i + lnphi_i(Y) - d_i
    // Hessian matrix H_ij = dg_i/dY_j = dij/Y_i + dlnphi_i(Y)/dY_j
    //                     = U + PHI
    std::shared_ptr<EoSParams> params{ flash_params.eos_params[trial_composition.eos_name] };

    // Construct Jacobian
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd Uinv = this->construct_Uinv();
    Eigen::MatrixXd J = this->construct_J(PHI, Uinv);

    // Calculate Newton step: dlnY = -J^-1 g = -U H^-1 g = -U L^-T L^-1 g
    // J = HU^-1
    Eigen::VectorXd dlnY = J.partialPivLu().solve(g);

    // Integrate line search:
    // When the solution pass is not minimizing the objective function TPD, perform line-search Halfing
    // Store previous value of lnY and TPD
    std::vector<double> lnY_old = lnY;
    double tpd_old = tpd;

    double lamb = 1.;
    int line_iter = 0;
    bool accepted = false;
    while (line_iter < params->stability_line_iter)
    {
        while (true)
        {
            // Update vector of lnY's
            for (int i = 0; i < ns_nonzero; i++)
            {
                int ii = nonzero_comp[i];
                lnY[ii] = lnY_old[ii] - lamb*dlnY(i);
                trial_composition.Y[ii] = std::exp(lnY[ii]);
            }
            if (params->eos->eos_in_range(trial_composition.Y.begin()))
            {
                break;
            }
            lamb *= 0.95;
            line_iter++;
        }
            
        // Update fugacities and TPD
        bool second_order = false;
        this->update_fugacities(trial_composition, second_order);
        tpd = this->calc_modtpd();

        if (tpd - tpd_old < params->stability_line_tol)
        {
            accepted = true;
            break;
        }
        lamb *= 0.95;
        line_iter++;
    }

    if (!accepted)
    {
        // for (int i = 0; i < ns_nonzero; i++)
        // {
        //     int ii = nonzero_comp[i];
        //     lnY[ii] = lnY_old[i];
        //     trial_composition.Y[ii] = std::exp(lnY[ii]);
        // }
        // error++;
        // return;
    }

    // Update 2nd order fugacities and gradient vector
    bool second_order = true;
    this->update_fugacities(trial_composition, second_order);
    this->update_g();

    newton_iter++;

    return;
}

void Stability::perform_alpha()
{
    // Perform Newton step with alpha = 2 sqrt(Yi) as primary variables

    // Newton iteration: da = -J^-1 g, where J = H U^-1 and dlnY = U dY
    //                      = -U L^-T L^-1 g

    // Gradient vector g_a_i = dTPD/da_i = g_i sqrt(Yi) = sqrt(Yi) * (lnY_i + lnphi_i(Y) - d_i)
    // Hessian matrix H_a_ij = dg_i/da_j = dij + 1/4 a_i a_j dlnphi_i(Y)/dY_j + 1/2 dij g_a_i
    std::shared_ptr<EoSParams> params{ flash_params.eos_params[trial_composition.eos_name] };

    // Construct gradient vector and Hessian
    Eigen::VectorXd g_a = this->construct_ga(alpha);
    Eigen::MatrixXd H_a = this->construct_H(alpha);

    // Perform LL^T Cholesky decomposition of H
    Eigen::LLT<Eigen::MatrixXd> lltOfH(H_a);
    Eigen::VectorXd alpha_vec(ns_nonzero), da(ns_nonzero);
    for (int i = 0; i < ns_nonzero; i++)
    {
        alpha_vec(i) = alpha[nonzero_comp[i]];
    }

    // When H is not positive definite, there will be a problem in performing Cholesky decomposition
    bool sec_ord = true;
    if (lltOfH.info() == Eigen::NumericalIssue)
    {
        // Modified Cholesky applied
        int error_modchol = 0;
        if (flash_params.modChol_stability)
        {
            ModifiedCholeskyS99 mod_chol;
            error_modchol += mod_chol.initialize(H_a, 1);
            error_modchol += mod_chol.solve(g_a, da);
        }
        // Switch back to SSI applied
        if (!flash_params.modChol_stability || error_modchol)
        {
            this->perform_ssi();
            switch_back_iter++;
            sec_ord = false;
        }
    }
    else
    {
        // Calculate Newton step: da = -H^-1 g = -L^-T L^-1 g
        da = lltOfH.solve(g_a);
    }

    if (sec_ord)
    {
        // Integrate line search:
        // When the solution pass is not minimizing the objective function TPD, perform line-search Halfing
        // Store previous value of alpha and TPD
        // std::vector<double> alpha_old = alpha;
        Eigen::VectorXd alpha_old = alpha_vec;
        double tpd_old = tpd;

        LineSearch linesearch{ns_nonzero};
        double lamb = 1.;
        int line_iter = 0;
        bool check = linesearch.init(lamb, alpha_old, tpd_old, g_a, da, 1.0);

        bool accepted = false;
        while (line_iter < params->stability_line_iter && check)
        {
            // Update vector of Y's
            bool negative_a = false;
            alpha_vec = alpha_old - lamb*da;
            for (int i = 0; i < ns_nonzero; i++)
            {
                // alpha[ii] = alpha_old[ii] - lamb*da(i);
                if (alpha_vec(i) < 0.)
                {
                    negative_a = true;
                    break;
                }
                int ii = nonzero_comp[i];
                alpha[ii] = alpha_vec(i);
                trial_composition.Y[ii] = 0.25 * std::pow(alpha[ii], 2);
                lnY[ii] = std::log(trial_composition.Y[ii]);
            }

            // Cut step to remain within each EoS range
            bool in_range = true;
            if (!params->eos->eos_in_range(trial_composition.Y.begin()))
            {
                in_range = false;
            }

            if (!negative_a && in_range)
            {
                // Update fugacities and TPD
                bool second_order = false;
                this->update_fugacities(trial_composition, second_order);
                tpd = this->calc_modtpd();

                if (tpd - tpd_old < params->stability_line_tol)
                {
                    accepted = true;
                    break;
                }
                else
                {
                    check = linesearch.process(alpha_vec, tpd);
                    lamb = linesearch.get_alam();
                    line_iter++;
                }
            }
            else
            {
                lamb *= 0.5;
                line_iter++;
            }
        }
        if (!accepted)
        {
            for (int i = 0; i < ns_nonzero; i++)
            {
                int ii = nonzero_comp[i];
                alpha[ii] = alpha_old(i);
                trial_composition.Y[ii] = 0.25 * std::pow(alpha[ii], 2);
                lnY[ii] = std::log(trial_composition.Y[ii]);
            }
            error++;
            return;
        }
        newton_iter++;
    }

    // Update 2nd order fugacities and gradient vector
    bool second_order = true;
    this->update_fugacities(trial_composition, second_order);
    this->update_g();

    return;
}

void Stability::update_g()
{
    // Update gradient vector with (ln)Yi as primary variables
    g = Eigen::VectorXd(ns_nonzero);
    for (int i = 0; i < ns_nonzero; i++)
    {
        int ii = nonzero_comp[i];
        g(i) = lnY[ii] + lnphi[ii] - h[ii];
    }
    return;
}

Eigen::VectorXd Stability::construct_ga(std::vector<double>& alpha_)
{
    // Construct gradient vector with alpha as primary variables from gi: g_ai = gi * sqrt(Yi)
    Eigen::VectorXd g_a(ns_nonzero);
    for (int i = 0; i < ns_nonzero; i++)
    {
        int ii = nonzero_comp[i];
        g_a(i) = 0.5*alpha_[ii] * g(i);
    }
    return g_a;
}

Eigen::MatrixXd Stability::construct_U()
{
    // Construct U
    Eigen::MatrixXd U = Eigen::MatrixXd::Identity(ns_nonzero, ns_nonzero);
    for (int i = 0; i < ns_nonzero; i++)
    {
        int ii = nonzero_comp[i];
        U(i, i) /= trial_composition.Y[ii];
    }
    return U;
}

Eigen::MatrixXd Stability::construct_Uinv()
{
    // Construct U
    Eigen::MatrixXd Uinv = Eigen::MatrixXd::Identity(ns_nonzero, ns_nonzero);
    for (int i = 0; i < ns_nonzero; i++)
    {
        int ii = nonzero_comp[i];
        Uinv(i, i) *= trial_composition.Y[ii];
    }
    return Uinv;
}

Eigen::MatrixXd Stability::construct_PHI()
{
    // Construct PHI
    Eigen::MatrixXd PHI = Eigen::MatrixXd(ns_nonzero, ns_nonzero);
    for (int j = 0; j < ns_nonzero; j++)
    {
        int jj = nonzero_comp[j];
        for (int i = j; i < ns_nonzero; i++)
        {
            int ii = nonzero_comp[i];
            PHI(i, j) = dlnphidn[ii*ns + jj];
        }
    }
    return PHI;
}

Eigen::MatrixXd Stability::construct_H(Eigen::MatrixXd& U, Eigen::MatrixXd& PHI)
{
    Eigen::MatrixXd H = PHI.selfadjointView<Eigen::Lower>();
    for (int i = 0; i < ns_nonzero; i++)
    {
        H(i, i) += U(i, i);
    }
    return H;
}

Eigen::MatrixXd Stability::construct_H(std::vector<double>& alpha_)
{
    // Construct H with alpha as primary variables
    Eigen::MatrixXd H_a = Eigen::MatrixXd(ns_nonzero, ns_nonzero);
    for (int j = 0; j < ns_nonzero; j++)
    {
        int jj = nonzero_comp[j];
        for (int i = j; i < ns_nonzero; i++)
        {
            int ii = nonzero_comp[i];
            H_a(i, j) = 0.25 * alpha_[ii] * alpha_[jj] * dlnphidn[ii*ns + jj];
        }
        H_a(j, j) += 1.;
    }
    H_a = H_a.selfadjointView<Eigen::Lower>();
    return H_a;
}

Eigen::MatrixXd Stability::construct_J(Eigen::MatrixXd& PHI, Eigen::MatrixXd& Uinv)
{
    // Eigen::MatrixXd J = PHI.selfadjointView<Eigen::Lower>() * Uinv.selfadjointView<Eigen::Lower>();
    PHI = PHI.selfadjointView<Eigen::Lower>();
    Uinv = Uinv.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd J = PHI * Uinv;
    for (int i = 0; i < ns_nonzero; i++)
    {
        J(i, i) += 1.;
    }
    return J;
}

double Stability::calc_modtpd()
{
    // Calculate modified tangent plane distance (Michelsen, 1982b) of trial composition Y relative to x
    double modtpd = 1.;

    for (int i = 0; i < ns; i++)
    {
        modtpd += trial_composition.Y[i] * (lnY[i] + lnphi[i] - h[i] - 1.);
    }
    return modtpd;
}

double Stability::calc_modtpd(TrialPhase& trial_comp)
{
    // Calculate modified tangent plane distance (Michelsen, 1982b) of trial composition Y relative to x    
    double modtpd = 1.;

    this->update_fugacities(trial_comp, false);
    for (int i = 0; i < ns; i++)
    {
        modtpd += trial_comp.Y[i] * (std::log(trial_comp.Y[i]) + lnphi[i] - h[i] - 1.);
    }
    return modtpd;
}

void Stability::update_fugacities(TrialPhase& comp, bool second_order)
{
    // Update fugacity coefficients and derivatives
    flash_params.timer.start(Timer::timer::EOS);

    std::shared_ptr<EoSParams> params{ flash_params.eos_params[comp.eos_name] };
    params->eos->set_root_flag(EoS::RootFlag::STABLE);
    params->eos->solve_PT(comp.Y.begin(), second_order);
    comp.is_stable_root = params->eos->is_stable();
    comp.is_preferred_root = params->eos->select_root(comp.Y.begin());

    // Find root type
    bool is_below_spinodal = false;
    EoS::RootFlag root_type = params->eos->is_root_type(is_below_spinodal);
    comp.root = (root_type > EoS::RootFlag::STABLE) ? root_type : EoS::RootFlag::STABLE;

    // First-order
    for (int i = 0; i < nc; i++)
    {
        lnphi[i] = params->eos->lnphii(i);
    }

    // Second-order
    if (second_order)
    {
        dlnphidn = params->eos->dlnphi_dn();
    }
    flash_params.timer.stop(Timer::timer::EOS);
    return;
}

double Stability::calc_condition_number()
{
    Eigen::VectorXd eigen(nc);
    if (flash_params.stability_variables == FlashParams::Y)
    {
        // Construct Hessian
        Eigen::MatrixXd U = this->construct_U();
        Eigen::MatrixXd PHI = this->construct_PHI();
        Eigen::MatrixXd H = this->construct_H(U, PHI);

        // Get the condition number for stability if Newton was used
        eigen = H.eigenvalues().real();
    }
    else if (flash_params.stability_variables == FlashParams::alpha)
    {
        // Construct gradient vector and Hessian
        Eigen::VectorXd g_a = this->construct_ga(alpha);
        Eigen::MatrixXd H_a = this->construct_H(alpha);

        // Get the condition number for stability if Newton was used
        eigen = H_a.eigenvalues().real();
    }
    else  // if(flash_params.stability_variables == FlashParams::lnY)
    {
        // Construct gradient vector and Hessian
        Eigen::MatrixXd U = this->construct_U();
        Eigen::MatrixXd PHI = this->construct_PHI();
        Eigen::MatrixXd H = this->construct_H(U, PHI);
        Eigen::MatrixXd J = H*this->construct_Uinv();

        // Get the condition number for stability if Newton was used
        eigen = J.eigenvalues().real();
    }

    double min_eigen = *std::min_element(eigen.begin(), eigen.end());
    double max_eigen = *std::max_element(eigen.begin(), eigen.end());
    return std::sqrt(max_eigen/min_eigen);
}

int Stability::test_matrices()
{
    // Compare U.inverse() and analytically obtained matrix Uinv
    Eigen::MatrixXd U = this->construct_U();
    Eigen::MatrixXd Uinv = U.inverse();

    Eigen::MatrixXd Uinv_analytical = this->construct_Uinv();
    Uinv_analytical = Uinv_analytical.selfadjointView<Eigen::Lower>();

    // If norm of difference matrix > tol, return 1
    double norm = (Uinv - Uinv_analytical).norm();
    if (norm > 1e-7)
    {
        std::cout << "Stability matrix inverse U not correct\n";
        print("Uinv", Uinv);
        print("Uinv analytical", Uinv_analytical);
        print("norm", norm);
        return 1;
    }

    // Compare J = I + PHI*U^-1 = H*U^-1
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd H = this->construct_H(U, PHI);
    Eigen::MatrixXd J = this->construct_J(PHI, Uinv_analytical);
    Eigen::MatrixXd Jh = H*Uinv;
    
    // If norm of difference matrix > tol, return 1
    norm = (J - Jh).norm();
    if (norm > 1e-12)
    {
        std::cout << "Stability matrix inverse J = I + PHI*Uinv != H*Uinv\n";
        print("J = I + PHI*Uinv", J);
        print("J = H*Uinv", Jh);
        print("norm", norm);
        return 1;
    }
    return 0;
}
