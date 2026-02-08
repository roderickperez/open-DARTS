#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iterator>

#include "dartsflash/global/timer.hpp"
#include "dartsflash/maths/modifiedcholeskys99.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/phase-split/basesplit.hpp"

#include <Eigen/Dense>

BaseSplit::BaseSplit(FlashParams& flashparams, int np_)
{
    // Initialize multiphase split calculations
    this->flash_params = flashparams;
    this->nc = flash_params.nc;
    this->ns = flash_params.ns;
    this->np = np_;

    // Allocate memory for lnphi and derivatives, vectors g and dx, matrices H, U, PHI
    this->z.resize(ns);
    this->nonzero_comp.resize(ns);
    this->lnK.reserve((np-1)*ns);
    this->trial_compositions.resize(np);
    this->eos_names.resize(np);
    this->roots.resize(np);
    this->nu.resize(np);
    this->X.resize(np*ns);
    this->n_ik.resize(np*ns);
    this->lnphi.resize(np*ns);
    this->dlnphidn.resize(np*ns*ns);
    this->g = Eigen::VectorXd::Zero((np-1)*ns);

    // Phase ordering
    this->reference_phase = std::vector<int>(ns, 0);
    this->k_idxs.resize(ns * np);
    for (int i = 0; i < ns; i++)
    {
        auto begin = k_idxs.begin() + i * (np-1);
        std::iota(begin, begin + np, 0);
        k_idxs.erase(begin + reference_phase[i]);
    }
}

int BaseSplit::init(std::vector<double>& z_, std::vector<double>& lnk, std::vector<TrialPhase>& trial_comps, bool set_root_flags)
{
    // Initialise multiphase split
    this->error = 0;
    this->trial_compositions = trial_comps;
    for (int j = 0; j < np; j++)
    {
        this->eos_names[j] = trial_compositions[j].eos_name;
        this->roots[j] = (set_root_flags) ? trial_compositions[j].root : EoS::RootFlag::STABLE;
    }

    // Check if feed composition needs to be corrected for 0 values
    this->z = z_;
    for (int i = 0; i < ns; i++)
    {
        nonzero_comp[i] = (z[i] > flash_params.min_z) ? 1 : 0;
    }

    if (flash_params.verbose)
    {
        std::cout << "Running phase split with z, lnK and EoS:\n";
        print("z", z_);
        print("lnK", lnk);
        print("eos", eos_names);
    }

    // Initialise multiphase split with initial guess of lnK-values
    // Solve RR with lnK - this will give corresponding phase compositions of initial guess    
    this->lnK = lnk;
    this->solve_rr();

    // Check if norm is small enough
    bool second_order = false;
    this->update_fugacities(second_order);
    this->update_g();
    this->gibbs = this->calc_gibbs();

    this->norm = this->l2norm();
    if (norm < flash_params.split_switch_tol)
    {
        return 0;
    }

    // If for EoS gmix is preferred over stationary point, set initial composition equal to local minimum
    bool recompute_lnk = false;
    for (size_t j = 0; j < trial_comps.size(); j++)
    {
        if (flash_params.eos_params[trial_comps[j].eos_name]->use_gmix)
        {
            std::copy(trial_comps[j].ymin.begin(), trial_comps[j].ymin.end(), this->X.begin() + j * ns);
            recompute_lnk = true;
        }
        else
        {
            double sumX = std::accumulate(X.begin() + j*ns, X.begin() + (j+1)*ns, 0.);
            if (std::fabs(sumX - 1.) > 1e-3)
            {
                for (int i = 0; i < ns; i++)
                {
                    X[j*ns + i] /= sumX;
                }
                recompute_lnk = true;
            }
        }
    }
    // And recompute lnK
    if (recompute_lnk)
    {
        for (int j = 1; j < np; j++)
        {
            for (int i = 0; i < ns; i++)
            {
                lnK[(j-1)*ns + i] = std::log(X[j*ns + i]) - std::log(X[i]);
            }
        }
        this->solve_rr();
    }

    // Update fugacities, gradient vector and Gibbs free energy
    this->update_fugacities(second_order);
    this->update_g();
    this->gibbs = this->calc_gibbs();
    ssi_iter++;

    // If initial K-values not valid, RR has not converged to feasible solution (sum X = 1); return 1
    double sumX = std::accumulate(X.begin(), X.begin() + ns, 0.);
    if (std::fabs(sumX - 1.) > 1e-3)
    {
        if (flash_params.verbose)
        {
            print("K-values not valid, RR has not converged to feasible solution (sum X = 1)", X, np);
        }
        return 1;
    }
    return 0;
}

int BaseSplit::run(std::vector<double>& z_, std::vector<double>& lnk, std::vector<TrialPhase>& trial_comps, bool set_root_flags)
{
    // Run phase split algorithm at p, T, z, with initial estimates lnK
    flash_params.timer.start(Timer::timer::SPLIT);
    ssi_iter = newton_iter = switch_back_iter = loose_iter = 0;
    (void) this->init(z_, lnk, trial_comps, set_root_flags);

    // Perform successive substitution steps
    norm = this->l2norm();
    double old_norm = norm;
    while (ssi_iter < flash_params.split_max_iter)
    {
        // Check if within tolerance, or else if SSI step is still effective
        if (norm < flash_params.split_tol || 
           (norm < flash_params.split_switch_tol && std::log(old_norm) - std::log(norm) < flash_params.split_switch_diff))
        {
            break;
        }

        // Perform SSI step
        this->perform_ssi();
        old_norm = norm;
        norm = this->l2norm();
        if (error != 0 || std::isinf(norm) || std::isnan(norm))
        {
            // If some error occur during SSI procedure, return 1
            if (flash_params.verbose)
            {
                print("Error in split (SSI), iterations, norm", {static_cast<double>(ssi_iter), norm});
            }
            return 1;
        }
        else if (ssi_iter == flash_params.split_negative_flash_iter)
        {
            for (double nuj: this->nu)
            {
                if (nuj < 0.)
                {
                    int error_output = this->output(trial_comps);
                    (void) error_output;
                    return -1;
                }
            }
        }
    }

    if (norm >= flash_params.split_tol && ssi_iter < flash_params.split_max_iter)
    {
        // Perform Newton steps
        bool second_order = true;
        this->update_fugacities(second_order);

        switch (flash_params.split_variables)
        {
            case FlashParams::nik:
            {
                while (norm > flash_params.split_tol &&
                       (newton_iter + switch_back_iter) < flash_params.split_max_iter && 
                       loose_iter < flash_params.split_loose_iter)
                {
                    this->perform_nik();
                    norm = this->l2norm();
                    if (norm < flash_params.split_tol * flash_params.split_loose_tol_multiplier)
                    {
                        loose_iter++;
                    }
                    if (error != 0 || std::isinf(norm) || std::isnan(norm))
                    {
                        // If some error occur during Newton procedure, return 1
                        if (flash_params.verbose)
                        {
                            print("Error in split (Newton nik), iterations (SSI/Newton), norm", 
                                {static_cast<double>(ssi_iter+switch_back_iter), static_cast<double>(newton_iter), norm});
                        }
                        return 1;
                    }
                    else if ((newton_iter + switch_back_iter) == flash_params.split_negative_flash_iter)
                    {
                        if (norm > flash_params.split_negative_flash_tol)
                        {
                            if (flash_params.verbose)
                            {
                                std::cout << "Split not converged, close to phase boundary\n";
                            }
                            return 1;
                        }
                        else
                        {
                            for (double nuj: this->nu)
                            {
                                if (nuj < 0.)
                                {
                                    int error_output = this->output(trial_comps);
                                    (void) error_output;
                                    return -1;
                                }
                            }
                        }
                    }
                }
                break;
            }
            case FlashParams::lnK:
            {
                while (norm > flash_params.split_tol &&
                      (newton_iter + switch_back_iter) < flash_params.split_max_iter && 
                      loose_iter < flash_params.split_loose_iter)
                {
                    this->perform_lnK();
                    norm = this->l2norm();
                    if (norm < flash_params.split_tol * flash_params.split_loose_tol_multiplier)
                    {
                        loose_iter++;
                    }
                    if (error != 0 || std::isinf(norm) || std::isnan(norm))
                    {
                        // If some error occur during Newton procedure, return 1
                        if (flash_params.verbose)
                        {
                            print("Error in split (Newton lnK), iterations (SSI/Newton), norm", 
                                {static_cast<double>(ssi_iter+switch_back_iter), static_cast<double>(newton_iter), norm});
                        }
                        return 1;
                    }
                    else if ((newton_iter + switch_back_iter) == flash_params.split_negative_flash_iter)
                    {
                        if (norm > flash_params.split_negative_flash_tol)
                        {
                            if (flash_params.verbose)
                            {
                                std::cout << "Split not converged, close to phase boundary\n";
                            }
                            return 1;
                        }
                        else
                        {
                            for (double nuj: this->nu)
                            {
                                if (nuj < 0.)
                                {
                                    int error_output = this->output(trial_comps);
                                    (void) error_output;
                                    return -1;
                                }
                            }
                        }
                    }
                }
                break;
            }
            case FlashParams::lnK_chol:
            {
                while (norm > flash_params.split_tol &&
                      (newton_iter + switch_back_iter) < flash_params.split_max_iter &&
                      loose_iter < flash_params.split_loose_iter)
                {
                    this->perform_lnK_chol();
                    norm = this->l2norm();
                    if (norm < flash_params.split_tol * flash_params.split_loose_tol_multiplier)
                    {
                        loose_iter++;
                    }
                    if (error != 0 || std::isinf(norm) || std::isnan(norm))
                    {
                        // If some error occur during Newton procedure, return 1
                        if (flash_params.verbose)
                        {
                            print("Error in split (Newton lnK-Cholesky), iterations (SSI/Newton), norm", 
                                {static_cast<double>(ssi_iter+switch_back_iter), static_cast<double>(newton_iter), norm});
                        }
                        return 1;
                    }
                    else if ((newton_iter + switch_back_iter) == flash_params.split_negative_flash_iter)
                    {
                        if (norm > flash_params.split_negative_flash_tol)
                        {
                            if (flash_params.verbose)
                            {
                                std::cout << "Split not converged, close to phase boundary\n";
                            }
                            return 1;
                        }
                        else
                        {
                            for (double nuj: this->nu)
                            {
                                if (nuj < 0.)
                                {
                                    int error_output = this->output(trial_comps);
                                    (void) error_output;
                                    return -1;
                                }
                            }
                        }
                    }
                }
                break;
            }
            default:
            {
                std::cout << "Invalid split variables defined\n";
                exit(1);
            }
        }
    }

    int error_output = this->output(trial_comps);

    flash_params.timer.stop(Timer::timer::SPLIT);

    if (norm < flash_params.split_tol * flash_params.split_loose_tol_multiplier)
    {
        for (double nuj: this->nu)
        {
            if (nuj < 0.)
            {
                return -1;
            }
        }
        return error_output;
    }
    else
    {
        if (flash_params.verbose)
        {
            print("Phase split not converged, iterations (SSI/Newton), norm", 
                                {static_cast<double>(ssi_iter+switch_back_iter), static_cast<double>(newton_iter), norm});
        }
        return 1;
    }
}

int BaseSplit::output(std::vector<TrialPhase>& comps)
{
    int error_output = 0;
    if (flash_params.verbose)
    {
        print("Phase split", "===============");
        print("ssi iterations", ssi_iter);
        print("of which switch back from Newton", switch_back_iter);
        print("Newton iterations", newton_iter);
        print("norm", this->l2norm());
        print("nu", nu);
        print("X", X, np);
    }
    for (size_t j = 0; j < comps.size(); j++)
    {
        double sumX = std::accumulate(X.begin() + j*ns, X.begin() + (j+1)*ns, 0.);
        if (std::fabs(sumX-1.) > 1e-3) { error_output++; }

        flash_params.eos_params[comps[j].eos_name]->eos->solve_PT(X.begin() + j*ns, false);
        comps[j].is_preferred_root = flash_params.eos_params[comps[j].eos_name]->eos->select_root(X.begin() + j*ns);
        error_output += (comps[j].is_preferred_root < EoS::RootSelect::ACCEPT) ? 1 : 0;
        
        comps[j].set_equilibrium_phase(X.begin() + j*ns, nu[j]);
        comps[j].set_stationary_point(X.begin() + j*ns, 0.);
    }

    // Check if duplicate compositions have been found
    for (size_t j = 0; j < comps.size(); j++)
    {
        for (size_t jj = j + 1; jj < comps.size(); jj++)
        {
            // Check if duplicate compositions have been found
            error_output += (compare_compositions(comps[j].X, comps[jj].X, flash_params.comp_tol)) ? 1 : 0;
        }
    }
    // Only if no errors have occurred
    // if (error_output == 0)
    // {
    //     for (size_t j = 0; j < comps.size(); j++)
    //     {
    //         comps[j].set_equilibrium_phase(X.begin() + j*ns, nu[j]);
    //         comps[j].set_stationary_point(X.begin() + j*ns, 0.);
    //     }
    // }
    return error_output;
}

std::vector<double> BaseSplit::get_lnk()
{
    for (int j = 1; j < np; j++)
    {
        for (int i = 0; i < nc; i++)
        {
            lnK[(j-1)*ns + i] = lnphi[i] - lnphi[j * ns + i];
        }
    }
    return lnK;
}

void BaseSplit::solve_rr()
{
    // Solve RR with lnK
    std::vector<double> K((np-1)*ns);
    std::transform(lnK.begin(), lnK.end(), K.begin(), [](double lnKi) { return std::exp(lnKi); });

    error += rr->solve_rr(z, K, nonzero_comp);
    nu = rr->getnu();
    X = rr->getx();

    // Update vector of mole numbers nik
    for (int k = 0; k < np; k++)
    {
        double nuk = (std::fabs(nu[k]) > flash_params.min_z) ? nu[k] : flash_params.min_z;
        for (int i = 0; i < ns; i++)
        {
            n_ik[k*ns + i] = nuk * X[k*ns + i];
        }
    }

    return;
}

void BaseSplit::perform_ssi() {
    // Perform successive substitution step for multiphase split
    std::vector<double> dlnK((np-1)*ns, 0.);
    std::vector<double> lnK_old = lnK;
    double lamb = 1.;

    // Calculate new lnK
    for (int k = 1; k < np; k++)
    {
        for (int i: flash_params.eos_params[eos_names[k]]->comp_idxs)
        {
            if (nonzero_comp[i] && i < nc)
            {
                dlnK[(k-1)*ns + i] = lnphi[i] - lnphi[k*ns + i] - lnK[(k-1)*ns + i];
            }
        }
    }
    
    // Cut lnK step to remain within each EoS range
    bool in_range = false;
    while (!in_range)
    {
        for (int i = 0; i < (np-1)*ns; i++)
        {
            lnK[i] = lnK_old[i] + lamb*dlnK[i];
        }

        // Solve RR with lnK - Equivalent to dlnK = -g
        this->solve_rr();

        // Check if each EoS is within range
        for (int k = 0; k < np; k++)
        {
            if (!this->flash_params.eos_params[eos_names[k]]->eos->eos_in_range(X.begin() + k*ns))
            {
                lamb *= 0.5;
                in_range = false;
                break;
            }
            else
            {   
                in_range = true;
            }
        }
    }

    // Update fugacities, gradient vector and Gibbs free energy
    bool second_order = false;
    this->update_fugacities(second_order);
    this->update_g();
    this->gibbs = this->calc_gibbs();

    ssi_iter++;
    return;
}

void BaseSplit::perform_nik() 
{
    // Perform Newton step with nik as variables (Petitfrere and Nichita, 2015)

    // Newton iteration: dnik = -H^-1 g
    //                        = -L^-T L^-1 g
    // (H_ij)_kp = d2G/dn_ik dn_jp
    //           = dkp (1/nu_p (dij/x_ip - 1) + dlnphi_ik/dn_jp)
    //             + (1/nu_R (dij/x_iR - 1) + dlnphi_iR/dn_jR)
    //           = U + PHI
    // (U_ij)_kp = dlnK_ik/dn_jp
    //           = dij (dkp/(nu_k x_ik) + 1/(nu_R x_iR)) - (dkp/nu_k + 1/nu_R)
    // (PHI_ij)_kp = d2G/dn_ik dn_jp
    //             = dkp dlnphi_ik/dn_jp + dlnphi_iR/dn_jR    

    // Construct Hessian
    // For the Hessian, it is important to select the right reference phase for each component (largest phase composition)
    // this->find_reference_phases();
    Eigen::MatrixXd U = this->construct_U();
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd H = this->construct_H(U, PHI);

    // Perform LL^T Cholesky decomposition of H
    Eigen::LLT<Eigen::MatrixXd> lltOfH(H);
    Eigen::VectorXd nik((np-1)*ns), dnik((np-1)*ns);
    for (int i = 0; i < ns; i++)
    {
        for (int k = 0; k < np-1; k++)
        {
            int kk = k_idxs[i * (np-1) + k];
            nik(k*ns + i) = n_ik[kk*ns + i];
        }
    }

    // When H is not positive definite, there will be a problem in performing Cholesky decomposition
    bool sec_ord = true;
    if (lltOfH.info() == Eigen::NumericalIssue )
    {
        // Modified Cholesky applied
        int error_modchol = 0;
        if (flash_params.modChol_split && switch_back_iter > 1)
        {
            ModifiedCholeskyS99 mod_chol;
            error_modchol += mod_chol.initialize(H, 2);
            error_modchol += mod_chol.solve(g, dnik);
        }
        if (!flash_params.modChol_split || error_modchol || switch_back_iter <= 1)
        {
            this->perform_ssi();
            switch_back_iter++;
            sec_ord = false;
        }
    }
    else
    {
        // Calculate Newton step: dnik = -H^-1 g = -L^-T L^-1 g
        dnik = lltOfH.solve(g);
    }

    if (sec_ord)
    {
        // Integrate line search:
        // When the solution pass is not minimizing the objective function Gibbs, perform line-search Halfing
        // Store previous value of nik and Gibbs energy
        // std::vector<double> n_ik_old = n_ik;
        Eigen::VectorXd nik_old = nik;
        double gibbs_old = gibbs;

        LineSearch linesearch{(np-1)*ns};
        double lamb = 1.;
        int line_iter = 0;
        bool check = linesearch.init(lamb, nik_old, gibbs_old, g, dnik, 1.0);

        bool accepted = false;
        while (line_iter < flash_params.split_line_iter && check)
        {
            // Update vector of nij's
            nik = nik_old - lamb * dnik;
            for (int i = 0; i < ns; i++)
            {
                int R = reference_phase[i];
                n_ik[R*ns + i] = z[i];  // n_iR = z_i - Σ n_ik

                for (int k = 0; k < np-1; k++)
                {
                    int kk = k_idxs[i * (np-1) + k];
                    
                    // n_ik[kk*ns + i] = nik_old[kk*ns + i] - lamb*dnik(k*ns + i);
                    n_ik[kk*ns + i] = nik(k*ns + i);
                    n_ik[R*ns + i] -= n_ik[kk*ns + i];
                }
            }

            // Update nu and X
            bool in_range = true;
            bool negative_X = false;
            for (int k = 0; k < np; k++)
            {
                // Cut step to remain within each EoS range
                if (!this->flash_params.eos_params[eos_names[k]]->eos->eos_in_range(n_ik.begin() + k*ns))
                {
                    in_range = false;
                    break;
                }

                nu[k] = std::accumulate(n_ik.begin() + k*ns, n_ik.begin() + (k+1)*ns, 0.);

                for (int i = 0; i < ns; i++)
                {
                    X[k*ns + i] = n_ik[k*ns + i]/nu[k];
                    if (X[k*ns + i] < 0 )
                    {
                        negative_X = true;
                        break;
                    }
                }
            }
            if (!in_range || negative_X)
            {
                lamb *= 0.5;
                line_iter++;
            }
            else
            {
                // Update fugacities and Gibbs energy
                bool second_order = false;
                this->update_fugacities(second_order);
                this->gibbs = this->calc_gibbs();

                if (gibbs - gibbs_old < flash_params.split_line_tol)
                {
                    accepted = true;
                    break;
                }
                else
                {
                    check = linesearch.process(nik, gibbs);
                    lamb = linesearch.get_alam();
                    line_iter++;
                }
            }
        }
        if (!accepted)
        {
            error++;
            return;
        }
        newton_iter++;
    }
    // Update 2nd order fugacities and gradient vector
    bool second_order = true;
    this->update_fugacities(second_order);
    this->update_g();

    return;
}

void BaseSplit::perform_lnK() 
{
    // Perform Newton step with lnK as variables (Petitfrere and Nichita, 2015)

    // Gradient vector g_ik = dG/dn_ik = lnK_ik + lnphi_ik - lnphi_iR, for k!=R
    // Hessian matrix (H_ij)_kp = d2G/dn_ik dn_jp
    //                          = dkp (1/nu_p (dij/x_ip - 1) + dlnphi_ik/dn_jp)
    //                            + (1/nu_R (dij/x_iR - 1) + dlnphi_iR/dn_jR)
    //                          = U + PHI

    // U and PHI
    // (U_ij)_kp = dlnK_ik/dn_jp
    //           = dij (dkp/(nu_k x_ik) + 1/(nu_R x_iR)) - (dkp/nu_k + 1/nu_R)
    // (PHI_ij)_kp = d2G/dn_ik dn_jp
    //             = dkp dlnphi_ik/dn_jp + dlnphi_iR/dn_jR

    // With lnK as independent variables, the Newton iterations are
    // J dlnK = -g
    // where J = HU^-1 = UU^-1 + PHI U^-1 = I + PHI U^-1

    // (U_ij)^-1 = dn_ik/dlnK_j

    // Construct U^-1, PHI and J
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd Uinv = this->construct_Uinv();
    Eigen::MatrixXd J = this->construct_J(PHI, Uinv);

    // Calculate Newton step, with : dlnK = U dn
    Eigen::VectorXd dlnK = J.partialPivLu().solve(g);

    // Integrate line search:
    // When the solution pass is not minimizing the objective function Gibbs, perform line-search Halfing
    // Store previous value of nik and Gibbs energy
    std::vector<double> lnK_old = lnK;
    double gibbs_old = gibbs;

    double lamb = 1.;
    int line_iter = 0;
    bool accepted = false;
    while (line_iter < flash_params.split_line_iter)
    {
        // Cut lnK step to remain within each EoS range
        bool in_range = false;
        while (!in_range)
        {
            for (int i = 0; i < (np-1)*ns; i++)
            {
                lnK[i] = lnK_old[i] - lamb*dlnK(i);
            }

            // Solve v, x with RR(z, K)
            this->solve_rr();

            // Check if each EoS is within range
            for (int k = 0; k < np; k++)
            {
                if (!this->flash_params.eos_params[eos_names[k]]->eos->eos_in_range(X.begin() + k*ns))
                {
                    lamb *= 0.5;
                    in_range = false;
                    break;
                }
                else
                {   
                    in_range = true;
                }
            }
        }

        // Update fugacities and gradient vector
        bool second_order = false;
        this->update_fugacities(second_order);
        this->gibbs = this->calc_gibbs();

        if (gibbs - gibbs_old < flash_params.split_line_tol)
        {
            accepted = true;
            break;
        }
        else
        {
            lamb *= 0.5;
            line_iter++;
        }
    }

    if (!accepted)
    {
        error++;
        return;
    }

    // Update 2nd order fugacities and gradient vector
    bool second_order = true;
    this->update_fugacities(second_order);
    this->update_g();
    newton_iter++;
    return;
}

void BaseSplit::perform_lnK_chol() 
{
    // Perform Newton step with nij as variables (Petitfrere and Nichita, 2015)

    // Newton iteration: dnij = -H^-1 g
    //                        = -L^-T L^-1 g
    // (H_ij)_kp = d2G/dn_ik dn_jp
    //           = dkp (1/nu_p (dij/x_ip - 1) + dlnphi_ik/dn_jp)
    //             + (1/nu_R (dij/x_iR - 1) + dlnphi_iR/dn_jR)
    //           = U + PHI
    // (U_ij)_kp = dlnK_ik/dn_jp
    //           = dij (dkp/(nu_k x_ik) + 1/(nu_R x_iR)) - (dkp/nu_k + 1/nu_R)
    // (PHI_ij)_kp = d2G/dn_ik dn_jp
    //             = dkp dlnphi_ik/dn_jp + dlnphi_iR/dn_jR

    // Construct Hessian
    // (H_ij)_kp = U + PHI
    // For the Hessian, it is important to select the right reference phase for each component (largest phase composition)
    this->find_reference_phases();
    Eigen::MatrixXd U = this->construct_U();
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd H = this->construct_H(U, PHI);

    // Perform LL^T Cholesky decomposition of H
    Eigen::LLT<Eigen::MatrixXd> lltOfH(H);
    Eigen::VectorXd dnik(H.rows());

    // When H is not positive definite, there will be a problem in performing Cholesky decomposition
    bool sec_ord = true;
    if (lltOfH.info() == Eigen::NumericalIssue)
    {
        // Modified Cholesky applied
        int error_modchol = 0;
        if (flash_params.modChol_split && switch_back_iter > 1)
        {
            ModifiedCholeskyS99 mod_chol;
            error_modchol += mod_chol.initialize(H, 1);
            error_modchol += mod_chol.solve(g, dnik);
        }
        // Switch back to SSI applied
        if (!flash_params.modChol_split || error_modchol || switch_back_iter <= 1)
        {
            this->perform_ssi();
            switch_back_iter++;
            sec_ord = false;
        }
    }
    else
    {
        // Calculate Newton step: dnij = -H^-1 g = -L^-T L^-1 g
        dnik = lltOfH.solve(g);
    }

    if (sec_ord)
    {
        // Calculate dlnK from U dnik
        std::vector<double> lnK_old = lnK;
        Eigen::VectorXd dlnK = U*dnik;

        // Cut lnK step to remain within each EoS range
        double lamb = 1.;
        bool in_range = false;
        while (!in_range)
        {
            for (int i = 0; i < (np-1)*ns; i++)
            {
                lnK[i] = lnK_old[i] - lamb*dlnK(i);
            }

            // Solve v, x with RR(z, K)
            this->solve_rr();

            // Check if each EoS is within range
            for (int k = 0; k < np; k++)
            {
                if (!this->flash_params.eos_params[eos_names[k]]->eos->eos_in_range(X.begin() + k*ns))
                {
                    lamb *= 0.5;
                    in_range = false;
                    break;
                }
                else
                {   
                    in_range = true;
                }
            }
        }

        // Update fugacities, gradient vector and Gibbs free energy
        bool second_order = false;
        this->update_fugacities(second_order);
        this->gibbs = this->calc_gibbs();

        newton_iter++;
    }

    // Update gradient vector
    bool second_order = true;
    this->update_fugacities(second_order);
    this->update_g();

    return;
}

void BaseSplit::update_g() {
    // Update lnf's (= gradient vector g)
    g = Eigen::VectorXd((np-1)*ns);
    for (int i = 0; i < nc; i++)
    {
        int iR = reference_phase[i]*ns + i;
        for (int k = 0; k < np-1; k++)
        {
            int kk = k_idxs[i*(np-1) + k];
            int ik = kk*ns + i;
            g(k*ns + i) = (nonzero_comp[i]) ? lnphi[ik] + std::log(X[ik]) - lnphi[iR] - std::log(X[iR]) : 0;
        }
    }
    return;
}

Eigen::MatrixXd BaseSplit::construct_H(Eigen::MatrixXd& U, Eigen::MatrixXd& PHI) 
{
    // (H_ij)_kp = d2G/dn_ik dn_jp
    //           = dkp (1/nu_p (dij/x_ip - 1) + dlnphi_ik/dn_jp)
    //             + (1/nu_R (dij/x_iR - 1) + dlnphi_iR/dn_jR)
    //           = U + PHI
    Eigen::MatrixXd H((np-1)*nc, (np-1)*nc);
    for (int j = 0; j < (np-1)*nc; j++)
    {
        for (int i = j; i < (np-1)*nc; i++)
        {
            H(i, j) = U(i, j) + PHI(i, j);
        }
    }
    H = H.selfadjointView<Eigen::Lower>();
    return H;
}

Eigen::MatrixXd BaseSplit::construct_J(Eigen::MatrixXd& PHI, Eigen::MatrixXd& Uinv) 
{
    // J = HU^-1 = UU^-1 + PHI U^-1 = I + PHI U^-1
    PHI = PHI.selfadjointView<Eigen::Lower>();
    Uinv = Uinv.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd J = PHI * Uinv;
    for (int i = 0; i < (np-1)*nc; i++)
    {
        J(i, i) += 1.;
    }
    return J;
}

void BaseSplit::update_fugacities(bool second_order)
{
    // Update fugacity coefficients in each phase
    for (int k = 0; k < np; k++) {
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[eos_names[k]] };
        // params->eos->set_root_flag(EoS::RootFlag::STABLE);
        params->eos->set_root_flag(static_cast<EoS::RootFlag>(this->roots[k]));

        // Solve EoS
        flash_params.timer.start(Timer::timer::EOS);
        params->eos->solve_PT(n_ik.begin() + k*ns, second_order);
        trial_compositions[k].is_stable_root = params->eos->is_stable();
        trial_compositions[k].is_preferred_root = params->eos->select_root(n_ik.begin());

        // Find root type
        bool is_below_spinodal = false;
        EoS::RootFlag root_type = params->eos->is_root_type(is_below_spinodal);
        trial_compositions[k].root = (root_type > EoS::RootFlag::STABLE) ? root_type : EoS::RootFlag::STABLE;

        // Fugacity coefficient
        for (int i = 0; i < nc; i++)
        {
            lnphi[k*ns + i] = params->eos->lnphii(i);
        }

		// If second-order, update derivatives
		if (second_order)
		{
            std::vector<double> dlnphik_dn = params->eos->dlnphi_dn();
            for (int i = 0; i < ns; i++)
            {
                for (int j = i; j < nc; j++)
                {
                    dlnphidn[k*ns*ns + j*ns + i] = dlnphik_dn[j*ns + i];
                }
            }
		}
        flash_params.timer.stop(Timer::timer::EOS);
    }

    return;
}

double BaseSplit::calc_gibbs()
{
    double G = 0.;
    for (int k = 0; k < np; k++)
    {
        for (int i = 0; i < ns; i++)
        {
            G += n_ik[k*ns + i] * (std::log(X[k*ns + i]) + lnphi[k*ns + i]);
        }
    }
    return G;
}

double BaseSplit::calc_condition_number()
{
    double condition_number;

    if(flash_params.split_variables == FlashParams::lnK)
    {
        // Construct gradient vector and Hessian
        Eigen::MatrixXd PHI = this->construct_PHI();
        Eigen::MatrixXd U_inv = this->construct_Uinv();
        Eigen::MatrixXd J = this->construct_J(PHI, U_inv);

        // Get the condition number for stability if Newton was used
        Eigen::VectorXd eigen = J.eigenvalues().real();
        std::sort(eigen.begin(),eigen.end());
        condition_number = std::sqrt(eigen(eigen.size()-1)/eigen(0));
    }
    else
    {
        // Construct Hessian
        Eigen::MatrixXd U = this->construct_U();
        Eigen::MatrixXd PHI = this->construct_PHI();
        Eigen::MatrixXd H = this->construct_H(U, PHI);

        // Get the condition number for stability if Newton was used
        Eigen::VectorXd eigen = H.eigenvalues().real();
        std::sort(eigen.begin(),eigen.end());
        condition_number = std::sqrt(eigen(eigen.size()-1)/eigen(0));
    }

    return condition_number;
}

int BaseSplit::test_matrices()
{
    // Compare U.inverse() and analytically obtained matrix U_inv
    Eigen::MatrixXd U = this->construct_U();
    Eigen::MatrixXd Uinv = U.inverse();

    Eigen::MatrixXd Uinv_analytical = this->construct_Uinv();
    Uinv_analytical = Uinv_analytical.selfadjointView<Eigen::Lower>();

    // If norm of difference matrix > tol, return 1
    double l2norm = (Uinv - Uinv_analytical).norm();
    if (l2norm > 1e-7)
    {
        std::cout << "PhaseSplit matrix inverse U not correct\n";
        print("Uinv", Uinv);
        print("Uinv analytical", Uinv_analytical);
        print("norm", l2norm);
        return 1;
    }
    
    // Compare J = I + PHI*U^-1 = H*U^-1
    Eigen::MatrixXd PHI = this->construct_PHI();
    Eigen::MatrixXd H = this->construct_H(U, PHI);
    Eigen::MatrixXd J = this->construct_J(PHI, Uinv_analytical);
    Eigen::MatrixXd Jh = H*Uinv;
    
    // If norm of difference matrix > tol, return 1
    l2norm = (J - Jh).norm();
    if (l2norm > 1e-7)
    {
        std::cout << "PhaseSplit matrix inverse J = I + PHI*Uinv != H*Uinv\n";
        print("J = I + PHI*Uinv", J);
        print("J = H*Uinv", Jh);
        print("norm", l2norm);
        return 1;
    }
    return 0;
}
