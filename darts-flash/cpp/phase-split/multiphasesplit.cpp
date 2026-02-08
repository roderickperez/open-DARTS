#include <iostream>
#include <vector>
#include <numeric>

#include "dartsflash/phase-split/multiphasesplit.hpp"
#include <Eigen/Dense>

MultiPhaseSplit::MultiPhaseSplit(FlashParams& flashparams, int np_) : BaseSplit(flashparams, np_)
{
    // Create Rachford-Rice object
    if (np > 2)
    {
        rr = std::make_shared<RR_Min>(RR_Min(flashparams, ns, np));
    }
    else
    {
        rr = std::make_shared<RR_EqConvex2>(RR_EqConvex2(flashparams, ns));
    }
}

void MultiPhaseSplit::find_reference_phases()
{
    // For each component, reference phase (idx 0) should be the phase with the largest composition
    // If not, Hessian matrix for Newton with mole numbers (nij) could become ill-conditioned due to 1/njR term
    // bool same_order = true;
    // for (int i = 0; i < nc; i++)
    // {
    //     int max_idx = 0;
    //     for (int k = 1; k < np; k++)
    //     {
    //         if (n_ik[k*nc + i] > n_ik[max_idx*nc + i])
    //         {
    //             max_idx = k;
    //         }
    //     }
    //     if (max_idx != reference_phase[i])
    //     {
    //         same_order = false;
    //         reference_phase[i] = max_idx;
    //         int kk = 0;
    //         for (int k = 0; k < np; k++)
    //         {
    //             if (k != max_idx)
    //             {
    //                 k_idxs[i*(np-1) + kk] = k;
    //                 kk++;
    //             }
    //         }
    //     }
    // }

    // // If not the same order as before, reconstruct the gradient vector
    // if (!same_order)
    // {
    //     print("not the same order", reference_phase);
    //     this->update_g();
    // }

    return;
}

Eigen::MatrixXd MultiPhaseSplit::construct_U() {
    // (U_ij)_kp = dlnK_ik/dn_jp
    //           = dkp (dij/n_ik - 1/nu_k) + (dij/n_iR - 1/nu_R)
    //           = dij (dkp/n_ik + 1/n_iR) - (dkp/nu_k + 1/nu_R)

    // Construct U
    Eigen::MatrixXd U = Eigen::MatrixXd((np-1)*nc, (np-1)*nc);
    for (int i = 0; i < nc; i++)
    {
        // Reference phase for component i
        int R = reference_phase[i];

        double nuR_inv = 1./nu[R];
        double niR_inv = 1./n_ik[R*nc + i];

        for (int k = 0; k < np-1; k++)
        {
            int kk = k_idxs[i*(np-1) + k];
            double nuk_inv = 1./nu[kk];

            // if i == j
            double Uii = niR_inv - nuR_inv;

            // if i == j & k == p: U(i, k) = 1/n[k,i] + 1/n[R,i] - 1/nu[k] - 1/nu[R]
            U(k*nc + i, k*nc + i) = Uii + 1./n_ik[kk*nc + i] - nuk_inv;

            for (int p = k+1; p < np-1; p++)
            {
                // if i == j & k != p: U(i, k) = 1/n[R,i] - 1/nu[R]
                U(k*nc + i, p*nc + i) = Uii;
                U(p*nc + i, k*nc + i) = Uii;
            }

            // if i != j
            for (int j = i+1; j < nc; j++)
            {
                // if i != j & k == p: U(i, k) = - 1/nu[k] - 1/nu[R]
                double Uijkk = -nuk_inv - nuR_inv;
                U(k*nc + i, k*nc + j) = Uijkk;
                U(k*nc + j, k*nc + i) = Uijkk;
                for (int p = k+1; p < np-1; p++)
                {
                    // if i != j & k != p: U(i, k) = - 1/nu[R]
                    U(k*nc + i, p*nc + j) = -nuR_inv;
                    U(k*nc + j, p*nc + i) = -nuR_inv;
                    U(p*nc + i, k*nc + j) = -nuR_inv;
                    U(p*nc + j, k*nc + i) = -nuR_inv;
                }
            }
        }
    }

    return U;
}

Eigen::MatrixXd MultiPhaseSplit::construct_Uinv() 
{
    // (U_ij^-1)_kp = dn_ik/dlnK_jp = (dn_ik/dlnK_jp)_θ + Σ_np (dn_ik/dθ_m)_K dθ_m/dlnK_jp
    //              = dij n_ik (dkp - n_ip/z_i) + Σ_np (dkp x_ik - nik * (x_ik-x_iR)/z_i) dθ_m/dlnK_jp
    // dθ_m/dlnK_jp = S^-1 f
    Eigen::MatrixXd U_inv = Eigen::MatrixXd::Zero((np-1)*nc, (np-1)*nc);
    
    /*
    // Precalculate some terms for later use
    std::vector<double> zi_inv(nc);
    std::vector<double> wik((np-1)*nc);
    for (int i = 0; i < nc; i++)
    {
        int R = reference_phase[i];
        int iR = R*nc + i;
        zi_inv[i] = 1./z[i];
        for (int k = 0; k < np-1; k++)
        {
            int kk = k_idxs[i*(np-1) + k];
            wik[i * (np-1) + k] = (X[kk*nc + i] - X[iR]) * zi_inv[i];
        }
    }

    // Calculate matrix S for derivatives w.r.t. RR system of equations
    Eigen::MatrixXd S = Eigen::MatrixXd(np-1, np-1);
    for (int k = 0; k < np-1; k++)
    {
        for (int m = k; m < np-1; m++)
        {
            double Skm = 0.;
            for (int i = 0; i < nc; i++)
            {
                Skm += z[i] * wik[i * (np-1) + k] * wik[i * (np-1) + m];
            }
            S(k, m) = Skm;
            S(m, k) = Skm;
        }
    }
    Eigen::LLT<Eigen::MatrixXd> lltOfS(S);
    if (lltOfS.info() == Eigen::NumericalIssue)
    {
        print("ERROR in LLT decomposition of S matrix", 1);
        exit(1);
    }

    // Construct U^-1
    Eigen::VectorXd f_jp = Eigen::VectorXd(np-1);
    for (int j = 0; j < nc; j++)
    {
        int R = reference_phase[j];
        for (int p = 0; p < np-1; p++)
        {
            // Assemble (U_ij^-1)_jp = (dn_ik/dlnK_jp)_θ + Σ_np (dn_ik/dθ_m)_K dθ_m/dlnK_jp
            int pp = k_idxs[j*(np-1) + p];
            int jp = p * nc + j;
            int jR = R * nc + j;

            // Second term
            // Σ_np (dn_ik/dθ_m)_K dθ_m/dlnK_jp = Σ_np (dkp x_ik - nik * (x_ik-x_iR)/z_i) dθ_m/dlnK_jp

            // Solve dθ/dlnK_jp
            // Construct vector f_jp
            for (int k = 0; k < np-1; k++)
            {
                f_jp(k) = -X[jR] * nu[pp] * wik[j * (np-1) + k];
            }
            f_jp(p) += X[jR];

            // Solve dnu/dlnK_jp
            Eigen::VectorXd dnu_dlnKjp = lltOfS.solve(f_jp);

            // Add 
            for (int i = j; i < nc; i++)
            {
                pp = k_idxs[i*(np-1) + p];
                int ip = p * nc + i;

                // i, k == p
                double dnik_dthetam = X[pp*nc + i] - n_ik[pp*nc + i] * wik[i * (np-1) + p];
                for (int m = 0; m < np-1; m++)
                {
                    U_inv(ip, jp) += dnik_dthetam * dnu_dlnKjp(m);
                    // U_inv(jp, ip) += dnik_dthetam * dnu_dlnKjp(m);
                }

                // i, k != p
                for (int k = p+1; k < np-1; k++)
                {
                    int kk = k_idxs[i*(np-1) + k];
                    int ik = k*nc + i;
                    dnik_dthetam = -n_ik[kk*nc + i] * wik[i * (np-1) + p];

                    for (int m = 0; m < np-1; m++)
                    {
                        U_inv(ik, jp) += dnik_dthetam * dnu_dlnKjp(m);
                        // U_inv(jk, ip) += dnik_dthetam * dnu_dlnKjp(m);
                    }
                }
            }

            // First term
            // (dn_ik/dlnK_jp)_θ = dij n_ik (dkp - n_ip/z_i)
            // i == j, k == p
            U_inv(jp, jp) += n_ik[pp*nc + j] * (1. - n_ik[pp*nc + j] * zi_inv[j]);
            for (int k = p+1; k < np-1; k++)
            {
                // i == j, k != p
                int kk = k_idxs[j*(np-1) + k];
                int jk = k * nc + j;
                U_inv(jk, jp) -= n_ik[kk*nc + j] * n_ik[pp*nc + j] * zi_inv[j];
                // U_inv(jp, jk) -= n_ik[kk*nc + j] * n_ik[pp*nc + j] * zi_inv[j];
            }
        }
    }
    // print("Uinv analytical", U_inv);
    U_inv = U_inv.selfadjointView<Eigen::Lower>();
    */
    
    Eigen::MatrixXd U = this->construct_U();
    U_inv = U.inverse();
    return U_inv;
}

Eigen::MatrixXd MultiPhaseSplit::construct_PHI() {
    // (PHI_ij)_kp = d2G/dn_ik dn_jp
    //             = dkp dlnphi_ik/dn_jp + dlnphi_iR/dn_jR

    // Construct PHI
    Eigen::MatrixXd PHI = Eigen::MatrixXd((np-1)*nc, (np-1)*nc);
    for (int i = 0; i < nc; i++)
    {
        // Reference phase for component i
        int R = reference_phase[i];

        for (int k = 0; k < np-1; k++)
        {
            int kk = k_idxs[i*(np-1) + k];
            for (int j = i; j < nc; j++)
            {
                double dlnphiiRdnjR = dlnphidn[R*nc*nc + j*nc + i];

                // if k == p: PHI(i, k) = dlnphi_ik/dn_jk + dlnphi_iR/dn_jR
                double PHIijkk = dlnphidn[kk*nc*nc + j*nc + i] + dlnphiiRdnjR;
                PHI(k*nc + i, k*nc + j) = PHIijkk;
                PHI(k*nc + j, k*nc + i) = PHIijkk;

                for (int p = k+1; p < np-1; p++)
                {
                    // if k != p: PHI(i, k) = dlnphi_iR/dn_jR
                    PHI(k*nc + i, p*nc + j) = dlnphiiRdnjR;
                    PHI(k*nc + j, p*nc + i) = dlnphiiRdnjR;
                    PHI(p*nc + i, k*nc + j) = dlnphiiRdnjR;
                    PHI(p*nc + j, k*nc + i) = dlnphiiRdnjR;
                }
            }
        }
    }

    return PHI;
}

int MultiPhaseSplit::test_matrices()
{
    // Run base class test
    if (BaseSplit::test_matrices())
    {
        return 1;
    }

    // If two phases, test if matrix U and Uinv
    if (np == 2)
    {
        // Compute U from MultiPhaseSplit class
        Eigen::MatrixXd Un = this->construct_U();

        // Construct U for TwoPhaseSplit case
        double VL_inv = 1./(nu[0]*nu[1]);
        Eigen::MatrixXd U2 = Eigen::MatrixXd::Constant(nc, nc, -VL_inv);
        for (int i = 0; i < nc; i++)
        {
            U2(i, i) += VL_inv * z[i] / (X[i] * X[nc+i]);
        }

        // If norm of difference matrix > tol, return 1
        double l2norm = (Un - U2).norm();
        if (l2norm > 1e-7)
        {
            std::cout << "MultiPhaseSplit and TwoPhaseSplit matrices U are not the same\n";
            print("Un", Un);
            print("U2", U2);
            print("norm", l2norm);
            return 1;
        }

        // Compute Uinv from MultiPhaseSplit class
        Eigen::MatrixXd Uinvn = this->construct_Uinv();

        // Construct U for TwoPhaseSplit case
        // Calculate u_i and s
        double s = 1.;
        std::vector<double> u(nc);
        for (int i = 0; i < nc; i++)
        {
            u[i] = X[i] * X[nc+i] / z[i];
            s -= u[i];
        }
        double s_inv = 1./s;

        // Construct U^-1
        Eigen::MatrixXd Uinv2 = Eigen::MatrixXd(nc, nc);
        double VL = nu[0]*nu[1];
        for (int j = 0; j < nc; j++)
        {
            Uinv2(j, j) = VL * u[j] * (1. + u[j] * s_inv);
            for (int i = j+1; i < nc; i++)
            {
                double Uinvij = VL * u[i] * u[j] * s_inv;
                Uinv2(i, j) = Uinvij;
                Uinv2(j, i) = Uinvij;
            }
        }

        // If norm of difference matrix > tol, return 1
        l2norm = (Uinvn - Uinv2).norm();
        if (l2norm > 1e-7)
        {
            std::cout << "MultiPhaseSplit and TwoPhaseSplit matrices U^-1 are not the same\n";
            print("U^-1 n", Uinvn);
            print("U^-1 2", Uinv2);
            print("norm", l2norm);
            return 1;
        }
    }
    return 0;
}