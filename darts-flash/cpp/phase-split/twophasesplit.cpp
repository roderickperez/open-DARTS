#include <iostream>
#include <vector>

#include "dartsflash/phase-split/twophasesplit.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include <Eigen/Dense>

TwoPhaseSplit::TwoPhaseSplit(FlashParams& flashparams) : BaseSplit(flashparams, 2)
{
    // Create Rachford-Rice object
    rr = std::make_shared<RR_EqConvex2>(RR_EqConvex2(flashparams, ns));
}

Eigen::MatrixXd TwoPhaseSplit::construct_U() {
    // (U_ij)_kp = dlnK_ik/dn_jp
    //           = dij (dkp/(nu_k x_ik) + 1/(nu_R x_iR)) - (dkp/nu_k + 1/nu_R)
    // U_ij = dlnK_i/dn_jV = 1/VL(dij/u_i - 1)

    // Construct U
    double VL_inv = 1./(nu[0]*nu[1]);
    Eigen::MatrixXd U = Eigen::MatrixXd::Constant(nc, nc, -VL_inv);
    for (int i = 0; i < nc; i++)
    {
        U(i, i) += VL_inv * z[i] / (X[i] * X[nc+i]);
    }
    return U;
}

Eigen::MatrixXd TwoPhaseSplit::construct_Uinv() {
    // U_ij^-1 = dn_iV/dlnK_j = VL u_i(dij + u_j/s) with s = 1-sum(u_i)

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
    Eigen::MatrixXd U_inv = Eigen::MatrixXd(nc, nc);
    double VL = nu[0]*nu[1];
    for (int j = 0; j < nc; j++)
    {
        U_inv(j, j) = VL * u[j] * (1. + u[j] * s_inv);
        for (int i = j+1; i < nc; i++)
        {
            U_inv(i, j) = VL * u[i] * u[j] * s_inv;
        }
    }
    return U_inv;
}

Eigen::MatrixXd TwoPhaseSplit::construct_PHI() {
    // (PHI_ij)_kp = d2G/dn_ik dn_jp
    //             = dkp dlnphi_ik/dn_jp + dlnphi_iR/dn_jR
    // PHI_ij = dlnphi_iL/dn_jL + dlnphi_iV/dn_jV

    // Construct PHI
    Eigen::MatrixXd PHI = Eigen::MatrixXd(nc, nc);
    for (int i = 0; i < nc; i++)
    {
        for (int j = i; j < nc; j++)
        {
            PHI(j, i) = dlnphidn[j*nc + i] + dlnphidn[nc*nc + j*nc + i];
        }
    }
    return PHI;
}
