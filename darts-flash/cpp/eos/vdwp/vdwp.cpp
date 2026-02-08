#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <unordered_map>

#include "dartsflash/eos/vdwp/vdwp.hpp"

namespace vdwp {
    double R = 8.314472;
    
    std::unordered_map<std::string, int> n_cages = {{"sI", 2}, {"sII", 2}, {"sH", 3}};
    std::unordered_map<std::string, std::vector<double>> Nm = {{"sI", {2, 6}}, {"sII", {16, 8}}, {"sH", {3, 2, 1}}}; // number per unit cell
    std::unordered_map<std::string, double> nH2O = {{"sI", 46.}, {"sII", 136.}, {"sH", 34.}}; // total number of H2O molecules in hydrate structure (sI, sII, sH)
    std::unordered_map<std::string, std::vector<int>> zm = {{"sI", {20, 24}}, {"sII", {20, 28}}, {"sH", {20, 20, 36}}}; // #waters in cage sI
    
    std::unordered_map<std::string, std::vector<double>> vm = { // number of cages per H2O molecule per unit cell
        {"sI", {Nm["sI"][0]/nH2O["sI"], Nm["sI"][1]/nH2O["sI"]}}, 
        {"sII", {Nm["sII"][0]/nH2O["sII"], Nm["sII"][1]/nH2O["sII"]}},
        {"sH", {Nm["sH"][0]/nH2O["sH"], Nm["sH"][1]/nH2O["sH"], Nm["sH"][2]/nH2O["sH"]}},
    };
    std::unordered_map<std::string, double> xwH_full = {
        {"sI", 1. - (Nm["sI"][0] + Nm["sI"][1]) / (nH2O["sI"] + Nm["sI"][0] + Nm["sI"][1])},
        {"sII", 1. - (Nm["sII"][0] + Nm["sII"][1]) / (nH2O["sII"] + Nm["sII"][0] + Nm["sII"][1])},
        {"sH", 1. - (Nm["sH"][0] + Nm["sH"][1] + Nm["sH"][2]) / (nH2O["sH"] + Nm["sH"][0] + Nm["sH"][1] + Nm["sH"][2])},
    };

} // namespace vdwp

VdWP::VdWP(CompData& comp_data, std::string hydrate_type) : EoS(comp_data) 
{
	this->phase = hydrate_type;
	water_index = comp_data.water_index;
    this->multiple_minima = false;

    zm = vdwp::zm[phase];
    Nm = vdwp::Nm[phase];
    vm = vdwp::vm[phase];
    n_cages = vdwp::n_cages[phase];
    nH2O = vdwp::nH2O[phase];

    // Define range of applicability (lower bound corresponds to xwH with full cage occupancy, upper bound is hypothetical empty hydrate lattice)
    this->eos_range[water_index] = {vdwp::xwH_full[phase], 1.};
    this->stationary_point_range[water_index] = {vdwp::xwH_full[phase] * 1.01, 1.};

    x.resize(nc);
    Nk.resize(nc);
    alpha.resize(nc);
}

void VdWP::solve_PT(std::vector<double>::iterator n_it, bool second_order)
{
    // Calculate mixture parameters with composition n
    (void) second_order;
    this->n_iterator = n_it;
    this->N = std::accumulate(n_it, n_it + this->ns, 0.);
    double nT_inv = 1./this->N;
    std::transform(n_it, n_it + this->nc, this->x.begin(),
                   [&nT_inv](double element) { return element *= nT_inv; });

    // Calculate fugacities
    f = this->fi();
    f[water_index] = this->fw(f);

    return;
}

void VdWP::solve_VT(std::vector<double>::iterator, bool)
{
	std::cout << "No implementation of volume-based calculations exists for VdWP, aborting.\n";
	exit(1);
}

double VdWP::fw(double p_, double T_, std::vector<double>& fi)
{
    // Calculate fugacity of water
    this->init_PT(p_, T_);
    return this->fw(fi);
}

double VdWP::alphai(int i)
{
    // alpha[i] = Ci1 / Ci2
    return C_km[i] / C_km[nc + i];
}
double VdWP::dalphai_dP(int i)
{
    // da[i]/dP = d/dP (Ci1/Ci2) = dCi1/dP / Ci2 - Ci1 * dCi2/dP / Ci2^2
    return dCkmdP[i] / C_km[nc+i] - C_km[i] * dCkmdP[nc+i] / std::pow(C_km[nc+i], 2);
}
double VdWP::dalphai_dT(int i)
{
    // da[i]/dT = d/dT (Ci1/Ci2) = dCi1/dT / Ci2 - Ci1 * dCi2/dT / Ci2^2
    return dCkmdT[i] / C_km[nc+i] - C_km[i] * dCkmdT[nc+i] / std::pow(C_km[nc+i], 2);
}
double VdWP::d2alphai_dPdT(int i)
{
    // d2a[i]/dPdT = d/dT (da[i]/dP) = d/dT (dCi1/dP / Ci2 - Ci1 * dCi2/dP / Ci2^2) 
    //                               = d2Ci1/dPdT / Ci2 - dCi1/dP / Ci2^2 dCi2/dT 
    //                                  - dCi1/dT * dCi2/dP / Ci2^2 - Ci1 d2Ci2/dPdT / Ci2^2 + 2 Ci1 * dCi2/dP / Ci2^3 dCi2/dT
    return d2CkmdPdT[i] / C_km[nc+i] - dCkmdP[i] / std::pow(C_km[nc+i], 2) * dCkmdT[nc+i]
            - dCkmdT[i] * dCkmdP[nc+i] / std::pow(C_km[nc+i], 2) - C_km[i] * d2CkmdPdT[nc+i] / std::pow(C_km[nc+i], 2) 
            + 2. * C_km[i] * dCkmdP[nc+i] / std::pow(C_km[nc+i], 3) * dCkmdT[nc+i];
}
double VdWP::d2alphai_dT2(int i)
{
    // d2a[i]/dT2 = d/dT (da[i]/dT) = d2Ci1/dT2 / Ci2 - dCi1/dT / Ci2^2 * dCi2/dT 
    //            - (dCi1/dT * dCi2/dT / Ci2^2 + Ci1 * d2Ci2/dT / Ci2^2 - 2 Ci1 (dCi2/dT)^2 / Ci2^3)
    return d2CkmdT2[i] / C_km[nc+i] - dCkmdT[i] / std::pow(C_km[nc+i], 2) * dCkmdT[nc+i] 
            - (dCkmdT[i] * dCkmdT[nc+i] / std::pow(C_km[nc+i], 2) + C_km[i] * d2CkmdT2[nc+i] / std::pow(C_km[nc+i], 2)
            - 2. * C_km[i] * std::pow(dCkmdT[nc+i], 2) / std::pow(C_km[nc+i], 3));
}

std::vector<double> VdWP::fi() 
{
    // Calculate guest fugacity from p, T, n
    // Cole (1990) and Michelsen (1990)
    f = std::vector<double>(nc);

    // Introduce alpha[k] = Ck1/Ck2 and N[k]
    alpha = std::vector<double>(nc);
    Nk = std::vector<double>(nc);  // number of guest molecules per water molecule in unit cell
    for (int k = 0; k < nc; k++)
    {
        if (k == water_index)
        {
            Nk[k] = 0.;
            alpha[k] = 0.;
        }
        else
        {
            Nk[k] = x[k] / x[water_index];
            alpha[k] = this->alphai(k);
        }
    }
    double sumNk = std::accumulate(Nk.begin(), Nk.end(), 0.);
    N0 = vm[0] + vm[1] - sumNk;

    // Calculate eta
    eta = this->calc_eta();

    // Calculate guest fugacities
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            f[k] = Nk[k]/N0 / (C_km[nc+k] * (eta + alpha[k] * (1-eta)));
        }
    }

    return f;
}
std::vector<double> VdWP::dfi_dP() 
{
    // Derivative of guest fugacities w.r.t. P

    // Calculate derivatives of Langmuir constants C_km w.r.t. P
    dCkmdP = this->dCkm_dP();

    // Calculate derivatives of eta w.r.t. P
    double detadP = this->deta_dP();

    // Calculate derivative of guest fugacities
    std::vector<double> dfidP(nc);
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // da[i]/dP = d/dP (Ci1/Ci2) = dCi1/dP / Ci2 - Ci1 * dCi2/dP / Ci2^2
            double dalphaidP = this->dalphai_dP(i);

            // df[i]/dP = Ni/N0 * (-1/Ci2^2 * dCi2/dP * 1/(n + a[i](1-n)) 
            //                     -1/Ci2 * 1/(n+a[i](1-n))^2 * (dn/dP*(1-a[i])+da[i]/dP*(1-n))
            double denom = 1./(C_km[nc+i] * (eta+alpha[i]*(1.-eta)));
            double ddenom = dCkmdP[nc + i] * (eta+alpha[i]*(1.-eta)) + C_km[nc+i] * (detadP * (1.-alpha[i]) + dalphaidP * (1.-eta));

            dfidP[i] = -Nk[i]/N0 * std::pow(denom, 2) * ddenom;
        }
    }
    return dfidP;
}
std::vector<double> VdWP::dfi_dT() 
{
    // Derivative of guest fugacities w.r.t. T

    // Calculate derivatives of Langmuir constants C_km w.r.t. T
    dCkmdT = this->dCkm_dT();

    // Calculate derivatives of eta w.r.t. T
    double detadT = this->deta_dT();

    // Calculate derivative of guest fugacities
    std::vector<double> dfidT(nc);
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // da[i]/dT = d/dT (Ci1/Ci2) = dCi1/dT / Ci2 - Ci1 * dCi2/dT / Ci2^2
            double dalphaidT = this->dalphai_dT(i);

            // df[i]/dT = Ni/N0 * (-1/Ci2^2 * dCi2/dT * 1/(n + a[i](1-n)) 
            //                     -1/Ci2 * 1/(n+a[i](1-n))^2 * (dn/dT*(1-a[i])+da[i]/dT*(1-n))
            double denom = 1./(C_km[nc+i] * (eta+alpha[i]*(1.-eta)));
            double ddenom = dCkmdT[nc + i] * (eta+alpha[i]*(1.-eta)) + C_km[nc+i] * (detadT + dalphaidT * (1.-eta) - alpha[i] * detadT);
            dfidT[i] = -Nk[i]/N0 * std::pow(denom, 2) * ddenom;
        }
    }
    return dfidT;
}
std::vector<double> VdWP::d2fi_dPdT() 
{
    // Second derivative of guest fugacities w.r.t. P and T

    // Calculate derivatives of Langmuir constants C_km w.r.t. P
    dCkmdP = this->dCkm_dP();
    dCkmdT = this->dCkm_dT();
    d2CkmdPdT = this->d2Ckm_dPdT();

    // Calculate derivatives of eta w.r.t. P
    double detadP = this->deta_dP();
    double detadT = this->deta_dT();
    double d2etadPdT = this->d2eta_dPdT(detadT);

    // Calculate derivative of guest fugacities
    std::vector<double> dfidP(nc), d2fidPdT(nc);
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // da[i]/dP = d/dP (Ci1/Ci2) = dCi1/dP / Ci2 - Ci1 * dCi2/dP / Ci2^2
            double dalphaidP = this->dalphai_dP(i);
            double dalphaidT = this->dalphai_dT(i);
            double d2alphaidPdT = this->d2alphai_dPdT(i);

            // df[i]/dP = Ni/N0 * (-1/Ci2^2 * dCi2/dP * 1/(n + a[i](1-n)) 
            //                     -1/Ci2 * 1/(n+a[i](1-n))^2 * (dn/dP*(1-a[i])+da[i]/dP*(1-n))
            // d2f[i]/dPdT = Ni/N0 * (-1/Ci2^2 * dCi2/dP * 1/(n + a[i](1-n)) 
            //                        -1/Ci2 * 1/(n+a[i](1-n))^2 * (dn/dP*(1-a[i])+da[i]/dP*(1-n))
            double denom = 1./(C_km[nc+i] * (eta+alpha[i]*(1.-eta)));
            double ddenom = dCkmdP[nc + i] * (eta+alpha[i]*(1.-eta)) + C_km[nc+i] * (detadP * (1.-alpha[i]) + dalphaidP * (1.-eta));
            double d2denom = d2CkmdPdT[nc + i] * (eta+alpha[i]*(1.-eta)) + dCkmdP[nc + i] * (detadT + dalphaidT * (1.-eta) - alpha[i] * detadT) 
                            + dCkmdT[nc+i] * (detadP * (1.-alpha[i]) + dalphaidP * (1.-eta)) + C_km[nc+i] * (d2etadPdT * (1.-alpha[i]) - detadP * dalphaidT + d2alphaidPdT * (1.-eta) - dalphaidP * detadT);

            dfidP[i] = -Nk[i]/N0 * std::pow(denom, 2) * ddenom;
            d2fidPdT[i] = Nk[i]/N0 * (2. * std::pow(denom, 3) * std::pow(ddenom, 2) - std::pow(denom, 2) * d2denom);
        }
    }
    return d2fidPdT;
}
std::vector<double> VdWP::d2fi_dT2() 
{
    // Second derivative of guest fugacities w.r.t. T

    // Calculate derivatives of Langmuir constants C_km w.r.t. T
    dCkmdT = this->dCkm_dT();
    d2CkmdT2 = this->d2Ckm_dT2();

    // Calculate derivatives of eta w.r.t. T
    double detadT = this->deta_dT();
    double d2etadT2 = this->d2eta_dT2(detadT);

    // Calculate derivative of guest fugacities
    std::vector<double> dfidT(nc), d2fidT2(nc);
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // f[i] = Nk[i] / (C_km[nc+i] * N0 * (eta + alpha[i] * (1-eta));
            // df[i]/dT = Ni/N0 * (-1/Ci2^2 * dCi2/dT * 1/(n + a[i](1-n)) 
            //                     -1/Ci2 * 1/(n+a[i](1-n))^2 * (dn/dT*(1-a[i])+da[i]/dT*(1-n))
            // d2f[i]/dT2 = Ni/N0 * d/dT (ddenom/dT)
            double dalphaidT = this->dalphai_dT(i);
            double d2alphaidT2 = this->d2alphai_dT2(i);

            double denom = 1./(C_km[nc+i] * (eta+alpha[i]*(1.-eta)));
            double ddenom = dCkmdT[nc + i] * (eta+alpha[i]*(1.-eta)) + C_km[nc+i] * (detadT + dalphaidT * (1.-eta) - alpha[i] * detadT);
            double d2denom = d2CkmdT2[nc + i] * (eta+alpha[i]*(1.-eta)) + 2. * dCkmdT[nc+i] * (detadT + dalphaidT * (1.-eta) - alpha[i] * detadT) 
                            + C_km[nc+i] * (d2etadT2 + d2alphaidT2 * (1.-eta) - dalphaidT * detadT - dalphaidT * detadT - alpha[i] * d2etadT2);

            dfidT[i] = -Nk[i]/N0 * std::pow(denom, 2) * ddenom;
            d2fidT2[i] = Nk[i]/N0 * (2. * std::pow(denom, 3) * std::pow(ddenom, 2) - std::pow(denom, 2) * d2denom);
        }
    }
    return d2fidT2;
}
std::vector<double> VdWP::dfi_dxj() 
{
    // Derivative of guest fugacities w.r.t. x_j

    // Calculate derivatives of eta w.r.t. x_j
    std::vector<double> detadxj = this->deta_dxj();

    // Calculate derivative of guest fugacities
    std::vector<double> dfidxj(nc*nc);
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            double denom = 1./(eta+alpha[i]*(1.-eta));
            for (int j = 0; j < nc; j++)
            {
                dfidxj[i*nc + j] = (this->dNjdxk(i, j)/N0 - this->dN0dxk(j)*Nk[i]/std::pow(N0, 2)) * 1./C_km[nc+i] * denom
                                    - Nk[i]/N0 * 1./C_km[nc+i] * std::pow(denom, 2) * detadxj[j]*(1.-alpha[i]);
            }
        }
    }
    return dfidxj;
}
std::vector<double> VdWP::d2fi_dTdxj() 
{
    // Second derivative of guest fugacities w.r.t. x_j and temperature

    // Calculate derivatives of eta w.r.t. T and x_j
    double detadT = this->deta_dT();
    std::vector<double> detadxj = this->deta_dxj();
    std::vector<double> d2etadTdxj = this->d2eta_dTdxj(detadT);

    // Calculate derivative of guest fugacities
    std::vector<double> d2fidTdxj(nc*nc);
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // da[i]/dT = d/dT (Ci1/Ci2) = dCi1/dT / Ci2 - Ci1 * dCi2/dT / Ci2^2
            double dalphaidT = this->dalphai_dT(i);

            double denom = 1./(eta+alpha[i]*(1.-eta));
            double ddenomdT = (1.-alpha[i]) * detadT + dalphaidT * (1.-eta);
            for (int j = 0; j < nc; j++)
            {
                d2fidTdxj[i*nc + j] = (this->dNjdxk(i, j)/N0 - this->dN0dxk(j)*Nk[i]/std::pow(N0, 2)) * 
                                    (-dCkmdT[nc+i]/std::pow(C_km[nc+i], 2) * denom - 1./C_km[nc+i] * std::pow(denom, 2) * ddenomdT)
                                    - Nk[i]/N0 * (-dCkmdT[nc+i]/std::pow(C_km[nc+i], 2) * std::pow(denom, 2) * detadxj[j]*(1.-alpha[i])
                                                - 2./C_km[nc+i] * std::pow(denom, 3) * ddenomdT * detadxj[j]*(1.-alpha[i])
                                                + 1./C_km[nc+i] * std::pow(denom, 2) * d2etadTdxj[j]*(1.-alpha[i])
                                                + 1./C_km[nc+i] * std::pow(denom, 2) * detadxj[j]*-dalphaidT);
            }
        }
    }
    return d2fidTdxj;
}

std::vector<double> VdWP::calc_theta() 
{
    // Fractional occupancy of cage m by component i
    std::vector<double> theta(n_cages*nc, 0.);
    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            sum_cf += C_km[nc*m + j] * f[j];
        }
        double denom = 1./(1.+sum_cf);
        for (int i = 0; i < nc; i++)
        {
            theta[nc*m + i] = C_km[nc*m + i] * f[i] * denom;
        }
    }
    return theta;
}
std::vector<double> VdWP::dtheta_dP(std::vector<double>& dfidP) 
{
    // Derivative of theta_im w.r.t. P
    std::vector<double> dtheta_km(n_cages*nc);

    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. };
        double sum_dcf{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            sum_cf += C_km[nc*m + j] * f[j];
            sum_dcf += dCkmdP[nc*m + j] * f[j] + C_km[nc*m + j] * dfidP[j];
        }
        double denom = 1./(1. + sum_cf);
        for (int i = 0; i < nc; i++)
        {
            dtheta_km[nc*m + i] = (dCkmdP[nc*m + i] * f[i] + C_km[nc*m + i] * dfidP[i]) * denom
                                    - C_km[nc*m + i] * f[i] * std::pow(denom, 2) * sum_dcf;
        }
    }
    return dtheta_km;
}
std::vector<double> VdWP::dtheta_dT(std::vector<double>& dfidT) 
{
    // Derivative of theta_im w.r.t. T
    std::vector<double> dtheta_km(n_cages*nc);

    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. };
        double sum_dcf{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            if (j != water_index)
            {
                sum_cf += C_km[nc*m + j] * f[j];
                sum_dcf += dCkmdT[nc*m + j] * f[j] + C_km[nc*m + j] * dfidT[j];
            }
        }
        double denom = 1./(1. + sum_cf);
        for (int i = 0; i < nc; i++)
        {
            dtheta_km[nc*m + i] = (dCkmdT[nc*m + i] * f[i] + C_km[nc*m + i] * dfidT[i]) * denom
                                    - C_km[nc*m + i] * f[i] * std::pow(denom, 2) * sum_dcf;
        }
    }
    return dtheta_km;
}
std::vector<double> VdWP::d2theta_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) 
{
    // Second derivative of theta_im w.r.t. P and T
    std::vector<double> d2theta_km(n_cages*nc);
    (void) dfidT;

    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. };
        double sum_dcfdP{ 0. }, sum_dcfdT{ 0. }, sum_d2cfdPdT{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            sum_cf += C_km[nc*m + j] * f[j];
            sum_dcfdP += dCkmdP[nc*m + j] * f[j] + C_km[nc*m + j] * dfidP[j];
            sum_dcfdT += dCkmdP[nc*m + j] * f[j] + C_km[nc*m + j] * dfidP[j];
            sum_d2cfdPdT += d2CkmdPdT[nc*m + j] * f[j] + dCkmdP[nc*m + j] * dfidT[j] + dCkmdT[nc*m + j] * d2fidPdT[j] + dCkmdT[nc*m + j] * dfidP[j];
        }
        double denom = 1./(1. + sum_cf);
        for (int i = 0; i < nc; i++)
        {
            d2theta_km[nc*m + i] = (d2CkmdPdT[nc*m + i] * f[i] + dCkmdP[nc*m + i] * dfidT[i] + C_km[nc*m + i] * dfidP[i] + dCkmdT[nc*m + i] * d2fidPdT[i]) * denom
                                 - (dCkmdP[nc*m + i] * f[i] + C_km[nc*m + i] * dfidP[i]) * std::pow(denom, 2) * sum_dcfdT
                                    - (dCkmdT[nc*m + i] * f[i] * std::pow(denom, 2) * sum_dcfdP
                                     + C_km[nc*m + i] * dfidT[i] * std::pow(denom, 2) * sum_dcfdP
                                     - 2.* C_km[nc*m + i] * f[i] * std::pow(denom, 3) * sum_dcfdT * sum_dcfdP
                                     + C_km[nc*m + i] * f[i] * std::pow(denom, 2) * sum_d2cfdPdT);
        }
    }
    return d2theta_km;
}
std::vector<double> VdWP::d2theta_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2) 
{
    // Second derivative of theta_im w.r.t. T
    std::vector<double> d2theta_km(n_cages*nc);

    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. }, sum_dcf{ 0. }, sum_d2cf{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            if (j != water_index)
            {
                sum_cf += C_km[nc*m + j] * f[j];
                sum_dcf += dCkmdT[nc*m + j] * f[j] + C_km[nc*m + j] * dfidT[j];
                sum_d2cf += d2CkmdT2[nc*m + j] * f[j] + dCkmdT[nc*m + j] * dfidT[j]
                            + dCkmdT[nc*m + j] * dfidT[j] + C_km[nc*m + j] * d2fidT2[j];
            }
        }
        double denom = 1./(1. + sum_cf);
        // double d2denom = -2. * denom * ddenom * sum_dcf - std::pow(denom, 2) * sum_d2cf;
        for (int i = 0; i < nc; i++)
        {
            d2theta_km[nc*m + i] = (d2CkmdT2[nc*m + i] * f[i] + dCkmdT[nc*m + i] * dfidT[i] + dCkmdT[nc*m + i] * dfidT[i] + C_km[nc*m + i] * d2fidT2[i]) * denom
                                    - (dCkmdT[nc*m + i] * f[i] + C_km[nc*m + i] * dfidT[i]) * std::pow(denom, 2) * sum_dcf
                                    - (dCkmdT[nc*m + i] * f[i] * std::pow(denom, 2) * sum_dcf 
                                        + C_km[nc*m + i] * dfidT[i] * std::pow(denom, 2) * sum_dcf
                                        - 2. * C_km[nc*m + i] * f[i] * std::pow(denom, 3) * std::pow(sum_dcf, 2)
                                        + C_km[nc*m + i] * f[i] * std::pow(denom, 2) * sum_d2cf);
        }
    }
    return d2theta_km;
}
std::vector<double> VdWP::dtheta_dxj(std::vector<double>& dfidxj) 
{
    // Derivative of theta_im w.r.t. x_j
    std::vector<double> dthetadxj(n_cages*nc*nc);
    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. };
        std::vector<double> sum_dcf(nc, 0.);
        for (int j = 0; j < nc; j++) 
        {
            sum_cf += C_km[m*nc + j] * f[j];
            for (int l = 0; l < nc; l++)
            {
                sum_dcf[j] += dfidxj[l*nc + j] * C_km[m*nc + l];
            }
        }
        double denom = 1./(1.+sum_cf);
        for (int i = 0; i < nc; i++)
        {
            for (int k = 0; k < nc; k++)
            {
                dthetadxj[m*nc*nc + i*nc + k] = C_km[m*nc + i] * dfidxj[i*nc+k] * denom 
                                                - C_km[m*nc + i] * f[i] * std::pow(denom, 2) * sum_dcf[k];
            }
        }
    }
    return dthetadxj;
}
std::vector<double> VdWP::d2theta_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj) 
{
    // Derivative of theta_im w.r.t. x_j
    std::vector<double> d2thetadTdxj(n_cages*nc*nc);
    for (int m = 0; m < n_cages; m++) 
    {
        double sum_cf{ 0. };
        double dsum_cfdT{ 0. };
        std::vector<double> sum_dcf(nc, 0.);
        std::vector<double> sum_d2cfdT(nc, 0.);
        for (int j = 0; j < nc; j++) 
        {
            sum_cf += C_km[m*nc + j] * f[j];
            dsum_cfdT += dCkmdT[m*nc + j] * f[j] + C_km[m*nc + j] * dfidT[j];
            for (int l = 0; l < nc; l++)
            {
                sum_dcf[j] += dfidxj[l*nc + j] * C_km[m*nc + l];
                sum_d2cfdT[j] += d2fidTdxj[l*nc + j] * C_km[m*nc + l] + dfidxj[l*nc + j] * dCkmdT[m*nc + l];
            }
        }
        double denom = 1./(1.+sum_cf);
        for (int i = 0; i < nc; i++)
        {
            for (int k = 0; k < nc; k++)
            {
                d2thetadTdxj[m*nc*nc + i*nc + k] = dCkmdT[m*nc + i] * dfidxj[i*nc+k] * denom 
                                                 + C_km[m*nc + i] * d2fidTdxj[i*nc+k] * denom 
                                                 - C_km[m*nc + i] * dfidxj[i*nc+k] * std::pow(denom, 2) * dsum_cfdT
                                                - (dCkmdT[m*nc + i] * f[i] * std::pow(denom, 2) * sum_dcf[k]
                                                 + C_km[m*nc + i] * dfidT[i] * std::pow(denom, 2) * sum_dcf[k]
                                                 - 2 * C_km[m*nc + i] * f[i] * std::pow(denom, 3) * dsum_cfdT * sum_dcf[k]
                                                 + C_km[m*nc + i] * f[i] * std::pow(denom, 2) * sum_d2cfdT[k]);
            }
        }
    }
    return d2thetadTdxj;
}

double VdWP::calc_eta() 
{
    // Calculate eta in Newton loop. See Michelsen (1991): Calculation of hydrate fugacities
    // F = Σj Nj eta / (eta + alpha[j]*(1-eta)) + eta * N0 - v2 = 0
    // dF/deta = Σj Nj alpha_j / (eta + alpha[j]*(1-eta))^2 + N0 = 0
    double eta_ = 0.;
    while (true)
    {
        // Calculate F and dF
        double F = eta_ * N0 - vm[1];
        double dF = N0;
        for (int k = 0; k < nc; k++)
        {
            if (alpha[k] > 0.)
            {
                double denom = 1./(eta_ + alpha[k] * (1. - eta_));
                F += Nk[k] * eta_ * denom;
                dF += Nk[k] * alpha[k] * std::pow(denom, 2);
            }
            else
            {
                F += Nk[k];
            }
        }

        if (std::fabs(F) < 1e-13)
        {
            break;
        }
        else
        {
            eta_ -= F/dF;
        }
    }
    return eta_;
}
double VdWP::deta_dP() 
{
    // Differentiate F = 0 implicitly w.r.t. P to obtain dn/dP (n = eta, a = alpha)
    // F = Σj Nj n / (n + a[j]*(1-n)) + n * N0 - v2 = 0

    // Σj d/dP Nj n / (n + a[j]*(1-n)) + dn/dP * N0 = 0
    // Σj Nj * d/dP (n / (n + a[j]*(1-n))) + dn/dP * N0 = 0
    // Σj Nj * (dn/dP / (n + a[j]*(1-n)) - n * (dn/dP + da[j]/dP * (1-n) - a[j] * dn/dP)/(n+a[j]*(1-n))^2) + dn/dP * N0 = 0
    // Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2) + N0) * dn/dP = Σj n * (1-n) * da[j]/dP/(n+a[j]*(1-n))^2

    // denom = Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2)) + N0
    double denom = N0;
    // num = Σj Nj * n * (1-n) * da[j]/dP / (n+a[j]*(1-n))^2
    double num = 0.;

    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {
            double denominator = 1. / (eta+alpha[j]*(1.-eta));
            // denom += Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)
            denom += Nk[j] * (denominator - eta*(1.-alpha[j]) * std::pow(denominator, 2));

            // num += n * (1-n) * da[j]/dP / (n+a[j]*(1-n))^2
            // da[i]/dP = d/dP (Ci1/Ci2) = dCi1/dP / Ci2 - Ci1 * dCi2/dP / Ci2^2
            double dalphaj_dP = this->dalphai_dP(j);
            
            num += Nk[j] * eta * (1.-eta) * dalphaj_dP * std::pow(denominator, 2);
        }
    }

    double detadP = num/denom;
    return detadP;
}
double VdWP::deta_dT() 
{
    // Differentiate F = 0 implicitly w.r.t. T to obtain dn/dT (n = eta, a = alpha)
    // F = Σj Nj n / (n + a[j]*(1-n)) + n * N0 - v2 = 0

    // Σj d/dT Nj n / (n + a[j]*(1-n)) + dn/dT * N0 = 0
    // Σj Nj * d/dT (n / (n + a[j]*(1-n))) + dn/dT * N0 = 0
    // Σj Nj * (dn/dT / (n + a[j]*(1-n)) - n * (dn/dT + da[j]/dT * (1-n) - a[j] * dn/dT)/(n+a[j]*(1-n))^2) + dn/dT * N0 = 0
    // Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2) + N0) * dn/dT = n * (1-n) * da[j]/dT/(n+a[j]*(1-n))^2

    // denom = Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2)) + N0
    double denom = N0;
    // num = Σj Nj * n * (1-n) * da[j]/dT / (n+a[j]*(1-n))^2
    double num = 0.;

    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {
            double denominator = 1. / (eta+alpha[j]*(1.-eta));
            // denom += Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)
            denom += Nk[j] * (denominator - eta*(1.-alpha[j]) * std::pow(denominator, 2));

            // num += Nj * n * (1-n) * da[j]/dT / (n+a[j]*(1-n))^2
            // da[i]/dT = d/dT (Ci1/Ci2) = dCi1/dT / Ci2 - Ci1 * dCi2/dT / Ci2^2
            double dalphaj_dT = this->dalphai_dT(j);
            
            num += Nk[j] * eta * (1.-eta) * dalphaj_dT * std::pow(denominator, 2);
        }
    }

    double detadT = num/denom;
    return detadT;
}
double VdWP::d2eta_dPdT(double detadT) 
{
    // Differentiate F = 0 implicitly w.r.t. P and T to obtain d2n/dPdT (n = eta, a = alpha)
    // F = Σj Nj n / (n + a[j]*(1-n)) + n * N0 - v2 = 0

    // Σj d/dP Nj n / (n + a[j]*(1-n)) + dn/dP * N0 = 0
    // Σj Nj * d/dP (n / (n + a[j]*(1-n))) + dn/dP * N0 = 0
    // Σj Nj * (dn/dP / (n + a[j]*(1-n)) - n * (dn/dP + da[j]/dP * (1-n) - a[j] * dn/dP)/(n+a[j]*(1-n))^2) + dn/dP * N0 = 0
    // Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2) + N0) * dn/dP = Σj n * (1-n) * da[j]/dP/(n+a[j]*(1-n))^2

    // denom = Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2)) + N0
    double denom = N0;
    double ddenom = 0.;
    // num = Σj Nj * n * (1-n) * da[j]/dP / (n+a[j]*(1-n))^2
    double num = 0.;
    double dnum = 0.;

    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {            
            // da[i]/dP = d/dP (Ci1/Ci2) = dCi1/dP / Ci2 - Ci1 * dCi2/dP / Ci2^2
            double dalphaj_dP = this->dalphai_dP(j);
            double dalphaj_dT = this->dalphai_dT(j);
            double d2alphaj_dPdT = this->d2alphai_dPdT(j);

            double denominator = 1. / (eta+alpha[j]*(1.-eta));
            double ddenominator = detadT + dalphaj_dT * (1.-eta) - alpha[j] * detadT;
            
            // denom += Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)
            denom += Nk[j] * (denominator - eta*(1.-alpha[j]) * std::pow(denominator, 2));
            ddenom += Nk[j] * (-std::pow(denominator, 2) * ddenominator 
                            - (detadT*(1.-alpha[j]) - eta * dalphaj_dT) * std::pow(denominator, 2) 
                            + 2. * eta*(1.-alpha[j]) * std::pow(denominator, 3) * ddenominator);

            // num += n * (1-n) * da[j]/dP / (n+a[j]*(1-n))^2
            num += Nk[j] * eta * (1.-eta) * dalphaj_dP * std::pow(denominator, 2);
            dnum += Nk[j] * (detadT * (1.-eta) * dalphaj_dP * std::pow(denominator, 2) - eta * detadT * dalphaj_dP * std::pow(denominator, 2)
                            + eta * (1.-eta) * d2alphaj_dPdT * std::pow(denominator, 2) - 2. * eta * (1.-eta) * dalphaj_dP * std::pow(denominator, 3) * ddenominator);
        }
    }

    double d2etadPdT = (dnum * denom - num * ddenom)/std::pow(denom, 2);
    return d2etadPdT;
}
double VdWP::d2eta_dT2(double detadT) 
{
    // Differentiate F = 0 implicitly w.r.t. T to obtain d2n/dT2 (n = eta, a = alpha)
    // F = Σj Nj n / (n + a[j]*(1-n)) + n * N0 - v2 = 0

    // 1st derivative: dn/dT
    // Σj d/dT Nj n / (n + a[j]*(1-n)) + dn/dT * N0 = 0
    // Σj Nj * d/dT (n / (n + a[j]*(1-n))) + dn/dT * N0 = 0
    // Σj Nj * (dn/dT / (n + a[j]*(1-n)) - n * (dn/dT + da[j]/dT * (1-n) - a[j] * dn/dT)/(n+a[j]*(1-n))^2) + dn/dT * N0 = 0
    // Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2) + N0) * dn/dT = n * (1-n) * da[j]/dT/(n+a[j]*(1-n))^2

    // denom = Σj (Nj * (1/(n + a[j]*(1-n)) - (n-a[j])/(n+a[j]*(1-n))^2)) + N0
    double denom = N0;
    double ddenom = 0.;
    // num = Σj Nj * n * (1-n) * da[j]/dT / (n+a[j]*(1-n))^2
    double num = 0.;
    double dnum = 0.;

    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {
            // num += Nj * n * (1-n) * da[j]/dT / (n+a[j]*(1-n))^2
            // da[i]/dT = d/dT (Ci1/Ci2) = dCi1/dT / Ci2 - Ci1 * dCi2/dT / Ci2^2
            // d2a[i]/dT2 = d/dT (da[i]/dT) = d2Ci1/dT2 / Ci2 - dCi1/dT / Ci2^2 * dCi2/dT 
            //            - (dCi1/dT * dCi2/dT / Ci2^2 + Ci1 * d2Ci2/dT / Ci2^2 - 2 Ci1 (dCi2/dT)^2 / Ci2^3)
            double dalphaj_dT = this->dalphai_dT(j);
            double d2alphaj_dT2 = this->d2alphai_dT2(j);

            double denominator = 1. / (eta+alpha[j]*(1.-eta));
            double ddenominator = detadT + dalphaj_dT * (1.-eta) - alpha[j] * detadT;
            
            // denom += Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)
            denom += Nk[j] * (denominator - eta*(1.-alpha[j]) * std::pow(denominator, 2));

            // ddenom += Nk[j] * (ddenominator/dT - (deta/dT*(1.-alpha[j]) - eta dalpha[j]/dT) * denominator^2)
            //          - 2 denominator * ddenominator/dT / denominator^3
            ddenom += Nk[j] * (-std::pow(denominator, 2) * ddenominator 
                            - (detadT*(1.-alpha[j]) - eta * dalphaj_dT) * std::pow(denominator, 2) 
                            + 2. * eta*(1.-alpha[j]) * std::pow(denominator, 3) * ddenominator);
            
            // num += Nj * n * (1-n) * da[j]/dT / (n+a[j]*(1-n))^2
            // da[i]/dT = d/dT (Ci1/Ci2) = dCi1/dT / Ci2 - Ci1 * dCi2/dT / Ci2^2
            num += Nk[j] * eta * (1.-eta) * dalphaj_dT * std::pow(denominator, 2);
            dnum += Nk[j] * (detadT * (1.-eta) * dalphaj_dT * std::pow(denominator, 2) - eta * detadT * dalphaj_dT * std::pow(denominator, 2)
                            + eta * (1.-eta) * d2alphaj_dT2 * std::pow(denominator, 2) - 2. * eta * (1.-eta) * dalphaj_dT * std::pow(denominator, 3) * ddenominator);
        }
    }

    double d2etadT2 = (dnum * denom - num * ddenom)/std::pow(denom, 2);
    return d2etadT2;
}
std::vector<double> VdWP::deta_dxj() 
{
    // Differentiate F = 0 implicitly w.r.t. xk to obtain dn/dxk (n = eta, a = alpha)
    // F = Σj Nj n / (n + a[j]*(1-n)) + n * N0 - v2 = 0
    std::vector<double> detadxj(nc);
    
    // Σj d/dxk (Nj n / (n + a[j]*(1-n)) + dn/dxk N0 + eta * dN0/dxk = 0
    // Σj (dNj/dxk n / (n + a[j]*(1-n)) + Nj dn/dxk / (n + a[j]*(1-n)) - n * (1-a[j]) * dn/dxk / (n + a[j]*(1-n))^2) + dn/dxk N0 + n dN0/dxk = 0
    // dn/dxk (Σj (Nj / (n + a[j]*(1-n)) - n*(1-a[j])/(n + a[j]*(1-n))^2) + N0) = -n dN0/dxk - Σj dNj/dxk n / (n+a[j]*(1-n))
    // dn/dxk = num / denom
    
    // Calculate denominator
    // denom = Σj [Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)] + N0
    double denom = N0;
    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {
            double denominator = 1. / (eta+alpha[j]*(1.-eta));
            // += Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)
            denom += Nk[j] * (denominator - eta*(1.-alpha[j]) * std::pow(denominator, 2));
        }
    }

    // Calculate numerator
    // num = -n dN0/dxk - Σj [dNj/dxk * n / (n + a[j]*(1-n))]
    for (int k = 0; k < nc; k++)
    {
        double num = -eta * this->dN0dxk(k);
        for (int j = 0; j < nc; j++)
        {
            if (j != water_index)
            {
                num -= this->dNjdxk(j, k) * eta / (eta + alpha[j]*(1.-eta));
            }
        }
        detadxj[k] = num / denom;
    }
    return detadxj;
}
std::vector<double> VdWP::d2eta_dTdxj(double detadT) 
{
    // Differentiate F = 0 implicitly w.r.t. xk to obtain dn/dxk (n = eta, a = alpha)
    // F = Σj Nj n / (n + a[j]*(1-n)) + n * N0 - v2 = 0
    std::vector<double> d2etadTdxj(nc);
    
    // Σj d/dxk (Nj n / (n + a[j]*(1-n)) + dn/dxk N0 + eta * dN0/dxk = 0
    // Σj (dNj/dxk n / (n + a[j]*(1-n)) + Nj dn/dxk / (n + a[j]*(1-n)) - n * (1-a[j]) * dn/dxk / (n + a[j]*(1-n))^2) + dn/dxk N0 + n dN0/dxk = 0
    // dn/dxk (Σj (Nj / (n + a[j]*(1-n)) - n*(1-a[j])/(n + a[j]*(1-n))^2) + N0) = -n dN0/dxk - Σj dNj/dxk n / (n+a[j]*(1-n))
    // dn/dxk = num / denom
    
    // Calculate denominator
    // denom = Σj [Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)] + N0
    double denom = N0;
    double ddenomdT = 0.;
    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {
            double dalphajdT = this->dalphai_dT(j);
            double denominator = 1. / (eta+alpha[j]*(1.-eta));
            double ddenominatordT = (1.-alpha[j]) * detadT + dalphajdT*(1.-eta);
            // += Nj * (1 / (n+a[j]*(1-n)) - n*(1-a[j]) / (n + a[j]*(1-n))^2)
            denom += Nk[j] * (denominator - eta*(1.-alpha[j]) * std::pow(denominator, 2));
            ddenomdT += Nk[j] * (-std::pow(denominator, 2) * ddenominatordT 
                                                - (detadT*(1.-alpha[j]) * std::pow(denominator, 2)
                                                 - eta*dalphajdT * std::pow(denominator, 2)
                                                 - 2 * eta*(1.-alpha[j]) * std::pow(denominator, 3) * ddenominatordT));
        }
    }

    // Calculate numerator
    // num = -n dN0/dxk - Σj [dNj/dxk * n / (n + a[j]*(1-n))]
    for (int k = 0; k < nc; k++)
    {
        double num = -eta * this->dN0dxk(k);
        double dnumdT = -detadT * this->dN0dxk(k);
        for (int j = 0; j < nc; j++)
        {
            if (j != water_index)
            {
                double dalphajdT = this->dalphai_dT(j);
                double denominator = 1. / (eta+alpha[j]*(1.-eta));
                double ddenominatordT = (1.-alpha[j]) * detadT + dalphajdT*(1.-eta);

                num -= this->dNjdxk(j, k) * eta * denominator;
                dnumdT -= this->dNjdxk(j, k) * (detadT * denominator - eta * std::pow(denominator, 2) * ddenominatordT);
            }
        }
        d2etadTdxj[k] = dnumdT / denom - num / std::pow(denom, 2) * ddenomdT;
    }
    return d2etadTdxj;
}

double VdWP::dN0dxk(int k) 
{
    // N0 = v0 + v1 - Σj Nj = v0 + v1 - Σj xj/xw
    if (k == water_index)
    {
        // dN0/dxw = Σj xj/xw^2 = (1-xw)/xw^2
        return (1.-x[water_index]) / std::pow(x[water_index], 2);
    }
    else
    {
        // dN0/dxk = -1/xw
        return -1. / x[water_index];
    }
}
double VdWP::dNjdxk(int j, int k)
{
    // Nj = xj/xw
    if (j == water_index)
    {
        // Nw = 0
        return 0.;
    }
    else if (j == k)
    {
        // dNj/dxj = 1/xw
        return 1./x[water_index];
    }
    else if (k == water_index)
    {
        // dNj/dxw = -xj/xw^2
        return -x[j] / std::pow(x[water_index], 2);
    }
    else
    {
        // dNj/dxk = 0 if j != k
        return 0.;
    }
}

double VdWP::calc_dmuH() 
{  
    // Fractional occupancy of cage m by component i
    theta_km = this->calc_theta();

    // Contribution of cage occupancy to chemical potential
    double dmu_H{ 0. };
    for (int m = 0; m < n_cages; m++) 
    {
        double sumtheta{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            sumtheta += theta_km[nc*m + j];
        }
        dmu_H += vm[m] * std::log(1.-sumtheta);
    }
    return dmu_H;
}
double VdWP::ddmuH_dP(std::vector<double>& dfidP) 
{
    // Derivative of Δmu_wH w.r.t. P

    // Calculate derivatives of theta_km w.r.t. P
    std::vector<double> dthetadP = this->dtheta_dP(dfidP);

    double ddmuHdP{ 0. };
    for (int m = 0; m < n_cages; m++)
    {
        double sum_theta = std::accumulate(theta_km.begin()+m*nc, theta_km.begin()+(m+1)*nc, 0.);
        double sum_dtheta{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            sum_dtheta += dthetadP[nc*m + j];
        }
        ddmuHdP -= vm[m] / (1.-sum_theta) * sum_dtheta;
    }
    return ddmuHdP;
}
double VdWP::ddmuH_dT(std::vector<double>& dfidT) 
{
    // Derivative of Δmu_wH w.r.t. T
    
    // Calculate derivatives of theta_km w.r.t. T
    std::vector<double> dthetadT = this->dtheta_dT(dfidT);

    double ddmuHdT{ 0. };
    for (int m = 0; m < n_cages; m++)
    {
        double sum_theta = std::accumulate(theta_km.begin()+m*nc, theta_km.begin()+(m+1)*nc, 0.);
        double sum_dtheta{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            if (j != water_index)
            {
                sum_dtheta += dthetadT[nc*m + j];   
            }
        }
        ddmuHdT -= vm[m] / (1.-sum_theta) * sum_dtheta;
    }
    return ddmuHdT;
}
double VdWP::d2dmuH_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) 
{
    // Second derivative of Δmu_wH w.r.t. P and T

    // Calculate derivatives of theta_km w.r.t. P
    std::vector<double> dthetadP = this->dtheta_dP(dfidP);
    std::vector<double> dthetadT = this->dtheta_dT(dfidT);
    std::vector<double> d2thetadPdT = this->d2theta_dPdT(dfidP, dfidT, d2fidPdT);
    (void) dfidT;

    double d2dmuHdPdT{ 0. };
    for (int m = 0; m < n_cages; m++)
    {
        double sum_theta = std::accumulate(theta_km.begin()+m*nc, theta_km.begin()+(m+1)*nc, 0.);
        double sum_dthetadP{ 0. }, sum_dthetadT{ 0. }, sum_d2thetadPdT{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            sum_dthetadP += dthetadP[nc*m + j];
            sum_dthetadT += dthetadT[nc*m + j];
            sum_dthetadP += d2thetadPdT[nc*m + j];
        }
        d2dmuHdPdT += vm[m] / std::pow(1.-sum_theta, 2) * sum_dthetadT * sum_dthetadP - vm[m] / (1.-sum_theta) * sum_d2thetadPdT;
    }
    return d2dmuHdPdT;
}
double VdWP::d2dmuH_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2) 
{
    // Second derivative of Δmu_wH w.r.t. T
    
    // Calculate derivatives of theta_km w.r.t. T
    std::vector<double> dthetadT = this->dtheta_dT(dfidT);
    std::vector<double> d2thetadT2 = this->d2theta_dT2(dfidT, d2fidT2);

    double d2dmuHdT2{ 0. };
    for (int m = 0; m < n_cages; m++)
    {
        double sum_theta = std::accumulate(theta_km.begin()+m*nc, theta_km.begin()+(m+1)*nc, 0.);
        double sum_dtheta{ 0. };
        double sum_d2theta{ 0. };
        for (int j = 0; j < nc; j++) 
        {
            if (j != water_index)
            {
                sum_dtheta += dthetadT[nc*m + j];
                sum_d2theta += d2thetadT2[nc*m + j];
            }
        }
        // ddmuHdT -= vm[m] / (1.-sum_theta) * sum_dtheta;
        d2dmuHdT2 -= vm[m] / std::pow(1.-sum_theta, 2) * std::pow(sum_dtheta, 2) + vm[m] / (1.-sum_theta) * sum_d2theta;
    }
    return d2dmuHdT2;
}
std::vector<double> VdWP::ddmuH_dxj(std::vector<double>& dfidxj) 
{
    // Derivative of Δmu_wH w.r.t. x_j

    // Calculate derivatives of theta_km w.r.t. x_j
    std::vector<double> dthetadxj = this->dtheta_dxj(dfidxj);

    std::vector<double> ddmuHdxj(nc, 0.);
    for (int m = 0; m < n_cages; m++)
    {
        double sum_theta = std::accumulate(theta_km.begin()+m*nc, theta_km.begin()+(m+1)*nc, 0.);
        for (int k = 0; k < nc; k++)
        {
            double sum_dtheta{ 0. };
            for (int j = 0; j < nc; j++)
            {
                sum_dtheta += dthetadxj[m*nc*nc + j*nc + k];
            }
            ddmuHdxj[k] -= vm[m] / (1.-sum_theta) * sum_dtheta;
        }
    }
    return ddmuHdxj;
}
std::vector<double> VdWP::d2dmuH_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj) 
{
    // Second derivative of Δmu_wH w.r.t. x_j and T

    // Calculate derivatives of theta_km w.r.t. x_j and T
    std::vector<double> dthetadT = this->dtheta_dT(dfidT);
    std::vector<double> dthetadxj = this->dtheta_dxj(dfidxj);
    std::vector<double> d2thetadTdxj = this->d2theta_dTdxj(dfidT, dfidxj, d2fidTdxj);

    std::vector<double> d2dmuHdTdxj(nc, 0.);
    for (int m = 0; m < n_cages; m++)
    {
        double sum_theta = std::accumulate(theta_km.begin()+m*nc, theta_km.begin()+(m+1)*nc, 0.);
        for (int k = 0; k < nc; k++)
        {
            double dsum_thetadT{ 0. };   // d/dT
            double sum_dtheta{ 0. };     // d/dxj
            double dsum_dthetadT{ 0. };  // d2/dTdxj
            for (int j = 0; j < nc; j++)
            {
                dsum_thetadT += dthetadT[nc*m + j];
                sum_dtheta += dthetadxj[m*nc*nc + j*nc + k];
                dsum_dthetadT += d2thetadTdxj[m*nc*nc + j*nc + k];
            }
            d2dmuHdTdxj[k] -= vm[m] / std::pow(1.-sum_theta, 2) * dsum_thetadT * sum_dtheta + vm[m] / (1.-sum_theta) * dsum_dthetadT;
        }
    }
    return d2dmuHdTdxj;
}

std::vector<double> VdWP::xH() 
{
    std::vector<double> xH(nc, 0.);

    double denominator{ 1. }; // denominator of eq. 3.50
    for (int m = 0; m < n_cages; m++) 
    {
        for (int i = 0; i < nc; i++) 
        {
            denominator += vm[m] * theta_km[nc*m + i];
        }
    }
    
    xH[water_index] = 1.;
    for (int i = 0; i < nc; i++) 
    {
        if (i != water_index)
        {
            double numerator{ 0. }; // numerator of eq. 3.50
            for (int m = 0; m < n_cages; m++) 
            { 
                numerator += vm[m] * theta_km[nc*m + i]; 
            }
            xH[i] = numerator / denominator;
            xH[water_index] -= xH[i];
        }
    }
    return xH;
}

double VdWP::lnphii(int i) 
{
    return std::log(f[i]/(x[i]*p));
}
std::vector<double> VdWP::dlnphi_dP() 
{
    // Calculate derivative of lnphi's w.r.t. P

    // Calculate derivatives of guest fugacities
    std::vector<double> dfidP = this->dfi_dP();
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // Derivative of lnphii w.r.t. P for guest molecules
            // dlnphi/dP = 1/f[i] * df[i]/dP - 1/p
            dlnphidP[i] = 1./f[i] * dfidP[i] - 1./p;
        }
        else
        {
            double dfwdP = this->dfw_dP(dfidP);
            dlnphidP[i] = 1./f[i] * dfwdP - 1./p;
        }
    }

    return dlnphidP;
}
std::vector<double> VdWP::dlnphi_dT() 
{
    // Calculate derivative of lnphi's w.r.t. T

    // Calculate derivative of guest fugacities
    std::vector<double> dfidT = this->dfi_dT();
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // Derivative of lnphii w.r.t. T for guest molecules
            // dlnphii/dT = 1/f[i] * df[i]/dT
            dlnphidT[i] = 1./f[i] * dfidT[i];
        }
        else
        {
            double dfwdT = this->dfw_dT(dfidT);
            dlnphidT[i] = 1./f[i] * dfwdT;
        }
    }
    return dlnphidT;
}
std::vector<double> VdWP::d2lnphi_dPdT() 
{
    // Calculate derivative of lnphi's w.r.t. P

    // Calculate derivatives of guest fugacities
    std::vector<double> dfidP = this->dfi_dP();
    std::vector<double> dfidT = this->dfi_dT();
    std::vector<double> d2fidPdT = this->d2fi_dPdT();
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // Derivative of lnphii w.r.t. P for guest molecules
            d2lnphidPdT[i] = 1./f[i] * d2fidPdT[i] - 1./std::pow(f[i], 2) * dfidT[i] * dfidP[i];
        }
        else
        {
            double dfwdP = this->dfw_dP(dfidP);
            double dfwdT = this->dfw_dT(dfidT);
            double d2fwdPdT = this->d2fw_dPdT(dfidP, dfidT, d2fidPdT);
            d2lnphidPdT[i] = 1./f[i] * d2fwdPdT - 1./std::pow(f[i], 2) * dfwdT * dfwdP;
        }
    }

    return d2lnphidPdT;
}
std::vector<double> VdWP::d2lnphi_dT2() 
{
    // Calculate second derivative of lnphi's w.r.t. T

    // Calculate derivative of guest fugacities
    std::vector<double> dfidT = this->dfi_dT();
    std::vector<double> d2fidT2 = this->d2fi_dT2();
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            // Derivative of lnphii w.r.t. T for guest molecules
            // dlnphii/dT = 1/f[i] * df[i]/dT
            // d2lnphii/dT2 = -1/f[i]^2 * (df[i]/dT)^2 + 1/f[i] * d2f[i]/dT2
            d2lnphidT2[i] = d2fidT2[i]/f[i] - std::pow(dfidT[i] / f[i], 2);
        }
        else
        {
            double dfwdT = this->dfw_dT(dfidT);
            double d2fwdT2 = this->d2fw_dT2(dfidT, d2fidT2);
            d2lnphidT2[i] = d2fwdT2/f[i] - std::pow(dfwdT / f[i], 2);
        }
    }
    return d2lnphidT2;
}
double VdWP::dlnphii_dnj(int i, int k) 
{
	// Evaluate derivative of fugacity coefficient of component i w.r.t. n_k
    std::vector<double> dlnphiidxj(nc);

    // Calculate derivative of guest fugacities w.r.t. x_j
    std::vector<double> dfidxj = this->dfi_dxj();
	if (i == water_index)
	{
		// Calculate derivative of fwH w.r.t. x_j
        std::vector<double> dfwdxj = this->dfw_dxj(dfidxj);
        for (int j = 0; j < nc; j++)
        {
            dlnphiidxj[i*nc + j] = 1./f[i] * dfwdxj[i*nc + j];
        }
        dlnphiidxj[i*nc + i] -= 1./x[i];
	}
	else
	{
		for (int j = 0; j < nc; j++)
        {
            // Derivative of lnphii w.r.t. x_j for guest molecules
            // dlnphi/dxj = 1/f[i] * df[i]/dxj - dij/xj
            dlnphiidxj[j] = 1./f[i] * dfidxj[i*nc + j];
        }
        dlnphiidxj[i] -= 1./x[i];
	}

	// Translate from dxj to dnj
	return this->dxj_to_dnk(dlnphiidxj, this->n_iterator, k);
}
std::vector<double> VdWP::dlnphi_dx()
{
    // Evaluate derivative of fugacity coefficient of component i w.r.t. x_j
    std::vector<double> dlnphidx(nc*nc);
    
    // Calculate derivative of guest fugacities w.r.t. x_j
    std::vector<double> dfidxj = this->dfi_dxj();
    std::vector<double> dfwdxj = this->dfw_dxj(dfidxj);
    std::copy(dfwdxj.begin(), dfwdxj.end(), dfidxj.begin() + water_index * nc);

    for (int i = 0; i < nc; i++)
    {
        // Calculate derivative of fwH w.r.t. x_j
        for (int j = 0; j < nc; j++)
        {
            dlnphidx[i*nc + j] = 1./f[i] * dfidxj[i*nc + j];
            if (i == j)
            {
                dlnphidx[i*nc + j] -= (1.-x[i])/x[i];
            }
            else
            {
                dlnphidx[i*nc + j] += 1.;
            }
        }
    }
    return dlnphidx;
}
std::vector<double> VdWP::d2lnphi_dTdx()
{
    // Evaluate derivative of fugacity coefficient of component i w.r.t. x_j
    std::vector<double> d2lnphidTdx(nc*nc);
    
    // Calculate derivative of guest fugacities w.r.t. x_j
    std::vector<double> dfidT = this->dfi_dT();
    std::vector<double> dfidxj = this->dfi_dxj();
    std::vector<double> d2fidTdxj = this->d2fi_dTdxj();
    std::vector<double> d2fwdTdxj = this->d2fw_dTdxj(dfidT, dfidxj, d2fidTdxj);
    std::copy(d2fwdTdxj.begin(), d2fwdTdxj.end(), d2fidTdxj.begin() + water_index * nc);

    for (int i = 0; i < nc; i++)
    {
        // Calculate derivative of fwH w.r.t. x_j
        for (int j = 0; j < nc; j++)
        {
            d2lnphidTdx[i*nc + j] = -1./std::pow(f[i], 2) * dfidT[i] * dfidxj[i*nc + j] + 1./f[i] * d2fidTdxj[i*nc + j];
        }
    }
    return d2lnphidTdx;
}
std::vector<double> VdWP::dlnphi_dn() 
{
	// Evaluate derivative of fugacity coefficients w.r.t. composition
    std::vector<double> dlnphidx = this->dlnphi_dx();
    return this->dxj_to_dnk(dlnphidx, this->n_iterator);
}
std::vector<double> VdWP::d2lnphi_dTdn()
{
	// Evaluate derivative of fugacity coefficients w.r.t. composition
    std::vector<double> d2lnphidTdx = this->d2lnphi_dTdx();
    return this->dxj_to_dnk(d2lnphidTdx, this->n_iterator);
}

int VdWP::derivatives_test(double p_, double T_, std::vector<double>& x_, double tol, bool verbose)
{
    // Test analytical derivatives of eta, theta and dmuH with respect to P, T and composition (mole fractions)
    int error_output = 0;

    double p0 = p_;
    double T0 = T_;
    std::vector<double> x0 = x_;

    // Calculate eta, theta, dmuH at p, T
    this->init_PT(p0, T0);
    this->solve_PT(x0.begin(), false);
    std::vector<double> f0 = f;
    std::vector<double> a0 = alpha;

    std::vector<double> Ckm0 = this->C_km;
    std::vector<double> dCkmdP0 = this->dCkm_dP();
    std::vector<double> dCkmdT0 = this->dCkm_dT();
    std::vector<double> d2CkmdPdT0 = this->d2Ckm_dPdT();
    std::vector<double> d2CkmdT20 = this->d2Ckm_dT2();
    double eta0 = this->calc_eta();
    double detadP = this->deta_dP();
    double detadT = this->deta_dT();
    double d2etadPdT = this->d2eta_dPdT(detadT);
    double d2etadT2 = this->d2eta_dT2(detadT);
    std::vector<double> detadxj = this->deta_dxj();
    std::vector<double> d2etadTdxj = this->d2eta_dTdxj(detadT);
    
    std::vector<double> dfidP = this->dfi_dP();
    std::vector<double> dfidT = this->dfi_dT();
    std::vector<double> d2fidPdT = this->d2fi_dPdT();
    std::vector<double> d2fidT2 = this->d2fi_dT2();
    std::vector<double> dfidxj = this->dfi_dxj();
    std::vector<double> d2fidTdxj = this->d2fi_dTdxj();
    
    std::vector<double> theta0 = this->calc_theta();
    std::vector<double> dthetadP = this->dtheta_dP(dfidP);
    std::vector<double> dthetadT = this->dtheta_dT(dfidT);
    std::vector<double> d2thetadPdT = this->d2theta_dPdT(dfidP, dfidT, d2fidPdT);
    std::vector<double> d2thetadT2 = this->d2theta_dT2(dfidT, d2fidT2);
    std::vector<double> dthetadxj = this->dtheta_dxj(dfidxj);
    std::vector<double> d2thetadTdxj = this->d2theta_dTdxj(dfidT, dfidxj, d2fidTdxj);
    double dmu_H0 = this->calc_dmuH();
    double ddmuHdP = this->ddmuH_dP(dfidP);
    double ddmuHdT = this->ddmuH_dT(dfidT);
    double d2dmuHdPdT = this->d2dmuH_dPdT(dfidP, dfidT, d2fidPdT);
    double d2dmuHdT2 = this->d2dmuH_dT2(dfidT, d2fidT2);
    std::vector<double> ddmuHdxj = this->ddmuH_dxj(dfidxj);
    std::vector<double> d2dmuHdTdxj = this->d2dmuH_dTdxj(dfidT, dfidxj, d2fidTdxj);
    
    double dfwHdP = this->dfw_dP(dfidP);
    double dfwHdT = this->dfw_dT(dfidT);
    double d2fwHdPdT = this->d2fw_dPdT(dfidP, dfidT, d2fidPdT);
    double d2fwHdT2 = this->d2fw_dT2(dfidT, d2fidT2);
    std::vector<double> dfwHdxj = this->dfw_dxj(dfidxj);
    std::vector<double> d2fwHdTdxj = this->d2fw_dTdxj(dfidT, dfidxj, d2fidTdxj);

    // Calculate eta, theta, dmuH at p1, T
    double dp = 1e-5;
    this->init_PT(p0+dp, T0);
    this->solve_PT(x0.begin(), false);
    std::vector<double> f1 = f;

    double eta1 = this->calc_eta();
    double dmu_H1 = this->calc_dmuH();
    std::vector<double> theta1 = this->calc_theta();
    std::vector<double> Ckm1 = C_km;
    
    double deta_num = (eta1-eta0)/dp;
    double d = (deta_num-detadP)/detadP;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(detadP) > 0.))
    {
        print("deta/dP", {detadP, deta_num, d});
        error_output++;
    }
    double ddmuH_num = (dmu_H1-dmu_H0)/dp;
    d = (ddmuH_num-ddmuHdP)/ddmuHdP;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(ddmuHdP) > 0.))
    {
        print("ddmuH/dP", {ddmuHdP, ddmuH_num, d});
        error_output++;
    }

    for (int m = 0; m < n_cages; m++)
    {
        for (int i = 0; i < nc; i++)
        {
            double dtheta_num = (theta1[m*nc + i] - theta0[m*nc + i])/dp;
            d = (dtheta_num-dthetadP[m*nc + i])/dthetadP[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dthetadP[m*nc+i]) > 0.))
            {
                print("dtheta/dP", {dthetadP[m*nc+i], dtheta_num, d});
                error_output++;
            }
            double dCkm_num = (Ckm1[m*nc + i] - Ckm0[m*nc + i])/dp;
            d = (dCkm_num-dCkmdP0[m*nc + i])/dCkmdP0[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dCkmdP0[m*nc+i]) > 0.))
            {
                print("dCkm/dP", {dCkmdP0[m*nc+i], dCkm_num, d});
                error_output++;
            }
        }
    }

    for (int i = 0; i < nc; i++)
    {
        double df_num = (f1[i]-f0[i])/dp;
        if (i == water_index)
        {
            d = (df_num-dfwHdP)/dfwHdP;
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dfwHdP) > 0.))
            {
                print("dfwH/dP", {dfwHdP, df_num, d});
                error_output++;
            }
        }
        else
        {
            d = (df_num-dfidP[i])/dfidP[i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dfidP[i]) > 0.))
            {
                print("dfi/dP", {dfidP[i], df_num, d});
                error_output++;
            }
        }
    }

    // Calculate eta, theta, dmuH at p, T1
    double dT = 1e-5;
    this->init_PT(p0, T0+dT);
    this->solve_PT(x0.begin(), true);
    f1 = f;

    Ckm1 = C_km;
    std::vector<double> dCkmdT1 = this->dCkm_dT();
    std::vector<double> dCkmdP1 = this->dCkm_dP();
    eta1 = this->calc_eta();
    double detadP1 = this->deta_dP();
    double detadT1 = this->deta_dT();
    dmu_H1 = this->calc_dmuH();
    theta1 = this->calc_theta();

    std::vector<double> dfidT1 = this->dfi_dT();
    std::vector<double> dthetadT1 = this->dtheta_dT(dfidT1);
    double ddmuHdT1 = this->ddmuH_dT(dfidT1);

    double dfwHdT1 = this->dfw_dT(dfidT1);

    std::vector<double> dfidP1 = this->dfi_dP();
    std::vector<double> dthetadP1 = this->dtheta_dP(dfidP1);
    double ddmuHdP1 = this->ddmuH_dP(dfidP1);

    double dfwHdP1 = this->dfw_dP(dfidP1);
    
    deta_num = (eta1-eta0)/dT;
    d = (deta_num-detadT)/detadT;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(detadT) > 0.))
    {
        print("deta/dT", {detadT, deta_num, d});
        error_output++;
    }
    double d2eta_num = (detadP1-detadP)/dT;
    d = (d2eta_num-d2etadPdT)/d2etadPdT;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2etadPdT) > 0.))
    {
        print("d2eta/dPdT", {d2etadPdT, d2eta_num, d});
        error_output++;
    }
    d2eta_num = (detadT1-detadT)/dT;
    d = (d2eta_num-d2etadT2)/d2etadT2;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2etadT2) > 0.))
    {
        print("d2eta/dT2", {d2etadT2, d2eta_num, d});
        error_output++;
    }
    ddmuH_num = (dmu_H1-dmu_H0)/dT;
    d = (ddmuH_num-ddmuHdT)/ddmuHdT;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(ddmuHdT) > 0.))
    {
        print("ddmuH/dT", {ddmuHdT, ddmuH_num, d});
        error_output++;
    }
    double d2dmuH_num = (ddmuHdP1-ddmuHdP)/dT;
    d = (d2dmuH_num-d2dmuHdPdT)/d2dmuHdPdT;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2dmuHdPdT) > 0.))
    {
        print("d2dmuH/dPdT", {d2dmuHdPdT, d2dmuH_num, d});
        error_output++;
    }
    d2dmuH_num = (ddmuHdT1-ddmuHdT)/dT;
    d = (d2dmuH_num-d2dmuHdT2)/d2dmuHdT2;
    if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2dmuHdT2) > 0.))
    {
        print("d2dmuH/dT2", {d2dmuHdT2, d2dmuH_num, d});
        error_output++;
    }

    for (int m = 0; m < n_cages; m++)
    {
        for (int i = 0; i < nc; i++)
        {
            double dCkm_num = (Ckm1[m*nc + i] - Ckm0[m*nc + i])/dT;
            d = (dCkm_num-dCkmdT0[m*nc + i])/dCkmdT0[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dCkmdT0[m*nc+i]) > 0.))
            {
                print("dCkm/dT", {dCkmdT0[m*nc+i], dCkm_num, d});
                error_output++;
            }
            double d2Ckm_num = (dCkmdP1[m*nc + i] - dCkmdP0[m*nc + i])/dT;
            d = (d2Ckm_num-d2CkmdPdT0[m*nc + i])/d2CkmdPdT0[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2CkmdPdT0[m*nc+i]) > 0.))
            {
                print("d2Ckm/dPdT", {d2CkmdPdT0[m*nc+i], d2Ckm_num, d});
                error_output++;
            }
            d2Ckm_num = (dCkmdT1[m*nc + i] - dCkmdT0[m*nc + i])/dT;
            d = (d2Ckm_num-d2CkmdT20[m*nc + i])/d2CkmdT20[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2CkmdT20[m*nc+i]) > 0.))
            {
                print("d2Ckm/dT2", {d2CkmdT20[m*nc+i], d2Ckm_num, d});
                error_output++;
            }

            double dtheta_num = (theta1[m*nc + i] - theta0[m*nc + i])/dT;
            d = (dtheta_num-dthetadT[m*nc + i])/dthetadT[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dthetadT[m*nc+i]) > 0.))
            {
                print("dtheta/dT", {dthetadT[m*nc+i], dtheta_num, d});
                error_output++;
            }
            double d2theta_num = (dthetadP1[m*nc + i] - dthetadP[m*nc + i])/dT;
            d = (d2theta_num-d2thetadPdT[m*nc + i])/d2thetadPdT[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2thetadPdT[m*nc+i]) > 0.))
            {
                print("d2theta/dPdT", {d2thetadPdT[m*nc+i], d2theta_num, d});
                error_output++;
            }
            d2theta_num = (dthetadT1[m*nc + i] - dthetadT[m*nc + i])/dT;
            d = (d2theta_num-d2thetadT2[m*nc + i])/d2thetadT2[m*nc+i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2thetadT2[m*nc+i]) > 0.))
            {
                print("d2theta/dT2", {d2thetadT2[m*nc+i], d2theta_num, d});
                error_output++;
            }
        }
    }

    for (int i = 0; i < nc; i++)
    {
        double df_num = (f1[i]-f0[i])/dT;
        if (i == water_index)
        {
            d = (df_num-dfwHdT)/dfwHdT;
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dfwHdT) > 0.))
            {
                print("dfwH/dT", {dfwHdT, df_num, d});
                error_output++;
            }
        }
        else
        {
            d = (df_num-dfidT[i])/dfidT[i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dfidT[i]) > 0.))
            {
                print("dfi/dT", {dfidT[i], df_num, d});
                error_output++;
            }
        }

        if (i == water_index)
        {
            double d2f_num = (dfwHdP1-dfwHdP)/dT;
            d = (d2f_num-d2fwHdPdT)/d2fwHdPdT;
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2fwHdPdT) > 0.))
            {
                print("d2fwH/dPdT", {d2fwHdPdT, d2f_num, d});
                error_output++;
            }

            d2f_num = (dfwHdT1-dfwHdT)/dT;
            d = (d2f_num-d2fwHdT2)/d2fwHdT2;
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2fwHdT2) > 0.))
            {
                print("d2fwH/dT2", {d2fwHdT2, d2f_num, d});
                error_output++;
            }
        }
        else
        {
            double d2f_num = (dfidP1[i]-dfidP[i])/dT;
            d = (d2f_num-d2fidPdT[i])/d2fidPdT[i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2fidPdT[i]) > 0.))
            {
                print("d2fi/dPdT", {d2fidPdT[i], d2f_num, d});
                error_output++;
            }

            d2f_num = (dfidT1[i]-dfidT[i])/dT;
            d = (d2f_num-d2fidT2[i])/d2fidT2[i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2fidT2[i]) > 0.))
            {
                print("d2fi/dT2", {d2fidT2[i], d2f_num, d});
                error_output++;
            }
        }
    }

    // Test analytical derivatives with respect to composition (mole fractions)
    double dx = 1e-6;
    this->init_PT(p0, T0);
	for (int k = 0; k < nc; k++)
	{
		// Transform to +dx
		double dxk = dx * x0[k];
		x0[k] += dxk;
        for (int ii = 0; ii < nc; ii++)
        {
            x0[ii] /= (1. + dxk);
        }
		
		// Numerical derivative of lnphi w.r.t. xk
        this->solve_PT(x0.begin(), true);

        f1 = f;

        Ckm1 = C_km;
        dCkmdT1 = this->dCkm_dT();
        eta1 = this->calc_eta();
        detadT1 = this->deta_dT();
        dmu_H1 = this->calc_dmuH();
        theta1 = this->calc_theta();

        dfidT1 = this->dfi_dT();
        dthetadT1 = this->dtheta_dT(dfidT1);
        ddmuHdT1 = this->ddmuH_dT(dfidT1);

        dfwHdT1 = this->dfw_dT(dfidT1);
    
		// Compare analytical and numerical
        deta_num = (eta1-eta0)/dxk;
        d = std::log(std::fabs(deta_num + 1e-15)) - std::log(std::fabs(detadxj[k] + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && std::fabs(detadxj[k]) > 0.))
        {
            print("deta/dxj", {detadxj[k], deta_num, d});
            error_output++;
        }
        d2eta_num = (detadT1-detadT)/dxk;
        d = std::log(std::fabs(d2eta_num + 1e-15)) - std::log(std::fabs(d2etadTdxj[k] + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2etadTdxj[k]) > 0.))
        {
            print("d2eta/dTdxj", {d2etadTdxj[k], d2eta_num, d});
            error_output++;
        }
        ddmuH_num = (dmu_H1-dmu_H0)/dxk;
        d = std::log(std::fabs(ddmuH_num + 1e-15)) - std::log(std::fabs(ddmuHdxj[k] + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && std::fabs(ddmuHdxj[k]) > 0.))
        {
            print("ddmuH/dxj", {ddmuHdxj[k], ddmuH_num, d});
            error_output++;
        }
        d2dmuH_num = (ddmuHdT1-ddmuHdT)/dxk;
        d = std::log(std::fabs(d2dmuH_num + 1e-15)) - std::log(std::fabs(d2dmuHdTdxj[k] + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2dmuHdTdxj[k]) > 0.))
        {
            print("d2dmuH/dTdxj", {d2dmuHdTdxj[k], d2dmuH_num, d});
            error_output++;
        }

        for (int m = 0; m < n_cages; m++)
        {
            for (int i = 0; i < nc; i++)
            {
                double dtheta_num = (theta1[m*nc+i]-theta0[m*nc + i])/dxk;
                d = std::log(std::fabs(dtheta_num + 1e-15)) - std::log(std::fabs(dthetadxj[m*nc*nc + i*nc + k] + 1e-15));
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(dthetadxj[m*nc*nc + i*nc + k]) > 0.))
                {
                    print("dtheta/dxj", {dthetadxj[m*nc*nc + i*nc + k], dtheta_num, d});
                    error_output++;
                }
                double d2theta_num = (dthetadT1[m*nc + i] - dthetadT[m*nc + i])/dxk;
                d = std::log(std::fabs(d2theta_num + 1e-15)) - std::log(std::fabs(d2thetadTdxj[m*nc*nc + i*nc + k] + 1e-15));
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2thetadTdxj[m*nc*nc + i*nc + k]) > 0.))
                {
                    print("d2theta/dTdxj", {d2thetadTdxj[m*nc*nc + i*nc + k], d2theta_num, d});
                    error_output++;
                }
            }
        }

        for (int i = 0; i < nc; i++)
        {
            double df_num = (f1[i]-f0[i])/dxk;
            if (i == water_index)
            {
                d = std::log(std::fabs(df_num + 1e-15)) - std::log(std::fabs(dfwHdxj[k] + 1e-15));
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(dfwHdxj[k]) > 0.))
                {
                    print("dfwH/dxj", {dfwHdxj[k], df_num, d});
                    error_output++;
                }
            }
            else
            {
                d = std::log(std::fabs(df_num + 1e-15)) - std::log(std::fabs(dfidxj[i*nc + k] + 1e-15));
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(dfidxj[i*nc + k]) > 0.))
                {
                    print("dfi/dxj", {dfidxj[i*nc + k], df_num, d});
                    error_output++;
                }
            }

            if (i == water_index)
            {
                double d2f_num = (dfwHdT1-dfwHdT)/dxk;
                d = std::log(std::fabs(d2f_num + 1e-15)) - std::log(std::fabs(d2fwHdTdxj[k] + 1e-15));
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2fwHdTdxj[k]) > 0.))
                {
                    print("d2fwH/dTdxj", {d2fwHdTdxj[k], d2f_num, d});
                    error_output++;
                }
            }
            else
            {
                double d2f_num = (dfidT1[i]-dfidT[i])/dxk;
                d = std::log(std::fabs(d2f_num + 1e-15)) - std::log(std::fabs(d2fidTdxj[i*nc + k] + 1e-15));
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(d2fidTdxj[i*nc + k]) > 0.))
                {
                    print("d2fi/dTdxj", {d2fidTdxj[i*nc + k], d2f_num, d});
                    error_output++;
                }
            }
        }

		// Return to original x
        for (int ii = 0; ii < ns; ii++)
        {
            x0[ii] *= (1. + dxk);
        }
        x0[k] -= dxk;
	}

    return error_output;
}
