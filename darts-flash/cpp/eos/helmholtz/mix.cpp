#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include "dartsflash/eos/helmholtz/mix.hpp"
#include "dartsflash/global/components.hpp"

Mix::Mix(CompData& data, double omegaa, double omegab, std::vector<double>& kappa_, bool volume_shift_, bool is_srk_)
{
    this->nc = data.nc;
    this->volume_shift = volume_shift_;
    this->is_srk = is_srk_;

    this->Pc = data.Pc;
	this->Tc = data.Tc;
    this->Z_ra = data.Z_ra;
	this->kij = data.kij;
    this->omegaA = omegaa;
    this->omegaB = omegab;
    this->kappa = kappa_;
    
    a_c = this->ac();

    a_i.resize(nc);
    b_i.resize(nc);
    c_i.resize(nc);
    a_ij.resize(nc*nc);
    b_ij.resize(nc*nc);

    B_i.resize(nc);
    D_i.resize(nc);
    B_ij.resize(nc*nc);
    D_ij.resize(nc*nc);
    D_iT.resize(nc);
}

double Mix::ac(int i) {
    return omegaA * std::pow(Tc[i], 2) / Pc[i];
}

std::vector<double> Mix::ac() {
    a_c = std::vector<double>(nc);
    for (int i = 0; i < nc; i++)
    {
        a_c[i] = this->ac(i);
    }
    return a_c;
}

double Mix::ai(double T, int i) {
    return a_c[i] * this->alpha(T, i);
}
std::vector<double> Mix::ai(double T) {
    for (int i = 0; i < nc; i++)
    {
        a_i[i] = this->ai(T, i);
    }
    return a_i;
}
double Mix::aij(int i, int j)
{
    return std::sqrt(a_i[i] * a_i[j]) * (1. - kij[i*nc + j]);
}
std::vector<double> Mix::aij(double T) {
    a_i = this->ai(T);

    for (int i = 0; i < nc; i++)
    {
        for (int j = i; j < nc; j++)
        {
            double a_ij_ = this->aij(i, j);
            a_ij[i*nc + j] = a_ij_;
            a_ij[j*nc + i] = a_ij_;
        }
    }
    return a_ij;
}

double Mix::bi(int i) {
    return omegaB * Tc[i] / Pc[i];
}
std::vector<double> Mix::bi() {
    for (int i = 0; i < nc; i++)
    {
        b_i[i] = this->bi(i);
    }
    return b_i;
}
double Mix::bij(int i, int j)
{
    return 0.5 * (b_i[i] + c_i[i] + b_i[j] + c_i[j]) * (1. - 0.);
}
std::vector<double> Mix::bij() {
    b_i = this->bi();
    c_i = this->ci();

    for (int i = 0; i < nc; i++)
    {
        for (int j = i; j < nc; j++)
        {
            double b_ij_ = this->bij(i, j);
            b_ij[i*nc + j] = b_ij_;
            b_ij[j*nc + i] = b_ij_;
        }
    }
    return b_ij;
}
double Mix::ci(int i)
{
    // Volume shift
    if (this->volume_shift)
    {
        // Peneloux (1982) volume shift parameter c -> v_true = v_eos + c
        if (this->is_srk)
        {
            return Tc[i] / Pc[i] * (0.1156 - 0.4077 * Z_ra[i]);
        }
        else
        {
            return Tc[i] / Pc[i] * (0.1154 - 0.4406 * Z_ra[i]);
        }
    }
    else
    {
        return 0.;
    }
}
std::vector<double> Mix::ci() {
    for (int i = 0; i < nc; i++)
    {
        c_i[i] = this->ci(i);
    }
    return c_i;
}

double Mix::alpha(double T, int i) {
    return std::pow(1. + kappa[i] * (1. - std::sqrt(T / Tc[i])), 2);
}
double Mix::dalpha_dT(double T, int i) {
    // 
    // double d1 = 1. + kappa[i] * (1. - std::sqrt(T / Tc[i]));
    // double d1_dT = -0.5 * kappa[i] / std::sqrt(T * Tc[i]);
    return 2 * (1. + kappa[i] * (1. - std::sqrt(T / Tc[i]))) * -0.5 * kappa[i] / std::sqrt(T * Tc[i]);
}
double Mix::d2alpha_dT2(double T, int i) {
    // 
    double d1 = 1. + kappa[i] * (1. - std::sqrt(T / Tc[i]));
    double d1_dT = -0.5 * kappa[i] / std::sqrt(T * Tc[i]);
    double d2 = -0.5 * kappa[i] / std::sqrt(T * Tc[i]);
    double d2_dT = 0.25 * kappa[i] / std::pow(T, 1.5) / std::sqrt(Tc[i]);
    return 2 * (d1_dT * d2 + d1 * d2_dT);
    // return 2 * -0.5 * kappa[i] / std::sqrt(T * Tc[i]) +
            // 2 * (1. + kappa[i] * (1. - std::sqrt(T / Tc[i]))) * 0.25 * kappa[i] * pow(T, -1.5) * pow(Tc[i], -0.5);
}
double Mix::dai_dT(double T, int i) {
    // ai = aci * alpha_i
    // dai/dT = aci * dalpha_i/dT
    return a_c[i] * this->dalpha_dT(T, i);
}
double Mix::d2ai_dT2(double T, int i) {
    // ai = aci * alpha_i
    // dai/dT = aci * dalpha_i/dT
    // d2ai/dT2 = aci * d2alpha_i/dT2
    return a_c[i] * this->d2alpha_dT2(T, i);
}
double Mix::daij_dT(double T, int i, int j) {
    // aij = sqrt(ai*aj) * (1 - kij)
    //     = sqrt(aci*acj) * sqrt(alpha_i) * sqrt(alpha_j) * (1 - kij)
    // daijdT = sqrt(aci*acj) * (1-kij) * (0.5 / sqrt(alpha(i)) * dalphadT(i) * sqrt(alpha(j))
    //                                  + 0.5 / sqrt(alpha(j)) * dalphadT(j) * sqrt(alpha(i)))
    double alpha_i = this->alpha(T, i);
    double alpha_j = this->alpha(T, j);
    double dalphaidT = this->dalpha_dT(T, i);
    double dalphajdT = this->dalpha_dT(T, j);
    double sqrt_ai = std::sqrt(alpha_i);
    double sqrt_aj = std::sqrt(alpha_j);
    
    // double dkijdT = dkij_dT(T, i, j);
    return std::sqrt(a_c[i]*a_c[j]) * (1.-kij[i*nc+j]) * (0.5 / sqrt_ai * dalphaidT * sqrt_aj
                                                        + 0.5 / sqrt_aj * dalphajdT * sqrt_ai);
}
double Mix::d2aij_dT2(double T, int i, int j) {
    // aij = sqrt(ai*aj) * (1 - kij)
    //     = sqrt(aci*acj) * sqrt(alpha_i) * sqrt(alpha_j) * (1 - kij)
    // daijdT = sqrt(aci*acj) * (1-kij) * (0.5 / sqrt(alpha(i)) * dalphadT(i) * sqrt(alpha(j))
    //                                   + 0.5 / sqrt(alpha(j)) * dalphadT(j) * sqrt(alpha(i)))
    // d2aijdT2 = sqrt(aci*acj) * (1-kij) * (-0.25 * alpha(i)^-1.5 * dalphadT(i)^2 * sqrt(alpha(j))
    //                                      + 0.5 / sqrt(alpha(i)) * d2alphadT2(i) * sqrt(alpha(j))
    //                                      + 0.5 / sqrt(alpha(i)) * dalphadT(i) * 0.5 / sqrt(alpha(j)) * dalphadT(j)
    //                                      + ...
    double alpha_i = this->alpha(T, i);
    double alpha_j = this->alpha(T, j);
    double dalphaidT = this->dalpha_dT(T, i);
    double dalphajdT = this->dalpha_dT(T, j);
    double d2alphaidT2 = this->d2alpha_dT2(T, i);
    double d2alphajdT2 = this->d2alpha_dT2(T, j);
    double sqrt_ai = std::sqrt(alpha_i);
    double sqrt_aj = std::sqrt(alpha_j);
    
    // double dkijdT = this->dkij_dT(T, i, j);
    // double d2kijdT2 = this->d2kij_dT2(T, i, j);
    return std::sqrt(a_c[i] * a_c[j]) * (-0.25 * std::pow(alpha_i, -1.5) * std::pow(dalphaidT, 2) * sqrt_aj
                                        + 0.5 / sqrt_ai * d2alphaidT2 * sqrt_aj
                                        + 0.5 / sqrt_ai * dalphaidT * 0.5 / sqrt_aj * dalphajdT
                                        -0.25 * std::pow(alpha_j, -1.5) * std::pow(dalphajdT, 2) * sqrt_ai
                                        + 0.5 / sqrt_aj * d2alphajdT2 * sqrt_ai
                                        + 0.5 / sqrt_aj * dalphajdT * 0.5 / sqrt_ai * dalphaidT) * (1.-kij[i*nc+j]) ;
}

double Mix::B(std::vector<double>::iterator n_it) {
    double nB = 0.;
    for (int i = 0; i < nc; i++)
    {
        double ni = *(n_it + i);
        nB += ni * ni * b_ij[i*nc + i];
        for (int j = i+1; j < nc; j++) 
        {
            double nj = *(n_it + j);
            nB += 2 * ni * nj * b_ij[i*nc + j];
        }
    }
    
    return nB/N;
}

double Mix::D(std::vector<double>::iterator n_it) {
    double DD = 0.;
    for (int i = 0; i < nc; i++)
    {
        double ni = *(n_it + i);
        DD += ni * ni * a_ij[i*nc + i];
        for (int j = i+1; j < nc; j++) 
        {
            double nj = *(n_it + j);
            DD += 2 * ni * nj * a_ij[i*nc + j];
        }
    }
    return DD;
}

void Mix::zeroth_order(std::vector<double>::iterator n_it) {
    // Calculate zeroth order mixing rule parameters: B, D
    N = std::accumulate(n_it, n_it + nc, 0.);

    B_ = this->B(n_it);
    D_ = this->D(n_it);

    return;
}
void Mix::first_order(std::vector<double>::iterator n_it) {
    // Calculate first order mixing rule parameters: (B, D), B_i, D_i
    for (int i = 0; i < nc; i++)
    {
        double B_ii = 0.;
        double D_ii = 0.;
        for (int j = 0; j < nc; j++)
        {
            double nj = *(n_it + j);
            B_ii += 2 * nj * b_ij[i*nc + j];
            D_ii += 2 * nj * a_ij[i*nc + j];
        }
        B_i[i] = (B_ii - B_)/N;
        D_i[i] = D_ii;
    }
    return;
}
void Mix::second_order(double T, std::vector<double>::iterator n_it) {
    // Calculate second order mixing rule parameters: 
    D_T = 0.;
    D_TT = 0.;

    for (int i = 0; i < nc; i++)
    {
        double ni = *(n_it + i);
        D_iT[i] = 0;
        for (int j = 0; j < nc; j++)
        {
            double nj = *(n_it + j);
            B_ij[i*nc + j] = (2 * b_ij[i*nc + j] - B_i[i] - B_i[j])/N;
            D_ij[i*nc + j] = 2 * a_ij[i*nc + j];
            D_iT[i] += 2 * nj * this->daij_dT(T, i, j);
            D_TT += ni * nj * this->d2aij_dT2(T, i, j);
        }
        D_T += 0.5 * ni * D_iT[i];
    }
    return;
}
