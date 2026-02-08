#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <unordered_map>

#include "dartsflash/maths/maths.hpp"
#include "dartsflash/eos/vdwp/klauda.hpp"

namespace klauda {
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Ps_ABC = {
        {"sI", {
            {"C1", {4.64130, -5366.1, -8.332e-3}},
            {"C2", {4.76152, -5419.38, -9.774e-3}},
            {"C3", {0., 0., 0.}},
            {"N2", {4.7328, -5400.61, -9.5e-3}}, 
            {"H2", {4.06453, -5869.39, -8.424e-3}}, 
            {"CO2", {4.59071, -5345.28, -7.522e-3}}}, },
        {"sII", {
            {"C1", {4.60893, -5397.85, -7.776e-3}},
            {"C2", {4.71591, -5492.66, -8.997e-3}},
            {"C3", {4.70796, -5449.89, -9.233e-3}},
            {"N2", {4.69009, -5354.38, -9.346e-3}}, 
            {"H2", {4.69736, -5458.15, -9.235e-3}}, 
            {"CO2", {4.84222, -5621.08, -9.199e-3}}}, },
    };

    std::unordered_map<std::string, std::vector<int>> zn = {{"sI", {8, 12, 8, 4, 8, 4}}, {"sII", {2, 6, 12, 12, 12, 4}}, }; // #water in layers
    std::unordered_map<std::string, std::vector<int>> n_shells = {{"sI", {2, 4}}, {"sII", {3, 3}}, };
    std::unordered_map<std::string, std::unordered_map<char, std::vector<double>>> Rn = {
        {"sI", {{'S', {3.906e-10, 6.593e-10, 8.086e-10}}, 
                {'L', {4.326e-10, 7.078e-10, 8.285e-10}}}},
        {"sII", {{'S', {3.902e-10, 6.667e-10, 8.079e-10}},
                {'L', {4.682e-10, 7.464e-10, 8.782e-10}}}},  // radius of layers S0, S1, S2, L0, L1, L2 (sII)
    };

    std::unordered_map<std::string, double> ai = {{"CO2", 0.677e-10}, {"N2", 0.341e-10},  {"H2S", 0.3508e-10}, {"C1", 0.28e-10}, {"C2", 0.574e-10}, {"C3", 0.6502e-10}, {"iC4", 0.859e-10}, {"H2O", 0.}, }; // hard core radius
    std::unordered_map<std::string, double> sigma = {{"CO2", 3.335e-10}, {"N2", 3.469e-10}, {"H2S", 3.607e-10}, {"C1", 3.505e-10}, {"C2", 4.022e-10}, {"C3", 4.519e-10}, {"iC4", 4.746e-10}, {"H2O", 3.564e-10}, }; // soft core radius
    std::unordered_map<std::string, double> eik = {{"CO2", 513.85}, {"N2", 142.1}, {"H2S", 459.6}, {"C1", 232.2}, {"C2", 404.3}, {"C3", 493.71}, {"iC4", 628.6}, {"nC4", 197.254}, {"H2O", 102.134}, }; // potential well depth/k

    double Kihara::f(double X, std::string component) {
        // hydrate cage cell potential w(r) = omega(r) [eq. 3.43a]
        double w{ 0. };
        // hard core, soft core radius and potential well depth of guest molecule
        double ai_ = ai[component]; double sigmai = sigma[component]; double eik_ = eik[component];

        // Loop over shell layers of cage m [eq. 4.45]
        for (int l = 0; l < n_shells[phase][cage_index]; l++) 
        {
            double Rn_ = Rn[phase][R1_index + l];
            int zn_ = zn[phase][R1_index + l];
            double delta10 = 1. / 10 * (std::pow((1. - X / Rn_ - ai_ / Rn_), -10.) - std::pow((1. + X / Rn_ - ai_ / Rn_), -10.));
            double delta11 = 1. / 11 * (std::pow((1. - X / Rn_ - ai_ / Rn_), -11.) - std::pow((1. + X / Rn_ - ai_ / Rn_), -11.));
            double delta4 = 1. / 4 * (std::pow((1. - X / Rn_ - ai_ / Rn_), -4.) - std::pow((1. + X / Rn_ - ai_ / Rn_), -4.));
            double delta5 = 1. / 5 * (std::pow((1. - X / Rn_ - ai_ / Rn_), -5.) - std::pow((1. + X / Rn_ - ai_ / Rn_), -5.));
            w += 2.0 * zn_ * (std::pow(sigmai, 12.) / (std::pow(Rn_, 11.) * X) * (delta10 + ai_ / Rn_ * delta11) -
                            std::pow(sigmai, 6.) / (std::pow(Rn_, 5.) * X) * (delta4 + ai_ / Rn_ * delta5));    
        }
        // term in integral for Langmuir constant C_im [eq. 3.42]
        return std::exp(-eik_ / TT * w) * std::pow(X, 2);
    }

    double Kihara::F(double p, double T, std::string component) {
        this->pp = p;
        this->TT = T;

        // integrals solved numerically with simpson's rule
        double s = 0.;
        int steps = 20;
        double h = (R1-R0)/steps;  // interval
    	double X;
    
    	// hi
        for (int i = 0; i < steps; i++) 
        {
            X = R0 + h*i;
            double hix = this->f(X, component);
            double hixh2 = this->f(X+h*0.5, component);
            double hixh = this->f(X+h, component);
            s += h*((hix + 4*hixh2 + hixh) / 6);
            if (hixh < 1e-200) { break; } // otherwise, integral becomes too large
        }
    	return s;
    }

    double Kihara::dFdP(double p, double T, std::string component) {
        (void) p; (void) T; (void) component;
        // double dP = 1e-5;
		// double F1 = this->F(p+dP, T, component);
		// double F0 = this->F(p, T, component);
		// return (F1-F0)/dP;
        return 0.;
    }

    double Kihara::dFdT(double p, double T, std::string component) {
        double dT = 1e-6;
		double F1 = this->F(p, T+dT, component);
		double F0 = this->F(p, T, component);
		return (F1-F0)/dT;
    }

} // namespace klauda

Klauda::Klauda(CompData& comp_data, std::string hydrate_type) : VdWP(comp_data, hydrate_type) { }

void Klauda::init_PT(double p_, double T_)
{
    // Fugacity of water in hydrate phase following modified VdW-P (Ballard, 2002)
    // Initializes all composition-independent parameters:
    // - Ideal gas Gibbs energy of water [eq. 3.1-3.4]
    // - Gibbs energy of water in empty hydrate lattice [eq. 3.47]
    // Each time calculating fugacity of water in hydrate, evaluate:
    // - Contribution of cage occupancy to total energy of hydrate [eq. 3.44]
    // - Activity of water in the hydrate phase [eq. 4.38]
    // - Chemical potential of water in hydrate [eq. 4.35]
    if (p_ != p || T_ != T)
    {
        this->p = p_;
        this->T = T_;

        // Calculate Langmuir constant of each guest k in each cage m
        C_km = this->calc_Ckm();

        // 
        double V_wB;
        double N_wB = 1.;
        if (phase == "sI")
        {
            V_wB = (11.835 + 2.217e-5 * T + 2.242e-6 * std::pow(T, 2)) * M_NA * 1e-30 / N_wB 
                    - 8.006e-10 * p + 5.448e-12 * std::pow(0.1*p, 2);
        }
        else
        {
            V_wB = (17.13 + 2.249e-4 * T + 2.013e-6 * std::pow(T, 2) + 1.009e-9 * std::pow(T, 3)) * M_NA * 1e-30 / N_wB 
                    - 8.006e-10 * p + 5.448e-12 * std::pow(0.1*p, 2);
        }
    }
}

void Klauda::solve_PT(std::vector<double>::iterator n_it, bool second_order)
{
    // Calculate mixture parameters with composition n
    (void) second_order;
    this->n_iterator = n_it;
    double nT_inv = 1./std::accumulate(n_it, n_it + this->nc, 0.);
    std::transform(n_it, n_it + this->nc, this->x.begin(),
                   [&nT_inv](double element) { return element *= nT_inv; });

    // Calculate fugacities
    f = this->fi();
    f[water_index] = this->fw(f);

    return;
}

double Klauda::fw(std::vector<double>& fi)
{
    // Fugacity of water in hydrate following Klauda (2000, 2003)

    // Contribution of cage occupancy to total energy of hydrate
    this->f = fi;

    // Calculate fugacity of water in hypothetical empty hydrate lattice f_wβ
    double A_mix{ 0. }, B_mix{ 0. }, D_mix{ 0. };
    for (int j = 0; j < nc; j++)
    {
        if (j != water_index)
        {
            A_mix += x[j] * klauda::Ps_ABC[phase][components[j]][0];
            B_mix += x[j] * klauda::Ps_ABC[phase][components[j]][1];
            D_mix += x[j] * klauda::Ps_ABC[phase][components[j]][2];
        }
    }
    double Ps_wB = std::exp(A_mix * std::log(T) + B_mix/T + klauda::C + D_mix * T);
    double phis_wB = 1.;  // taken to be unity, because vapour pressure of the water phases is low (Klauda, 2000)

    double f_wB = Ps_wB * phis_wB * std::exp(V_wB * (p - Ps_wB)/(klauda::R*T));

    return f_wB * std::exp(-this->calc_dmuH());
}

std::vector<double> Klauda::dfw_dxj(std::vector<double>& dfidxj)
{
    // dfw/dxk = exp(dmu/RT) * d/dxk (dmu/RT)
    std::vector<double> dfwdxk(nc);
    std::vector<double> ddmuHdxk = this->ddmuH_dxj(dfidxj);
    for (int k = 0; k < nc; k++)
    {
        dfwdxk[k] = f[water_index] * ddmuHdxk[k];
    }
    return dfwdxk;
}

double Klauda::dfw_dP(std::vector<double>& dfidP) 
{
    // dfw/dP = exp(dmu/RT) * d/dP (dmu/RT)
    munck::VB v = munck::VB(ref_phase);
    double ddV = v.dFdP(p, T, phase);
    double ddmuH = this->ddmuH_dP(dfidP);
    return f[water_index] * (ddV + ddmuH);
}

double Klauda::dfw_dT(std::vector<double>& dfidT)
{
    // dfw/dT = exp(dmu/RT) * d/dP (dmu/RT)
    munck::HB h = munck::HB(ref_phase);
    munck::VB v = munck::VB(ref_phase);

    double ddH = h.dFdT(p, T, phase);
    double ddV = v.dFdT(p, T, phase);
    double ddmuH = this->ddmuH_dT(dfidT);
    return f[water_index] * (-ddH + ddV + ddmuH);
}

std::vector<double> Klauda::calc_Ckm() 
{
    // Calculate Langmuir constant of each guest k in each cage m
    std::vector<double> Ckm(n_cages * nc);
    
    double invT = 1./T;
    double atm_to_bar = 1./1.01325;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            std::vector<double> Aki = munck::A_km[phase][components[k]];
            std::vector<double> Bki = munck::B_km[phase][components[k]];
            for (int m = 0; m < n_cages; m++)
            {
                Ckm[nc*m + k] = Aki[m] * atm_to_bar * invT * std::exp(Bki[m] * invT);
            }
        }
    }
    return Ckm;
}

std::vector<double> Klauda::dCkm_dP()
{
    return std::vector<double>(n_cages*nc, 0.);
}

std::vector<double> Klauda::dCkm_dT()
{
    // Calculate derivative of Langmuir constants w.r.t. T
    std::vector<double> dCkmdT_(n_cages * nc);
    
    double invT = 1./T;
    double atm_to_bar = 1./1.01325;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            std::vector<double> Aki = munck::A_km[phase][components[k]];
            std::vector<double> Bki = munck::B_km[phase][components[k]];
            for (int m = 0; m < n_cages; m++)
            {
                dCkmdT_[nc*m + k] = Aki[m] * invT * atm_to_bar * std::exp(Bki[m] * invT) * (-invT - Bki[m]*std::pow(invT, 2));
            }
        }
    }
    return dCkmdT_;
}
