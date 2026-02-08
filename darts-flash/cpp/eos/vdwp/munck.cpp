#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <unordered_map>

#include "dartsflash/eos/vdwp/munck.hpp"

namespace munck {
    double T_0 = 273.15;
    double R = 8.3145;

    // Langmuir constants (Munck [1988])
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> A_km = {
        {"sI", {{"C1", {0.7228e-3, 23.35e-3}}, {"C2", {0., 3.039e-3}}, {"C3", {0., 0.}}, {"iC4", {0., 0.}}, {"nC4", {0., 0.}}, {"N2", {1.617e-3, 6.078e-3}}, {"CO2", {0.2474e-3, 42.46e-3}}, {"H2S", {0.025e-3, 16.34e-3}}}},
        {"sII", {{"C1", {0.2207e-3, 100.e-3}}, {"C2", {0., 240.e-3}}, {"C3", {0., 5.455e-3}}, {"iC4", {0., 189.3e-3}}, {"nC4", {0., 30.51e-3}}, {"N2", {0.1742e-3, 18.e-3}}, {"CO2", {0.0845e-3, 851.e-3}}, {"H2S", {0.0298e-3, 87.2e-3}}}},
    };
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> B_km = {
        {"sI", {{"C1", {3187., 2653.}}, {"C2", {0., 3861.}}, {"C3", {0., 0.}}, {"iC4", {0., 0.}}, {"nC4", {0., 0.}}, {"N2", {2905., 2431.}}, {"CO2", {3410., 2813.}}, {"H2S", {4568., 3737.}}}},
        {"sII", {{"C1", {3453., 1916.}}, {"C2", {0., 2967.}}, {"C3", {0., 4638.}}, {"iC4", {0., 3800.}}, {"nC4", {0., 3699.}}, {"N2", {3082., 1728.}}, {"CO2", {3615., 2025.}}, {"H2S", {4878., 2633.}}}},
    };

    // Physical constants in liquid/ice reference states
    std::unordered_map<std::string, std::unordered_map<std::string, double>> dmu0 = { // Δmu_0 [J/mol]
        {"sI", {{"W", 1264.}, {"I", 0.}}}, {"sII", {{"W", 883.}, {"I", 0.}}},
    };
    std::unordered_map<std::string, std::unordered_map<std::string, double>> dH0 = { // ΔH_0 [J/mol]
        {"sI", {{"W", -4858.}, {"I", 1151.}}}, {"sII", {{"W", -5201.}, {"I", 808.}}},
    };
    std::unordered_map<std::string, std::unordered_map<std::string, double>> dV0 = { // ΔV_0 [cm3/mol]
        {"sI", {{"W", 4.6}, {"I", 3.0}}}, {"sII", {{"W", 5.0}, {"I", 3.4}}},
    };
    std::unordered_map<std::string, std::unordered_map<std::string, double>> dCp = { // ΔCp_0 [J/mol.K]
        {"sI", {{"W", 39.16}, {"I", 0.}}}, {"sII", {{"W", 39.16}, {"I", 0.}}},
    };

    double HB::f(double T) 
    {
        double h_beta = dH0[phase][ref_phase] + dCp[phase][ref_phase] * (T - munck::T_0);
        return h_beta / (R * std::pow(T, 2)); // molar enthalpy of empty hydrate lattice
    }
    double HB::F(double T) 
    {
        double R_inv = 1./R;
        double H_beta = - dH0[phase][ref_phase] * R_inv * (1./T-1./munck::T_0) 
                        + dCp[phase][ref_phase] * R_inv * (std::log(T)-std::log(munck::T_0) + munck::T_0/T - 1.);
        return H_beta;
    }
    double HB::dFdT(double T) 
    {
        return this->f(T);
    }
    double HB::d2FdT2(double T) 
    {
        double h_beta = dH0[phase][ref_phase] + dCp[phase][ref_phase] * (T - munck::T_0);
        double dh_beta = dCp[phase][ref_phase];
        return dh_beta / (R * std::pow(T, 2)) - 2. * h_beta / (R * std::pow(T, 3)); // molar enthalpy of empty hydrate lattice
    }
    int HB::test_derivatives(double T, double tol)
    {
        int error_output = 0;
        double dF = this->dFdT(T);
        double d2F = this->d2FdT2(T);

        double dT = 1e-5;
        double F_ = this->F(T-dT);
        double F1 = this->F(T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(T-dT);
        double dF1 = this->dFdT(T+dT);
        double d2F_num = (dF1-dF_)/(2*dT);

        if (std::fabs(dF_num - dF) > tol) { print("Munck HB dF/dT", {dF_num, dF}); error_output++; }
        if (std::fabs(d2F_num - d2F) > tol) { print("Munck HB d2F/dT2", {d2F_num, d2F}); error_output++; }
        
        return error_output;
    }

    double VB::f(double p, double T) 
    {
        (void) p;
        return dV0[phase][ref_phase] / (R * 0.5*(T + 273.15)); // molar enthalpy of empty hydrate lattice
    }
    double VB::dfdT(double p, double T) 
    {
        (void) p;
        return -dV0[phase][ref_phase] / (R * 0.5* std::pow(T + 273.15, 2)); // molar enthalpy of empty hydrate lattice
    }
    double VB::F(double p, double T) 
    {
		return dV0[phase][ref_phase] * p / (R * 0.5*(T + 273.15)); // molar enthalpy of empty hydrate lattice, p_0 = 0
    }
    double VB::dFdP(double p, double T) 
    {
        return this->f(p, T);
    }
    double VB::dFdT(double p, double T) 
    {
        return -dV0[phase][ref_phase] * p / (R * 0.5*std::pow(T + 273.15, 2)); // molar enthalpy of empty hydrate lattice
    }
    double VB::d2FdPdT(double p, double T) 
    {
        return this->dfdT(p, T);
    }
    double VB::d2FdT2(double p, double T) 
    {
        return 2. * dV0[phase][ref_phase] * p / (R * 0.5 * std::pow(T + 273.15, 3)); // molar enthalpy of empty hydrate lattice
    }
    int VB::test_derivatives(double p, double T, double tol)
    {
        int error_output = 0;
        double df = this->dfdT(p, T);
        double dF = this->dFdT(p, T);
        double d2F = this->d2FdT2(p, T);

        double dT = 1e-5;
        double f_ = this->f(p, T-dT);
        double f1 = this->f(p, T+dT);
        double df_num = (f1-f_)/(2*dT);
        double F_ = this->F(p, T-dT);
        double F1 = this->F(p, T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(p, T-dT);
        double dF1 = this->dFdT(p, T+dT);
        double d2F_num = (dF1-dF_)/(2*dT);

        if (std::fabs(df_num - df) > tol) { print("Munck VB df/dT", {df_num, df}); error_output++; }
        if (std::fabs(dF_num - dF) > tol) { print("Munck VB dF/dT", {dF_num, dF}); error_output++; }
        if (std::fabs(d2F_num - d2F) > tol) { print("Munck VB d2F/dT2", {d2F_num, d2F}); error_output++; }
        
        return error_output;
    }
} // namespace munck

Munck::Munck(CompData& comp_data, std::string hydrate_type) : VdWP(comp_data, hydrate_type) { }

void Munck::init_PT(double p_, double T_)
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

        // Calculate difference in chemical potential between empty hydrate lattice and reference phase (W or I)
        ref_phase = (T > 273.15) ? "W" : "I";  // water or ice

        // Chemical potential at reference conditions
        dmu = munck::dmu0[phase][ref_phase]/(munck::R * munck::T_0);

        // Enthalpy contribution
        munck::HB hh = munck::HB(phase, ref_phase);
        dH = hh.F(T);
        
        // Volume contribution
        munck::VB vv = munck::VB(phase, ref_phase);
        dV = vv.F(p, T);
    }
}

void Munck::init_VT(double, double)
{
	std::cout << "No implementation of volume-based calculations exists for Munck, aborting.\n";
	exit(1);
}

double Munck::V(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Molar volume of hydrate
    (void) p_;
    (void) n_;
    (void) start_idx;
    (void) pt;
    ref_phase = (T_ > 273.15) ? "W" : "I";  // water or ice
    return munck::dV0[phase][ref_phase];
}

double Munck::fw(std::vector<double>& fi)
{
    // Fugacity of water in hydrate following Munck (1988)

    // Contribution of cage occupancy to total energy of hydrate
    this->f = fi;
    double dmu_H = this->calc_dmuH();

    f[water_index] = std::exp(dmu - dH + dV + dmu_H);
    return f[water_index];
}
double Munck::dfw_dP(std::vector<double>& dfidP) 
{
    // dfw/dP = exp(dmu/RT) * d/dP (dmu/RT)
    munck::VB vv = munck::VB(phase, ref_phase);
    double ddV = vv.dFdP(p, T);
    double ddmuH = this->ddmuH_dP(dfidP);
    dfidP[water_index] = f[water_index] * (ddV + ddmuH);
    return dfidP[water_index];
}
double Munck::dfw_dT(std::vector<double>& dfidT)
{
    // dfw/dT = exp(dmu/RT) * d/dP (dmu/RT)
    munck::HB hh = munck::HB(phase, ref_phase);
    munck::VB vv = munck::VB(phase, ref_phase);

    double ddH = hh.dFdT(T);
    double ddV = vv.dFdT(p, T);
    double ddmuH = this->ddmuH_dT(dfidT);
    dfidT[water_index] = f[water_index] * (-ddH + ddV + ddmuH);
    return dfidT[water_index];
}
double Munck::d2fw_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) 
{
    // dfw/dP = exp(dmu/RT) * d/dP (dmu/RT)
    (void) dfidT;
    munck::VB vv = munck::VB(phase, ref_phase);
    double ddV = vv.dFdP(p, T);
    double d2dV = vv.d2FdPdT(p, T);
    double ddmuH = this->ddmuH_dP(dfidP);
    double d2dmuH = this->d2dmuH_dPdT(dfidP, dfidT, d2fidPdT);
    d2fidPdT[water_index] = dfidT[water_index] * (ddV + ddmuH) + f[water_index] * (d2dV + d2dmuH);
    return d2fidPdT[water_index];
}
double Munck::d2fw_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2)
{
    // dfw/dT = exp(dmu/RT) * d/dP (dmu/RT)
    munck::HB hh = munck::HB(phase, ref_phase);
    munck::VB vv = munck::VB(phase, ref_phase);

    double ddH = hh.dFdT(T);
    double d2dH = hh.d2FdT2(T);
    double ddV = vv.dFdT(p, T);
    double d2dV = vv.d2FdT2(p, T);
    double ddmuH = this->ddmuH_dT(dfidT);
    double d2dmuH = this->d2dmuH_dT2(dfidT, d2fidT2);
    d2fidT2[water_index] = dfidT[water_index] * (-ddH + ddV + ddmuH) + f[water_index] * (-d2dH + d2dV + d2dmuH);
    return d2fidT2[water_index];
}
std::vector<double> Munck::dfw_dxj(std::vector<double>& dfidxj)
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
std::vector<double> Munck::d2fw_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj)
{
    // d2fw/dTdxk = exp(dmu/RT) * d/dxk (dmu/RT)
    std::vector<double> d2fwdTdxk(nc);
    std::vector<double> ddmuHdxk = this->ddmuH_dxj(dfidxj);
    std::vector<double> d2dmuHdTdxk = this->d2dmuH_dTdxj(dfidT, dfidxj, d2fidTdxj);
    for (int k = 0; k < nc; k++)
    {
        d2fwdTdxk[k] = dfidT[water_index] * ddmuHdxk[k] + f[water_index] * d2dmuHdTdxk[k];
    }
    return d2fwdTdxk;
}

std::vector<double> Munck::calc_Ckm() 
{
    // Calculate Langmuir constant of each guest k in each cage m
    this->C_km = std::vector<double>(n_cages * nc, 0.);
    
    double invT = 1./T;
    double atm_to_bar = 1./1.01325;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            std::vector<double> Aki = munck::A_km[phase][this->compdata.components[k]];
            std::vector<double> Bki = munck::B_km[phase][this->compdata.components[k]];
            for (int m = 0; m < n_cages; m++)
            {
                C_km[nc*m + k] = Aki[m] * atm_to_bar * invT * std::exp(Bki[m] * invT);
            }
        }
    }
    return C_km;
}
std::vector<double> Munck::dCkm_dP()
{
    this->dCkmdP = std::vector<double>(n_cages*nc, 0.);
    return dCkmdP;
}
std::vector<double> Munck::dCkm_dT()
{
    // Calculate derivative of Langmuir constants w.r.t. T
    this->dCkmdT = std::vector<double>(n_cages * nc, 0.);
    
    double invT = 1./T;
    double atm_to_bar = 1./1.01325;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            std::vector<double> Aki = munck::A_km[phase][this->compdata.components[k]];
            std::vector<double> Bki = munck::B_km[phase][this->compdata.components[k]];
            for (int m = 0; m < n_cages; m++)
            {
                dCkmdT[nc*m + k] = Aki[m] * atm_to_bar * std::exp(Bki[m] * invT) * (-std::pow(invT, 2) - Bki[m] * std::pow(invT, 3));
            }
        }
    }
    return dCkmdT;
}
std::vector<double> Munck::d2Ckm_dPdT()
{
    this->d2CkmdPdT = std::vector<double>(n_cages*nc, 0.);
    return d2CkmdPdT;
}
std::vector<double> Munck::d2Ckm_dT2()
{
    // Calculate second derivative of Langmuir constants w.r.t. T
    this->d2CkmdT2 = std::vector<double>(n_cages * nc, 0.);
    
    double invT = 1./T;
    double atm_to_bar = 1./1.01325;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            std::vector<double> Aki = munck::A_km[phase][this->compdata.components[k]];
            std::vector<double> Bki = munck::B_km[phase][this->compdata.components[k]];
            for (int m = 0; m < n_cages; m++)
            {
                d2CkmdT2[nc*m + k] = Aki[m] * atm_to_bar * std::exp(Bki[m] * invT) 
                                    * (2. * std::pow(invT, 3) + 4. * Bki[m] * std::pow(invT, 4) + std::pow(Bki[m], 2) * std::pow(invT, 5));
            }
        }
    }
    return d2CkmdT2;
}

int Munck::derivatives_test(double p_, double T_, std::vector<double>& x_, double tol, bool verbose)
{
    int error_output = VdWP::derivatives_test(p_, T_, x_, tol, verbose);

    munck::HB hh = munck::HB(phase, ref_phase);
    munck::VB vv = munck::VB(phase, ref_phase);

    error_output += hh.test_derivatives(T_, tol);
    error_output += vv.test_derivatives(p_, T_, tol);

    return error_output;
}
