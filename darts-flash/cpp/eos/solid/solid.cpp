#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/maths/maths.hpp"
#include "dartsflash/eos/solid/solid.hpp"

namespace solid_par {
    double T_0{ 298.15 }; // reference temperature
    double P_0{ 1. }; // reference pressure [bar]

    // gibbs energy of ideal gas at p0, T0
    std::unordered_map<std::string, double> gi0 = {
        {"H2O", -228700}, {"CO2", -394600}, {"N2", 0}, {"H2S", -33100}, 
        {"C1", -50830}, {"C2", -32900}, {"C3", -23500}, {"iC4", -20900}, {"nC4", -17200}, {"iC5", -15229}, {"nC5", -8370}, {"nC6", -290}, {"nC7", 8120},
        {"NaCl", -384138.00}, {"CaCl2", -748100.00}, {"KCl", -409140.00} 
    };
    // molar enthalpy of ideal gas at p0, T0
    std::unordered_map<std::string, double> hi0 = {
        {"H2O", -242000}, {"CO2", -393800}, {"N2", 0}, {"H2S", -20200}, 
        {"C1", -74900}, {"C2", -84720}, {"C3", -103900}, {"iC4", -134600}, {"nC4", -126200}, {"iC5", -165976}, {"nC5", -146500}, {"nC6", -167300}, {"nC7", -187900} ,
        {"NaCl", -411153.00}, {"CaCl2", -795800.00}, {"KCl", -436747.00}
    };

    std::unordered_map<std::string, std::string> pure_comp = {{"Ice", "H2O"}, {"NaCl", "NaCl"}, {"CaCl2", "CaCl2"}, {"KCl", "KCl"}};

    std::unordered_map<std::string, double> gp0 = {{"Ice", -236539.24}, {"NaCl", -384138.00}, {"CaCl2", -748100.00}, {"KCl", -409140.00}};
    std::unordered_map<std::string, double> hp0 = {{"Ice", -292714.43}, {"NaCl", -411153.00}, {"CaCl2", -795800.00}, {"KCl", -436747.00}};
    std::unordered_map<std::string, double> v0 = {{"Ice", 19.7254}, {"NaCl", 26.9880}, {"CaCl2", 51.5270}, {"KCl", 37.5760}};
    std::unordered_map<std::string, double> kappa = {{"Ice", 1.3357E-5}, {"NaCl", 2.0000E-6}, {"CaCl2", 2.0000E-6}, {"KCl", 2.0000E-6}};

    std::unordered_map<std::string, std::vector<double>> cp = {
        {"Ice", {0.735409713, 1.4180551e-2, -1.72746e-5, 63.5104e-9}},
        {"NaCl", {5.526, 0.1963e-2, 0., 0.}},
        {"CaCl2", {8.646, 0.153e-2, 0., 0.}},
        {"KCl", {6.17, 0., 0., 0.}}
    };
    std::unordered_map<std::string, std::vector<double>> alpha = {
        {"Ice", {1.522300E-4, 1.660000E-8, 0.}},
        {"NaCl", {2.000000E-5, 0., 0.}},
        {"CaCl2", {2.000000E-5, 0., 0.}},
        {"KCl", {2.000000E-5, 0., 0.}}
    };

    IG::IG(std::string component_) : Integral(component_)
    {
        this->gi0 = solid_par::gi0[component_];
        this->hi0 = solid_par::hi0[component_];
        this->cpi = comp_data::cpi[component_];
    }
    double IG::H(double T)
    {
        // Integral of H(T)/RT^2 dT from T_0 to T
        return (-(this->hi0 / M_R
                - this->cpi[0] * solid_par::T_0 
                - 1. / 2 * this->cpi[1] * std::pow(solid_par::T_0, 2) 
                - 1. / 3 * this->cpi[2] * std::pow(solid_par::T_0, 3)
                - 1. / 4 * this->cpi[3] * std::pow(solid_par::T_0, 4)) * (1./T - 1./solid_par::T_0)
                + (this->cpi[0] * (std::log(T) - std::log(solid_par::T_0))
                + 1. / 2 * this->cpi[1] * (T - solid_par::T_0)
                + 1. / 6 * this->cpi[2] * (std::pow(T, 2) - std::pow(solid_par::T_0, 2))
                + 1. / 12 * this->cpi[3] * (std::pow(T, 3) - std::pow(solid_par::T_0, 3))));
    }
    double IG::dHdT(double T)
    {
        // Derivative of integral w.r.t. temperature
        return (this->hi0 / M_R + 
                this->cpi[0] * (T-solid_par::T_0) 
                + 1. / 2 * this->cpi[1] * (std::pow(T, 2)-std::pow(solid_par::T_0, 2)) 
                + 1. / 3 * this->cpi[2] * (std::pow(T, 3)-std::pow(solid_par::T_0, 3))
                + 1. / 4 * this->cpi[3] * (std::pow(T, 4)-std::pow(solid_par::T_0, 4))) / std::pow(T, 2);
    }
    double IG::d2HdT2(double T)
    {
        // Derivative of integral w.r.t. temperature
        return (this->cpi[0] + this->cpi[1] * T + this->cpi[2] * std::pow(T, 2) + this->cpi[3] * std::pow(T, 3)) / std::pow(T, 2) - 2. * this->dHdT(T) / T;
    }
    int IG::test_derivatives(double T, double tol, bool verbose)
    {
        int error_output = 0;
        double dF = this->dHdT(T);
        double d2F = this->d2HdT2(T);

        double d, dT{ 1e-5 };
        double F_ = this->H(T-dT);
        double F1 = this->H(T+dT);
        double dF1 = this->dHdT(T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double d2F_num = (dF1-dF)/dT;

        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid IG dH/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid IG d2H/dT2", {d2F, d2F_num, d}); error_output++; }

        return error_output;
    }

    double H::f(double T) 
    {
        // H/RT^2: molar enthalpy of pure phase [eq. 3.38], J/mol
        return (hp0[phase] / M_R
                + cp[phase][0] * (T-solid_par::T_0)
                + 1. / 2 * cp[phase][1] * (std::pow(T, 2)-std::pow(solid_par::T_0, 2))
                + 1. / 3 * cp[phase][2] * (std::pow(T, 3)-std::pow(solid_par::T_0, 3))
                + 1. / 4 * cp[phase][3] * (std::pow(T, 4)-std::pow(solid_par::T_0, 4))) / std::pow(T, 2);
    }
    double H::dfdT(double T)
    {
        // d(H/RT^2)/dT
        return (cp[phase][0] + cp[phase][1] * T + cp[phase][2] * std::pow(T, 2) + cp[phase][3] * std::pow(T, 3)) / std::pow(T, 2)
                - 2.*(hp0[phase] / M_R
                + cp[phase][0] * (T-solid_par::T_0) 
                + 1. / 2 * cp[phase][1] * (std::pow(T, 2)-std::pow(solid_par::T_0, 2)) 
                + 1. / 3 * cp[phase][2] * (std::pow(T, 3)-std::pow(solid_par::T_0, 3)) 
                + 1. / 4 * cp[phase][3] * (std::pow(T, 4)-std::pow(solid_par::T_0, 4))) / std::pow(T, 3);
    }
    double H::F(double T) 
    {
        // Integral of H(T)/RT^2 dT from T_0 to T
        return -(hp0[phase] / M_R
                - cp[phase][0] * solid_par::T_0 
                - 1. / 2 * cp[phase][1] * std::pow(solid_par::T_0, 2) 
                - 1. / 3 * cp[phase][2] * std::pow(solid_par::T_0, 3)
                - 1. / 4 * cp[phase][3] * std::pow(solid_par::T_0, 4)) * (1./T - 1./solid_par::T_0)
                + cp[phase][0] * (std::log(T) - std::log(solid_par::T_0))
                + 1. / 2 * cp[phase][1] * (T - solid_par::T_0)
                + 1. / 6 * cp[phase][2] * (std::pow(T, 2) - std::pow(solid_par::T_0, 2))
                + 1. / 12 * cp[phase][3] * (std::pow(T, 3) - std::pow(solid_par::T_0, 3));
    }
    double H::dFdT(double T) 
    {
        // Derivative of integral w.r.t. T
        return this->f(T);
    }
    double H::d2FdT2(double T) 
    {
        // f(x) = h(T)/RT^2
        // df(x)/dT = dh(T)/dT/RT^2 - 2 h(T)/RT^3
        double dhdT = cp[phase][0] + cp[phase][1] * T + cp[phase][2] * std::pow(T, 2) + cp[phase][3] * std::pow(T, 3);
		
        return dhdT / std::pow(T, 2) - 2 * this->dFdT(T) / T;
    }
    int H::test_derivatives(double T, double tol, bool verbose)
    {
        int error_output = 0;
        double df = this->dfdT(T);
        double dF = this->dFdT(T);
        double d2F = this->d2FdT2(T);

        double d, dT{ 1e-5 };
        double f_ = this->f(T-dT);
        double f1 = this->f(T+dT);
        double df_num = (f1-f_)/(2*dT);
        double F_ = this->F(T-dT);
        double F1 = this->F(T+dT);
        double dF1 = this->dFdT(T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double d2F_num = (dF1-dF)/dT;

        d = std::log(std::fabs(df + 1e-15)) - std::log(std::fabs(df_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid H df/dT", {df, df_num, d}); error_output++; }
        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid H dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid H d2F/dT2", {d2F, d2F_num, d}); error_output++; }
        
        return error_output;
    }

    double V::f(double p, double T) 
    {
        // molar volume of pure phase [eq. 3.39], m3/mol
        double a = alpha[phase][0] * (T-T_0) 
                + alpha[phase][1] * std::pow((T-T_0), 2) 
                + alpha[phase][2] * std::pow((T-T_0), 3);
        double b = - kappa[phase] * (p-P_0);
        return v0[phase]*1e-6 * std::exp(a+b) / (M_R * 1e-5 * T);
    }
    double V::dfdP(double p, double T)
    {
        // dV/dP: derivative of molar volume of pure phase w.r.t. pressure [eq. 3.39], m3/mol
        double a = alpha[phase][0] * (T-T_0) 
                + alpha[phase][1] * std::pow((T-T_0), 2) 
                + alpha[phase][2] * std::pow((T-T_0), 3);
        double b = -kappa[phase] * (p-P_0);
        double dbdp = -kappa[phase];
        return v0[phase]*1e-6 * std::exp(a+b) * dbdp / (M_R * 1e-5 * T);
    }
    double V::dfdT(double p, double T)
    {
        // dV/dT: derivative of molar volume of pure phase w.r.t. temperature [eq. 3.39], m3/mol
        double a = alpha[phase][0] * (T-T_0) 
                + alpha[phase][1] * std::pow((T-T_0), 2) 
                + alpha[phase][2] * std::pow((T-T_0), 3);
        double dadT = alpha[phase][0] + 
                + 2. * alpha[phase][1] * (T-T_0)
                + 3. * alpha[phase][2] * std::pow((T-T_0), 2);
        double b = -kappa[phase] * (p-P_0);
        return v0[phase]*1e-6 * std::exp(a+b) * (dadT - 1./T) / (M_R * 1e-5 * T);
    }
    double V::F(double p, double T)
    {
        // Integral of V(T,P)/RT dP from P_0 to P
        // int e^cx dx = 1/c e^cx
        // int v0 exp(a + b*p)/RT dp = v0 exp(a)/RT * int exp(b*p) dp = v0 exp(a)/bRT * exp(b*p)
        double a = alpha[phase][0] * (T-T_0)
                + alpha[phase][1] * std::pow((T-T_0), 2)
                + alpha[phase][2] * std::pow((T-T_0), 3);
        double b = -kappa[phase];
        return v0[phase]*1e-6 * std::exp(a-b*P_0) / (M_R * 1e-5 * T) * (std::exp(b*p) / b - std::exp(b*P_0) / b);
    }
    double V::dFdP(double p, double T) 
    {
        // Derivative of integral w.r.t. pressure
        return this->f(p, T);
    }
    double V::dFdT(double p, double T) 
    {
        // Derivative of integral w.r.t. temperature
        double a = alpha[phase][0] * (T-T_0) 
                + alpha[phase][1] * std::pow(T-T_0, 2) 
                + alpha[phase][2] * std::pow(T-T_0, 3);
        double b = -kappa[phase];

        double da_dT = alpha[phase][0] 
                    + 2*alpha[phase][1] * (T-T_0) 
                    + 3*alpha[phase][2] * std::pow(T-T_0, 2);
        
        double d_dT = (T * da_dT * std::exp(a-b*P_0) - std::exp(a-b*P_0)) / std::pow(T, 2);
        
        return v0[phase]*1e-6 / (M_R*1e-5) * d_dT * (std::exp(b*p) / b - std::exp(b*P_0) / b);
    }
    double V::d2FdPdT(double p, double T)
    {
        // Second derivative of integral w.r.t. pressure and temperature
        return this->dfdT(p, T);
    }
    double V::d2FdT2(double p, double T) 
    {
        // Second derivative of integral w.r.t. temperature
        double a = alpha[phase][0] * (T-T_0) 
                + alpha[phase][1] * std::pow(T-T_0, 2) 
                + alpha[phase][2] * std::pow(T-T_0, 3);
        double b = -kappa[phase];

        double da_dT = alpha[phase][0]
                    + 2*alpha[phase][1] * (T-T_0)
                    + 3*alpha[phase][2] * std::pow(T-T_0, 2);
        double d2a_dT2 = 2*alpha[phase][1] + 6*alpha[phase][2]*(T-T_0);

        double d_dT = (T * da_dT * std::exp(a-b*P_0) - std::exp(a-b*P_0)) / std::pow(T, 2);
        double d2_dT2 = ((T * d2a_dT2 + T * std::pow(da_dT, 2)) * std::exp(a-b*P_0)) / std::pow(T, 2) - 2. * d_dT / T;
        
        return v0[phase]*1e-6 / (M_R*1e-5) * d2_dT2 * (std::exp(b*p) / b - std::exp(b*P_0) / b);
    }
    int V::test_derivatives(double p, double T, double tol, bool verbose)
    {
        int error_output = 0;
        double f_, f1, df;
        double F_, F1, dF1, dF, d2F;
        double df_num, dF_num, d2F_num;

        // Test derivatives w.r.t. pressure
        double d, dp{ 1e-5 };
        df = this->dfdP(p, T);
        dF = this->dFdP(p, T);
        f_ = this->f(p-dp, T);
        f1 = this->f(p+dp, T);
        df_num = (f1-f_)/(2*dp);
        F_ = this->F(p-dp, T);
        F1 = this->F(p+dp, T);
        dF_num = (F1-F_)/(2*dp);

        d = std::log(std::fabs(df + 1e-15)) - std::log(std::fabs(df_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid V df/dP", {df, df_num, d}); error_output++; }
        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid V dF/dP", {dF, dF_num, d}); error_output++; }

        // Test derivatives w.r.t. pressure
        double dT = 1e-5;
        df = this->dfdT(p, T);
        dF = this->dFdT(p, T);
        d2F = this->d2FdT2(p, T);
        f_ = this->f(p, T-dT);
        f1 = this->f(p, T+dT);
        df_num = (f1-f_)/(2*dT);
        F_ = this->F(p, T-dT);
        F1 = this->F(p, T+dT);
        dF1 = this->dFdT(p, T+dT);
        dF_num = (F1-F_)/(2*dT);
        d2F_num = (dF1-dF)/dT;

        d = std::log(std::fabs(df + 1e-15)) - std::log(std::fabs(df_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid V df/dT", {df, df_num, d}); error_output++; }
        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid V dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Solid V d2F/dT2", {d2F, d2F_num, d}); error_output++; }

        return error_output;
    }
}

PureSolid::PureSolid(CompData& comp_data, std::string phase_) : EoS(comp_data)
{
    this->phase = phase_;
    std::string pure_comp = solid_par::pure_comp[phase];
    this->pure_comp_idx = std::distance(compdata.components.begin(), std::find(compdata.components.begin(), compdata.components.end(), pure_comp));

    this->eos_range[this->pure_comp_idx] = std::vector<double>{1., 1.};
}

void PureSolid::init_PT(double p_, double T_) {
    if (p_ != p || T_ != T)
    {
        this->p = p_; this->T = T_;

        // ideal gas Gibbs energy
        solid_par::IG ig = solid_par::IG(solid_par::pure_comp[phase]);
        double gio = ig.G();  // gi0/RT0
        double hio = ig.H(T);  // eq. 3.3
        double gi0 = gio - hio;  // eq. 3.2

        // Gibbs energy of ions in aqueous phase
        solid_par::H hh = solid_par::H(phase);
        solid_par::V vv = solid_par::V(phase);
        double gs0 = solid_par::gp0[phase] / (M_R * solid_par::T_0);
        double hs = hh.F(T);  // integral of H(T)/RT^2 from T0 to T
        double vs = vv.F(p, T);  // integral of V(T,P)/RT from P0 to P
        double gs = gs0 - hs + vs;
        
        // Calculate fugacity coefficient
        lnfS = gs - gi0;
    }
}
void PureSolid::solve_PT(std::vector<double>::iterator n_it, bool second_order)
{
    std::copy(n_it, n_it + ns, this->n.begin());
    this->N = std::accumulate(n_it, n_it + this->ns, 0.);
    (void) second_order;
    return;
}

void PureSolid::init_VT(double, double)
{
	std::cout << "No implementation of volume-based calculations exists for PureSolid, aborting.\n";
	exit(1);
}
void PureSolid::solve_VT(std::vector<double>::iterator, bool)
{
	std::cout << "No implementation of volume-based calculations exists for PureSolid, aborting.\n";
	exit(1);
}

double PureSolid::P(double V_, double T_, std::vector<double>& n_)
{
    // Find pressure at given (T, V, n)
    this->p = 1.;

    // Newton loop to find root
    int it = 0;
    while (it < 10)
    {
        double res = this->V(p, T_, n_) - V_;
        double dres_dp = this->dV_dP(p, T_, n_);
        p -= res/dres_dp;

        if (std::fabs(res) < 1e-14)
        {
            break;
        }
        it++;
    }
    return p;
}

double PureSolid::V(double p_, double T_, std::vector<double>& n_)
{
    // Calculate volume at (P, T, n)
    (void) n_;
    solid_par::V vv = solid_par::V(phase);
    return vv.f(p_, T_) * (M_R * 1E-5 * T_);
}

double PureSolid::dV_dP(double p_, double T_, std::vector<double>& n_)
{
    // Calculate pressure derivative of volume at (P, T, n)
    (void) n_;
    solid_par::V vv = solid_par::V(phase);
    return vv.dfdP(p_, T_) * (M_R * 1E-5 * T_);
}

double PureSolid::dV_dT(double p_, double T_, std::vector<double>& n_)
{
    // Calculate temperature derivative of volume at (P, T, n)
    (void) n_;
    solid_par::V vv = solid_par::V(phase);
    return vv.dfdT(p_, T_) * (M_R * 1E-5 * T_);
}

double PureSolid::dV_dni(double p_, double T_, std::vector<double>& n_, int i)
{
    // Calculate temperature derivative of volume at (P, T, n)
    (void) p_;
    (void) T_;
    (void) n_;
    (void) i;
    return 0.;
}

double PureSolid::lnphii(int i) 
{
    return (i == pure_comp_idx) ? lnfS - std::log(p) : NAN;
}
std::vector<double> PureSolid::dlnphi_dP()
{
    solid_par::V vv = solid_par::V(phase);
    double dVs_dP = vv.dFdP(p, T);
    
    dlnphidP = std::vector<double>(ns, NAN);
    dlnphidP[pure_comp_idx] = dVs_dP - 1./p;
    return dlnphidP;
}
std::vector<double> PureSolid::dlnphi_dT() {
    solid_par::H hh = solid_par::H(phase);
    solid_par::V vv = solid_par::V(phase);
    double dHs_dT = hh.dFdT(T);
    double dVs_dT = vv.dFdT(p, T);

    solid_par::IG ig = solid_par::IG(solid_par::pure_comp[phase]);
    double dHi_dT = ig.dHdT(T);

    dlnphidT = std::vector<double>(ns, NAN);
    dlnphidT[pure_comp_idx] = -dHs_dT + dVs_dT + dHi_dT;
    return dlnphidT;
}
std::vector<double> PureSolid::d2lnphi_dPdT()
{
    solid_par::V vv = solid_par::V(phase);
    double d2Vs_dPdT = vv.d2FdPdT(p, T);
    
    d2lnphidPdT = std::vector<double>(ns, NAN);
    d2lnphidPdT[pure_comp_idx] = d2Vs_dPdT;
    return d2lnphidPdT;
}
std::vector<double> PureSolid::d2lnphi_dT2() {
    solid_par::H hh = solid_par::H(phase);
    solid_par::V vv = solid_par::V(phase);
    double d2Hs_dT2 = hh.d2FdT2(T);
    double d2Vs_dT2 = vv.d2FdT2(p, T);

    solid_par::IG ig = solid_par::IG(solid_par::pure_comp[phase]);
    double d2Hi_dT2 = ig.d2HdT2(T);

    d2lnphidT2 = std::vector<double>(ns, NAN);
    d2lnphidT2[pure_comp_idx] = -d2Hs_dT2 + d2Vs_dT2 + d2Hi_dT2;
    return d2lnphidT2;
}
std::vector<double> PureSolid::dlnphi_dn() 
{
    dlnphidn = std::vector<double>(ns*ns, 0.);
    return dlnphidn;
}
std::vector<double> PureSolid::d2lnphi_dTdn()
{
    d2lnphidTdn = std::vector<double>(ns*ns, 0.);
    return d2lnphidTdn;
}

std::vector<double> PureSolid::lnphi0(double X, double T_, bool pt)
{
    (void) pt;
    this->init_PT(X, T_);

    std::vector<double> lnphi0_(ns, NAN);
    lnphi0_[this->pure_comp_idx] = this->lnphii(this->pure_comp_idx);
    return lnphi0_;
}

int PureSolid::derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Test derivatives of ig, h and v terms
    int error_output = 0;
    (void) n_;
    
    solid_par::IG ig = solid_par::IG("H2O");
    solid_par::H hh = solid_par::H(phase);
    solid_par::V vv = solid_par::V(phase);
	
	error_output += ig.test_derivatives(T_, tol, verbose);
	error_output += hh.test_derivatives(T_, tol, verbose);
	error_output += vv.test_derivatives(p_, T_, tol, verbose);

    return error_output;
}

int PureSolid::pvt_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Consistency of PVT: Calculate volume at (P, T, n) and find P at (T, V, n)
    int error_output = 0;

    // Calculate volume at P, T, n
    this->v = this->V(p_, T_, n_);

    // Evaluate P(T,V,n_)
    double pp = this->P(this->v, T_, n_);
    if (verbose || std::fabs(pp - p_) > tol)
    {
        print("P(T, V, n) != p", {pp, p_, std::fabs(pp-p_)});
        error_output++;
    }
    
    return error_output;
}