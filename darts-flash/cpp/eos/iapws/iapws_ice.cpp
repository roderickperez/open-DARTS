#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory>
#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/maths/maths.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"
#include "dartsflash/eos/iapws/iapws_ice.hpp"

namespace iapws_ice {
    double p_0{ 101325. }, T_0{ 273.15 };
    double p_t{ 611.654771007894 }, T_t{ 273.16 };  // pressure and temperature at triple point
    double s0{ -0.332733756492168e4 };
    std::vector<double> g_0i = {-0.632020233449497e6, 0.655022213658955, -0.189369929326131e-7, 0.339746123271053e-14, -0.556464869058991e-21};
    std::complex<double> r1(0.447050716285388e2, 0.656876847463481e2);
    std::complex<double> t1(0.368017112855051e-1, 0.510878114959572e-1);
    std::complex<double> t2(0.337315741065416, 0.335449415919309);
    std::vector<std::complex<double>> r_2i = {{-0.72597457432922e2, -0.78100842711287e2}, 
                                              {-0.557107698030123e-4, 0.464578634580806e-4}, 
                                              {0.234801409215913e-10, -0.285651142904972e-10}};
}

namespace iapws_ice_ref {
    std::vector<double> g = {0.611784135, 0.10134274069e3, -0.222296513088e6};                   // J/kg
    std::vector<double> g_p = {0.109085812737e-2, 0.109084388214e-2, 0.106193389260e-2};         // m3/kg
    std::vector<double> g_T = {0.122069433940e4, 0.122076932550e4, 0.261195122589e4};            // J/kg.K
    std::vector<double> g_pp = {-0.128495941571e-12, -0.128485364928e-12, -0.941807981761e-13};  // m3/kg.Pa
    std::vector<double> g_TT = {-0.767602985875e1, -0.767598233365e1, -0.866333195517e1};        // m3/kg.K
    std::vector<double> g_pT = {0.174387964700e-6, 0.174362219972e-6, 0.274505162488e-7};        // J/kg.K^2

    std::vector<double> h = {-0.333444253966e6, -0.333354873637e6, -0.483491635676e6};           // J/kg
    std::vector<double> f = {-0.55446875e-1, -0.918701567e1, -0.328489902347e6};                 // J/kg
    std::vector<double> u = {0.333444921197e6, -0.333465403393e6, -0.589685024936e6};            // J/kg
    std::vector<double> s = {-0.122069433940e4, -0.122076932550e4, -0.261195122589e4};           // J/kg.K
    std::vector<double> c_p = {0.209678431622e4, 0.209671391024e4, 0.866333195517e3};            // J/kg.K
    std::vector<double> v = {0.916709492200e3, 0.916721463419e3, 0.941678203297e3};              // kg/m3
}

IAPWSIce::IAPWSIce(CompData& comp_data, bool iapws_ideal_) : EoS(comp_data)
{
    this->iapws_ideal = iapws_ideal_;
    this->iapws = std::make_shared<IAPWS95>(IAPWS95(comp_data, iapws_ideal_));
    this->n = std::vector<double>(ns, 0.);
}

void IAPWSIce::init_PT(double p_, double T_)
{
    if (p_ != p || T_ != T)
    {
        this->p = p_; this->T = T_;

        // NANs if T out of range
        if (T_ > 273.16)
        {
            this->g = this->g_p = this->g_pp = this->g_T = this->g_TT = this->g_TP = NAN;
        }
        else
        {
            // Calculate g0 and r2 variables and their pressure derivatives        
            // Auxiliary variables
            std::vector<double> pow_p_pr(5);
            pow_p_pr[0] = 1.;
            pow_p_pr[1] = p*1e5 / iapws_ice::p_t - iapws_ice::p_0 / iapws_ice::p_t;
            for (int k = 2; k < 5; k++)
            {
                pow_p_pr[k] = pow_p_pr[k-1] * pow_p_pr[1];
            }

            // g0
            double g0 = 0.;
            for (int k = 0; k < 5; k++)
            {
                g0 += iapws_ice::g_0i[k] * pow_p_pr[k];
            }
            // g0_p
            double g0_p = 0.;
            for (int k = 1; k < 5; k++)
            {
                g0_p += static_cast<double>(k) * iapws_ice::g_0i[k] / iapws_ice::p_t * pow_p_pr[k-1];
            }
            // g0_pp
            double g0_pp = 0.;
            for (int k = 2; k < 5; k++)
            {
                g0_pp += static_cast<double>(k) * (static_cast<double>(k)-1.) * iapws_ice::g_0i[k] / std::pow(iapws_ice::p_t, 2) * pow_p_pr[k-2];
            }
            // r2
            std::complex<double> r2(0., 0.);
            for (int k = 0; k < 3; k++)
            {
                r2 += iapws_ice::r_2i[k] * pow_p_pr[k];
            }
            // r2_p
            std::complex<double> r2_p(0., 0.);
            for (int k = 1; k < 3; k++)
            {
                r2_p += iapws_ice::r_2i[k] * static_cast<double>(k) / iapws_ice::p_t * pow_p_pr[k-1];
            }
            // r2_pp
            std::complex<double> r2_pp = iapws_ice::r_2i[2] * 2. / std::pow(iapws_ice::p_t, 2);

            // Auxiliary variables
            double tr = T / iapws_ice::T_t;
            std::complex<double> ln_t1mt = std::log(iapws_ice::t1 - tr);  // ln(t1 - tr)
            std::complex<double> ln_t1pt = std::log(iapws_ice::t1 + tr);  // ln(t1 + tr)
            std::complex<double> r1_term = (iapws_ice::t1 - tr) * ln_t1mt + (iapws_ice::t1 + tr) * ln_t1pt - 2.*iapws_ice::t1 * std::log(iapws_ice::t1) - std::pow(tr, 2)/iapws_ice::t1;
            std::complex<double> ln_t2mt = std::log(iapws_ice::t2 - tr);  // ln(t2 - tr)
            std::complex<double> ln_t2pt = std::log(iapws_ice::t2 + tr);  // ln(t2 + tr)
            std::complex<double> r2_term = (iapws_ice::t2 - tr) * ln_t2mt + (iapws_ice::t2 + tr) * ln_t2pt - 2.*iapws_ice::t2 * std::log(iapws_ice::t2) - std::pow(tr, 2)/iapws_ice::t2;
            
            double s0 = this->iapws_ideal ? iapws_ice::s0 : iapws_ice::s0;
            // g [J/kg]
            this->g = g0 - s0 * iapws_ice::T_t * tr + iapws_ice::T_t * std::real(iapws_ice::r1 * r1_term + r2 * r2_term);
            // dg/dp [m3/kg]
            this->g_p = g0_p + iapws_ice::T_t * std::real(r2_p * r2_term);
            // dg/dT [J/kg.K]
            this->g_T = -s0 + std::real(iapws_ice::r1 * (-ln_t1mt + ln_t1pt - 2. * tr/iapws_ice::t1) + r2 * (-ln_t2mt + ln_t2pt - 2. * tr/iapws_ice::t2));
            // d2g/dp2 [m3/kg.Pa]
            this->g_pp = g0_pp + iapws_ice::T_t * std::real(r2_pp * r2_term);
            // d2g/dT2 [J/kg.K^2]
            this->g_TT = 1/iapws_ice::T_t * std::real(iapws_ice::r1 * (1./(iapws_ice::t1-tr) + 1./(iapws_ice::t1+tr) - 2./iapws_ice::t1) + r2 * (1./(iapws_ice::t2-tr) + 1./(iapws_ice::t2+tr) - 2./iapws_ice::t2));
            // d2g/dTdp [m3/kg.K]
            this->g_TP = std::real(r2_p * (-ln_t2mt + ln_t2pt - 2.*tr/iapws_ice::t2));
        }
    }
}
void IAPWSIce::solve_PT(std::vector<double>::iterator n_it, bool second_order) 
{
    std::copy(n_it, n_it + ns, this->n.begin());
    this->N = std::accumulate(n_it, n_it + ns, 0.);
    (void) second_order;

    // Calculate molar volume
    this->v = N * this->g_p * this->compdata.Mw[this->compdata.water_index] * 1e-3;  // g_p [m3/kg] * M [kg/mol] = [m3/mol]
    return;
}

void IAPWSIce::init_VT(double V_, double T_) 
{
    this->v = V_;
    this->T = T_;
    return;
}
void IAPWSIce::solve_VT(std::vector<double>::iterator n_it, bool second_order) 
{
    std::copy(n_it, n_it + ns, this->n.begin());
    this->N = std::accumulate(n_it, n_it + ns, 0.);
    this->p = this->P(this->v, this->T, this->n);
    (void) second_order;
    return;
}

double IAPWSIce::P(double V_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Find pressure at given (T, V, n)
    (void) pt;
    double p_ = 1.;

    // Newton loop to find root
    int it = 0;
    while (it < 10)
    {
        this->v = this->V(p_, T_, n_, start_idx);
        double res = this->v - V_;
        double dres_dp = this->dV_dP();
        p_ -= res/dres_dp;

        if (std::fabs(res) < 1e-14)
        {
            break;
        }
        it++;
    }
    return p;
}

double IAPWSIce::V(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate volume at (P, T, n)
    this->solve_PT(p_, T_, n_, start_idx, false);
    (void) pt;
    return this->v;
}

double IAPWSIce::dV_dP()
{
    // Calculate pressure derivative of volume at (P, T, n)
    return N * this->compdata.Mw[this->compdata.water_index] * 1e-3 * this->g_pp * 1e5; // g_p^-1 [kg/m3] / M [kg/mol]
}

double IAPWSIce::dV_dT()
{
    // Calculate temperature derivative of volume at (P, T, n)
    return N * this->compdata.Mw[this->compdata.water_index] * 1e-3 * this->g_TP;
}

double IAPWSIce::dV_dni(int i)
{
    // Calculate temperature derivative of volume at (P, T, n)
    return (i == this->compdata.water_index) ? this->v/N : 0.;
}

double IAPWSIce::rho(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    return this->compdata.get_molar_weight(n_) * 1e-3 / this->V(p_, T_, n_, start_idx, pt);
}

double IAPWSIce::lnphii(int i)
{
    if (i == this->compdata.water_index)
    {
        return (this->g / iapws95::R - this->Gi(this->p, this->T, n, 0, true)/N) / this->T;  // lnphii = Gr/RT
    }
    else
    {
        return NAN;
    }
}
std::vector<double> IAPWSIce::dlnphi_dP()
{
    dlnphidP = std::vector<double>(ns, NAN);

    dlnphidP[this->compdata.water_index] = (this->g_p / iapws95::R - this->T / (this->p*1e5)) / this->T * 1e5;
    return dlnphidP;
}
std::vector<double> IAPWSIce::dlnphi_dT()
{
    dlnphidT = std::vector<double>(ns, NAN);
    
    int i = this->compdata.water_index;
    dlnphidT[i] = (this->g_T / iapws95::R + this->si(this->p, this->T, i, true)) / this->T
                - (this->g / iapws95::R - this->Gi(this->p, this->T, n, 0, true)/N) / std::pow(this->T, 2);
    return dlnphidT;
}
std::vector<double> IAPWSIce::d2lnphi_dPdT()
{
    d2lnphidPdT = std::vector<double>(ns, NAN);

    d2lnphidPdT[this->compdata.water_index] = ((this->g_TP / iapws95::R - 1. / (this->p*1e5)) / this->T
                                             - (this->g_p / iapws95::R - this->T / (this->p*1e5)) / std::pow(this->T, 2)) * 1e5;
    return d2lnphidPdT;
}
std::vector<double> IAPWSIce::d2lnphi_dT2()
{
    d2lnphidT2 = std::vector<double>(ns, NAN);

    int i = this->compdata.water_index;
    d2lnphidT2[i] = (this->g_TT / iapws95::R + this->dsi_dT(this->p, this->T, i, true)) / this->T
                  - 2. * (this->g_T / iapws95::R + this->si(this->p, this->T, i, true)) / std::pow(this->T, 2)
                  + 2. * (this->g / iapws95::R - this->Gi(this->p, this->T, n, 0, true)/N) / std::pow(this->T, 3);
    return d2lnphidT2;
}
std::vector<double> IAPWSIce::dlnphi_dn()
{
    dlnphidn = std::vector<double>(ns*ns, 0.);
    // double gi = this->Gi(this->p, this->T, this->n, 0, true);
    // std::vector<double> dGidni = this->dGi_dni(this->p, this->T, this->n, 0, true);

    // int i = this->compdata.water_index;
    // dlnphidn[i * ns + i] = -(dGidni[i]/N - gi/std::pow(N, 2)) / this->T;
    return dlnphidn;
}
std::vector<double> IAPWSIce::d2lnphi_dTdn()
{
    d2lnphidTdn = std::vector<double>(ns*ns, 0.);
    // double gi = this->Gi(this->p, this->T, this->n, 0, true);
    // std::vector<double> dGidni = this->dGi_dni(this->p, this->T, this->n, 0, true);
    // std::vector<double> d2GidTdni = this->d2Gi_dTdni(this->p, this->T, this->n, 0, true);

    // int i = this->compdata.water_index;
    // d2lnphidTdn[i * ns + i] = -(d2GidTdni[i]/N + this->si(this->p, this->T, i, true)/std::pow(N, 2)) / this->T
    //                         + (dGidni[i]/N - gi/std::pow(N, 2)) / std::pow(T, 2);
    return d2lnphidTdn;
}

double IAPWSIce::cpi(double T_, int i)
{
    // Ideal gas heat capacity at constant pressure from IAPWS-95 EoS
    return this->iapws->cpi(T_, i);  // Cpi/R
}
double IAPWSIce::hi(double T_, int i)
{
    // Ideal gas enthalpy from IAPWS-95 EoS
    return this->iapws->hi(T_, i);  // Hi/R
}
double IAPWSIce::si(double X, double T_, int i, bool pt)
{
    // Ideal gas entropy from IAPWS-95 EoS
    if (this->iapws_ideal)
    {
        // Ideal gas entropy from IAPWS-95 reference
        if (pt)
        {
            double siVT = this->si(this->v, T_, i, false);
            return siVT + std::log(T_ / this->compdata.T_0) + std::log(this->v/N / this->compdata.V_0) 
                        - std::log(X / this->compdata.P_0);  // si(PT)/R
        }
        else
        {
            this->iapws->solve_VT(X, T_, this->n, 0, true);
            this->iapws->set_d(X);
            this->iapws->set_tau(T_);
            return this->iapws->get_tau() * this->iapws->phi0_t() - this->iapws->phi0();  // si(VT)/R
        }
    }
    else
    {
        return EoS::si(X, T_, i, pt);  // Si/R
    }
}
double IAPWSIce::dsi_dP(double X, double T_, int i, bool pt)
{
    if (this->iapws_ideal)
    {
        // Ideal gas entropy from IAPWS-95 reference
        if (pt)
        {
            this->iapws->set_d(this->v);
            this->iapws->set_tau(T_);
            double dsiVT = (this->iapws->get_tau() * this->iapws->phi0_dt() - this->iapws->phi0_dd()) * this->iapws->dd_dV() * this->dV_dP();
            return dsiVT - 1./X + this->dV_dP()/this->v;  // 1/R dsi(PT)/dP
        }
        else
        {
            this->iapws->set_d(X);
            this->iapws->set_tau(T_);
            return (this->iapws->get_tau() * this->iapws->phi0_dt() - this->iapws->phi0_d()) * this->iapws->dd_dV() * this->dV_dP();  // 1/R dsi(VT)/dP
        }
    }
    else
    {
        return EoS::dsi_dP(X, T_, i, pt);
    }
}
double IAPWSIce::dsi_dT(double X, double T_, int i, bool pt)
{
    if (this->iapws_ideal)
    {
        // Ideal gas entropy from IAPWS-95 reference
        if (pt)
        {
            this->iapws->set_d(this->v);
            this->iapws->set_tau(T_);
            double dsiVT = this->iapws->dtau_dT() * (this->iapws->get_tau() * this->iapws->phi0_tt()) - this->iapws->phi0_d() * this->iapws->dd_dV() * this->dV_dT();
            return dsiVT + 1./T_ + this->dV_dT()/this->v;  // 1/R dsi(PT)/dT
        }
        else
        {
            // this->iapws->solve_VT(X, T_, this->n, 0, false);
            this->iapws->set_d(X);
            this->iapws->set_tau(T_);
            return this->iapws->dtau_dT() * this->iapws->get_tau() * this->iapws->phi0_tt();  // 1/R dsi(VT)/dT
        }
    }
    else
    {
        return EoS::dsi_dT(X, T_, i, pt);
    }
}
std::vector<double> IAPWSIce::dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate partial molar ideal gas entropy
    if (this->iapws_ideal)
    {
        std::vector<double> dSidn(ns, 0.);
        
        // Volume term contribution to entropy
        int i = this->compdata.water_index;
        dSidn[i] = this->si(X, T_, i, pt); //  + n_[start_idx + i] * this->dV_dni(i)/this->v;
        return dSidn;  // 1/R dSi/dn
    }
    else
    {
        return EoS::dSi_dni(X, T_, n_, start_idx, pt);
    }
}

std::vector<double> IAPWSIce::lnphi0(double X, double T_, bool pt)
{
    (void) pt;
    this->init_PT(X, T_);
    std::vector<double> lnphi0_(nc, NAN);
    lnphi0_[this->compdata.water_index] = this->lnphii(this->compdata.water_index);
    return lnphi0_;
}

int IAPWSIce::derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Test analytical derivatives
    int error_output = 0;
    
    double p0 = p_;
    double T0 = T_;
    std::vector<double> n0 = n_;

    // Calculate properties at P0, T0
    this->solve_PT(p0, T0, n0, 0, true);
    double G0 = this->g;
    double G0_P = this->g_p * unit_conversion::bar_to_output[Units::PRESSURE::PA];
    double G0_PP = this->g_pp * std::pow(unit_conversion::bar_to_output[Units::PRESSURE::PA], 2);
    double G0_T = this->g_T;
    double G0_TT = this->g_TT;
    double G0_TP = this->g_TP * unit_conversion::bar_to_output[Units::PRESSURE::PA];

    double d, G1, G1_P, G1_T;

    // Calculate pressure derivatives
    double dp = 1e-5;
    this->solve_PT(p0 + dp, T0, n0, 0, true);
    G1 = this->g;
    G1_P = this->g_p * unit_conversion::bar_to_output[Units::PRESSURE::PA];
    G1_T = this->g_T;

    double dp_num = (G1 - G0) / dp;
    d = std::log(std::fabs(G0_P + 1e-15)) - std::log(std::fabs(dp_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dg/dP != dg/dP", {G0_P, dp_num, d}); error_output++; }

    double d2p_num = (G1_P - G0_P) / dp;
    d = std::log(std::fabs(G0_PP + 1e-15)) - std::log(std::fabs(d2p_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2g/dP2 != d2g/dP2", {G0_PP, d2p_num, d}); error_output++; }

    double d2tp_num = (G1_T - G0_T) / dp;
    d = std::log(std::fabs(G0_TP + 1e-15)) - std::log(std::fabs(d2tp_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2g/dTdP != d2g/dTdP", {G0_TP, d2tp_num, d}); error_output++; }

    // Calculate temperature derivatives
    double dT = 1e-5;
    this->solve_PT(p0, T0 + dT, n0, 0, true);
    G1 = this->g;
    G1_P = this->g_p * unit_conversion::bar_to_output[Units::PRESSURE::PA];
    G1_T = this->g_T;

    double dT_num = (G1 - G0) / dT;
    d = std::log(std::fabs(G0_T + 1e-15)) - std::log(std::fabs(dT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dg/dT != dg/dT", {G0_T, dT_num, d}); error_output++; }

    double d2T_num = (G1_T - G0_T) / dT;
    d = std::log(std::fabs(G0_TT + 1e-15)) - std::log(std::fabs(d2T_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2g/dT2 != d2g/dT2", {G0_TT, d2T_num, d}); error_output++; }

    double d2pt_num = (G1_P - G0_P) / dT;
    d = std::log(std::fabs(G0_TP + 1e-15)) - std::log(std::fabs(d2pt_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dg/dPdT != dg/dPdT", {G0_TP, d2pt_num, d}); error_output++; }

    return error_output;
}

int IAPWSIce::pvt_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Consistency of PVT: Calculate volume at (P, T, n) and find P at (T, V, n)
    int error_output = 0;

    // Calculate volume at P, T, n
    // Evaluate properties at P, T, n
    this->solve_PT(p_, T_, n_, 0, true);
    double p0 = p_;
    double T0 = T_;
    double V0 = this->v;
    std::vector<double> n0 = n_;

    // Evaluate P(T,V,n)
    double pp = this->P(this->v, T0, n0);
    if (verbose || std::fabs(pp - p0) > tol)
    {
        print("P(T, V, n) != p", {pp, p_, std::fabs(pp-p0)});
        error_output++;
    }

    // Calculate derivatives of P and V w.r.t. V, T and composition
    this->solve_PT(p0, T0, n0, 0, true);
    // double dPdV = this->dP_dV();
    // double d2PdV2 = this->d2P_dV2();
	// double dPdT = this->dP_dT();
    double dVdT = this->dV_dT();
    std::vector<double> dVdn(nc);
    for (int i = 0; i < nc; i++)
    {
	    // dPdn[i] = this->dP_dni(i);
	    dVdn[i] = this->dV_dni(i);
    }
    double d, dX{ 1e-5 };

    // // Numerical derivative with respect to volume
    // this->solve_VT(V0 + dX*V0, T0, n0, 0, true);
    // double dPdV_num = (this->p - p0) / (dX*V0);
    // d = std::log(std::fabs(dPdV + 1e-15)) - std::log(std::fabs(dPdV_num + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("dP/dV != dP/dV", {dPdV, dPdV_num, d}); error_output++; }
    // double d2PdV2_num = (this->dP_dV() - dPdV) / (dX*V0);
    // d = std::log(std::fabs(d2PdV2 + 1e-15)) - std::log(std::fabs(d2PdV2_num + 1e-15));
    // // if (verbose || !(std::fabs(d) < tol)) { print("d2P/dV2 != d2P/dV2", {d2PdV2, d2PdV2_num, d}); error_output++; }

    // // Numerical derivative with respect to temperature
    // this->solve_VT(V0, T0 + dX*T0, n0, 0, true);
    // double dPdT_num = (this->p - p0) / (dX*T0);
    // d = std::log(std::fabs(dPdT + 1e-15)) - std::log(std::fabs(dPdT_num + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("dP/dT != dP/dT", {dPdT, dPdT_num, d}); error_output++; }

    this->solve_PT(p0, T0 + dX*T0, n0, 0, true);
    double dVdT_num = (this->v - V0) / (dX*T0);
    d = std::log(std::fabs(dVdT + 1e-15)) - std::log(std::fabs(dVdT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dV/dT != dV/dT", {dVdT, dVdT_num, d}); error_output++; }

    // Numerical derivative with respect to composition
    for (int i = 0; i < nc; i++)
    {
        double dni = dX*n0[i];
        n_[i] += dni;

        // this->solve_VT(V0, T0, n_, 0, true);
        // double dPdn_num = (this->p - p0) / dni;
        // d = std::log(std::fabs(dPdn[i] + 1e-15)) - std::log(std::fabs(dPdn_num + 1e-15));
        // if (verbose || (!(std::fabs(d) < tol) && n0[i] > 0.)) { print("dP/dni != dP/dni", {dPdn[i], dPdn_num, d}); error_output++; }

        this->solve_PT(p0, T0, n_, 0, true);
        double dVdn_num = (this->v - V0) / dni;
        d = std::log(std::fabs(dVdn[i] + 1e-15)) - std::log(std::fabs(dVdn_num + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && n0[i] > 0.)) { print("dV/dni != dV/dni", {dVdn[i], dVdn_num, d}); error_output++; }

        n_[i] -= dni;
    }
    
    return error_output;
}

int IAPWSIce::references_test(double tol, bool verbose)
{
    // Test reference values for g, derivatives and thermodynamic properties
    int error_output = 0;

    // Check values at triple point (611.657 Pa, 273.16 K), normal 
    std::vector<double> pp = {611.657e-5, 1.01325, 1000.};
    std::vector<double> TT = {273.16, 273.152519, 100.};
    std::vector<double> nn = {1.};

    for (int i = 0; i < 3; i++)
    {
        this->solve_PT(pp[i], TT[i], nn);

        double d = std::log(std::fabs(iapws_ice_ref::g[i] + 1e-15)) - std::log(std::fabs(this->g + 1e-15));;
        if (verbose || !(std::fabs(d) < tol)) { print("IAPWSIce g", {iapws_ice_ref::g[i], this->g}); error_output++; }
        d = std::log(std::fabs(iapws_ice_ref::g_p[i] + 1e-15)) - std::log(std::fabs(this->g_p + 1e-15));;
        if (verbose || !(std::fabs(d) < tol)) { print("IAPWSIce dg/dP", {iapws_ice_ref::g_p[i], this->g_p}); error_output++; }
        d = std::log(std::fabs(iapws_ice_ref::g_T[i] + 1e-15)) - std::log(std::fabs(this->g_T + 1e-15));;
        if (verbose || !(std::fabs(d) < tol)) { print("IAPWSIce dg/dT", {iapws_ice_ref::g_T[i], this->g_T}); error_output++; }
        d = std::log(std::fabs(iapws_ice_ref::g_pp[i] + 1e-15)) - std::log(std::fabs(this->g_pp + 1e-15));;
        if (verbose || !(std::fabs(d) < tol)) { print("IAPWSIce d2g/dP2", {iapws_ice_ref::g_pp[i], this->g_pp}); error_output++; }
        d = std::log(std::fabs(iapws_ice_ref::g_TT[i] + 1e-15)) - std::log(std::fabs(this->g_TT + 1e-15));;
        if (verbose || !(std::fabs(d) < tol)) { print("IAPWSIce d2g/dT2", {iapws_ice_ref::g_TT[i], this->g_TT}); error_output++; }
        d = std::log(std::fabs(iapws_ice_ref::g_pT[i] + 1e-15)) - std::log(std::fabs(this->g_TP + 1e-15));;
        if (verbose || !(std::fabs(d) < tol)) { print("IAPWSIce d2g/dPdT", {iapws_ice_ref::g_pT[i], this->g_TP}); error_output++; }
    }

    return error_output;
}
