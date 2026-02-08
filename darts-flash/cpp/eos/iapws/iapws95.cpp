#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/maths/maths.hpp"
#include "dartsflash/maths/root_finding.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"

namespace iapws95 {
    double Mw{ 18.015 } /*g/mol*/, Tc{ 647.096 } /*K*/, rhoc{ 322. } /*kg/m3*/, Vc{ Mw * 1e-3 / rhoc } /*m3/mol*/, Pc{ 220.50 } /*bar*/, Zc{ Pc * Vc / (M_R * 1e-5 * Tc) } /*[-]*/, R{ 0.46151805 } /*kJ/kg.K*/;

    // Table 1: i = [1, 8]
    std::vector<double> ni0 = {-8.3204464837497, 6.6832105275932, 3.00632, 0.012436, 0.97315, 1.27950, 0.96956, 0.24873}; // i = [1, 8]
    std::vector<double> ji0 = {0, 0, 0, 1.28728967, 3.53734222, 7.74073708, 9.24437796, 27.5075105};  // i = [4, 8]

    // Table 2: i = [1, 56]
    std::vector<int> ci = {0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           3, 3, 3, 3, 4, 6, 6, 6, 6};  // i = [8, 51]
    std::vector<int> di = {1, 1, 1, 2, 2, 3, 4, 
                           1, 1, 1, 2, 2, 3, 4, 4, 5, 7, 9, 10, 11, 13, 15,
                           1, 2, 2, 2, 3, 4, 4, 4, 5, 6, 6, 7, 9, 9, 9, 9, 9, 10, 10, 12,
                           3, 4, 4, 5, 14, 3, 6, 6, 6,
                           3, 3, 3};  // i = [1, 54]
    std::vector<double> ti = {-0.5, 0.875, 1., 0.5, 0.75, 0.375, 1, 
                              4, 6, 12, 1, 5, 4, 2, 13, 9, 3, 4, 11, 4, 13, 1,
                              7, 1, 9, 10, 10, 3, 7, 10, 10, 6, 10, 10, 1, 2, 3, 4, 8, 6, 9, 8,
                              16, 22, 23, 23, 10, 50, 44, 46, 50,
                              0, 1, 4};  // i = [1, 54]
    std::vector<double> ni = {0.12533547935523e-1, 0.78957634722828e1, -0.87803203303561e1, 0.31802509345418, -0.26145533859358, -0.78199751687981e-2, 0.88089493102134e-2,
                             -0.66856572307965, 0.20433810950965, -0.66212605039687e-4, -0.19232721156002, -0.25709043003438, 0.16074868486251, -0.40092828925807e-1, 0.39343422603254e-6, -0.75941377088144e-5, 0.56250979351888e-3, -0.15608652257135e-4, 0.11537996422951e-8, 0.36582165144204e-6, -0.13251180074668e-11, -0.62639586912454e-9,
                             -0.10793600908932, 0.17611491008752e-1, 0.22132295167546, -0.40247669763528, 0.58083399985759, 0.49969146990806e-2, -0.31358700712549e-1, -0.74315929710341, 0.47807329915480, 0.20527940895948e-1, -0.13636435110343, 0.14180634400617e-1, 0.83326504880713e-2, -0.29052336009585e-1, 0.38615085574206e-1, -0.20393486513704e-1, -0.16554050063734e-2, 0.19955571979541e-2, 0.15870308324157e-3, -0.16388568342530e-4, 0.43613615723811e-1, 0.34994005463765e-1, -0.76788197844621e-1, 0.22446277332006e-1, -0.62689710414685e-4, -0.55711118565645e-9, -0.19905718354408, 0.31777497330738, -0.11841182425981,
                             -0.31306260323435e2, 0.31546140237781e2, -0.25213154341695e4, -0.14874640856724, 0.31806110878444};
    std::vector<double> ai = {20, 20, 20, 3.5, 3.5};  // i = [52, 56]
    std::vector<double> bi = {150, 150, 250, 0.85, 0.95}; // i = [52, 56]
    std::vector<double> ji = {1.21, 1.21, 1.25};  // i = 52,53,54
    std::vector<int> ei = {1, 1, 1}; // i = 52, 53, 54
    std::vector<double> Ai = {0.32, 0.32};  // i = 55, 56
    std::vector<double> Bi = {0.2, 0.2};  // i = 55, 56
    std::vector<int> Ci = {28, 32};  // i = 55, 56
    std::vector<int> Di = {700, 800};  // i = 55, 56
    std::vector<double> betai = {0.3, 0.3};  // i = 55, 56
}

namespace iapws95ref {
    // Reference values for phi0, phir and derivatives at T = 500K and rho = 838.025 kg/m3
    double phi0{ 0.204797733e1 }, phi0_d{ 0.384236747 }, phi0_dd{ -0.147637878 }, phi0_t{ 0.904611106e1 }, phi0_tt{ -0.193249185e1 }, phi0_dt{ 0. };
    double phir{ -0.342693206e1 }, phir_d{ -0.364366650 }, phir_dd{ 0.856063701 }, phir_t{ -0.581403435e1 }, phir_tt{ -0.223440737e1 }, phir_dt{ -0.112176915e1 };
    
    // Reference values for thermodynamic properties at selected conditions: P, Cv, sound speed and S
    std::vector<double> T = {300., 300., 300., 500., 500., 500., 500., 647., 900., 900., 900.};  // K
    std::vector<double> rho = {0.9965560e3, 0.1005308e4, 0.1188202e4, 0.4350000, 0.4532000e1, 0.8380250e3, 0.1084564e4, 0.3580000e3, 0.2410000, 0.5261500e2, 0.8707690e3};  // kg/m3
    std::vector<double> p = {0.992418352e-1, 0.200022515e2, 0.700004704e3, 0.999679423e-1, 0.999938125, 0.100003858e2, 0.700000405e3, 0.220384756e2, 0.100062559, 0.200000690e2, 0.700000006e3};  // MPa
    std::vector<double> cv = {0.413018112e1, 0.406798347e1, 0.346135580e1, 0.150817541e1, 0.166991025e1, 0.322106219e1, 0.307437693e1, 0.618315728e1, 0.175890657e1, 0.193510526e1, 0.266422350e1};  // kJ/kg.K
    std::vector<double> w = {0.150151914e4, 0.153492501e4, 0.244357992e4, 0.548314253e3, 0.535739001e3, 0.127128441e4, 0.241200877e4, 0.252145078e3, 0.724027147e3, 0.698445674e3, 0.201933608e4};   // m/s
    std::vector<double> s = {0.393062643, 0.387405401, 0.132609616, 0.794488271e1, 0.682502725e1, 0.256690919e1, 0.203237509e1, 0.432092307e1, 0.916653194e1, 0.659070225e1, 0.417223802e1};   // kJ/kg.K
}

IAPWS95::IAPWS95(CompData& comp_data, bool iapws_ideal_) : HelmholtzEoS(comp_data)
{
    this->iapws_ideal = iapws_ideal_;
}

void IAPWS95::init_VT(double V_, double T_)
{
    this->v = V_;  // m3/mol
    this->set_tau(T_);
    return;
}
void IAPWS95::set_tau(double T_)
{
    this->T = T_;  // K
    this->tau = iapws95::Tc / T_;
    return;
}
void IAPWS95::set_d(double V_)
{
    this->v = V_;
    this->vm = V_ / N;
    double rho = iapws95::Mw * 1e-3 / this->vm;  // kg/mol / m3/mol = kg/m3
    this->dv = rho / iapws95::rhoc;
    return;
}
double IAPWS95::get_v(double d_, double N_)
{
    return iapws95::Mw * 1e-3 * N_ / (d_ * iapws95::rhoc);
}

void IAPWS95::init_PT(double p_, double T_)
{
    if (T_ <= 0.)
    {
        this->tau = NAN;
    }
    else if (T_ != this->T_for_ig)
    {
        this->set_tau(T_);

        // At subcritical T, calculate minimum P possible for liquid root and maximum P possible for vapour root to exist
        if (this->tau > 1.)
        {
            this->T_for_ig = T_;
            this->calc_initial_guesses();
        }
    }
    this->p = p_;
    return;
}

void IAPWS95::calc_initial_guesses()
{
    // Find minimum P and d for liquid root to exist and maximum P and d for vapour root to exist
    double d_min, d_mid, d_max;
    RootFinding rf;
    auto fn = std::bind(&IAPWS95::dP_obj, this, std::placeholders::_1);
    double tol_d = 1e-14;
    double tol_f = 1e-15;

    // Find minimum of P-d curve in liquid root range
    d_min = 1.; d_max = 4.;  // Minimum bound: Intermediate root d=1; maximum bound: d=4 (d~4 at 1000 MPa and 1273 K)
    d_mid = (d_min + d_max) * 0.5;

    // Apply bisection until we have located (P'(d_min) < 0 & P''(d_min) > 0) and (P'(d_max) > 0 & P''(d_max) > 0)
    bool d1_min{ false }, d2_min{ false }, d1_max{ false }, d2_max{ false };
    bool converged = false;
    int it, it_min{ 0 }, it_max{ 0 };
    for (it = 1; it < 50; it++)
    {
        (void) this->Pd(d_mid, this->T, this->n);

        // If dP/dd < 0 -> update d_min
        if (this->dP_dd() < 0)
        {
            d_min = d_mid;
            d1_min = true;
            d2_min = (this->d2P_dd2() > 0);
        }
        else
        {
            d_max = d_mid;
            d1_max = true;
            d2_max = (this->d2P_dd2() > 0);
        }
        d_mid = (d_min + d_max) * 0.5;

        // Check if all conditions of curvature at d_min and d_max have been satisfied
        if (d1_min && d2_min && d1_max && d2_max)
        {
            converged = true;
            break;
        }
    }
    if (!converged) { print("IAPWS95.V() LIQUID MINIMUM BISECTION not converged. T, dmin, dmid, dmax", {T, d_min, d_mid, d_max}); exit(1); }
    it_min += it;

    // Use Brent's method to find dL for which dP/dd = 0
    int brent_output = rf.brent(fn, d_mid, d_min, d_max, tol_f, tol_d);
    if (brent_output > 0) { print("IAPWS95.V() LIQUID MINIMUM BRENT not converged. T", T); exit(1); }
    else
    {
        this->p_minL = this->p;
        this->d_minL = rf.getx();
    }

    // Find maximum of P-d curve in vapour root range
    d_min = 1.e-15; d_max = 1.;  // Minimum bound: d=1e-15 (almost vacuum); maximum bound: intermediate root d=1
    d_mid = (d_min + d_max) * 0.5;

    // Apply bisection until we have located (P'(d_min) > 0 & P''(d_min) < 0) and (P'(d_max) < 0 & P''(d_max) < 0)
    d1_min = false; d2_min = false; d1_max = false; d2_max = false;
    converged = false;
    for (it = 1; it < 50; it++)
    {
        (void) this->Pd(d_mid, this->T, this->n);

        // If dP/dd < 0 -> update d_min
        if (this->dP_dd() > 0)
        {
            d_min = d_mid;
            d1_min = true;
            d2_min = (this->d2P_dd2() < 0);
        }
        else
        {
            d_max = d_mid;
            d1_max = true;
            d2_max = (this->d2P_dd2() < 0);
        }
        d_mid = (d_min + d_max) * 0.5;

        // Check if all conditions of curvature at d_min and d_max have been satisfied
        if (d1_min && d2_min && d1_max && d2_max)
        {
            converged = true;
            break;
        }
    }
    if (!converged) { print("IAPWS95.V() VAPOUR MAXIMUM BISECTION not converged. T, dmin, dmid, dmax", {T, d_min, d_mid, d_max}); exit(1); }
    it_max += it;

    // Use Brent's method to find dV for which dP/dd = 0
    brent_output = rf.brent(fn, d_mid, d_max, d_min, tol_f, tol_d);
    if (brent_output > 0) { print("IAPWS95.V() VAPOUR MAXIMUM BRENT not converged. T", T); exit(1); }
    else
    {
        this->p_maxV = this->p;
        this->d_maxV = rf.getx();
    }

    this->n_iterations += it_min + it_max;
}
double IAPWS95::V()
{
    // Solve volume for specified pressure and temperature
    // Solve P from initial guess of V
    double p_spec = this->p;
    this->Z_roots = std::vector<std::complex<double>>(2, NAN);

    // Initialize Brent's method to find roots
    RootFinding rf;
    auto fn = std::bind(&IAPWS95::P_obj, this, std::placeholders::_1);
    double tol_d = 1e-14;
    double tol_f = 1e-15;

    if (std::isnan(tau))
    {
        return NAN;
    }
    else if (this->tau > 1.)
    {
        // For subcritical temperatures, use minimum and maximum in liquid/vapour root ranges to find roots
        double v_L = NAN;
        double g_L = NAN;
        if (p_spec >= this->p_minL && this->root_flag != EoS::RootFlag::MAX)  // only compute if root flag is not MAX
        {
            double d_min = this->d_minL;
            double d_max = 4.;
            double d_mid = (d_min + d_max) * 0.5;

            int brent_output = rf.brent(fn, d_mid, d_min, d_max, tol_f, tol_d);
            if (brent_output > 0) { print("IAPWS95.V() LIQUID BRENT not converged. T", T); exit(1); }
            else
            {
                d_mid = rf.getx();
                this->v = this->get_v(d_mid, N);

                this->set_d(this->v);
                this->set_phir();
                this->set_dphir();

                this->z = this->p * this->vm / (this->units.R * this->T);
                v_L = this->v;
                g_L = this->lnphii(compdata.water_index);
                this->Z_roots[0] = this->z;
            }
        }
        
        double v_V = NAN;
        double g_V = NAN;
        if (p_spec <= this->p_maxV && this->root_flag != EoS::RootFlag::MIN)  // only compute if root flag is not MIN
        {
            double v_ig = N * this->units.R * this->T / p_spec;

            // Find maximum volume for consistent use of set_d(v) and get_v(d)
            // Above the maximum volume, intermediate variables reach machine precision and therefore rounding errors occur
            double v_max = iapws95::Mw * 1e-3 * N / (std::numeric_limits<double>::epsilon() * iapws95::rhoc);
            if (v_ig > 5.e4)
            {
                this->set_d(v_max);
                this->z = 1.;
            }
            else
            {
                this->set_d(v_ig);
                double d_ig = this->dv;
                double d_min = d_ig;
                double d_max = this->d_maxV;
                double d_mid = (d_min + d_max) * 0.5;

                int brent_output = rf.brent(fn, d_mid, d_min, d_max, tol_f, tol_d);
                if (brent_output > 0) { print("IAPWS95.V() VAPOUR BRENT not converged. T", T); exit(1); }
                else
                {
                    d_mid = rf.getx();
                    this->set_d(this->get_v(d_mid, N));
                    this->z = this->p * this->vm / (this->units.R * this->T);
                }
            }

            this->set_phir();
            this->set_dphir();

            v_V = this->v;
            g_V = this->lnphii(compdata.water_index);
            this->Z_roots[1] = this->z;
        }

        // Find most stable root
        if (std::isnan(g_V)  // no V root, L
            || (!std::isnan(g_L) && !std::isnan(g_V) && g_L < g_V))  // V and L roots found, L more stable
        {
            this->set_d(v_L);
            this->z = this->Z_roots[0].real();
            this->root_type = EoS::RootFlag::MIN;
        }
        else
        {
            this->set_d(v_V);
            this->z = this->Z_roots[1].real();
            this->root_type = EoS::RootFlag::MAX;
        }
        this->set_phir();
        this->set_dphir();

        return this->v;
    }
    else
    {
        // For critical temperatures, use critical volume as initial guess
        double d_min{ 1e-12 }, d_mid{ 1. }, d_max{ 4. };

        int brent_output = rf.brent(fn, d_mid, d_min, d_max, tol_f, tol_d);
        if (brent_output > 0) { print("IAPWS95.V() CRITICAL BRENT not converged. T", T); exit(1); }
        else
        {
            d_mid = rf.getx();

            this->set_d(d_mid);
            this->set_phir();
            this->set_dphir();
            
            this->v = this->get_v(d_mid, N);
            this->vm = this->v / N;
            this->z = this->p * this->vm / (this->units.R * this->T);
            this->Z_roots[0] = this->z;
            this->root_type = (this->vm > iapws95::Vc) ? EoS::RootFlag::MAX : EoS::RootFlag::MIN;
            return this->v;
        }
    }
}
double IAPWS95::P()
{
    // Calculate pressure
    double rho = iapws95::rhoc * this->dv;
    double P_kPa = (1.+this->dv * phir_d) * rho * iapws95::R * this->T;  // kPa
    return P_kPa * unit_conversion::input_to_bar[Units::PRESSURE::KPA];
}
double IAPWS95::dP_dd()
{
    // First derivative dP/dd of pressure with respect to reduced volume d
    double rho = iapws95::rhoc * this->dv;
    double dP_kPa = (iapws95::rhoc * (1. + this->dv * phir_d) + rho * (phir_d + this->dv * phir_dd)) * iapws95::R * this->T;
    return dP_kPa * unit_conversion::input_to_bar[Units::PRESSURE::KPA];
}
double IAPWS95::d2P_dd2()
{
    // Second derivative d2P/dd2 of pressure with respect to reduced volume d
    double rho = iapws95::rhoc * this->dv;
    double d2P_kPa = (2. * iapws95::rhoc * (phir_d + this->dv * phir_dd) + rho * (2. * phir_dd + this->dv * phir_ddd)) * iapws95::R * this->T;
    return d2P_kPa * unit_conversion::input_to_bar[Units::PRESSURE::KPA];
}
double IAPWS95::Pd(double d_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate pressure as a function of reduced volume d
    (void) n_;
    (void) start_idx;
    (void) pt;

    this->dv = d_;
    this->set_tau(T_);
    
    this->set_phir();
    this->set_dphir();

    return this->P();
}
double IAPWS95::Zd(double d_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate compressibility factor as a function of reduced volume d
    this->p = this->Pd(d_, T_, n_, start_idx, pt);
    return this->p * this->get_v(d_, 1) / (this->units.R * T_);
}

double IAPWS95::P_obj(double d_)
{
    // Objective function P-Pspec = 0 for volume root finding
    return this->Pd(d_, this->T, this->n) - this->p;
}
double IAPWS95::dP_obj(double d_)
{
    // First derivative of P with respect to d for Brent method implementation
    this->p = this->Pd(d_, this->T, this->n);
    return this->dP_dd();
}

void IAPWS95::zeroth_order(double V_)
{
    // Calculate zeroth order contribution to Helmholtz function: phi = phi0 + phir
    this->set_d(V_);
    this->set_phir();
}
void IAPWS95::first_order(std::vector<double>::iterator n_it)
{
    (void) n_it;

    this->set_dphir();
    return;
}
void IAPWS95::second_order(std::vector<double>::iterator n_it)
{
    this->first_order(n_it);

    // if properties other than dP/dV are required

    // Derivatives of phir w.r.t. t, tt and dt
    this->phir_t = 0.;
    this->phir_tt = 0.;
    this->phir_dt = 0.;
    for (int i = 0; i < 7; i++)  // first term i=1-7
    {
        double n_i = iapws95::ni[i];
        int di = iapws95::di[i];
        double ti = iapws95::ti[i];
            
        double pow_ddi = std::pow(this->dv, di);
        double pow_tti = std::pow(this->tau, ti-1);
        this->phir_t += n_i * ti * pow_ddi * pow_tti;
        this->phir_tt += n_i * ti * (ti-1) * pow_ddi * pow_tti / this->tau;
        this->phir_dt += n_i * di * ti * pow_ddi/dv * pow_tti;
    }
    for (int i = 7; i < 51; i++) // second term i=8-51
    {
        double n_i = iapws95::ni[i];
        int ci = iapws95::ci[i];
        int di = iapws95::di[i];
        double ti = iapws95::ti[i];

        double pow_dci = std::pow(this->dv, ci);
        double pow_ddi = std::pow(this->dv, di);
        double exp_dci = std::exp(-pow_dci);
        double pow_tti = std::pow(this->tau, ti-1);

        this->phir_t += n_i * ti * pow_ddi * pow_tti * exp_dci;
        this->phir_tt += n_i * ti * (ti-1) * pow_ddi * pow_tti/this->tau * exp_dci;
        this->phir_dt += n_i * ti * pow_ddi/this->dv * pow_tti * (di - ci*pow_dci) * exp_dci;
    }
    for (int i = 0; i < 3; i++) // third term i=52-54
    {
        int ii = i + 51;

        double n_i = iapws95::ni[ii];
        int di = iapws95::di[ii];
        double ti = iapws95::ti[ii];
        double bi = iapws95::bi[i];
        double ji = iapws95::ji[i];
        double ai = iapws95::ai[0];
        int ei = iapws95::ei[0];

        double exp_ade = std::exp(-ai * std::pow(this->dv-ei, 2));  // exp(-ai (d-ei)^2
        double exp_bty = std::exp(-bi * std::pow(this->tau-ji, 2)); // exp(-bi (tau-ji)^2)
        double pow_ddi = std::pow(this->dv, di);
        double pow_tti = std::pow(this->tau, ti);
        double term = n_i * pow_ddi * pow_tti * exp_ade * exp_bty;

        this->phir_t += term * (ti/this->tau - 2*bi * (this->tau-ji));
        this->phir_tt += term * (std::pow(ti/this->tau - 2*bi * (this->tau-ji), 2) - ti/std::pow(this->tau, 2) - 2*bi);
        this->phir_dt += term * (di/this->dv - 2*ai*(this->dv-ei)) * (ti/this->tau - 2*bi*(this->tau-ji));
    }
    for (int i = 0; i < 2; i++) // fourth term i=55,56
    {
        int ii = i + 54;
        double n_i = iapws95::ni[ii];
        double ai = iapws95::ai[i+3];
        double bi = iapws95::bi[i+3];
        double Ai = iapws95::Ai[i];
        double Bi = iapws95::Bi[i];
        double betai = iapws95::betai[i];

        double d_1_squared = std::pow(this->dv-1., 2);
        double theta = (1.-tau) + Ai * std::pow(d_1_squared, 0.5/betai);
        double D = std::pow(theta, 2) + Bi * std::pow(d_1_squared, ai);
        double pow_Dbi = std::pow(D, bi);

        double dDdd = (this->dv-1)*(Ai*theta*2/betai*std::pow(d_1_squared, 0.5/betai-1) - 2*Bi*ai*std::pow(d_1_squared, ai-1));
        double dDbdd = bi * std::pow(D, bi-1) * dDdd;
        double dDbdt = -2*theta*bi*pow_Dbi/D;
        double d2Dbdt2 = 2*bi*pow_Dbi/D + 4*std::pow(theta, 2)*bi*(bi-1)*pow_Dbi/std::pow(D, 2);
        double d2Dbdddt = -Ai*bi*2/betai*pow_Dbi/D * (this->dv-1)*std::pow(d_1_squared, 0.5/betai-1) - 2*theta*bi*(bi-1)*pow_Dbi/std::pow(D, 2) * dDdd;
            
        int Ci = iapws95::Ci[i];
        int Di = iapws95::Di[i];
        double psi = std::exp(-Ci*d_1_squared - Di*std::pow(this->tau-1., 2));
        double dpsidd = -2*Ci * (this->dv-1)*psi;
        double dpsidt = -2*Di * (this->tau-1)*psi;
        double d2psidt2 = (2*Di * std::pow(this->tau-1, 2) - 1) * 2 * Di * psi;
        double d2psidddt = 4*Ci*Di * (this->dv-1) * (this->tau-1) * psi;

        this->phir_t += n_i * this->dv * (dDbdt * psi + pow_Dbi * dpsidt);
        this->phir_tt += n_i * this->dv * (d2Dbdt2 * psi + 2*dDbdt * dpsidt + pow_Dbi*d2psidt2);
        this->phir_dt += n_i * (pow_Dbi * (dpsidt + this->dv * d2psidddt) + this->dv * dDbdd * dpsidt + dDbdt * (psi + this->dv * dpsidd) + d2Dbdddt * this->dv * psi);
    }
    return;
}

std::vector<double> IAPWS95::lnphi0(double X, double T_, bool pt)
{
    (void) pt;
    this->init_PT(X, T_);
    std::vector<double> lnphi0_(nc, NAN); 
    lnphi0_[compdata.water_index] = this->lnphii(compdata.water_index);
    return lnphi0_;
}

double IAPWS95::F()
{
    // Reduced residual Helmholtz function: Ar/RT = N phi
    return N * this->phir;
}
double IAPWS95::dF_dV()
{
    // dF/dV = dF/dd dd/dV
    return N * this->phir_d * this->dd_dV();
}
double IAPWS95::dF_dT()
{
    // dF/dT = dF/dtau dtau/dT
    return N * this->phir_t * this->dtau_dT();
}
double IAPWS95::dF_dni(int i)
{
    // dF/dni = f + N df/dd dd/dni
    if (i == compdata.water_index)
    {
        return phir + N * phir_d * this->dd_dni();
    }
    else
    {
        return 0.;
    }
}
double IAPWS95::d2F_dnidnj(int i, int j)
{
    // d2F/dnidnj = df/dd dd/dnj + df/dd dd/dni + N d/dnj df/dd dd/dni
    if (i == compdata.water_index && j == compdata.water_index)
    {
        return 2. * this->phir_d * this->dd_dni() 
                + N * this->phir_dd * std::pow(this->dd_dni(), 2) + N * this->phir_d * this->d2d_dni2();
    }
    else
    {
        return 0.;
    }
}
double IAPWS95::d2F_dTdni(int i)
{
    if (i == compdata.water_index)
    {
        return phir_t * this->dtau_dT() + N * phir_dt * this->dd_dni() * this->dtau_dT();
    }
    else
    {
        return 0.;
    }
}
double IAPWS95::d2F_dVdni(int i)
{
    if (i == compdata.water_index)
    {
        return this->phir_d * this->dd_dV() 
                + N * this->phir_dd * this->dd_dV() * this->dd_dni() + N * this->phir_d * this->d2d_dVdni();
    }
    else
    {
        return 0.;
    }
}
double IAPWS95::d2F_dTdV()
{
    return N * this->phir_dt * this->dd_dV() * this->dtau_dT();
}
double IAPWS95::d2F_dV2()
{
    return N * (this->phir_dd * std::pow(this->dd_dV(), 2) + this->phir_d * this->d2d_dV2());
}
double IAPWS95::d2F_dT2()
{
    return N * (this->phir_tt * std::pow(this->dtau_dT(), 2) + this->phir_t * this->d2tau_dT2());
}
double IAPWS95::d3F_dV3()
{
    return 0.;
    // return N * (this->phir_ddd * std::pow(this->dd_dV(), 3) + this->phir_dd * 2. * this->dd_dV() * this->d2d_dV2() 
    //             + this->phir_dd * this->d2d_dV2() + this->phir_d * this->d3d_dV3());
}

void IAPWS95::set_phir()
{
    // Calculate residual contribution
    phir = 0.;
    for (int i = 0; i < 7; i++)  // first term i=1-7
    {
        phir += iapws95::ni[i] * std::pow(this->dv, iapws95::di[i]) * std::pow(this->tau, iapws95::ti[i]);
    }
    for (int i = 7; i < 51; i++) // second term i=8-51
    {
        phir += iapws95::ni[i] * std::pow(this->dv, iapws95::di[i]) * std::pow(this->tau, iapws95::ti[i]) * std::exp(-std::pow(this->dv, iapws95::ci[i]));
    }
    for (int i = 0; i < 3; i++) // third term i=52-54
    {
        int ii = i + 51;
        double ai = iapws95::ai[0];
        int ei = iapws95::ei[0];

        double exp_ade = std::exp(-ai * std::pow(this->dv-ei, 2));  // exp(-ai (d-ei)^2
        double exp_bty = std::exp(-iapws95::bi[i] * std::pow(this->tau-iapws95::ji[i], 2)); // exp(-bi (tau-ji)^2)
        phir += iapws95::ni[ii] * std::pow(this->dv, iapws95::di[ii]) * std::pow(this->tau, iapws95::ti[ii]) * exp_ade * exp_bty;
    }
    for (int i = 0; i < 2; i++) // fourth term i=55,56
    {
        int ii = i + 54;
        double d_1_squared = std::pow(this->dv-1., 2);
        double theta = (1.-tau) + iapws95::Ai[i] * std::pow(d_1_squared, 0.5/iapws95::betai[i]);
        double D = std::pow(theta, 2) + iapws95::Bi[i] * std::pow(d_1_squared, iapws95::ai[i+3]);
        double psi = std::exp(-iapws95::Ci[i]*d_1_squared - iapws95::Di[i]*std::pow(this->tau-1., 2));
        phir += iapws95::ni[ii] * std::pow(D, iapws95::bi[i+3]) * this->dv * psi;
    }
}
void IAPWS95::set_dphir()
{
    // Calculate derivatives of phir w.r.t. d and dd to calculate P, dP/dd and d2P/dd2 for solving for volume
    this->phir_d = 0.;
    this->phir_dd = 0.;
    this->phir_ddd = 0.;
    for (int i = 0; i < 7; i++)  // first term i=1-7
    {
        double n_i = iapws95::ni[i];
        int di = iapws95::di[i];
        double ti = iapws95::ti[i];

        double ni_di_ddi_tauti = n_i * di * std::pow(this->dv, di-1.) * std::pow(this->tau, ti);
        this->phir_d += ni_di_ddi_tauti;
        this->phir_dd += ni_di_ddi_tauti * (di-1) / dv;
        this->phir_ddd += ni_di_ddi_tauti * (di-1) * (di-2) / std::pow(dv, 2);
    }
    for (int i = 7; i < 51; i++) // second term i=8-51
    {
        double n_i = iapws95::ni[i];
        int ci = iapws95::ci[i];
        int di = iapws95::di[i];
        double ti = iapws95::ti[i];

        double pow_dci = std::pow(this->dv, ci);
        double pow_ddi = std::pow(this->dv, di-1);
        double pow_tti = std::pow(this->tau, ti);
        double ni_edc = n_i * std::exp(-pow_dci);
        this->phir_d += ni_edc * pow_ddi * pow_tti * (di - ci * pow_dci);
        
        double d2b = (di - ci * pow_dci) * (di - 1. - ci * pow_dci) - std::pow(ci, 2)*pow_dci;
        double dd2b = -std::pow(ci, 2) * pow_dci/this->dv * (di - 1. - ci * pow_dci + di - ci * pow_dci) - std::pow(ci, 3)*pow_dci/this->dv;
        double d2a = pow_tti * ni_edc * pow_ddi / this->dv * d2b;
        double dd2a = pow_tti * ni_edc * (-ci * pow_dci * pow_ddi / std::pow(this->dv, 2) * d2b 
                                        + (di-2.) * pow_ddi / std::pow(this->dv, 2) * d2b 
                                        + pow_ddi / this->dv * dd2b);
        this->phir_dd += d2a;
        this->phir_ddd += dd2a;
    }
    for (int i = 0; i < 3; i++) // third term i=52-54
    {
        int ii = i + 51;

        double n_i = iapws95::ni[ii];
        int di = iapws95::di[ii];
        double ti = iapws95::ti[ii];
        double bi = iapws95::bi[i];
        double ji = iapws95::ji[i];
        double ai = iapws95::ai[0];
        int ei = iapws95::ei[0];

        double exp_ade = std::exp(-ai * std::pow(this->dv-ei, 2));  // exp(-ai (d-ei)^2
        double exp_bty = std::exp(-bi * std::pow(this->tau-ji, 2)); // exp(-bi (tau-ji)^2)
        double pow_ddi = std::pow(this->dv, di);
        double ni_tauti_exp = n_i * std::pow(this->tau, ti) * exp_ade * exp_bty;
        this->phir_d += ni_tauti_exp * pow_ddi * (di/this->dv - 2*ai * (this->dv-ei));

        double d2 = -2*ai*pow_ddi + 4*std::pow(ai, 2) * pow_ddi * std::pow(this->dv-ei, 2) - 4*di*ai*pow_ddi/this->dv*(this->dv-ei) + di*(di-1)*pow_ddi/std::pow(this->dv, 2);
        this->phir_dd += ni_tauti_exp * d2;
        double dd2 = -2.*ai*di*pow_ddi/this->dv + 4*std::pow(ai, 2) * (di * pow_ddi/this->dv * std::pow(this->dv-ei, 2) + 2. * pow_ddi * (this->dv-ei))
                      - 4*di*ai*((di-1.) * pow_ddi/std::pow(this->dv, 2)*(this->dv-ei) + pow_ddi/this->dv)
                      + di*(di-1)*(di-2)*pow_ddi/std::pow(this->dv, 3);
        this->phir_ddd += ni_tauti_exp * (-ai * 2. * (this->dv-ei) * d2 + dd2);
    }
    for (int i = 0; i < 2; i++) // fourth term i=55,56
    {
        int ii = i + 54;
        double n_i = iapws95::ni[ii];
        double ai = iapws95::ai[i+3];
        double bi = iapws95::bi[i+3];
        double Ai = iapws95::Ai[i];
        double Bi = iapws95::Bi[i];
        double betai = iapws95::betai[i];

        double d_1_squared = std::pow(this->dv-1., 2);
        double theta = (1.-tau) + Ai * std::pow(d_1_squared, 0.5/betai);
        double dtheta = Ai * (0.5/betai) * std::pow(d_1_squared, 0.5/betai - 1.) * 2. * (this->dv-1.);
        double D = std::pow(theta, 2) + Bi * std::pow(d_1_squared, ai);
        double dDdd = (this->dv-1)*(Ai*theta*2./betai*std::pow(d_1_squared, 0.5/betai-1.) + 2*Bi*ai*std::pow(d_1_squared, ai-1.));
        double dDbdd = bi * std::pow(D, bi-1.) * dDdd;

        double d2 = 4*Bi*ai*(ai-1.)*std::pow(d_1_squared, ai-2.)
                    + 2.*std::pow(Ai, 2)*std::pow(betai, -2) * std::pow(std::pow(d_1_squared, 0.5/betai-1.), 2)
                    + Ai*theta*4/betai * (0.5/betai-1.)*std::pow(d_1_squared, 0.5/betai-2.);
        double dd2 = 4*Bi*ai*(ai-1.)*(ai-2.)*std::pow(d_1_squared, ai-3.) * 2.*(this->dv-1.)
                    + 2.*std::pow(Ai, 2)*std::pow(betai, -2) * 2. * std::pow(d_1_squared, 0.5/betai-1.) * (0.5/betai-1.) * std::pow(d_1_squared, 0.5/betai-2.) * 2. * (this->dv-1.)
                    + Ai*dtheta*4/betai * (0.5/betai-1.) * std::pow(d_1_squared, 0.5/betai-2.)
                    + Ai*theta*4/betai * (0.5/betai-1.) * (0.5/betai-2.) * std::pow(d_1_squared, 0.5/betai-3.) * 2. * (this->dv-1.);
        
        double d2Ddd2 = dDdd/(this->dv-1.) + d_1_squared * d2;
        double d2Dbdd2 = bi * (std::pow(D, bi-1) * d2Ddd2 + (bi-1)*std::pow(D, bi-2.)*std::pow(dDdd, 2));
        double d3Ddd3 = d2Ddd2/(this->dv-1.) - dDdd/d_1_squared 
                        + 2. * (this->dv-1.) * d2 + d_1_squared * dd2;
        double d3Dbdd3 = bi * ((bi-1.) * std::pow(D, bi-2) * dDdd * d2Ddd2 + std::pow(D, bi-1) * d3Ddd3
                             + (bi-1.)*(bi-2.)*std::pow(D, bi-3.)*std::pow(dDdd, 3) + 2. * (bi-1)*std::pow(D, bi-2.) * dDdd * d2Ddd2);
        
        int Ci = iapws95::Ci[i];
        int Di = iapws95::Di[i];
        double psi = std::exp(-Ci*d_1_squared - Di*std::pow(this->tau-1., 2));
        double dpsidd = -2.*Ci * (this->dv-1)*psi;
        double d2psidd2 = (2*Ci * d_1_squared - 1.) * 2. * Ci * psi;
        double d3psidd3 = 4*Ci * (this->dv-1.) * 2. * Ci * psi + (2*Ci * d_1_squared - 1.) * 2. * Ci * dpsidd;

        this->phir_d += n_i * (std::pow(D, bi) * (psi + this->dv * dpsidd) + dDbdd * this->dv * psi);
        this->phir_dd += n_i * (std::pow(D, bi) * (2*dpsidd + this->dv * d2psidd2) + 2*dDbdd*(psi + this->dv * dpsidd) + d2Dbdd2*this->dv*psi);
        this->phir_ddd += n_i * (bi * std::pow(D, bi-1.) * dDdd * (2*dpsidd + this->dv * d2psidd2) + std::pow(D, bi) * (2*d2psidd2 + d2psidd2 + this->dv * d3psidd3) 
                                + 2*d2Dbdd2*(psi + this->dv * dpsidd) + 2*dDbdd*(dpsidd + dpsidd + this->dv * d2psidd2) 
                                + d3Dbdd3*this->dv*psi + d2Dbdd2*psi + d2Dbdd2*this->dv*dpsidd);
    }
}
double IAPWS95::phi0()
{
    // Calculate ideal gas contribution
    double phi_0 = std::log(this->dv) + iapws95::ni0[0] + iapws95::ni0[1] * this->tau + iapws95::ni0[2] * std::log(this->tau);
    for (int i = 3; i < 8; i++)
    {
        phi_0 += iapws95::ni0[i] * std::log(1. - std::exp(-iapws95::ji0[i] * this->tau));
    }
    return phi_0;
}
double IAPWS95::phi0_d()
{
    // Derivative of phi0 w.r.t. d
    return 1./this->dv;
}
double IAPWS95::phi0_dd()
{
    // Derivatives of phi0 w.r.t. dd
    return -1./std::pow(this->dv, 2);
}
double IAPWS95::phi0_t()
{
    // Derivative of phi0 w.r.t. t
    double phi0t = iapws95::ni0[1] + iapws95::ni0[2] / this->tau;
    for (int i = 3; i < 8; i++)
    {
        phi0t += iapws95::ni0[i] * iapws95::ji0[i] * (1./(1. - std::exp(-iapws95::ji0[i] * this->tau)) - 1.);
    }
    return phi0t;
}
double IAPWS95::phi0_tt()
{
    // Derivative of phi0 w.r.t. tt
    double phi0tt = -iapws95::ni0[2] / std::pow(this->tau, 2);
    for (int i = 3; i < 8; i++)
    {
        phi0tt -= iapws95::ni0[i] * std::pow(iapws95::ji0[i], 2) * std::exp(-iapws95::ji0[i] * this->tau) / std::pow(1. - std::exp(-iapws95::ji0[i] * this->tau), 2);
    }   
    return phi0tt;
}

double IAPWS95::cpi(double T_, int i)
{
    if (this->iapws_ideal)
    {
        // Ideal gas heat capacity at constant pressure from IAPWS-95 reference
        this->set_tau(T_);
        return -std::pow(this->tau, 2) * this->phi0_tt() + 1.;  // cpi/R
    }
    else
    {
        return EoS::cpi(T_, i);
    }
}
double IAPWS95::hi(double T_, int i)
{
    if (this->iapws_ideal)
    {
        // Ideal gas enthalpy from IAPWS-95 reference
        this->set_tau(T_);
        return (1. + this->tau * this->phi0_t()) * T_;  // hi/R
    }
    else
    {
        return EoS::hi(T_, i);
    }
}
double IAPWS95::si(double X, double T_, int i, bool pt)
{
    if (this->iapws_ideal)
    {
        // Ideal gas entropy from IAPWS-95 reference
        if (pt)
        {
            double siVT = this->si(this->v, T_, i, false);
            return siVT + std::log(T_ / this->compdata.T_0) + std::log(this->vm / this->compdata.V_0) 
                        - std::log(X / this->compdata.P_0);  // si(PT)/R
        }
        else
        {
            this->set_d(X);
            this->set_tau(T_);
            return this->tau * this->phi0_t() - this->phi0();  // si(VT)/R
        }
    }
    else
    {
        return EoS::si(X, T_, i, pt);
    }
}
// double IAPWS95::dsi_dP(double X, double T_, int i, bool pt)
// {
//     if (this->iapws_ideal)
//     {
//         // Ideal gas entropy from IAPWS-95 reference
//         if (pt)
//         {
//             this->set_d(this->v);
//             this->set_tau(T_);
//             double dsiVT = (this->tau * this->phi0_dt() - this->phi0_dd()) * this->dd_dV() / this->dP_dV();
//             return dsiVT - 1./X + 1./this->dP_dV()/this->v;  // 1/R dsi(PT)/dP
//         }
//         else
//         {
//             this->set_d(X);
//             this->set_tau(T_);
//             return (this->tau * this->phi0_dt() - this->phi0_d()) * this->dd_dV() / this->dP_dV();  // 1/R dsi(VT)/dP
//         }
//     }
//     else
//     {
//         return HelmholtzEoS::dsi_dP(X, T_, i, pt);
//     }
// }
// double IAPWS95::dsi_dT(double X, double T_, int i, bool pt)
// {
//     if (this->iapws_ideal)
//     {
//         // Ideal gas entropy from IAPWS-95 reference
//         if (pt)
//         {
//             this->set_d(this->v);
//             this->set_tau(T_);
//             double dsiVT = this->dtau_dT() * (this->tau * this->phi0_tt()) - this->phi0_d() * this->dd_dV() * this->dV_dT();
//             return dsiVT + 1./T_ + this->dV_dT()/this->v;  // 1/R dsi(PT)/dT
//         }
//         else
//         {
//             this->set_d(X);
//             this->set_tau(T_);
//             return this->dtau_dT() * this->tau * this->phi0_tt();  // 1/R dsi(VT)/dT
//         }
//     }
//     else
//     {
//         return EoS::dsi_dT(X, T_, i, pt);
//     }
// }
std::vector<double> IAPWS95::dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
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

int IAPWS95::derivatives_test(double V_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Test derivatives of Helmholtz function F(T,V,n) = Ar(T,V,n)/RT
    double T0 = T_;
    double V0 = V_;
    std::vector<double> n0 = n_;
    int error_output = HelmholtzEoS::derivatives_test(V0, T0, n0, tol, verbose);
    double d;

    // Test derivatives of dv and tau
    this->solve_VT(V0, T0, n0, 0, true);
    double d0 = this->dv;
    double dV_an = this->dd_dV();
    double dV2_an = this->d2d_dV2();
    double dVdn_an = this->d2d_dVdni();
    double dn_an = this->dd_dni();
    // double dn2_an = this->d2d_dni2();

    double t0 = this->tau;
    double dT_an = this->dtau_dT();
    double dT2_an = this->d2tau_dT2();

    // Derivatives with respect to volume
    double dV = 1e-5 * V0;
    this->solve_VT(V0 + dV, T0, n0, 0, true);
    
    double dV_num = (this->dv - d0) / dV;
    d = std::log(std::fabs(dV_an + 1e-15)) - std::log(std::fabs(dV_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 dd/dV", {dV_an, dV_num, d}); error_output++; }

    double dV2_num = (this->dd_dV() - dV_an) / dV;
    d = std::log(std::fabs(dV2_an + 1e-15)) - std::log(std::fabs(dV2_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 d2d/dV2", {dV2_an, dV2_num, d}); error_output++; }

    double dVdn_num = (this->dd_dni() - dn_an) / dV;
    d = std::log(std::fabs(dVdn_an + 1e-15)) - std::log(std::fabs(dVdn_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 d2d/dVdni", {dVdn_an, dVdn_num, d}); error_output++; }

    // Derivatives with respect to temperature
    double dT = 1e-5 * T0;
    this->solve_VT(V0, T0 + dT, n0, 0, true);

    double dT_num = (this->tau - t0) / dT;
    d = std::log(std::fabs(dT_an + 1e-15)) - std::log(std::fabs(dT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 dtau/dT", {dT_an, dT_num, d}); error_output++; }

    double dT2_num = (this->dtau_dT() - dT_an) / dT;
    d = std::log(std::fabs(dT2_an + 1e-15)) - std::log(std::fabs(dT2_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 d2tau/dT2", {dT2_an, dT2_num, d}); error_output++; }

    // Derivatives with respect to composition
    double dni = 1e-5 * n0[0];
    n0[0] += dni;
    this->solve_VT(V0, T0, n0, 0, true);
    n0[0] -= dni;

    double dn_num = (this->dv - d0) / dni;
    d = std::log(std::fabs(dn_an + 1e-15)) - std::log(std::fabs(dn_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 dd/dni", {dn_an, dn_num, d}); error_output++; }

    // double dn2_num = (this->dd_dni() - dn_an) / dni;
    // d = std::log(std::fabs(dn2_an + 1e-15)) - std::log(std::fabs(dn2_num + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 d2d/dni2", {dn2_an, dn2_num, d}); error_output++; }

    // Test derivatives of P with respect to dv
    this->solve_VT(V0, T0, n0, 0, false);
    double p0 = this->p;
    double dd_an = this->dP_dd();

    double phir0 = this->phir;
    double phird_an = this->phir_d;
    double phirdd_an = this->phir_dd;
    double phirddd_an = this->phir_ddd;

    this->solve_VT(V0 + dV, T0, n0, 0, false);
    double p1 = this->p;
    double dd_num = (p1 - p0) / (this->dv - d0);

    double phird_num = (this->phir - phir0) / (this->dv - d0);
    double phirdd_num = (this->phir_d - phird_an) / (this->dv - d0);
    double phirddd_num = (this->phir_dd - phirdd_an) / (this->dv - d0);

    d = std::log(std::fabs(dd_an + 1e-15)) - std::log(std::fabs(dd_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 dP/dd", {dd_an, dd_num, d}); error_output++; }

    d = std::log(std::fabs(phird_an + 1e-15)) - std::log(std::fabs(phird_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 dphir/dd", {phird_an, phird_num, d}); error_output++; }

    d = std::log(std::fabs(phirdd_an + 1e-15)) - std::log(std::fabs(phirdd_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 d2phir/dd2", {phirdd_an, phirdd_num, d}); error_output++; }

    d = std::log(std::fabs(phirddd_an + 1e-15)) - std::log(std::fabs(phirddd_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("IAPWS95 d3phir/dd3", {phirddd_an, phirddd_num, d}); error_output++; }

    return error_output;
}

int IAPWS95::references_test(double tol, bool verbose)
{
    // Test reference values for phi0, phir and thermodynamic properties
    int error_output = 0;

    // Test phi0 and phir and derivatives
    double rho = 838.025;
    this->v = iapws95::Mw * 1e-3 / rho;
    this->T = 500.;
    this->n = {1.};
    this->solve_VT(v, T, n, 0, true);

    if (verbose || std::fabs(iapws95ref::phi0 - this->phi0()) > tol) { print("IAPWS95 phi0", {iapws95ref::phi0, this->phi0()}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phi0_d - this->phi0_d()) > tol) { print("IAPWS95 phi0/d", {iapws95ref::phi0_d, this->phi0_d()}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phi0_dd - this->phi0_dd()) > tol) { print("IAPWS95 phi0/dd", {iapws95ref::phi0_dd, this->phi0_dd()}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phi0_dt - this->phi0_dt()) > tol) { print("IAPWS95 phi0/dt", {iapws95ref::phi0_dt, this->phi0_dt()}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phi0_t - this->phi0_t()) > tol) { print("IAPWS95 phi0/t", {iapws95ref::phi0_t, this->phi0_t()}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phi0_tt - this->phi0_tt()) > tol) { print("IAPWS95 phi0/tt", {iapws95ref::phi0_tt, this->phi0_tt()}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phir - this->phir) > tol) { print("IAPWS95 phir", {iapws95ref::phir, this->phir}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phir_d - this->phir_d) > tol) { print("IAPWS95 phir/d", {iapws95ref::phir_d, this->phir_d}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phir_dd - this->phir_dd) > tol) { print("IAPWS95 phir/dd", {iapws95ref::phir_dd, this->phir_dd}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phir_dt - this->phir_dt) > tol) { print("IAPWS95 phir/dt", {iapws95ref::phir_dt, this->phir_dt}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phir_t - this->phir_t) > tol) { print("IAPWS95 phir/t", {iapws95ref::phir_t, this->phir_t}); error_output++; }
    if (verbose || std::fabs(iapws95ref::phir_tt - this->phir_tt) > tol) { print("IAPWS95 phir/tt", {iapws95ref::phir_tt, this->phir_tt}); error_output++; }

    // Test thermodynamic properties against reference values
    bool pt = false;
    for (size_t ith_condition = 0; ith_condition < iapws95ref::T.size(); ith_condition++)
    {
        double V_ = iapws95::Mw * 1e-3 / iapws95ref::rho[ith_condition];
        this->solve_VT(V_, iapws95ref::T[ith_condition], n, 0, true);

        // Reference values for P
        double pref = unit_conversion::input_to_bar[Units::PRESSURE::MPA] * iapws95ref::p[ith_condition];
        if (verbose || std::fabs(pref - this->p) > tol) { print("IAPWS95 p", {static_cast<double>(ith_condition), pref, this->p}); error_output++; }

        // Reference values for Cv
        double cvref = iapws95ref::cv[ith_condition];
        double cv = this->Cv(this->v, this->T, this->n, 0, pt) * iapws95::R;
        if (verbose || std::fabs(cvref - cv) > tol) { print("IAPWS95 Cv", {static_cast<double>(ith_condition), cvref, cv}); error_output++; }

        // Reference values for sound speed
        // double wref = iapws95ref::w[ith_condition];
        // double w = this->vs(this->v, this->T, this->n, 0, pt);
        // if (verbose || std::fabs(wref - w) > tol) { print("IAPWS95 W", {static_cast<double>(ith_condition), wref, w}); error_output++; }

        // Reference values for S
        double sref = iapws95ref::s[ith_condition];
        double s = this->S(this->v, this->T, this->n, 0, pt) * iapws95::R;
        if (verbose || std::fabs(sref - s) > tol) { print("IAPWS95 S", {static_cast<double>(ith_condition), sref, s}); error_output++; }
    }

    return error_output;
}
