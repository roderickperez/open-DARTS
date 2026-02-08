#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cassert>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/units.hpp"
#include "dartsflash/eos/eos.hpp"
#include <Eigen/Dense>

EoS::EoS(CompData& comp_data) 
{
    this->compdata = comp_data;
    this->nc = comp_data.nc;
    this->ni = comp_data.ni;
    this->ns = nc + ni;
    this->n.resize(ns);
	
    this->dlnphidP.resize(ns);
    this->dlnphidT.resize(ns);
    this->dlnphidn.resize(ns*ns);
    this->d2lnphidPdT.resize(ns);
    this->d2lnphidT2.resize(ns);
    this->d2lnphidTdn.resize(ns*ns);

    this->units = comp_data.units;
}

void EoS::set_eos_range(int i, const std::vector<double>& range)
{
    // Define range of applicability for specific EoS
    this->eos_range[i] = range;
    return;
}
bool EoS::eos_in_range(std::vector<double>::iterator n_it, bool in_stationary_point_range)
{
    // Check if EoS is applicable for specific state according to specified ranges
    if (!in_stationary_point_range)
    {
        for (auto& it: this->eos_range) 
        {
            this->N = std::accumulate(n_it, n_it + this->ns, 0.);
            double xi = *(n_it + it.first) / this->N;
    		if (xi < it.second[0] || xi > it.second[1])
            {
                return false;
            }
	    }
    }
    else
    {
        for (auto& it: this->stationary_point_range)
        {
            this->N = std::accumulate(n_it, n_it + this->ns, 0.);
            double xi = *(n_it + it.first) / this->N;
    		if (xi < it.second[0] || xi > it.second[1])
            {
                return false;
            }
	    }
    }

    return true;
}

void EoS::solve_PT(double p_, double T_, std::vector<double>& n_, int start_idx, bool second_order)
{
    // Calculate composition-independent and composition-dependent EoS-parameters
    this->init_PT(p_, T_);
    this->solve_PT(n_.begin() + start_idx, second_order);
    return;
}
void EoS::solve_VT(double V_, double T_, std::vector<double>& n_, int start_idx, bool second_order)
{
    // Calculate composition-independent and composition-dependent EoS-parameters
    this->init_VT(V_, T_);
    this->solve_VT(n_.begin() + start_idx, second_order);
    return;
}

EoS::RootFlag EoS::is_root_type(std::vector<double>::iterator n_it, bool& is_below_spinodal, bool pt)
{
    if (pt)
    {
        this->solve_PT(n_it, false); 
    }
    else
    {
        this->solve_VT(n_it, false); 
    }
    return this->is_root_type(is_below_spinodal); 
}
EoS::RootFlag EoS::is_root_type(double X, double T_, std::vector<double>& n_, bool& is_below_spinodal, int start_idx, bool pt) 
{ 
	if (pt)
	{
		this->solve_PT(X, T_, n_, start_idx, false);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, false);
    }
    return this->is_root_type(is_below_spinodal); 
}

std::vector<double> EoS::lnphi()
{
    // Return vector of lnphi for each component
    std::vector<double> ln_phi(ns);
    for (int i = 0; i < ns; i++)
    {
        ln_phi[i] = this->lnphii(i);
    }
    return ln_phi;
}
std::vector<double> EoS::lnphi(double p_, double T_, std::vector<double>& n_)
{
    this->solve_PT(p_, T_, n_, 0, false);
    return this->lnphi();
}
std::vector<double> EoS::dlnphi_dn()
{
    for (int i = 0; i < ns; i++)
    {
        for (int j = 0; j < ns; j++)
        {
            dlnphidn[i * ns + j] = this->dlnphii_dnj(i, j);
        }
    }
    return dlnphidn;
}
std::vector<double> EoS::dlnphi_dP()
{
    for (int i = 0; i < ns; i++)
    {
        dlnphidP[i] = this->dlnphii_dP(i);
    }
    return dlnphidP;
}
std::vector<double> EoS::dlnphi_dT()
{
    for (int i = 0; i < ns; i++)
    {
        dlnphidT[i] = this->dlnphii_dT(i);
    }
    return dlnphidT;
}
std::vector<double> EoS::d2lnphi_dPdT()
{
    for (int i = 0; i < ns; i++)
    {
        d2lnphidPdT[i] = this->d2lnphii_dPdT(i);
    }
    return d2lnphidPdT;
}
std::vector<double> EoS::d2lnphi_dT2()
{
    for (int i = 0; i < ns; i++)
    {
        d2lnphidT2[i] = this->d2lnphii_dT2(i);
    }
    return d2lnphidT2;
}
std::vector<double> EoS::d2lnphi_dTdn()
{
    for (int i = 0; i < ns; i++)
    {
        for (int j = 0; j < ns; j++)
        {
            d2lnphidTdn[i*ns + j] = this->d2lnphii_dTdnj(i, j);
        }
    }
    return d2lnphidTdn;
}

std::vector<double> EoS::fugacity(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate component fugacities in mixture
    if (this->eos_in_range(n_.begin() + start_idx))
    {
	    // Calculate fugacity coefficients and fugacity
        std::vector<double> fi(ns);
        double nT_inv = 1./std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);

        if (pt)
        {
            this->solve_PT(X, T_, n_, start_idx, false);
        }
        else
        {
            this->solve_VT(X, T_, n_, start_idx, false);
        }
        
        for (int i = 0; i < nc; i++)
        {
            fi[i] = std::exp(this->lnphii(i)) * n_[start_idx + i] * nT_inv * this->p;
        }
        return fi;
    }
    else
    {
        return std::vector<double>(ns, NAN);
    }
}

Eigen::MatrixXd EoS::calc_hessian(std::vector<double>::iterator n_it)
{
    // Calculate dlnphii/dnj
    this->solve_PT(n_it);
    this->dlnphi_dn();

    // Construct Hessian matrix of Gibbs energy surface
    Eigen::MatrixXd H_ = Eigen::MatrixXd(nc, nc);
    for (int j = 0; j < nc; j++)
    {
        for (int i = j; i < nc; i++)
        {
            H_(i, j) = dlnphidn[i*nc + j];  // PHI contribution
            H_(j, i) = dlnphidn[i*nc + j];  // PHI contribution
        }
        H_(j, j) += 1. / *(n_it + j);  // U contribution
    }

    return H_;
}
bool EoS::is_convex(std::vector<double>::iterator n_it)
{
    // Check if GE/TPD surface at composition n is convex
    Eigen::MatrixXd H_ = this->calc_hessian(n_it);

    // Perform Cholesky decomposition; only possible with positive definite matrix
    // Positive definite Hessian matrix corresponds to local minimum
    Eigen::LLT<Eigen::MatrixXd> lltOfA(H_);
    if(lltOfA.info() == Eigen::NumericalIssue)
    {
        return false;
    }
    else
    {
        return true;
    }
}
bool EoS::is_convex(double p_, double T_, std::vector<double>& n_, int start_idx)
{
    // Check if GE/TPD surface at P, T and composition n is convex
    this->init_PT(p_, T_);
    return this->is_convex(n_.begin() + start_idx);
}
double EoS::calc_condition_number(double p_, double T_, std::vector<double>& n_, int start_idx)
{
    if (this->eos_in_range(n_.begin() + start_idx))
    {
        // Calculate condition number of Hessian to evaluate curvature of GE/TPD surface at P, T and composition n
        this->init_PT(p_, T_);
        Eigen::MatrixXd H_ = this->calc_hessian(n_.begin() + start_idx);

        // Get the condition number for stability if Newton was used
        Eigen::VectorXd eigen = H_.eigenvalues().real();
        double min_eigen = *std::min_element(eigen.begin(), eigen.end());
        double max_eigen = *std::max_element(eigen.begin(), eigen.end());
        return (min_eigen > 0.) ? max_eigen/min_eigen : NAN;
    }
    else
    {
        return NAN;
    }
}

double EoS::cpi(double T_, int i)
{
    // cpi/R: Ideal gas heat capacity at constant pressure of component i
    return this->compdata.cpi[i][0]
         + this->compdata.cpi[i][1] * T_
         + this->compdata.cpi[i][2] * std::pow(T_, 2)
         + this->compdata.cpi[i][3] * std::pow(T_, 3);
}
double EoS::Cpi(double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Cpi/R: Ideal gas heat capacity at constant pressure
    (void) pt;
    double Cpi_ = 0.;
    for (int i = 0; i < ns; i++)
    {
        Cpi_ += n_[start_idx+i] * this->cpi(T_, i);
    }
    return Cpi_;
}
double EoS::cvi(double T_, int i)
{
    // cvi/R: Ideal gas heat capacity at constant volume of component i
    return this->cpi(T_, i) - 1.;
}
double EoS::Cvi(double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Cvi/R: Ideal gas heat capacity at constant volume
    (void) pt;
    double Cvi_ = 0.;
    for (int i = 0; i < ns; i++)
    {
        Cvi_ += n_[start_idx+i] * this->cvi(T_, i);
    }
    return Cvi_;
}
double EoS::hi(double T_, int i)
{
    // hi/R: Ideal gas enthalpy of component i
    return this->compdata.cpi[i][0] * (T_-this->compdata.T_0)
            + 1. / 2 * this->compdata.cpi[i][1] * (std::pow(T_, 2)-std::pow(this->compdata.T_0, 2))
            + 1. / 3 * this->compdata.cpi[i][2] * (std::pow(T_, 3)-std::pow(this->compdata.T_0, 3))
            + 1. / 4 * this->compdata.cpi[i][3] * (std::pow(T_, 4)-std::pow(this->compdata.T_0, 4));  // hi/R
}
double EoS::Hi(double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Hi/R: Ideal gas enthalpy
    (void) pt;
    double Hi_ = 0.;
    for (int i = 0; i < ns; i++)
    {
        Hi_ += n_[start_idx+i] * this->hi(T_, i);
    }
    return Hi_;
}
double EoS::dHi_dT(double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // 1/R dHi/dT = Cpi/R
    return this->Cpi(T_, n_, start_idx, pt);
}
std::vector<double> EoS::dHi_dni(double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of ideal gas enthalpy with respect to composition
    (void) pt;
    std::vector<double> dHidn(ns);
    for (int i = 0; i < ns; i++)
    {
        dHidn[i] = (n_[start_idx+i]/this->N > 0.) ? this->hi(T_, i) : 0.;
    }
    return dHidn;  // 1/R dHi/dn
}
std::vector<double> EoS::d2Hi_dTdni(double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of ideal gas enthalpy with respect to composition
    (void) pt;
    std::vector<double> d2HidTdn(ns);
    for (int i = 0; i < ns; i++)
    {
        d2HidTdn[i] = (n_[start_idx+i]/this->N > 0.) ? this->cpi(T_, i) : 0.;
    }
    return d2HidTdn;  // 1/R dHi/dn
}
double EoS::si(double X, double T_, int i, bool pt)
{
    if (pt)
    {   // si(P,T)/R: ideal gas entropy at constant pressure of component i
        return this->compdata.cpi[i][0] * std::log(T_ / this->compdata.T_0)
             + this->compdata.cpi[i][1] * (T_ - this->compdata.T_0)
             + 1. / 2 * this->compdata.cpi[i][2] * (std::pow(T_, 2)-std::pow(this->compdata.T_0, 2))
             + 1. / 3 * this->compdata.cpi[i][3] * (std::pow(T_, 3)-std::pow(this->compdata.T_0, 3))
             - std::log(X / this->compdata.P_0);
    }
    else
    {   // si(V,T)/R: ideal gas entropy at constant volume of component i
        return this->compdata.cpi[i][0] * std::log(T_ / this->compdata.T_0)
             + this->compdata.cpi[i][1] * (T_ - this->compdata.T_0)
             + 1. / 2 * this->compdata.cpi[i][2] * (std::pow(T_, 2)-std::pow(this->compdata.T_0, 2))
             + 1. / 3 * this->compdata.cpi[i][3] * (std::pow(T_, 3)-std::pow(this->compdata.T_0, 3))
             - std::log(T_ / this->compdata.T_0) - std::log(X/N / this->compdata.V_0);
    }
}
double EoS::dsi_dP(double X, double T_, int i, bool pt)
{
    (void) T_;
    (void) i;
    if (pt)
    {   // 1/R dsi(P,T)/dP = -1/p
        return -1./X;
    }
    else
    {   // 1/R dsi(V,T)/dT = -dV/dP / N / p
        // return -1./(X * this->dP_dV());
        return 0.;
    }
}
double EoS::dsi_dT(double X, double T_, int i, bool pt)
{
    (void) X;
    if (pt)
    {   // 1/R dsi(P,T)/dT = cpi/T
        return this->cpi(T_, i) / T_;
    }
    else
    {   // 1/R dsi(V,T)/dT = cvi/T
        return this->cvi(T_, i) / T_;
    }
}
double EoS::Si(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal gas entropy
    double Si_ = 0.;
    double nT_inv = 1./std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);
    for (int i = 0; i < ns; i++)
    {
        double xi = n_[start_idx+i] * nT_inv;
        if (xi > 0.)
        {
            Si_ += n_[start_idx+i] * (this->si(X, T_, i, pt) - std::log(xi));
        }
    }
    return Si_;  // Si/R
}
double EoS::dSi_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivative of ideal gas entropy with respect to pressure
    double dSidP = 0.;
    double nT_inv = 1./std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);
    for (int i = 0; i < ns; i++)
    {
        double xi = n_[start_idx+i] * nT_inv;
        if (xi > 0.)
        {
            dSidP += n_[start_idx+i] * this->dsi_dP(X, T_, i, pt);
        }
    }
    return dSidP;  // 1/R dSi/dP
}
double EoS::dSi_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivative of ideal gas entropy with respect to temperature
    double dSidT = 0.;
    double nT_inv = 1./std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);
    for (int i = 0; i < ns; i++)
    {
        double xi = n_[start_idx+i] * nT_inv;
        if (xi > 0.)
        {
            dSidT += n_[start_idx+i] * this->dsi_dT(X, T_, i, pt);
        }
    }
    return dSidT;  // 1/R dSi/dT
}
std::vector<double> EoS::dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate partial molar ideal gas entropy
    std::vector<double> dSidn(ns, 0.);

    double nT_inv = 1./std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);
    for (int i = 0; i < ns; i++)
    {
        // Ideal contribution to Gibbs free energy
        double xi = n_[start_idx+i] * nT_inv;

        if (xi > 0.)
        {
            dSidn[i] += this->si(X, T_, i, pt) - std::log(xi);
            for (int j = 0; j < ns; j++)
            {
                if (i == j)
                {
                    dSidn[i] -= n_[start_idx + i] / xi * (nT_inv - n_[start_idx + j] * std::pow(nT_inv, 2));
                }
                else
                {
                    dSidn[i] -= n_[start_idx + i] / xi * -n_[start_idx + j] * std::pow(nT_inv, 2);
                }
            }
        }
    }
    return dSidn;  // 1/R dSi/dn
}
std::vector<double> EoS::d2Si_dTdni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate partial molar ideal gas entropy
    std::vector<double> d2SidTdn(ns);

    for (int i = 0; i < ns; i++)
    {
        // Ideal contribution to Gibbs free energy
        double xi = n_[start_idx+i] / this->N;
        d2SidTdn[i] = (xi > 0) ? this->dsi_dT(X, T_, i, pt) : 0.;
    }
    return d2SidTdn;  // 1/R d2Si/dTdni
}
double EoS::Gi(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal gas Gibbs free energy
    return this->Hi(T_, n_, start_idx, pt) - T_ * this->Si(X, T_, n_, start_idx, pt);  // Gi/R
}
double EoS::dGi_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivative of ideal gas Gibbs free energy with respect to pressure
    return -T_ * this->dSi_dP(X, T_, n_, start_idx, pt);  // 1/R dGi/dP
}
double EoS::dGi_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivative of ideal gas Gibbs free energy with respect to temperature
    return this->dHi_dT(T_, n_, start_idx, pt) - this->Si(X, T_, n_, start_idx, pt) - T_ * this->dSi_dT(X, T_, n_, start_idx, pt);  // 1/R dGi/dT
}
std::vector<double> EoS::dGi_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivative of ideal gas Gibbs free energy with respect to composition
    std::vector<double> dHidn = this->dHi_dni(T_, n_, start_idx, pt);
    std::vector<double> dSidn = this->dSi_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dGidn(ns, 0.);

    for (int i = 0; i < ns; i++)
    {
        if (n_[start_idx + i]/this->N > 0.)
        {
            dGidn[i] = dHidn[i] - T_ * dSidn[i];
        }
    }
    return dGidn;  // 1/R dGi/dni
}
std::vector<double> EoS::d2Gi_dTdni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate partial molar ideal gas entropy
    std::vector<double> d2GidTdn(ns);

    for (int i = 0; i < ns; i++)
    {
        // Ideal contribution to Gibbs free energy
        double xi = n_[start_idx+i] / this->N;
        d2GidTdn[i] = (xi > 0) ? this->cpi(T_, i) - this->si(X, T_, i, pt) - T_ * this->dsi_dT(X, T_, i, pt) : 0.;
    }
    return d2GidTdn;  // 1/R d2Gi/dTdni
}
/*
double EoS::Ai(double X, double T_, std::vector<double>& n_, bool pt)
{
    // Ideal gas Helmholtz free energy
    return this->Gi(X, T_, n_, pt) + PV;
}
double EoS::Ui(double X, double T_, std::vector<double>& n_, bool pt)
{

}
*/

double EoS::Sr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Residual entropy
    if (pt)
    {
        // Sr(PT)/R = Hr(PT)/RT - Gr(PT)/RT
        double G_r = this->Gr(X, T_, n_, start_idx, pt);  // Gr/R
        double H_r = this->Hr(X, T_, n_, start_idx, pt);  // Hr/R
        return (H_r - G_r) / T;  // Sr/R = (Hr/R - Gr/R)/T
    }
    else
    {
        // Sr(VT)/R = Ur(VT)/RT - Ar(VT)/RT
        // double Ar = this->Ar(X, T_, n_, start_idx, pt);  // Ar/RT
        // double Ur = this->Ur(X, T_, n_, start_idx, pt);  // Ur/RT
        // return (Ur - Ar) / T;  // Sr/R = (Ur/R - Ar/R)/T
        return not_implemented_double();
    }
}
double EoS::Gr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Residual Gibbs free energy
    if (pt)
    {
        // Gr(PT)/RT = Σi ni lnphii
        if (this->eos_in_range(n_.begin() + start_idx))
        {
            this->solve_PT(X, T_, n_, start_idx, false);
            std::vector<double> lnphi_ = this->lnphi();

            double Gr = 0.;
            bool nans = false;
            for (int i = 0; i < ns; i++)
            {
                if (!std::isnan(lnphi_[i]))
                {
                    Gr += n_[start_idx + i] * lnphi_[i];  // partial molar Gibbs energy
                }
                else
                {
                    nans = true;
                }
            }
            // If all NANs, G will be equal to zero, so return NAN; else return G
            return (nans && Gr == 0.) ? NAN : Gr * T;  // Gr/R
        }
        else
        {
            // Outside of EoS range
            return NAN;
        }
    }
    else
    {
        return not_implemented_double();
    }
}
double EoS::Hr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Residual enthalpy
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
        this->dlnphidT = this->dlnphi_dT();

        double Hr = 0.;
        bool nans = false;
        for (int i = 0; i < ns; i++)
        {
            if (!std::isnan(dlnphidT[i]))
            {
                Hr -= n_[start_idx + i] * dlnphidT[i];  // partial molar enthalpy Hri/RT^2
            }
            else
            {
                nans = true;
            }
        }
        // If all NANs, H will be equal to zero, so return NAN; else return H
        return (nans && Hr == 0.) ? NAN : Hr * std::pow(T, 2);  // Hr/R
    }
    else
    {
        return not_implemented_double();
    }
}
/*
double EoS::Ar(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual Helmholtz free energy of mixture at P, T, x
}
double EoS::Ur(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual internal energy of mixture at P, T, x
}
*/

double EoS::dSr_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Derivative of residual entropy with respect to pressure
    if (pt)
    {
        // 1/R dSr(PT)/dP = 1/RT (dHr(PT)/dP - dGr(PT)/dP)
        double dGrdP = this->dGr_dP(X, T_, n_, start_idx, pt);  // 1/R dGr/dP
        double dHrdP = this->dHr_dP(X, T_, n_, start_idx, pt);  // 1/R dHr/dP
        return (dHrdP - dGrdP)/T;  // 1/R dSr/dP = (1/R dHr/dP - 1/R dGr/dP)/T
    }
    else
    {
        // 1/R dSr(VT)/dP = 1/RT (dAr(VT)/dP - dUr(VT)/dP)
        // double dArdP = this->Ar(X, T_, n_, start_idx, pt);  // 1/R dAr/dP
        // double dUrdP = this->Ur(X, T_, n_, start_idx, pt);  // 1/R dUr/dP
        // return (dUrdP - dArdP) / T;  // 1/R dSr/dP = (1/R dUr/dP - 1/R dAr/dP) / T
        return not_implemented_double();
    }
}
double EoS::dGr_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Derivative of residual Gibbs free energy with respect to pressure
    if (pt)
    {
        // Gr(PT)/RT = Σi ni lnphii
        if (this->eos_in_range(n_.begin() + start_idx))
        {
            this->solve_PT(X, T_, n_, start_idx, false);
            this->dlnphidP = this->dlnphi_dP();

            double dGrdP = 0.;
            bool nans = false;
            for (int i = 0; i < ns; i++)
            {
                if (!std::isnan(this->dlnphidP[i]))
                {
                    dGrdP += n_[start_idx + i] * this->dlnphidP[i];  // partial molar Gibbs energy
                }
                else
                {
                    nans = true;
                }
            }
            // If all NANs, G will be equal to zero, so return NAN; else return G
            return (nans && dGrdP == 0.) ? NAN : dGrdP * T;  // 1/R dGr/dP
        }
        else
        {
            // Outside of EoS range
            return NAN;
        }
    }
    else
    {
        return not_implemented_double();
    }
}
double EoS::dHr_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Derivative of residual enthalpy with respect to pressure
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
        this->d2lnphidPdT = this->d2lnphi_dPdT();

        double dHrdP = 0.;
        bool nans = false;
        for (int i = 0; i < ns; i++)
        {
            if (!std::isnan(d2lnphidPdT[i]))
            {
                dHrdP -= n_[start_idx + i] * d2lnphidPdT[i];  // partial molar enthalpy Hri/RT
            }
            else
            {
                nans = true;
            }
        }
        // If all NANs, H will be equal to zero, so return NAN; else return H
        return (nans && dHrdP == 0.) ? NAN : dHrdP * std::pow(T, 2);  // 1/R dHr/dP
    }
    else
    {
        return not_implemented_double();
    }
}
/*
double EoS::Ar(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual Helmholtz free energy of mixture at P, T, x
}
double EoS::Ur(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual internal energy of mixture at P, T, x
}
*/

double EoS::dSr_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Derivative of residual entropy with respect to temperature
    if (pt)
    {
        // 1/R dSr(PT)/dT = 1/RT (dHr(PT)/dT - dGr(PT)/dT)
        double G_r = this->Gr(X, T_, n_, start_idx, pt);  // Gr/R
        double H_r = this->Hr(X, T_, n_, start_idx, pt);  // Hr/R
        double dGrdT = this->dGr_dT(X, T_, n_, start_idx, pt);  // 1/R dGr/dT
        double dHrdT = this->dHr_dT(X, T_, n_, start_idx, pt);  // 1/R dHr/dT
        return (dHrdT - dGrdT) / T - (H_r - G_r) / std::pow(T, 2);  // 1/R dSr/dT = (1/R dHr/dT - 1/R dGr/dT)/T - (Hr/R - Gr/R) / T^2
    }
    else
    {
        // 1/R dSr(VT)/dT = 1/RT (dAr(VT)/dT - dUr(VT)/dT)
        // double dArdT = this->dAr_dT(X, T_, n_, start_idx, pt);  // 1/RT dAr/dT
        // double dUrdT = this->dUr_dT(X, T_, n_, start_idx, pt);  // 1/RT dUr/dT
        // return dUrdT - dArdT;  // 1/R dSr/dT = 1/RT (dUr/dT - dAr/dT)
        return not_implemented_double();
    }
}
double EoS::dGr_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Derivative of residual Gibbs free energy with respect to temperature
    if (pt)
    {
        // Gr(PT)/RT = Σi ni lnphii
        if (this->eos_in_range(n_.begin() + start_idx))
        {
            this->solve_PT(X, T_, n_, start_idx, true);
            std::vector<double> lnphi_ = this->lnphi();
            this->dlnphidT = this->dlnphi_dT();

            double dGrdT = 0.;
            bool nans = false;
            for (int i = 0; i < ns; i++)
            {
                if (!std::isnan(this->dlnphidT[i]))
                {
                    dGrdT += n_[start_idx + i] * (lnphi_[i] + T * this->dlnphidT[i]);
                }
                else
                {
                    nans = true;
                }
            }
            // If all NANs, G will be equal to zero, so return NAN; else return G
            return (nans && dGrdT == 0.) ? NAN : dGrdT;  // 1/R dGr/dT
        }
        else
        {
            // Outside of EoS range
            return NAN;
        }
    }
    else
    {
        return not_implemented_double();
    }
}
double EoS::dHr_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Derivative of residual enthalpy with respect to temperature
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }
    this->dlnphidT = this->dlnphi_dT();
    this->d2lnphidT2 = this->d2lnphi_dT2();

    // Calculate derivative of residual enthalpy Hr w.r.t T
    double dHrdT = 0.;
    for (int i = 0; i < this->ns; i++)
    {
        if (!std::isnan(dlnphidT[i]))
        {
            dHrdT -= n_[start_idx + i] * T * (2. * this->dlnphidT[i] + T * this->d2lnphidT2[i]);
        }
    }
    return dHrdT;  // 1/R dHr/dT = Cpr/R
}
/*
double EoS::Ar(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual Helmholtz free energy of mixture at P, T, x
}
double EoS::Ur(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual internal energy of mixture at P, T, x
}
*/

std::vector<double> EoS::dSr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual entropy with respect to composition
    std::vector<double> dSrdni(ns);
    
    if (pt)
    {
        // Sr(PT)/R = Hr(PT)/RT - Gr(PT)/RT
        std::vector<double> dGrdni = this->dGr_dni(X, T_, n_, start_idx, pt);  // 1/R dGr/dni
        std::vector<double> dHrdni = this->dHr_dni(X, T_, n_, start_idx, pt);  // 1/R dHr/dni

        for (int i = 0; i < ns; i++)
        {
            dSrdni[i] = (dHrdni[i] - dGrdni[i]) / T;
        }
        return dSrdni;  // 1/R dSr/dni = (1/R dHr/dni - 1/R dGr/dni) / T
    }
    else
    {
        // Sr(VT)/R = (1/R dUr/dni - 1/R dAr(VT)/dni) / T
        // std::vector<double> dArdni = this->dAr_dni(X, T_, n_, start_idx, pt);  // 1/R dAr/dni
        // std::vector<double> dUrdni = this->dUr_dni(X, T_, n_, start_idx, pt);  // 1/R dUr/dni

        // for (int i = 0; i < nc; i++)
        // {
        //     dSrdni[i] = (dUrdni[i] - dArdni[i]) / T;
        // }
        return not_implemented_vector(ns);  // 1/R dSr/dni = (1/R dHr/dni - 1/R dGr/dni) / T
    }
}
std::vector<double> EoS::dGr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual Gibbs free energy with respect to composition
    if (pt)
    {
        // Gr(PT)/RT = Σi ni lnphii
        std::vector<double> dGrdni(ns, 0.);
        if (this->eos_in_range(n_.begin() + start_idx))
        {
            this->solve_PT(X, T_, n_, start_idx, true);
            std::vector<double> lnphi_ = this->lnphi();
            this->dlnphidn = this->dlnphi_dn();

            for (int i = 0; i < ns; i++)
            {
                dGrdni[i] += T * lnphi_[i];  // partial molar Gibbs energy
                for (int j = 0; j < ns; j++)
                {
                    dGrdni[j] += n_[start_idx + i] * T * this->dlnphidn[i * ns + j];
                }
            }
            return dGrdni;  // 1/R dGr/dni
        }
        else
        {
            // Outside of EoS range
            return std::vector<double>(ns, NAN);
        }
    }
    else
    {
        return not_implemented_vector(ns);
    }
}
std::vector<double> EoS::dHr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate partial molar enthalpy at P,T,n
    std::vector<double> dHrdn(ns, 0.);

    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }

    dlnphidT = this->dlnphi_dT();
    d2lnphidTdn = this->d2lnphi_dTdn();
    for (int j = 0; j < ns; j++)
    {
        dHrdn[j] -= std::pow(T, 2) * dlnphidT[j];
        for (int i = 0; i < ns; i++)
        {
            dHrdn[i] -= n_[start_idx + j] * std::pow(T, 2) * d2lnphidTdn[j*ns + i];
        }
    }
    return dHrdn;  // 1/R dHr/dni
}
/*
double EoS::dAr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual Helmholtz free energy
    double Ai = this->ideal.A(T_, n_.begin() + start_idx);
    double Ar = this->Ar(X, T_, n_, start_idx, pt);
    return Ai + Ar;
}
double EoS::dUr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual internal energy
    double Ui = this->ideal.U(T_, n_.begin() + start_idx);
    double Ur = this->Ur(X, T_, n_, start_idx, pt);
    return Ui + Ur;
}
*/

double EoS::S(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual entropy
    return this->Si(X, T_, n_, start_idx, pt) + this->Sr(X, T_, n_, start_idx, pt);  // S/R
}
double EoS::G(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual Gibbs free energy
    return this->Gi(X, T_, n_, start_idx, pt) + this->Gr(X, T_, n_, start_idx, pt);  // G/R = Gi/R + Gr/R
}
double EoS::H(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual enthalpy
    return this->Hi(T_, n_, start_idx, pt) + this->Hr(X, T_, n_, start_idx, pt);  // H/R = Hi/R + Hr/R
}
/*
double EoS::A(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual Helmholtz free energy
    double Ai = this->ideal.A(T_, n_.begin() + start_idx);
    double Ar = this->Ar(X, T_, n_, start_idx, pt);
    return Ai + Ar;
}
double EoS::U(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual internal energy
    double Ui = this->ideal.U(T_, n_.begin() + start_idx);
    double Ur = this->Ur(X, T_, n_, start_idx, pt);
    return Ui + Ur;
}
*/

double EoS::dS_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total entropy with respect to pressure
    return this->dSi_dP(X, T_, n_, start_idx, pt) + this->dSr_dP(X, T_, n_, start_idx, pt);  // 1/R dS/dP
}
double EoS::dG_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Gibbs free energy with respect to pressure
    return this->dGi_dP(X, T_, n_, start_idx, pt) + this->dGr_dP(X, T_, n_, start_idx, pt);  // 1/R dG/dP
}
double EoS::dH_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total enthalpy with respect to pressure
    return this->dHr_dP(X, T_, n_, start_idx, pt);  // 1/R dH/dP
}
/*
double dA_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Helmholtz free energy with respect to pressure
    return this->dAi_dP(X, T_, n_, start_idx, pt) + this->dAr_dP(X, T_, n_, start_idx, pt);  // 1/R dA/dP
}
double dU_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total internal energy with respect to pressure
    return this->dUi_dP(X, T_, n_, start_idx, pt) + this->dUr_dP(X, T_, n_, start_idx, pt);  // 1/R dU/dP
}
*/

double EoS::dS_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total entropy with respect to temperature
    if (pt)
    {
        return this->Cp(X, T_, n_, start_idx, pt) / T_;  // 1/R (dS/dT)_P,n = C_p/RT
    }
    else
    {
        return this->Cv(X, T_, n_, start_idx, pt) / T_;  // 1/R (dS/dT)_V,n = C_v/RT
    }
}
double EoS::dG_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Gibbs free energy with respect to pressure
    if (pt)
    {
        return -this->S(X, T_, n_, start_idx, pt);  // 1/R (dG/dT)_P,n = -S/R
    }
    else
    {
        // return -this->S(X, T_, n_, start_idx, pt) + X * this->dP_dT();  // 1/R (dG/dT)_V,n = -S/R  = C_v/R + V (dP/dT)_V,n
        return not_implemented_double();
    }
}
double EoS::dH_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total enthalpy with respect to pressure
    if (pt)
    {
        return this->dHi_dT(T_, n_, start_idx, pt) + this->dHr_dT(X, T_, n_, start_idx, pt);  // 1/R (dH/dT)_P,n = C_p/R
        // return this->Cp(X, T_, n_, start_idx, pt);  // 1/R (dH/dT)_P,n = C_p/R
    }
    else
    {
        // return this->Cv(X, T_, n_, start_idx, pt) + X * this->dP_dT();  // 1/R (dH/dT)_V,n = C_v/R + V (dP/dT)_V,n
        return not_implemented_double();
    }
}
/*
double dA_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Helmholtz free energy with respect to pressure
    return this->dAi_dP(X, T_, n_, start_idx, pt) + this->dAr_dP(X, T_, n_, start_idx, pt);  // 1/R dA/dP
}
double dU_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total internal energy with respect to pressure
    return this->dUi_dP(X, T_, n_, start_idx, pt) + this->dUr_dP(X, T_, n_, start_idx, pt);  // 1/R dU/dP
}
*/

std::vector<double> EoS::dS_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total entropy with respect to composition
    std::vector<double> dSidn = this->dSi_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dSrdn = this->dSr_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dSdn(ns);

    for (int i = 0; i < ns; i++)
    {
        dSdn[i] = dSidn[i] + dSrdn[i];
    }
    return dSdn;  // 1/R dS/dni
}
std::vector<double> EoS::dG_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Gibbs free energy with respect to composition
    std::vector<double> dGidn = this->dGi_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dGrdn = this->dGr_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dGdn(ns);

    for (int i = 0; i < ns; i++)
    {
        dGdn[i] = dGidn[i] + dGrdn[i];
    }
    return dGdn;  // 1/R dG/dni
}
std::vector<double> EoS::dH_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total enthalpy with respect to composition
    std::vector<double> dHidn = this->dHi_dni(T_, n_, start_idx, pt);
    std::vector<double> dHrdn = this->dHr_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dHdn(ns);

    for (int i = 0; i < ns; i++)
    {
        dHdn[i] = dHidn[i] + dHrdn[i];
    }
    return dHdn;  // 1/R dH/dni
}
/*
double EoS::dA_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual Helmholtz free energy
    double Ai = this->ideal.A(T_, n_.begin() + start_idx);
    double Ar = this->Ar(X, T_, n_, start_idx, pt);
    return Ai + Ar;
}
double EoS::dU_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Ideal + residual internal energy
    double Ui = this->ideal.U(T_, n_.begin() + start_idx);
    double Ur = this->Ur(X, T_, n_, start_idx, pt);
    return Ui + Ur;
}
*/

double EoS::Cpr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate residual heat capacity at constant pressure: Cp/R = 1/R (dHr/dT)_P
    return this->dHr_dT(X, T_, n_, start_idx, pt);  // Cp/R = 1/R (dHr/dT)_P
}
double EoS::Cp(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate heat capacity at constant pressure: Cp/R = 1/R (dH/dT)_P
    return this->Cpi(T_, n_, start_idx, pt) + this->Cpr(X, T_, n_, start_idx, pt);  // Cp/R
}
double EoS::Cv(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Total heat capacity at constant volume Cv/R
    return this->Cvi(T_, n_, start_idx, pt) + this->Cvr(X, T_, n_, start_idx, pt);  // Cv/R
}

std::vector<double> EoS::lnphi0(double X, double T_, bool pt)
{
    // Calculate pure component fugacity coefficient at P, T
    std::vector<double> lnphi0_(nc, NAN);
    if (pt && (X != p || T_ != T))
    {
        this->init_PT(X, T_);

        // Loop over components
        for (int i = 0; i < nc; i++)
        {
            std::vector<double> n_(nc, 0.);
            n_[i] = 1.;

            if (this->eos_in_range(n_.begin()))
            {
                this->solve_PT(n_.begin(), false);

                if (this->select_root(n_.begin()))
                {
                    lnphi0_[i] = this->lnphii(i);
                }
                else
                {
                    lnphi0_[i] = NAN;
                }
            }
        }
    }
    return lnphi0_;
}

double EoS::dxi_dnj(std::vector<double>& n_, int i, int j)
{
    // Derivative of mole fractions with respect to composition
    // xi = ni/nT
    double nT_inv = 1./std::accumulate(n_.begin(), n_.end(), 0.);
    if (i == j)
    {
        // dxi/dni = 1/nT - ni/nT^2
        return nT_inv - n_[i] * std::pow(nT_inv, 2);
    }
    else
    {   
        // dxi/dnj = -ni/nT^2
        return -n_[i] * std::pow(nT_inv, 2);
    }
}
double EoS::dxj_to_dnk(std::vector<double>& dlnphiidxj, std::vector<double>::iterator n_it, int k) {
    // Translate from dlnphii/dxj to dlnphii/dnk
	// dlnphii/dnk = 1/V * [dlnphii/dxk - sum_j xj dlnphii/dxj]
    double nT_inv = 1./std::accumulate(n_it, n_it + this->ns, 0.);
	double dlnphiidnk = dlnphiidxj[k];
	for (int j = 0; j < ns; j++)
	{
        double nj = *(n_it + j);
		dlnphiidnk -= nj * nT_inv * dlnphiidxj[j];
	}
	dlnphiidnk *= nT_inv;
    return dlnphiidnk;
}
std::vector<double> EoS::dxj_to_dnk(std::vector<double>& dlnphiidxj, std::vector<double>::iterator n_it)
{
    // Translate from dlnphii/dxj to dlnphii/dnk
	// dlnphii/dnj = 1/V * [dlnphii/dxj - Σk xk dlnphii/dxk]
    double nT_inv = 1./std::accumulate(n_it, n_it + this->ns, 0.);
    std::vector<double> dlnphiidnk(ns*ns);

    for (int i = 0; i < ns; i++)
    {
        double sum_xj = 0.;
        for (int j = 0; j < ns; j++)
        {
            double nj = *(n_it + j);
            sum_xj += nj * nT_inv * dlnphiidxj[i*ns + j];
        }
        for (int k = 0; k < ns; k++)
        {
            dlnphiidnk[i*ns + k] = nT_inv * (dlnphiidxj[i*ns + k] - sum_xj);
        }
    }
    return dlnphiidnk;
}

int EoS::dlnphi_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose) 
{
    // Analytical derivatives w.r.t. composition, pressure and temperature
	int error_output = 0;

    double p0 = p_;
    double T0 = T_;
    std::vector<double> n0 = n_;

	this->solve_PT(p0, T0, n0, 0, true);
	std::vector<double> lnphi0 = this->lnphi();
    std::vector<double> dlnphidn0 = this->dlnphi_dn();
	std::vector<double> dlnphidP0 = this->dlnphi_dP();
	std::vector<double> dlnphidT0 = this->dlnphi_dT();
    std::vector<double> d2lnphidPdT0 = this->d2lnphi_dPdT();
	std::vector<double> d2lnphidT20 = this->d2lnphi_dT2();
    std::vector<double> d2lnphidTdn0 = this->d2lnphi_dTdn();

	// Calculate numerical derivatives w.r.t. composition
	double dn = 1e-4;
	std::vector<double> lnphi1;
    std::vector<double> nn = n0;
    for (int j = 0; j < ns; j++)
	{
        // Numerical derivative of lnphi w.r.t. nj
        double dnj = n0[j] * dn;
	    nn[j] += dnj;
		this->solve_PT(p0, T0, nn, 0, false);
	    lnphi1 = this->lnphi();
		nn[j] -= dnj;
        for (int i = 0; i < ns; i++)
	    {
            double dlnphidn_an = dlnphidn0[i*ns+j];
            double dlnphi = (lnphi1[i] - lnphi0[i]);
	        double dlnphidn_num = dlnphi/dnj;
			double d = std::log(std::fabs(dlnphidn_num + 1e-15)) - std::log(std::fabs(dlnphidn_an + 1e-15));
            if (verbose || (!(std::fabs(d) < tol) && (std::fabs(dlnphidn_an) > 1e-14 && std::fabs(dlnphidn_num) > 1e-14)))
	        {
	            print("comp", {i, j});
        	    print("dlnphi/dn", {dlnphidn_an, dlnphidn_num, d});
                error_output++;
	        }
        }
	}

	// Calculate numerical derivatives w.r.t. pressure
	double dp = 1e-6;
	this->solve_PT(p0 + dp, T0, n0, 0, false);
	lnphi1 = this->lnphi();

	// Compare analytical and numerical
	for (int i = 0; i < ns; i++)
	{
        double dlnphidP_num = (lnphi1[i] - lnphi0[i])/dp;
		// Use logarithmic scale to compare
		double d = std::log(std::fabs(dlnphidP_num + 1e-15)) - std::log(std::fabs(dlnphidP0[i] + 1e-15));
	    if (verbose || (!(std::fabs(d) < tol) && (std::fabs(dlnphidP0[i]) > 1e-8 && std::fabs(dlnphidP_num) > 1e-8)))
        {
        	print("comp", i);
            print("dlnphi/dP", {dlnphidP0[i], dlnphidP_num, d});
    	    error_output++;
	    }
    }

	// Calculate numerical derivatives w.r.t. temperature
	double dT = 1e-6;
	this->solve_PT(p0, T0 + dT, n0, 0, false);
	lnphi1 = this->lnphi();
    std::vector<double> dlnphidP1 = this->dlnphi_dP();
	std::vector<double> dlnphidT1 = this->dlnphi_dT();

	// Compare analytical and numerical
	for (int i = 0; i < ns; i++)
	{
		double dlnphidT_num = (lnphi1[i] - lnphi0[i])/dT;
		double d = std::log(std::fabs(dlnphidT_num + 1e-15)) - std::log(std::fabs(dlnphidT0[i] + 1e-15));
		if (verbose || (!(std::fabs(d) < tol) && (std::fabs(dlnphidT0[i]) > 1e-8 && std::fabs(dlnphidT_num) > 1e-8)))
		{
			print("comp", i);
			print("dlnphi/dT", {dlnphidT0[i], dlnphidT_num, d});
			error_output++;
		}

        double dPdT_an = d2lnphidPdT0[i];
		double d2lnphidPdT_num = (dlnphidP1[i] - dlnphidP0[i])/dT;
		d = std::log(std::fabs(d2lnphidPdT_num + 1e-15)) - std::log(std::fabs(dPdT_an + 1e-15));
		if (verbose || (!(std::fabs(d) < tol) && (std::fabs(dPdT_an) > 1e-8 && std::fabs(d2lnphidPdT_num) > 1e-8)))
		{
			print("comp", i);
			print("d2lnphi/dPdT", {dPdT_an, d2lnphidPdT_num, d});
			error_output++;
		}

        double dT2_an = d2lnphidT20[i];
		double d2lnphidT2_num = (dlnphidT1[i] - dlnphidT0[i])/dT;
		d = std::log(std::fabs(d2lnphidT2_num + 1e-15)) - std::log(std::fabs(dT2_an + 1e-15));
		if (verbose || (std::fabs(d) > tol && (std::fabs(dT2_an) > 1e-8 && std::fabs(d2lnphidT2_num) > 1e-8)))
		{
			print("comp", i);
			print("d2lnphi/dT2", {dT2_an, d2lnphidT2_num, d});
			error_output++;
		}

        // Second derivative of d2lnphi/dTdnj w.r.t. temperature and composition
        this->solve_PT(p0, T0 + dT, n0, 0, true);
        std::vector<double> dlnphidn1 = this->dlnphi_dn();
        for (int j = 0; j < ns; j++)
        {
            double dTdni_an = d2lnphidTdn0[i*ns+j];
            double dTdni = (dlnphidn1[i*ns+j] - dlnphidn0[i*ns+j]);
            double dTdni_num = dTdni / dT;
            d = std::log(std::fabs(dTdni_an + 1e-15)) - std::log(std::fabs(dTdni_num + 1e-15));
            if (verbose || (!(std::fabs(d) < tol) && (std::fabs(dT2_an) > 1e-14 && std::fabs(dTdni) > 1e-14)))
            {
                print("i, j", {i, j});
                print("d2lnphi/dTdnj != d2lnphi/dTdnj", {dTdni_an, dTdni_num, d});
                error_output++;
            }
        }
	}

	// Calculate lnphi and derivatives for -n
	std::vector<double> n_n = n0;
	std::transform(n0.begin(), n0.end(), n_n.begin(), [](double element) { return element *= -1; });
	this->solve_PT(p0, T0, n_n, 0, true);
	std::vector<double> lnphi_n = this->lnphi();
	std::vector<double> dlnphidP_n = this->dlnphi_dP();
	std::vector<double> dlnphidT_n = this->dlnphi_dT();
	std::vector<double> d2lnphidT2_n = this->d2lnphi_dT2();
	std::vector<double> dlnphidn_n = this->dlnphi_dn();
    std::vector<double> d2lnphidTdn_n = this->d2lnphi_dTdn();

	// Compare positive and negative
	for (int i = 0; i < ns; i++)
	{
        if (n0[i] > 0.)
        {
            double d = (lnphi0[i] - lnphi_n[i])/lnphi0[i];
            if (verbose || (!(std::fabs(d) < tol) && lnphi0[i] > 0.))
            {
                print("comp", i);
                print("lnphi negative", {lnphi0[i], lnphi_n[i], d});
                error_output++;
            }
            d = (dlnphidP0[i] - dlnphidP_n[i])/dlnphidP0[i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dlnphidP0[i]) > 0.))
            {
                print("comp", i);
                print("dlnphi/dP negative", {dlnphidP0[i], dlnphidP_n[i], d});
                error_output++;
            }
            d = (dlnphidT0[i] - dlnphidT_n[i])/dlnphidT0[i];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dlnphidT0[i]) > 0.))
            {
                print("comp", i);
                print("dlnphi/dT negative", {dlnphidT0[i], dlnphidT_n[i], d});
                error_output++;
            }
            d = (d2lnphidT20[i] - d2lnphidT2_n[i])/d2lnphidT20[i];
            if (verbose || std::fabs(d) > tol || (std::isnan(d) && std::fabs(d2lnphidT20[i]) > 0.))
            {
                print("comp", i);
                print("d2lnphi/dT2 negative", {d2lnphidT20[i], d2lnphidT2_n[i], d});
                error_output++;
            }
            for (int j = 0; j < ns; j++)
            {
                d = (dlnphidn0[i*ns + j] + dlnphidn_n[i*ns + j])/dlnphidn0[i*ns + j];
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(dlnphidn0[i*ns+j]) > 0.))
                {
                    print("comp", {i, j});
                    print("dlnphi/dn negative", {dlnphidn0[i*ns + j], dlnphidn_n[i*ns + j], d});
                    error_output++;
                }
            }
            for (int j = 0; j < ns; j++)
            {
                d = dlnphidn0[i*ns + j] + dlnphidn_n[i*ns + j];
                if (verbose || (!(std::fabs(d) < tol) && std::fabs(dlnphidn0[i*ns+j]) > 0.))
                {
                    print("comp", {i, j});
                    print("d2lnphi/dTdn negative", {d2lnphidTdn0[i*ns + j], d2lnphidTdn_n[i*ns + j], d});
                    error_output++;
                }
            }
        }
	}

    return error_output;
}

int EoS::properties_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Analytical derivatives w.r.t. composition, pressure and temperature
	int error_output = 0;

    double p0 = p_;
    double T0 = T_;
    std::vector<double> n0 = n_;

    // Thermal properties tests
    bool pt = true;
    double d, dP = p0 * 1e-6, dT = T0 * 1e-6;
    double Hi0, Hi1, Hr0, Hr1, H0, H1, Gi0, Gi1, Gr0, Gr1, G0, G1, Si0, Si1, Sr0, Sr1, S0, S1;
    std::vector<double> dHi0, dHr0, dH0, dGi0, dGr0, dG0, dSi0, dSr0, dS0;

    // Ideal gas heat capacity at constant pressure Cp
    Hi0 = this->Hi(T0, n0, 0, pt);
    double CPi = this->Cpi(T0, n0, 0, pt);
    Hi1 = this->Hi(T0+dT, n0, 0, pt);
    double CPi_num = (Hi1-Hi0)/dT;
    d = std::log(std::fabs(CPi + 1e-15)) - std::log(std::fabs(CPi_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("Cpi", {CPi, CPi_num, d}); error_output++; }

    // Residual heat capacity at constant pressure Cpr
    Hr0 = this->Hr(p0, T0, n0, 0, pt);
    double CPr = this->Cpr(p0, T0, n0, 0, pt);
    Hr1 = this->Hr(p0, T0+dT, n0, 0, pt);
    double CPr_num = (Hr1-Hr0)/dT;
    d = std::log(std::fabs(CPr + 1e-15)) - std::log(std::fabs(CPr_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("Cpr", {CPr, CPr_num, d}); error_output++; }

    // Heat capacity at constant pressure Cp
    H0 = this->H(p0, T0, n0, 0, pt);
    double CP = this->Cp(p0, T0, n0, 0, pt);
    H1 = this->H(p0, T0+dT, n0, 0, pt);
    double CP_num = (H1-H0)/dT;
    d = std::log(std::fabs(CP + 1e-15)) - std::log(std::fabs(CP_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("Cp", {CP, CP_num, d}); error_output++; }

    // Derivative of (residual) Gibbs energy, enthalpy and entropy w.r.t. pressure, temperature and composition
    this->solve_PT(p0, T0, n0, 0, true);
    Gi0 = this->Gi(p0, T0, n0, 0, pt);
    Gr0 = this->Gr(p0, T0, n0, 0, pt);
    G0 = this->G(p0, T0, n0, 0, pt);
    Hi0 = this->Hi(T0, n0, 0, pt);
    Hr0 = this->Hr(p0, T0, n0, 0, pt);
    H0 = this->H(p0, T0, n0, 0, pt);
    Si0 = this->Si(p0, T0, n0, 0, pt);
    Sr0 = this->Sr(p0, T0, n0, 0, pt);
    S0 = this->S(p0, T0, n0, 0, pt);

    // Pressure derivatives
    {
        double dGidP = this->dGi_dP(p0, T0, n0, 0, pt);
        double dGrdP = this->dGr_dP(p0, T0, n0, 0, pt);
        double dGdP = this->dG_dP(p0, T0, n0, 0, pt);
        double dHrdP = this->dHr_dP(p0, T0, n0, 0, pt);
        double dHdP = this->dH_dP(p0, T0, n0, 0, pt);
        double dSidP = this->dSi_dP(p0, T0, n0, 0, pt);
        double dSrdP = this->dSr_dP(p0, T0, n0, 0, pt);
        double dSdP = this->dS_dP(p0, T0, n0, 0, pt);
        this->solve_PT(p0 + dP, T0, n0, 0, true);

        // Derivative of ideal Gibbs free energy Gi w.r.t. pressure
        Gi1 = this->Gi(p0 + dP, T0, n0, 0, pt);
        double dGidP_num = (Gi1-Gi0)/dP;
        d = std::log(std::fabs(dGidP + 1e-15)) - std::log(std::fabs(dGidP_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dGi/dP", {dGidP, dGidP_num, d}); error_output++; }

        // Derivative of residual Gibbs free energy Gr w.r.t. pressure
        Gr1 = this->Gr(p0 + dP, T0, n0, 0, pt);
        double dGrdP_num = (Gr1-Gr0)/dP;
        d = std::log(std::fabs(dGrdP + 1e-15)) - std::log(std::fabs(dGrdP_num + 1e-15));
        if (verbose || (!std::isnan(dGrdP) && !(std::fabs(d) < tol))) { print("dGr/dP", {dGrdP, dGrdP_num, d}); error_output++; }

        // Derivative of Gibbs free energy G w.r.t. pressure
        G1 = this->G(p0 + dP, T0, n0, 0, pt);
        double dGdP_num = (G1-G0)/dP;
        d = std::log(std::fabs(dGdP + 1e-15)) - std::log(std::fabs(dGdP_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dG/dP", {dGdP, dGdP_num, d}); error_output++; }
        
        // Derivative of residual enthalpy Hr w.r.t. pressure
        Hr1 = this->Hr(p0 + dP, T0, n0, 0, pt);
        double dHrdP_num = (Hr1-Hr0)/dP;
        d = std::log(std::fabs(dHrdP + 1e-15)) - std::log(std::fabs(dHrdP_num + 1e-15));
        if (verbose || (!std::isnan(dHrdP) && !(std::fabs(d) < tol))) { print("dHr/dP", {dHrdP, dHrdP_num, d}); error_output++; }

        // Derivative of enthalpy Hr w.r.t. pressure
        H1 = this->H(p0 + dP, T0, n0, 0, pt);
        double dHdP_num = (H1-H0)/dP;
        d = std::log(std::fabs(dHdP + 1e-15)) - std::log(std::fabs(dHdP_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dH/dP", {dHdP, dHdP_num, d}); error_output++; }
                
        // Derivative of residual entropy Sr w.r.t. pressure
        Si1 = this->Si(p0 + dP, T0, n0, 0, pt);
        double dSidP_num = (Si1-Si0)/dP;
        d = std::log(std::fabs(dSidP + 1e-15)) - std::log(std::fabs(dSidP_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dSi/dP", {dSidP, dSidP_num, d}); error_output++; }

        // Derivative of residual entropy Sr w.r.t. pressure
        Sr1 = this->Sr(p0 + dP, T0, n0, 0, pt);
        double dSrdP_num = (Sr1-Sr0)/dP;
        d = std::log(std::fabs(dSrdP + 1e-15)) - std::log(std::fabs(dSrdP_num + 1e-15));
        if (verbose || (!std::isnan(dSrdP) && !(std::fabs(d) < tol))) { print("dSr/dP", {dSrdP, dSrdP_num, d}); error_output++; }

        // Derivative of entropy S w.r.t. temperature
        S1 = this->S(p0 + dP, T0, n0, 0, pt);
        double dSdP_num = (S1-S0)/dP;
        d = std::log(std::fabs(dSdP + 1e-15)) - std::log(std::fabs(dSdP_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dS/dP", {dSdP, dSdP_num, d}); error_output++; }
    }

    // Temperature derivatives
    {
        double dGidT = this->dGi_dT(p0, T0, n0, 0, pt);
        double dGrdT = this->dGr_dT(p0, T0, n0, 0, pt);
        double dGdT = this->dG_dT(p0, T0, n0, 0, pt);
        double dHidT = this->dHi_dT(T0, n0, 0, pt);
        double dHrdT = this->dHr_dT(p0, T0, n0, 0, pt);
        double dHdT = this->dH_dT(p0, T0, n0, 0, pt);
        double dSidT = this->dSi_dT(p0, T0, n0, 0, pt);
        double dSrdT = this->dSr_dT(p0, T0, n0, 0, pt);
        double dSdT = this->dS_dT(p0, T0, n0, 0, pt);
        this->solve_PT(p0, T0 + dT, n0, 0, true);

        // Derivative of ideal Gibbs free energy Gi w.r.t. temperature
        Gi1 = this->Gi(p0, T0 + dT, n0, 0, pt);
        double dGi_num = (Gi1-Gi0)/dT;
        d = std::log(std::fabs(dGidT + 1e-15)) - std::log(std::fabs(dGi_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dGi/dT", {dGidT, dGi_num, d}); error_output++; }

        // Derivative of residual Gibbs free energy Gr w.r.t. temperature
        Gr1 = this->Gr(p0, T0 + dT, n0, 0, pt);
        double dGr_num = (Gr1-Gr0)/dT;
        d = std::log(std::fabs(dGrdT + 1e-15)) - std::log(std::fabs(dGr_num + 1e-15));
        if (verbose || std::fabs(d) > tol) { print("dGr/dT", {dGrdT, dGr_num, d}); error_output++; }

        // Derivative of Gibbs free energy G w.r.t. temperature
        G1 = this->G(p0, T0 + dT, n0, 0, pt);
        double dG_num = (G1-G0)/dT;
        d = std::log(std::fabs(dGdT + 1e-15)) - std::log(std::fabs(dG_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dG/dT", {dGdT, dG_num, d}); error_output++; }
                
        // Derivative of ideal enthalpy Hi w.r.t. temperature
        Hi1 = this->Hi(T0 + dT, n0, 0, pt);
        double dHi_num = (Hi1-Hi0)/dT;
        d = std::log(std::fabs(dHidT + 1e-15)) - std::log(std::fabs(dHi_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dHi/dT", {dHidT, dHi_num, d}); error_output++; }

        // Derivative of residual enthalpy Hr w.r.t. temperature
        Hr1 = this->Hr(p0, T0+dT, n0, 0, pt);
        double dHr_num = (Hr1-Hr0)/dT;
        d = std::log(std::fabs(dHrdT + 1e-15)) - std::log(std::fabs(dHr_num + 1e-15));
        if (verbose || std::fabs(d) > tol) { print("dHr/dT", {dHrdT, dHr_num, d}); error_output++; }

        // Derivative of enthalpy Hr w.r.t. temperature
        H1 = this->H(p0, T0+dT, n0, 0, pt);
        double dH_num = (H1-H0)/dT;
        d = std::log(std::fabs(dHdT + 1e-15)) - std::log(std::fabs(dH_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dH/dT", {dHdT, dH_num, d}); error_output++; }
                
        // Derivative of residual entropy Sr w.r.t. temperature
        Si1 = this->Si(p0, T0+dT, n0, 0, pt);
        double dSi_num = (Si1-Si0)/dT;
        d = std::log(std::fabs(dSidT + 1e-15)) - std::log(std::fabs(dSi_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dSi/dT", {dSidT, dSi_num, d}); error_output++; }

        // Derivative of residual entropy Sr w.r.t. temperature
        Sr1 = this->Sr(p0, T0+dT, n0, 0, pt);
        double dSr_num = (Sr1-Sr0)/dT;
        d = std::log(std::fabs(dSrdT + 1e-15)) - std::log(std::fabs(dSr_num + 1e-15));
        if (verbose || std::fabs(d) > tol) { print("dSr/dT", {dSrdT, dSr_num, d}); error_output++; }

        // Derivative of enthalpy Hr w.r.t. temperature
        S1 = this->S(p0, T0+dT, n0, 0, pt);
        double dS_num = (S1-S0)/dT;
        d = std::log(std::fabs(dSdT + 1e-15)) - std::log(std::fabs(dS_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dS/dT", {dSdT, dS_num, d}); error_output++; }
    }

    // Composition derivatives
    {
        std::vector<double> dGidn = this->dGi_dni(p0, T0, n0, 0, pt);
        std::vector<double> dGrdn = this->dGr_dni(p0, T0, n0, 0, pt);
        std::vector<double> dGdn = this->dG_dni(p0, T0, n0, 0, pt);
        std::vector<double> dHidn = this->dHi_dni(T0, n0, 0, pt);
        std::vector<double> dHrdn = this->dHr_dni(p0, T0, n0, 0, pt);
        std::vector<double> dHdn = this->dH_dni(p0, T0, n0, 0, pt);
        std::vector<double> dSidn = this->dSi_dni(p0, T0, n0, 0, pt);
        std::vector<double> dSrdn = this->dSr_dni(p0, T0, n0, 0, pt);
        std::vector<double> dSdn = this->dS_dni(p0, T0, n0, 0, pt);

        for (int j = 0; j < nc; j++)
        {
            if (n0[j] != 0.)
            {
                std::vector<double> nn = n0;
                double dnj = 1e-5 * n0[j];
                nn[j] += dnj;
                this->solve_PT(p0, T0, nn, 0, true);

                // Derivative of ideal Gibbs free energy Gi w.r.t. composition
                Gi1 = this->Gi(p0, T0, nn, 0, pt);
                double dGi_num = (Gi1-Gi0)/dnj;
                d = std::log(std::fabs(dGidn[j] + 1e-15)) - std::log(std::fabs(dGi_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dGi/dnj", {static_cast<double>(j), dGidn[j], dGi_num, d}); error_output++; }

                // Derivative of residual Gibbs free energy Gr w.r.t. composition
                Gr1 = this->Gr(p0, T0, nn, 0, pt);
                double dGr_num = (Gr1-Gr0)/dnj;
                d = std::log(std::fabs(dGrdn[j] + 1e-15)) - std::log(std::fabs(dGr_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dGr/dnj", {static_cast<double>(j), dGrdn[j], dGr_num, d}); error_output++; }

                // Derivative of Gibbs free energy G w.r.t. composition
                G1 = this->G(p0, T0, nn, 0, pt);
                double dG_num = (G1-G0)/dnj;
                d = std::log(std::fabs(dGdn[j] + 1e-15)) - std::log(std::fabs(dG_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dG/dnj", {static_cast<double>(j), dGdn[j], dG_num, d}); error_output++; }
                
                // Derivative of ideal enthalpy Hi w.r.t. composition
                Hi1 = this->Hi(T0, nn, 0, pt);
                double dHi_num = (Hi1-Hi0)/dnj;
                d = std::log(std::fabs(dHidn[j] + 1e-15)) - std::log(std::fabs(dHi_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dHi/dnj", {static_cast<double>(j), dHidn[j], dHi_num, d}); error_output++; }

                // Derivative of residual enthalpy Hr w.r.t. composition
                Hr1 = this->Hr(p0, T0, nn, 0, pt);
                double dHr_num = (Hr1-Hr0)/dnj;
                d = std::log(std::fabs(dHrdn[j] + 1e-15)) - std::log(std::fabs(dHr_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dHr/dnj", {static_cast<double>(j), dHrdn[j], dHr_num, d}); error_output++; }

                // Derivative of enthalpy Hr w.r.t. composition
                H1 = this->H(p0, T0, nn, 0, pt);
                double dH_num = (H1-H0)/dnj;
                d = std::log(std::fabs(dHdn[j] + 1e-15)) - std::log(std::fabs(dH_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dH/dnj", {static_cast<double>(j), dHdn[j], dH_num, d}); error_output++; }
                
                // Derivative of residual entropy Sr w.r.t. composition
                Si1 = this->Si(p0, T0, nn, 0, pt);
                double dSi_num = (Si1-Si0)/dnj;
                d = std::log(std::fabs(dSidn[j] + 1e-15)) - std::log(std::fabs(dSi_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dSi/dnj", {static_cast<double>(j), dSidn[j], dSi_num, d}); error_output++; }

                // Derivative of residual entropy Sr w.r.t. composition
                Sr1 = this->Sr(p0, T0, nn, 0, pt);
                double dSr_num = (Sr1-Sr0)/dnj;
                d = std::log(std::fabs(dSrdn[j] + 1e-15)) - std::log(std::fabs(dSr_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dSr/dnj", {static_cast<double>(j), dSrdn[j], dSr_num, d}); error_output++; }

                // Derivative of enthalpy Hr w.r.t. composition
                S1 = this->S(p0, T0, nn, 0, pt);
                double dS_num = (S1-S0)/dnj;
                d = std::log(std::fabs(dSdn[j] + 1e-15)) - std::log(std::fabs(dS_num + 1e-15));
                if (verbose || !(std::fabs(d) < tol)) { print("dS/dnj", {static_cast<double>(j), dSdn[j], dS_num, d}); error_output++; }
            }
        }

        // Second derivative with respect to temperature and composition d2/dTdni
        std::vector<double> d2GidTdn = this->d2Gi_dTdni(p0, T0, n0, 0, pt);
        // std::vector<double> d2GrdTdn = this->d2Gr_dTdni(p0, T0, n0, 0, pt);
        // std::vector<double> d2GdTdn = this->d2G_dTdni(p0, T0, n0, 0, pt);
        std::vector<double> d2HidTdn = this->d2Hi_dTdni(T0, n0, 0, pt);
        // std::vector<double> d2HrdTdn = this->d2Hr_dTdni(p0, T0, n0, 0, pt);
        // std::vector<double> d2HdTdn = this->d2H_dTdni(p0, T0, n0, 0, pt);
        std::vector<double> d2SidTdn = this->d2Si_dTdni(p0, T0, n0, 0, pt);
        // std::vector<double> d2SrdTdn = this->d2Sr_dTdni(p0, T0, n0, 0, pt);
        // std::vector<double> d2SdTdn = this->d2S_dTdni(p0, T0, n0, 0, pt);

        std::vector<double> dGidn1 = this->dGi_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dGrdn1 = this->dGr_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dGdn1 = this->dG_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dHidn1 = this->dHi_dni(T0+dT, n0, 0, pt);
        std::vector<double> dHrdn1 = this->dHr_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dHdn1 = this->dH_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dSidn1 = this->dSi_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dSrdn1 = this->dSr_dni(p0, T0+dT, n0, 0, pt);
        std::vector<double> dSdn1 = this->dS_dni(p0, T0+dT, n0, 0, pt);

        for (int j = 0; j < nc; j++)
        {
            // Derivative of ideal molar entropy si(PT) w.r.t. temperature
            this->solve_PT(p0, T0, n0, 0, true);
            double dsidT = this->dsi_dT(p0, T0, j, true);
            double si0 = this->si(p0, T0, j, true);
            this->solve_PT(p0, T0+dT, n0, 0, true);
            double si1 = this->si(p0, T0+dT, j, true);
            double dsi_num = (si1-si0)/dT;
            d = std::log(std::fabs(dsidT + 1e-15)) - std::log(std::fabs(dsi_num + 1e-15));
            if (verbose || !(std::fabs(d) < tol)) { print("dsi/dT", {static_cast<double>(j), dsidT, dsi_num, d}); error_output++; }

            // // Second derivative of ideal Gibbs free energy Gi w.r.t. temperature and composition
            // double d2Gi_num = (dGidn1[j]-dGidn[j])/dT;
            // d = std::log(std::fabs(d2GidTdn[j] + 1e-15)) - std::log(std::fabs(d2Gi_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2Gi/dTdnj", {static_cast<double>(j), d2GidTdn[j], d2Gi_num, d}); error_output++; }

            // // Second derivative of residual Gibbs free energy Gr w.r.t. temperature and composition
            // double d2Gr_num = (dGrdn1[j]-dGrdn[j])/dT;
            // d = std::log(std::fabs(d2GrdTdn[j] + 1e-15)) - std::log(std::fabs(d2Gi_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2Gr/dTdnj", {static_cast<double>(j), d2GrdTdn[j], d2Gr_num, d}); error_output++; }

            // // Second derivative of total Gibbs free energy G w.r.t. temperature and composition
            // double d2G_num = (dGdn1[j]-dGdn[j])/dT;
            // d = std::log(std::fabs(d2GdTdn[j] + 1e-15)) - std::log(std::fabs(d2Gi_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2G/dTdnj", {static_cast<double>(j), d2GdTdn[j], d2G_num, d}); error_output++; }

            // Second derivative of ideal enthalpy Hi w.r.t. temperature and composition
            double d2Hi_num = (dHidn1[j]-dHidn[j])/dT;
            d = std::log(std::fabs(d2HidTdn[j] + 1e-15)) - std::log(std::fabs(d2Hi_num + 1e-15));
            if (verbose || !(std::fabs(d) < tol)) { print("d2Hi/dTdnj", {static_cast<double>(j), d2HidTdn[j], d2Hi_num, d}); error_output++; }

            // // Second derivative of residual enthalpy Hr w.r.t. temperature and composition
            // double d2Hr_num = (dHrdn1[j]-dGrdn[j])/dT;
            // d = std::log(std::fabs(d2HrdTdn[j] + 1e-15)) - std::log(std::fabs(d2Hr_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2Hr/dTdnj", {static_cast<double>(j), d2HrdTdn[j], d2Hr_num, d}); error_output++; }

            // // Second derivative of total enthalpy H w.r.t. temperature and composition
            // double d2H_num = (dGdn1[j]-dGdn[j])/dT;
            // d = std::log(std::fabs(d2HdTdn[j] + 1e-15)) - std::log(std::fabs(d2Gi_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2H/dTdnj", {static_cast<double>(j), d2HdTdn[j], d2H_num, d}); error_output++; }

            // Second derivative of ideal entropy Si w.r.t. temperature and composition
            double d2Si_num = (dSidn1[j]-dSidn[j])/dT;
            d = std::log(std::fabs(d2SidTdn[j] + 1e-15)) - std::log(std::fabs(d2Si_num + 1e-15));
            if (verbose || !(std::fabs(d) < tol)) { print("d2Si/dTdnj", {static_cast<double>(j), d2SidTdn[j], d2Si_num, d}); error_output++; }

            // // Second derivative of residual entropy Sr w.r.t. temperature and composition
            // double d2Sr_num = (dSrdn1[j]-dSrdn[j])/dT;
            // d = std::log(std::fabs(d2SrdTdn[j] + 1e-15)) - std::log(std::fabs(d2Sr_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2Sr/dTdnj", {static_cast<double>(j), d2SrdTdn[j], d2Sr_num, d}); error_output++; }

            // // Second derivative of total entropy S w.r.t. temperature and composition
            // double d2S_num = (dSdn1[j]-dSdn[j])/dT;
            // d = std::log(std::fabs(d2SdTdn[j] * T0 + 1e-15)) - std::log(std::fabs(d2S_num + 1e-15));
            // if (verbose || !(std::fabs(d) < tol)) { print("d2S/dTdnj", {static_cast<double>(j), d2SdTdn[j], d2S_num, d}); error_output++; }
        }
    }

    return error_output;
}

int EoS::derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Default implementation of derivatives_test() tests nothing and returns 0
    (void) p_;
    (void) T_;
    (void) n_;
    (void) tol;
    (void) verbose;
    return 0;
}
