#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>

#include "dartsflash/eos/helmholtz/helmholtz.hpp"
#include <Eigen/Dense>

HelmholtzEoS::HelmholtzEoS(CompData& comp_data) : EoS(comp_data) { }

EoS::RootSelect HelmholtzEoS::select_root(std::vector<double>::iterator n_it)
{
    for (auto it: this->preferred_roots)
    {
        // it.first: i
        // it.second: {x, root_flag}
        if (*(n_it + it.first) / N >= it.second.first)
        {
            if (this->root_flag == it.second.second && this->is_preferred_root >= RootSelect::ACCEPT)
            {
                // If particular root is the preferred root, return PREFER flag
                return RootSelect::PREFER;
            }
            else
            {
                return RootSelect::REJECT;
            }
        }
    }
    return this->is_stable() ? RootSelect::ACCEPT : RootSelect::REJECT;
}

// Evaluation of EoS at (T, V, n)
void HelmholtzEoS::solve_VT(std::vector<double>::iterator n_it, bool second_order)
{
    // Calculate zero'th order parameters for (V, T, n) specification
    std::copy(n_it, n_it + ns, this->n.begin());
    N = std::accumulate(n_it, n_it + this->ns, 0.);

    // Set root flag if in preferred root composition range
    this->root_type = EoS::RootFlag::NONE;
    for (auto it: this->preferred_roots)
    {
        if (n[it.first] / this->N > it.second.first)
        {
            this->set_root_flag(it.second.second);
        }
    }

    // Calculate zero'th/first/second order parameters for (V, T, n) specification
    this->zeroth_order(n.begin(), this->v);
    (second_order) ? this->second_order(n.begin()) : this->first_order(n.begin());

    // Calculate pressure
    this->p = P();
    this->z = p * v / (N * this->units.R * T);
    return;
}
void HelmholtzEoS::solve_PT(std::vector<double>::iterator n_it, bool second_order)
{
    // Calculate zero'th order parameters for (P, T, n) specification. This includes calculation of V(P, T, n)
    std::copy(n_it, n_it + ns, this->n.begin());
    N = std::accumulate(n_it, n_it + this->ns, 0.);

    // Set root flag if in preferred root composition range
    this->root_type = EoS::RootFlag::NONE;
    for (auto it: this->preferred_roots)
    {
        if (//root_flag <= EoS::RootFlag::STABLE && 
            n[it.first] / this->N >= it.second.first)
        {
            this->set_root_flag(it.second.second);
        }
    }

    // Calculate zeroth order parameters of reduced Helmholtz function F for solving volume
    this->zeroth_order(n.begin());  // Volume-independent parameters to calculate
    
    // Calculate V
    this->v = this->V();
    this->z = p * v / (N * this->units.R * T);

    // Calculate first/second order parameters for properties
    this->zeroth_order(this->v);  // Volume-dependent parameters of Helmholtz function
    (second_order) ? this->second_order(n.begin()) : this->first_order(n.begin());

    // if (!this->preferred_roots.empty())
    // {
    //     this->root_flag = EoS::RootFlag::STABLE;
    // }
    return;
}

// Overloaded pressure P(T, V, n) and volume V(P, T, n) functions
double HelmholtzEoS::P(double V_, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    (void) pt;
    this->solve_VT(V_, T_, n_, start_idx, false);
	return this->P();
}
double HelmholtzEoS::V(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    (void) pt;
    this->solve_PT(p_, T_, n_, start_idx, false);
	return this->V();
}
double HelmholtzEoS::Z(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, false);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, false);
    }
    return p * v / (N * this->units.R * T);
}
double HelmholtzEoS::rho(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    if (pt)
    {
        double p_ = X;
        return this->compdata.get_molar_weight(n_) * 1e-3 / this->V(p_, T_, n_, start_idx);   
    }
    else
    {
        this->v = X;
        return this->compdata.get_molar_weight(n_) * 1e-3 / this->v;
    }
}
int HelmholtzEoS::volume_iterations(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Number of iterations required for volume solver at P, T, n
    (void) pt;
    this->solve_PT(p_, T_, n_, start_idx, false);
    return this->n_iterations;
}

// Pressure function and derivatives
double HelmholtzEoS::P() 
{
    return this->units.R * (-T * this->dF_dV() + N * T / v);
}
double HelmholtzEoS::dP_dV() 
{
    return - this->units.R * (T * this->d2F_dV2() + N * T / std::pow(v, 2));
}
double HelmholtzEoS::dP_dT() 
{
    return - this->units.R * T * this->d2F_dTdV() + p / T;
}
double HelmholtzEoS::dP_dni(int i) 
{
    return this->units.R * (-T * this->d2F_dVdni(i) + T / v);
}
double HelmholtzEoS::dV_dni(int i) 
{
    return - this->dP_dni(i) / this->dP_dV();
}
double HelmholtzEoS::dV_dT() 
{
    return - this->dP_dT() / this->dP_dV();
}
double HelmholtzEoS::d2P_dV2()
{
    return this->units.R * (-T * this->d3F_dV3() + 2. * N * T / std::pow(v, 3));
}

// Fugacity coefficient and derivatives
double HelmholtzEoS::lnphii(int i) 
{
    return this->dF_dni(i) - std::log(z);
}
double HelmholtzEoS::dlnphii_dnj(int i, int j) 
{
    return this->d2F_dnidnj(i, j) + 1./N + 1./(this->units.R * T) * this->dP_dni(j) * this->dP_dni(i) / this->dP_dV();
}
double HelmholtzEoS::dlnphii_dT(int i) 
{
    return this->d2F_dTdni(i) + 1./T - this->dV_dni(i) / (this->units.R * T) * this->dP_dT();
}
double HelmholtzEoS::dlnphii_dP(int i) 
{
    return this->dV_dni(i) / (this->units.R * T) - 1./p;
}

// Ideal gas properties at VT
double HelmholtzEoS::dsi_dP(double X, double T_, int i, bool pt)
{
    (void) T_;
    (void) i;
    if (pt)
    {   // 1/R dsi(P,T)/dP = -1/p
        return -1./X;
    }
    else
    {   // 1/R dsi(V,T)/dP = -dV/dP / N / p
        return -1./(X * this->dP_dV());
    }
}

// Residual bulk properties
double HelmholtzEoS::Ar(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Residual Helmholtz free energy Ar/RT
    if (pt)
    {
        // Ar(PT)/R = Ar(VT)/R - n T ln Z
        this->solve_PT(X, T_, n_, start_idx, false);

        // Ar(VT)/R = F T
        double ArTV = this->F();

        return (ArTV - N * std::log(z)) * T;  // Ar(PT)/R [-]
    }
    else
    {
        // Ar(VT)/R = F T
        this->solve_VT(X, T_, n_, start_idx, false);
        return this->F() * T;  // A(VT)/R [-]
    }
}
double HelmholtzEoS::Sr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    if (pt)
    {
        // Sr(PT)/R = Sr(VT)/R + n ln Z
        this->solve_PT(X, T_, n_, start_idx, true);

        // Sr(VT)/R = -(T dF/dT + F)
        double SrTV = -(T * this->dF_dT() + this->F());

        return SrTV + N * std::log(z);  // Sr(PT)/R [-]
    }
    else
    {
        // Sr(VT)/R = -(T dF/dT + F)
        this->solve_VT(X, T_, n_, start_idx, true);
        return -(T * this->dF_dT() + this->F());  // S(VT)/R [-]
    }    
}
double HelmholtzEoS::Ur(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    if (pt)
    {
        // Ur(PT)/RT = Ur(VT)/RT
        this->solve_PT(X, T_, n_, start_idx, true);

        // Ur(VT)/R = Ar(VT)/R + T Sr(VT)/R
        // Ar(VT)/R = F T
        // Sr(VT)/R = -(T dF/dT + F)
        double ArTV = this->F();
        double SrTV = -(T * this->dF_dT() + this->F());
        
        return (ArTV + SrTV) * T;  // Ur(PT)/R [-]
    }
    else
    {
        // Ur(VT)/RT = Ar(VT)/RT + Sr(VT)/R
        this->solve_VT(X, T_, n_, start_idx, true);

        // Ar(VT)/RT = F [-]
        // Sr(VT)/R = -(T dF/dT + F) [-]
        double ArTV = this->F();
        double SrTV = -(T * this->dF_dT() + this->F());
        
        return (ArTV + SrTV) * T;  // U(VT)/R [-]
    }
}
double HelmholtzEoS::Hr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    if (pt)
    {
        // Hr(PT)/R = Gr(PT)/R + T Sr(PT)/R
        this->solve_PT(X, T_, n_, start_idx, true);

        // Gr(PT)/R = Gr(VT)/R - n T ln Z (= lnphii T)
        // Sr(PT)/R = Sr(VT)/R + n ln Z
        double GrPT = this->Gr(this->v, this->T, n_, start_idx, false);
        double SrPT = this->Sr(this->v, this->T, n_, start_idx, false);

        return GrPT + T * SrPT; // Hr(PT)/R [-]
    }
    else
    {
        // Hr(VT)/R = Ur(VT)/R + PV/R - nT
        double UrTV = this->Ur(X, T_, n_, start_idx, pt);  // Ur(VT)/R [-]

        return UrTV + p * v / this->units.R - N * T; // Hr(VT)/R [-]
    }
}
double HelmholtzEoS::Gr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    if (pt)
    {
        // Gr(PT)/RT = Gr(VT)/RT - n ln Z
        this->solve_PT(X, T_, n_, start_idx, false);

        // Gr(VT)/R = Ar(VT)/R + PV/R - nT
        // Ar(VT)/R = F T
        double ArTV = this->F() * T;
        double GrTV = ArTV + p * v / this->units.R - N * T;

        return GrTV - N * T * std::log(z);  // Gr(PT)/R [-]
    }
    else
    {
        // Gr(VT)/R = Ar(VT)/R + PV/R - nT
        // Ar(VT)/R = F T
        double ArTV = this->Ar(X, T_, n_, start_idx, pt);

        return ArTV + p * v / this->units.R - N * T;  // Gr(VT)/R [-]
    }
}

std::vector<double> HelmholtzEoS::dAr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual Helmholtz free energy with respect to composition
    std::vector<double> dArdni(nc);

    if (pt)
    {
        // Ar(PT)/RT = Ar(VT)/RT - n ln Z = F - n ln Z
        this->solve_PT(X, T_, n_, start_idx, true);

        // 1/RT (dF/dni)_PT = dF/dV dV/dni + dF/dni
        for (int i = 0; i < ns; i++)
        {
            dArdni[i] = (this->dF_dV() * dV_dni(i) + this->dF_dni(i)) * T;
            
            double dzdni = this->p / (this->units.R * this->T) * (this->dV_dni(i) - this->v / N);
            dArdni[i] -= (std::log(z) + dzdni/z) * T;
        }
    }
    else
    {
        // Ar(VT)/RT = F
        this->solve_VT(X, T_, n_, start_idx, true);
        for (int i = 0; i < ns; i++)
        {
            dArdni[i] = this->dF_dni(i) * T;
        }
    }
    return dArdni;  // 1/R dAr/dni [1/mol]
}
std::vector<double> HelmholtzEoS::dSr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual entropy with respect to composition
    std::vector<double> dSrdni(nc);
    if (pt)
    {
        // Sr(PT)/R = Sr(VT)/R + n ln Z = -(T dF/dT + F) + n ln Z
        this->solve_PT(X, T_, n_, start_idx, true);

        // d/dni (-(T dF/dT + F))_PT = - (T (d2F/dTdV dV/dni + d2F/dTdni) + dF/dV dV/dni + dF/dV)
        for (int i = 0; i < nc; i++)
        {
            double dzdni = this->p / (this->units.R * this->T) * (this->dV_dni(i) - this->v / N);
            dSrdni[i] = -(T * (this->d2F_dTdV() * dV_dni(i) + this->d2F_dTdni(i)) + this->dF_dV() * dV_dni(i) + this->dF_dni(i)) + std::log(z) + dzdni/z;
        }
    }
    else
    {
        // Sr(VT)/R = -(T dF/dT + F)
        this->solve_VT(X, T_, n_, start_idx, true);
        for (int i = 0; i < nc; i++)
        {
            dSrdni[i] = -(T * this->d2F_dTdni(i) + this->dF_dni(i));
        }
    }
    return dSrdni;  // 1/R dSr/dni [1/mol]
}
std::vector<double> HelmholtzEoS::dUr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual internal energy with respect to composition
    // 1/RT dUr(VT)/dni = 1/RT dUr(PT)/dni
    std::vector<double> dUrdni(nc);

    // 1/RT dUr(VT)/dni = 1/R dAr(VT)/dni + 1/R * dSr(VT)/dni
    std::vector<double> dArdni = this->dAr_dni(X, T_, n_, start_idx, pt);
    std::vector<double> dSrdni = this->dSr_dni(X, T_, n_, start_idx, pt);
        
    for (int i = 0; i < nc; i++)
    {
        dUrdni[i] = dArdni[i] + T * dSrdni[i];
    }
    return dUrdni;  // 1/R dUr/dni [1/mol]
}
std::vector<double> HelmholtzEoS::dHr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual enthalpy with respect to composition
    std::vector<double> dHrdni(nc);
    if (pt)
    {
        // 1/R dHr(PT)/dni = 1/R dHr(VT)/dni = 1/R dUr(VT)/dni + P/R dV/dni - T
        // 1/R dUr(VT)/dni = 1/R dUr(PT)/dni
        std::vector<double> dUrdni = this->dUr_dni(X, T_, n_, start_idx, pt);
        for (int i = 0; i < nc; i++)
        {
            dHrdni[i] = dUrdni[i] + this->p * this->dV_dni(i) / this->units.R - T;
        }
    }
    else
    {
        // 1/R dHr(PT)/dni = 1/R dHr(VT)/dni = 1/R dUr(VT)/dni + dP/dni V/R - T
        // 1/R dUr(VT)/dni = 1/R dUr(PT)/dni
        std::vector<double> dUrdni = this->dUr_dni(X, T_, n_, start_idx, pt);
        for (int i = 0; i < nc; i++)
        {
            dHrdni[i] = dUrdni[i] + this->dP_dni(i) * this->v / this->units.R - T;
        }
    }
    return dHrdni;  // 1/R dHr/dni [1/mol]
}
std::vector<double> HelmholtzEoS::dGr_dni(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of residual internal energy with respect to composition
    std::vector<double> dGrdni(nc);

    if (pt)
    {
        // Sr(PT)/R = Sr(VT)/R + n ln Z
        this->solve_PT(X, T_, n_, start_idx, true);

        // Sr(VT)/R = -(T dF/dT + F)
        for (int i = 0; i < nc; i++)
        {
            double dArdni = (this->dF_dV() * dV_dni(i) + this->dF_dni(i)) * T;
            dGrdni[i] = dArdni + p * this->dV_dni(i) / this->units.R - T;

            double dzdni = this->p / (this->units.R * this->T) * (this->dV_dni(i) - this->v / N);
            dGrdni[i] -= (std::log(z) + dzdni/z) * T;
        }
    }
    else
    {
        // Sr(VT)/R = -(T dF/dT + F)
        this->solve_VT(X, T_, n_, start_idx, true);
        
        for (int i = 0; i < nc; i++)
        {
            double dArdni = this->dF_dni(i) * T;
            dGrdni[i] = dArdni + this->dP_dni(i) * v / this->units.R - T;
        }
    }
    return dGrdni;  // 1/RT dUr/dni [1/mol]
}

double HelmholtzEoS::Cvr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Cvr/R = - T^2 d2F/dT2 - 2T dF/dT
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }
    
    return -std::pow(T, 2) * this->d2F_dT2() - 2. * T * this->dF_dT();
}
double HelmholtzEoS::Cpr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Cpr/R = Cvr/R - T/R (dP/dT)^2/ (dP/dV) - n
    double cvr = this->Cvr(X, T_, n_, start_idx, pt);
    double cpr = cvr - T/this->units.R * std::pow(this->dP_dT(), 2) / this->dP_dV() - N;
    return cpr;
}

double HelmholtzEoS::dS_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total entropy with respect to pressure
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }
    return -this->dV_dT() / this->units.R;  // 1/R (dS/dP)_T,n = -1/R (dV/dT)_P,n
}
double HelmholtzEoS::dG_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Gibbs free energy with respect to pressure
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, false);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, false);
    }
    return this->v / this->units.R;  // 1/R (dG/dP)_T,n = V/R
}
double HelmholtzEoS::dH_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total enthalpy with respect to pressure
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }
    return (this->v - T_ * this->dV_dT()) / this->units.R;  // 1/R (dH/dP)_T,n = V/R - T/R (dV/dT)_P,n
}
double HelmholtzEoS::dA_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Helmholtz free energy with respect to pressure
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }
    return -this->p / this->dP_dV() / this->units.R;  // 1/R (dA/dP)_T,n = -P (dV/dP)_T,n
}
double HelmholtzEoS::dU_dP(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total internal energy with respect to pressure
    if (pt)
    {
        this->solve_PT(X, T_, n_, start_idx, true);
    }
    else
    {
        this->solve_VT(X, T_, n_, start_idx, true);
    }
    return (-T_ * this->dV_dT() - this->p / this->dP_dV()) / this->units.R;  // 1/R (dU/dP)_T,n = - T (dV/dT)_P,n - P (dV/dP)_T,n
}

double HelmholtzEoS::dS_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total entropy with respect to temperature
    if (pt)
    {
        return this->Cp(X, T_, n_, start_idx, pt) / T_;  // 1/R (dS/dT)_P,n = C_p/T
    }
    else
    {
        return this->Cv(X, T_, n_, start_idx, pt) / T_;  // 1/R (dS/dT)_V,n = Cv/T
    }
}
double HelmholtzEoS::dG_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Gibbs free energy with respect to pressure
    if (pt)
    {
        return -this->S(X, T_, n_, start_idx, pt);  // 1/R (dG/dT)_P,n = -S
    }
    else
    {
        return -this->S(X, T_, n_, start_idx, pt) + X * this->dP_dT() / this->units.R;  // 1/R (dG/dT)_V,n = -S  = Cv + V (dP/dT)_V,n
    }
}
double HelmholtzEoS::dH_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total enthalpy with respect to pressure
    if (pt)
    {
        return this->Cp(X, T_, n_, start_idx, pt);  // 1/R (dH/dT)_P,n = C_p/R
    }
    else
    {
        return this->Cp(X, T_, n_, start_idx, pt) + X * this->dP_dT() / this->units.R;  // 1/R (dH/dT)_V,n = Cv/R + V/R (dP/dT)_V,n
    }
}
double HelmholtzEoS::dA_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total Helmholtz free energy with respect to pressure
    if (pt)
    {
        return -this->S(X, T_, n_, start_idx, pt) - this->p * this->dV_dT() / this->units.R;  // 1/R (dA/dT)_P,n = -S - P (dV/dT)_P,n
    }
    else
    {
        return -this->S(X, T_, n_, start_idx, pt);  // 1/R (dA/dT)_V,n = -S
    }
}
double HelmholtzEoS::dU_dT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Derivatives of total internal energy with respect to pressure
    if (pt)
    {
        return this->Cp(X, T_, n_, start_idx, pt) - this->p * this->dV_dT() / this->units.R;  // 1/R (dU/dT)_P,n = Cp - P (dV/dT)_P,n
    }
    else
    {
        return this->Cv(X, T_, n_, start_idx, pt);  // 1/R (dU/dT)_V,n = Cv
    }
}

double HelmholtzEoS::vs(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Sound speed vs^2 = v / (β Mw)
    // β = -1/v Cvr/Cpr / (dP/dV)
    double Mw = this->compdata.get_molar_weight(n_);
    double cv = this->Cv(X, T_, n_, start_idx, pt);
    double cp = this->Cp(X, T_, n_, start_idx, pt);
    double beta = -1./v * cv/cp / this->dP_dV();
    double VS2 = v / (beta * Mw * 1e-3) * M_R / this->units.R;
    return std::sqrt(VS2);
}
double HelmholtzEoS::JT(double X, double T_, std::vector<double>& n_, int start_idx, bool pt) 
{
    // Joule-Thomson coefficient μ_JT = -1/Cp (v + T (dP/dT)/(dP/dV))
    double cp = this->Cp(X, T_, n_, start_idx, pt) * this->units.R;
    return -1. / cp * (v - T * this->dV_dT());
}

// EoS CONSISTENCY TESTS
int HelmholtzEoS::derivatives_test(double V_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Test derivatives of Helmholtz function F(T,V,n) = Ar(T,V,n)/RT
    double T0 = T_;
    double V0 = V_;
    std::vector<double> n0 = n_;
    double p0 = this->P(V0, T0, n0, 0);
    double dX = 1e-5;
    int error_output = EoS::derivatives_test(p0, T0, n0, tol, verbose);

    // Analytical derivatives
    this->solve_VT(V0, T0, n0, 0, true);
    double F0 = this->F();
    double dV = this->dF_dV();
    double dT = this->dF_dT();
    double dTdV = this->d2F_dTdV();
    double dV2 = this->d2F_dV2();
    double dT2 = this->d2F_dT2();
    double dV3 = this->d3F_dV3();

    std::vector<double> dni(nc), dTdni(nc), dVdni(nc), dnidnj(nc*nc);
    for (int i = 0; i < nc; i++)
    {
        dni[i] = this->dF_dni(i);
        dTdni[i] = this->d2F_dTdni(i);
        dVdni[i] = this->d2F_dVdni(i);
        for (int j = 0; j < nc; j++)
        {
            dnidnj[i*nc + j] = this->d2F_dnidnj(i, j);
        }
    }

    // Numerical derivatives
    double d, F1;

    // dF/dV
    this->solve_VT(V0 + V0*dX, T0, n0, 0, true);
    F1 = this->F();
    double dV_num = (F1 - F0) / (V0*dX);
    d = std::log(std::fabs(dV + 1e-15)) - std::log(std::fabs(dV_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dF/dV != dF/dV", {dV, dV_num, d}); error_output++; }

    // dF/dT
    this->solve_VT(V0, T0 + T0*dX, n0, 0, true);
    F1 = this->F();
    double dT_num = (F1 - F0) / (T0*dX);
    d = std::log(std::fabs(dT + 1e-15)) - std::log(std::fabs(dT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dF/dT != dF/dT", {dT, dT_num, d}); error_output++; }

    // dF/dni
    for (int i = 0; i < nc; i++)
    {
        std::vector<double> nn = n0;
        nn[i] += dX*n0[i];
        this->solve_VT(V0, T0, nn, 0, true);
        F1 = this->F();
        double dni_num = (F1 - F0) / (n0[i]*dX);
        d = std::log(std::fabs(dni[i] + 1e-15)) - std::log(std::fabs(dni_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("i", i); print("dF/dni != dF/dni", {dni[i], dni_num, d}); error_output++; }

        // d2F/dnidnj
        double dni1, dni_1;
        for (int j = 0; j < nc; j++)
        {
            nn[j] += dX*n0[j];
            this->solve_VT(V0, T0, nn, 0, true);
            dni1 = this->dF_dni(i);
            nn[j] -= dX*n0[j];

            nn[j] -= dX*n0[j];
            this->solve_VT(V0, T0, nn, 0, true);
            dni_1 = this->dF_dni(i);
            nn[j] += dX*n0[j];

            double dnidnj_num = (dni1 - dni_1) / (2*n0[j]*dX);
            d = std::log(std::fabs(dnidnj[i*nc + j] + 1e-15)) - std::log(std::fabs(dnidnj_num + 1e-15));
            if (verbose || !(std::fabs(d) < tol))
            { 
                print("i, j", {i, j}); 
                print("d2F/dnidnj != d2F/dnidnj", {dnidnj[i*nc + j], dnidnj_num, d}); 
                error_output++; 
            }
        }

        // d2F/dTdni
        this->solve_VT(V0, T0 + T0*dX, nn, 0, true);
        dni1 = this->dF_dni(i);
        this->solve_VT(V0, T0 - T0*dX, nn, 0, true);
        dni_1 = this->dF_dni(i);
        double dTdni_num = (dni1 - dni_1) / (2*T0*dX);
        d = std::log(std::fabs(dTdni[i] + 1e-15)) - std::log(std::fabs(dTdni_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("i", i); print("d2F/dTdni != d2F/dTdni", {dTdni[i], dTdni_num, d}); error_output++; }

        // d2F/dVdni
        this->solve_VT(V0 + V0*dX, T0, nn, 0, true);
        dni1 = this->dF_dni(i);
        this->solve_VT(V0 - V0*dX, T0, nn, 0, true);
        dni_1 = this->dF_dni(i);
        double dVdni_num = (dni1 - dni_1) / (2*V0*dX);
        d = std::log(std::fabs(dVdni[i] + 1e-15)) - std::log(std::fabs(dVdni_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("i", i); print("d2F/dVdni != d2F/dVdni", {dVdni[i], dVdni_num, d}); error_output++; }

        nn[i] -= dX*n0[i];
    }

    // d2F/dVdT and d2F/dTdV
    this->solve_VT(V0, T0 - dX*T0, n0, 0, true);
    double dV_1 = this->dF_dV();
    this->solve_VT(V0, T0 + dX*T0, n0, 0, true);
    double dV1 = this->dF_dV();
    double dVdT_num = (dV1 - dV_1) / (2*dX*T0);
    d = std::log(std::fabs(dTdV + 1e-15)) - std::log(std::fabs(dVdT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2F/dVdT != d2F/dVdT", {dTdV, dVdT_num, d}); error_output++; }

    this->solve_VT(V0 - dX*V0, T0, n0, 0, true);
    double dT_1 = this->dF_dT();
    this->solve_VT(V0 + dX*V0, T0, n0, 0, true);
    double dT1 = this->dF_dT();
    double dTdV_num = (dT1 - dT_1) / (2*dX*V0);
    d = std::log(std::fabs(dTdV + 1e-15)) - std::log(std::fabs(dTdV_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2F/dTdV != d2F/dTdV", {dTdV, dTdV_num, d}); error_output++; }

    // d2F/dV2
    this->solve_VT(V0 - V0*dX, T0, n0, 0, true);
    dV_1 = this->dF_dV();
    this->solve_VT(V0 + V0*dX, T0, n0, 0, true);
    dV1 = this->dF_dV();
    double dV2_num = (dV1 - dV_1) / (2*dX*V0);
    d = std::log(std::fabs(dV2 + 1e-15)) - std::log(std::fabs(dV2_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2F/dV2 != d2F/dV2", {dV2, dV2_num, d}); error_output++; }

    // d2F/dT2
    this->solve_VT(V0, T0 - T0*dX, n0, 0, true);
    dT_1 = this->dF_dT();
    this->solve_VT(V0, T0 + T0*dX, n0, 0, true);
    dT1 = this->dF_dT();
    double dT2_num = (dT1 - dT_1) / (2*dX*T0);
    d = std::log(std::fabs(dT2 + 1e-15)) - std::log(std::fabs(dT2_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("d2F/dT2 != d2F/dT2", {dT2, dT2_num, d}); error_output++; }

    // d3F/dV3
    this->solve_VT(V0 - V0*dX, T0, n0, 0, true);
    dV_1 = this->d2F_dV2();
    this->solve_VT(V0 + V0*dX, T0, n0, 0, true);
    dV1 = this->d2F_dV2();
    double dV3_num = (dV1 - dV_1) / (2*dX*V0);
    d = std::log(std::fabs(dV3 + 1e-15)) - std::log(std::fabs(dV3_num + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("d3F/dV3 != d3F/dV3", {dV3, dV3_num, d}); error_output++; }

    // dZ/dP and d2Z/dP2
    this->solve_PT(p0, T0, n0, 0, true);
    double dZdP = this->dZ_dP();
    double d2ZdP2 = this->d2Z_dP2();

    this->solve_PT(p0 - p0*dX, T0, n0, 0, true);
    double Z_1 = this->z;
    double dP_1 = this->dZ_dP();
    this->solve_PT(p0 + p0*dX, T0, n0, 0, true);
    double Z1 = this->z;
    double dP1 = this->dZ_dP();
    double dP_num = (Z1 - Z_1) / (2*dX*p0);
    double d2P_num = (dP1 - dP_1) / (2*dX*p0);
    
    d = std::log(std::fabs(dZdP + 1e-15)) - std::log(std::fabs(dP_num + 1e-15));
    if (verbose || std::fabs(d) > tol) { print("dZ/dP != dZ/dP", {dZdP, dP_num, d}); error_output++; }
    d = std::log(std::fabs(d2ZdP2 + 1e-15)) - std::log(std::fabs(d2P_num + 1e-15));
    if (verbose || std::fabs(d) > tol) { print("d2Z/dP2 != d2Z/dP2", {d2ZdP2, d2P_num, d}); error_output++; }

    return error_output;
}

int HelmholtzEoS::lnphi_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose) 
{
    // d/dnj (sum_i n_i lnphii) = G_rj/RT = lnphij
    int error_output = 0;

    // Calculate d(Gri/RT)/dnj numerically
    this->solve_PT(p_, T_, n_, 0, false);
    std::vector<double> lnphi0 = this->lnphi();

    // at p, T, n
    double Gres0 = 0.;
    for (int i = 0; i < nc; i++) {
        Gres0 += n_[i] * lnphi0[i];
    }

    // add dn
    double dn = 1e-8;
    std::vector<double> dGres(nc, 0.);
    for (int j = 0; j < nc; j++)
    {
        n_[j] += dn;
        this->solve_PT(p_, T_, n_, 0, false);
        std::vector<double> lnphi1 = this->lnphi();

        double Gres1 = 0.;
        for (int i = 0; i < nc; i++)
        {
            Gres1 += n_[i] * lnphi1[i];
        }
        dGres[j] = (Gres1-Gres0)/dn;
        n_[j] -= dn;
    }

    // compare lnphi's
    for (int j = 0; j < nc; j++)
    {
        double d = std::log(std::fabs(dGres[j] + 1e-15)) - std::log(std::fabs(lnphi0[j] + 1e-15));
        if (verbose || !(std::fabs(d) < tol))
        {
            print("comp", j);
            print("lnphi consistency test", {dGres[j], lnphi0[j], d});
            error_output++;
        }
    }
    return error_output;
}

int HelmholtzEoS::pressure_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose) 
{
    // Pressure consistency test
    // d/dP sum_i dlnphii/dP = (Z-1)N/P
    int error_output = 0;

    // Calculate dlnphi/dP from EoS
    this->solve_PT(p_, T_, n_, 0, true);

    dlnphidP = this->dlnphi_dP();
    double ndlnphi_dp = std::inner_product(n_.begin(), n_.end(), dlnphidP.begin(), 0.);

    // compare dlnphi_dP with (Z-1)/P
    if (verbose || std::fabs(ndlnphi_dp - (this->z-1.)*N/p) > tol)
    {
        print("Pressure consistency test", {ndlnphi_dp, (this->z-1.)/p});
        error_output++;
    }
    return error_output;
}

int HelmholtzEoS::temperature_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose) 
{
    (void) p_;
    (void) T_;
    (void) n_;
    (void) tol;
    (void) verbose;
    return 0;
}

int HelmholtzEoS::composition_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose) 
{
    // Consistency of composition derivatives: symmetry of dlnphii/dnj and Gibbs-Duhem relation
    int error_output = 0;

    // Calculate all dlnphii/dnj
    this->solve_PT(p_, T_, n_, 0, true);
    dlnphidn = this->dlnphi_dn();

    // Symmetry of dlnphii_dnj: dlnphii/dnj == dlnphij/dni
    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            double d = (this->dlnphidn[i*nc+j]-this->dlnphidn[j*nc+i])/this->dlnphidn[i*nc+j];
            if (verbose || (!(std::fabs(d) < tol) && std::fabs(dlnphidn[i*nc+j]) > 0.))
            {
                print("symmetry", {dlnphidn[i*nc+j], dlnphidn[j*nc+i], d});
                error_output++;
            }
        }
    }

    // Gibbs-Duhem relation
    // sum_i n_i dlnphii/dnj = 0
    for (int j = 0; j < nc; j++)
    {
        double G_D = 0;
        for (int i = 0; i < nc; i++)
        {
            G_D += n_[i] * this->dlnphidn[i*nc+ j];
        }
        if (verbose || std::fabs(G_D) > tol)
        {
            print("Gibbs-Duhem", G_D);
            error_output++;
        }
    }
    return error_output;
}

int HelmholtzEoS::pvt_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Consistency of EoS PVT: Calculate volume roots at P, T, n and find P at roots (T, V, n)
    int error_output = 0;

    // Evaluate properties at P, T, n
    this->solve_PT(p_, T_, n_, 0, true);
    double p0 = p_;
    double T0 = T_;
    double V0 = this->v;
    std::vector<double> n0 = n_;

    // Consistency of PT solver
    for (std::complex<double> Z: this->Z_roots)
    {
        if (Z.imag() == 0.)
        {
            // Calculate total volume
            double V = Z.real() * N * this->units.R * T / p0;

            // Evaluate P(T,V,n)
            this->p = this->P(V, T0, n0);
            if (verbose || std::fabs(this->p - p0) > tol)
            {
                print("P(T, V, n) != p", {p, p0, std::fabs(p-p0)});
                error_output++;
            }
        }
    }

    // Calculate derivatives of P and V w.r.t. V, T and composition
    this->solve_PT(p0, T0, n0, 0, true);
    double dPdV = this->dP_dV();
    double d2PdV2 = this->d2P_dV2();
	double dPdT = this->dP_dT();
    double dVdT = this->dV_dT();
    std::vector<double> dPdn(nc), dVdn(nc);
    for (int i = 0; i < nc; i++)
    {
	    dPdn[i] = this->dP_dni(i);
	    dVdn[i] = this->dV_dni(i);
    }
    double d, dX{ 1e-5 };

    // Numerical derivative with respect to volume
    this->solve_VT(V0 + dX*V0, T0, n0, 0, true);
    double dPdV_num = (this->p - p0) / (dX*V0);
    d = std::log(std::fabs(dPdV + 1e-15)) - std::log(std::fabs(dPdV_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dP/dV != dP/dV", {dPdV, dPdV_num, d}); error_output++; }
    double d2PdV2_num = (this->dP_dV() - dPdV) / (dX*V0);
    d = std::log(std::fabs(d2PdV2 + 1e-15)) - std::log(std::fabs(d2PdV2_num + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("d2P/dV2 != d2P/dV2", {d2PdV2, d2PdV2_num, d}); error_output++; }

    // Numerical derivative with respect to temperature
    this->solve_VT(V0, T0 + dX*T0, n0, 0, true);
    double dPdT_num = (this->p - p0) / (dX*T0);
    d = std::log(std::fabs(dPdT + 1e-15)) - std::log(std::fabs(dPdT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dP/dT != dP/dT", {dPdT, dPdT_num, d}); error_output++; }

    this->solve_PT(p0, T0 + dX*T0, n0, 0, true);
    double dVdT_num = (this->v - V0) / (dX*T0);
    d = std::log(std::fabs(dVdT + 1e-15)) - std::log(std::fabs(dVdT_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("dV/dT != dV/dT", {dVdT, dVdT_num, d}); error_output++; }

    // Numerical derivative with respect to composition
    for (int i = 0; i < nc; i++)
    {
        double dni = dX*n0[i];
        n_[i] += dni;

        this->solve_VT(V0, T0, n_, 0, true);
        double dPdn_num = (this->p - p0) / dni;
        d = std::log(std::fabs(dPdn[i] + 1e-15)) - std::log(std::fabs(dPdn_num + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && n0[i] > 0.)) { print("dP/dni != dP/dni", {dPdn[i], dPdn_num, d}); error_output++; }

        this->solve_PT(p0, T0, n_, 0, true);
        double dVdn_num = (this->v - V0) / dni;
        d = std::log(std::fabs(dVdn[i] + 1e-15)) - std::log(std::fabs(dVdn_num + 1e-15));
        if (verbose || (!(std::fabs(d) < tol) && n0[i] > 0.)) { print("dV/dni != dV/dni", {dVdn[i], dVdn_num, d}); error_output++; }

        n_[i] -= dni;
    }

    return error_output;
}

int HelmholtzEoS::critical_point_test(std::vector<double>& n_, double tol, bool verbose)
{
    // Consistency of EoS PVT: Check if criticality conditions are satisfied at critical point
    int error_output = 0;

    // Calculate critical point and check if dP/dV = d2P/dV2 = 0
    HelmholtzEoS::CriticalPoint cp = this->critical_point(n_);
    this->solve_VT(cp.Vc, cp.Tc, n_, 0, true);
    double dPdV = this->dP_dV();
    double d2PdV2 = this->d2P_dV2();

    this->solve_VT(cp.Vc-1e-8, cp.Tc, n_, 0, true);
    double d2P_dV2_min = this->d2P_dV2();

    this->solve_VT(cp.Vc+1e-8, cp.Tc, n_, 0, true);
    double d2P_dV2_plus = this->d2P_dV2();

    if (verbose || cp.is_nan || std::fabs(dPdV) > tol || //std::fabs(d2PdV2) > tol * 1e5 || 
        !(d2P_dV2_plus <= 0. && d2P_dV2_min >= 0.))
    {
        print("Criticality conditions not satisfied, dP/dV", dPdV);
        print("d2P/dV2, min, plus", {d2PdV2, d2P_dV2_min, d2P_dV2_plus});
        cp.print_point();
        error_output++;
    }
    return error_output;
}

int HelmholtzEoS::properties_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Consistency of EoS residual properties at (P, T, n)
    int error_output = EoS::properties_test(p_, T_, n_, tol, verbose);
    double p0 = p_;
    double T0 = T_;
    std::vector<double> n0 = n_;
    double V0 = this->V(p0, T0, n0);
    double d;
    bool pt;

    // Gibbs free energy Gr(PT)(P, T, n)
    pt = true;
    double Gr = EoS::Gr(p0, T0, n0, 0, pt);
    double Grhh = this->Gr(p0, T0, n0, 0, pt);
    d = std::log(std::fabs(Gr + 1e-15)) - std::log(std::fabs(Grhh + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("HH::Gr != EoS::Gr", {Gr, Grhh, d}); error_output++; }

    // Enthalpy Hr(PT)(P, T, n)
    double Hr = EoS::Hr(p0, T0, n0, 0, pt);
    double Hrhh = this->Hr(p0, T0, n0, 0, pt);
    d = std::log(std::fabs(Hr + 1e-15)) - std::log(std::fabs(Hrhh + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("HH::Hr != EoS::Hr", {Hr, Hrhh, d}); error_output++; }

    // Entropy Sr(PT)(P, T, n)
    double Sr = EoS::Sr(p0, T0, n0, 0, pt);
    double Srhh = this->Sr(p0, T0, n0, 0, pt);
    d = std::log(std::fabs(Sr + 1e-15)) - std::log(std::fabs(Srhh + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("HH::Sr != EoS::Sr", {Sr, Srhh, d}); error_output++; }

    // Helmholtz free energy Ar(P, T, n)
    // double Ar = EoS::Ar(p0, T0, n0, 0, pt);
    // double Arhh = this->Ar(p0, T0, n0, 0, pt);
    // d = std::log(std::fabs(Ar + 1e-15)) - std::log(std::fabs(Arhh + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("HH::Ar != EoS::Ar", {Ar, Arhh, d}); error_output++; }

    // Internal energy Ur(P, T, n)
    // double Ur = EoS::Ur(p0, T0, n0, 0, pt);
    // double Urhh = this->Ur(p0, T0, n0, 0, pt);
    // d = std::log(std::fabs(Ur + 1e-15)) - std::log(std::fabs(Urhh + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("HH::Ur != EoS::Ur", {Ur, Urhh, d}); error_output++; }

    // Thermal properties tests
    double dT = 1e-3;

    // Residual heat capacity at constant volume Cvr
    pt = false;
    double CVr = this->Cvr(V0, T0, n0, 0, pt);
    double Ur0 = this->Ur(V0, T0, n0, 0, pt);
    double Ur1 = this->Ur(V0, T0 + dT, n0, 0, pt);
    double CVr_num = (Ur1-Ur0)/dT;
    d = std::log(std::fabs(CVr + 1e-15)) - std::log(std::fabs(CVr_num + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("Cvr", {CVr, CVr_num, d}); error_output++; }

    // // Heat capacity at constant volume Cv
    // double CV = this->Cv(p, T, n0, 0, pt);
    // double U0 = this->U(V0, T0, n0, 0, pt);
    // double U1 = this->U(V0, T0+dT, n0, 0, pt);
    // double CV_num = (U1-U0)/dT;
    // d = std::log(std::fabs(CV + 1e-15)) - std::log(std::fabs(CV_num + 1e-15));
    // if (verbose || !(std::fabs(d) < tol)) { print("Cv", {CV, CV_num, d}); error_output++; }

    // Heat capacity at constant pressure Cp
    pt = true;
    double CP = EoS::Cp(p0, T0, n0, 0, pt);
    double CPhh = this->Cp(p0, T0, n0, 0, pt);
    d = std::log(std::fabs(CP + 1e-15)) - std::log(std::fabs(CPhh + 1e-15));
    if (verbose || !(std::fabs(d) < tol)) { print("HH::Cp != EoS::Cp", {CP, CPhh, d}); error_output++; }

    // Derivatives w.r.t. composition
    pt = false;
    this->solve_VT(V0, T0, n0, 0, true);
    double SrVT0 = this->Sr(V0, T0, n0, 0, pt);
    double ArVT0 = this->Ar(V0, T0, n0, 0, pt);
    double UrVT0 = this->Ur(V0, T0, n0, 0, pt);
    double GrVT0 = this->Gr(V0, T0, n0, 0, pt);
    double HrVT0 = this->Hr(V0, T0, n0, 0, pt);
    
    std::vector<double> dSrVTdn = this->dSr_dni(V0, T0, n0, 0, pt);
    std::vector<double> dArVTdn = this->dAr_dni(V0, T0, n0, 0, pt);
    std::vector<double> dUrVTdn = this->dUr_dni(V0, T0, n0, 0, pt);
    std::vector<double> dGrVTdn = this->dGr_dni(V0, T0, n0, 0, pt);
    std::vector<double> dHrVTdn = this->dHr_dni(V0, T0, n0, 0, pt);

    pt = true;
    this->solve_PT(p0, T0, n0, 0, true);
    double SrPT0 = this->Sr(p0, T0, n0, 0, pt);
    double ArPT0 = this->Ar(p0, T0, n0, 0, pt);
    double UrPT0 = this->Ur(p0, T0, n0, 0, pt);
    double GrPT0 = this->Gr(p0, T0, n0, 0, pt);
    double HrPT0 = this->Hr(p0, T0, n0, 0, pt);

    std::vector<double> dSrPTdn = this->dSr_dni(p0, T0, n0, 0, pt);
    std::vector<double> dArPTdn = this->dAr_dni(p0, T0, n0, 0, pt);
    std::vector<double> dUrPTdn = this->dUr_dni(p0, T0, n0, 0, pt);
    std::vector<double> dGrPTdn = this->dGr_dni(p0, T0, n0, 0, pt);
    std::vector<double> dHrPTdn = this->dHr_dni(p0, T0, n0, 0, pt);

    for (int j = 0; j < nc; j++)
    {
        // VOLUME-based
        pt = false;

        // Derivative of ideal molar entropy si(PT) w.r.t. pressure and temperature
        // this->solve_VT(V0, T0, n0, 0, true);
        // double dsidP = this->dsi_dP(V0, T0, j, pt);
        // double dsidT = this->dsi_dT(V0, T0, j, pt);
        // double si0 = this->si(V0, T0, j, pt);

        // double V1 = this->V(p0+dP, T0, n0);
        // this->solve_VT(V1, T0, n0, 0, true);
        // double si1 = this->si(V1, T0, j, pt);
        // double dsi_num = (si1-si0)/dP;
        // d = std::log(std::fabs(dsidP + 1e-15)) - std::log(std::fabs(dsi_num + 1e-15));
        // if (verbose || !(std::fabs(d) < tol)) { print("dsi/dP VT", {static_cast<double>(j), dsidP, dsi_num, d}); error_output++; }

        // this->solve_VT(V0, T0+dT, n0, 0, true);
        // si1 = this->si(V0, T0+dT, j, pt);
        // dsi_num = (si1-si0)/dT;
        // d = std::log(std::fabs(dsidT + 1e-15)) - std::log(std::fabs(dsi_num + 1e-15));
        // if (verbose || !(std::fabs(d) < tol)) { print("dsi/dT VT", {static_cast<double>(j), dsidT, dsi_num, d}); error_output++; }

        // Composition derivatives
        std::vector<double> nn = n0;
        double dnj = 1e-5 * n0[j];
        nn[j] += dnj;

        // Derivative of residual entropy Sr w.r.t. composition
        double Sr1 = this->Sr(V0, T0, nn, 0, pt);
        double dSr_num = (Sr1-SrVT0)/dnj;
        d = std::log(std::fabs(dSrVTdn[j] + 1e-15)) - std::log(std::fabs(dSr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dSr/dnj VT", {dSrVTdn[j], dSr_num, d}); error_output++; }

        // Derivative of residual Helmholtz free energy Ar w.r.t. composition
        double Ar1 = this->Ar(V0, T0, nn, 0, pt);
        double dAr_num = (Ar1-ArVT0)/dnj;
        d = std::log(std::fabs(dArVTdn[j] + 1e-15)) - std::log(std::fabs(dAr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dAr/dnj VT", {dArVTdn[j], dAr_num, d}); error_output++; }

        // Derivative of residual internal energy Ur w.r.t. composition
        Ur1 = this->Ur(V0, T0, nn, 0, pt);
        double dUr_num = (Ur1-UrVT0)/dnj;
        d = std::log(std::fabs(dUrVTdn[j] + 1e-15)) - std::log(std::fabs(dUr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dUr/dnj VT", {dUrVTdn[j], dUr_num, d}); error_output++; }

        // Derivative of residual Gibbs free energy Ar w.r.t. composition
        double Gr1 = this->Gr(V0, T0, nn, 0, pt);
        double dGr_num = (Gr1-GrVT0)/dnj;
        d = std::log(std::fabs(dGrVTdn[j] + 1e-15)) - std::log(std::fabs(dGr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dGr/dnj VT", {dGrVTdn[j], dGr_num, d}); error_output++; }

        // Derivative of residual enthalpy Hr w.r.t. composition
        double Hr1 = this->Hr(V0, T0, nn, 0, pt);
        double dHr_num = (Hr1-HrVT0)/dnj;
        d = std::log(std::fabs(dHrVTdn[j] + 1e-15)) - std::log(std::fabs(dHr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dHr/dnj VT", {dHrVTdn[j], dHr_num, d}); error_output++; }

        // // Derivative of internal energy U w.r.t. composition
        // U1 = this->U(V0, T0, nn, 0, pt);
        // double dU_num = (U1-U0)/dnj;
        // d = std::log(std::fabs(dUdn[j] + 1e-15)) - std::log(std::fabs(dU_num + 1e-15));
        // if (verbose || !(std::fabs(d) < tol)) { print("dU/dnj VT", {dUdn[j], dU_num, d}); error_output++; }

        // // Derivative of enthalpy H w.r.t. composition
        // H1 = this->H(p0, T0, nn, 0, pt);
        // double dH_num = (H1-H0)/dnj;
        // d = std::log(std::fabs(dHdn[j] + 1e-15)) - std::log(std::fabs(dH_num + 1e-15));
        // if (verbose || !(std::fabs(d) < tol)) { print("dH/dnj VT", {dHdn[j], dH_num, d}); error_output++; }

        // PRESSURE-based
        pt = true;

        // Derivative of residual entropy Sr w.r.t. composition
        Sr1 = this->Sr(p0, T0, nn, 0, pt);
        dSr_num = (Sr1-SrPT0)/dnj;
        d = std::log(std::fabs(dSrPTdn[j] + 1e-15)) - std::log(std::fabs(dSr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dSr/dnj PT", {dSrPTdn[j], dSr_num, d}); error_output++; }

        // Derivative of residual Helmholtz free energy Ar w.r.t. composition
        Ar1 = this->Ar(p0, T0, nn, 0, pt);
        dAr_num = (Ar1-ArPT0)/dnj;
        d = std::log(std::fabs(dArPTdn[j] + 1e-15)) - std::log(std::fabs(dAr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dAr/dnj PT", {dArPTdn[j], dAr_num, d}); error_output++; }

        // Derivative of residual internal energy Ur w.r.t. composition
        Ur1 = this->Ur(p0, T0, nn, 0, pt);
        dUr_num = (Ur1-UrPT0)/dnj;
        d = std::log(std::fabs(dUrPTdn[j] + 1e-15)) - std::log(std::fabs(dUr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dUr/dnj PT", {dUrPTdn[j], dUr_num, d}); error_output++; }

        // Derivative of residual Gibbs free energy Ar w.r.t. composition
        Gr1 = this->Gr(p0, T0, nn, 0, pt);
        dGr_num = (Gr1-GrPT0)/dnj;
        d = std::log(std::fabs(dGrPTdn[j] + 1e-15)) - std::log(std::fabs(dGr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dGr/dnj PT", {dGrPTdn[j], dGr_num, d}); error_output++; }

        // Derivative of residual enthalpy Hr w.r.t. composition
        Hr1 = this->Hr(p0, T0, nn, 0, pt);
        dHr_num = (Hr1-HrPT0)/dnj;
        d = std::log(std::fabs(dHrPTdn[j] + 1e-15)) - std::log(std::fabs(dHr_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("dHr/dnj PT", {dHrPTdn[j], dHr_num, d}); error_output++; }

        // // Derivative of internal energy U w.r.t. composition
        // U1 = this->U(p0, T0, nn, 0, pt);
        // double dU_num = (U1-U0)/dnj;
        // d = std::log(std::fabs(dUdn[j] + 1e-15)) - std::log(std::fabs(dU_num + 1e-15));
        // if (verbose || !(std::fabs(d) < tol)) { print("dU/dnj", {dUdn[j], dU_num, d}); error_output++; }

        // // Derivative of enthalpy H w.r.t. composition
        // H1 = this->H(p0, T0, nn, 0, pt);
        // double dH_num = (H1-H0)/dnj;
        // d = std::log(std::fabs(dHdn[j] + 1e-15)) - std::log(std::fabs(dH_num + 1e-15));
        // if (verbose || !(std::fabs(d) < tol)) { print("dH/dnj", {dHdn[j], dH_num, d}); error_output++; }

    }

    return error_output;
}
