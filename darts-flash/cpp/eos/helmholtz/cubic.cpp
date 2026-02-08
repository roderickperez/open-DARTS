#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <complex>
#include <numeric>
#include <memory>

#include "dartsflash/eos/helmholtz/cubic.hpp"
#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/maths/maths.hpp"
#include "dartsflash/maths/root_finding.hpp"

CubicEoS::CubicEoS(CompData& comp_data, CubicEoS::CubicType type, bool volume_shift) : HelmholtzEoS(comp_data)
{
    // Define Cubic EoS version specific parameters
    double omegaA = 0.0;
    double omegaB = 0.0;
    std::vector<double> kappa(comp_data.nc);

    // Peng-Robinson EoS
    if (type == CubicEoS::PR)
    {
        this->d1 = 1. + std::sqrt(2.);
        this->d2 = 1. - std::sqrt(2.);
        double eta_c = 1. / (1. + std::cbrt(4.-std::sqrt(8.)) + std::cbrt(4.+std::sqrt(8.)));
        omegaA = (8. + 40.*eta_c) / (49. - 37.*eta_c);  // ~= 0.45724
        omegaB = eta_c / (3. + eta_c);  // ~= 0.0778

        // Calculate PR kappa
        for (int i = 0; i < comp_data.nc; i++)
	    {
            if (comp_data.ac[i] <= 0.49)
	    	{
		    	kappa[i] = 0.37464 + 1.54226 * comp_data.ac[i] - 0.26992 * std::pow(comp_data.ac[i], 2);
    		}
            else
		    {
			    kappa[i] = 0.379642 + 1.48503 * comp_data.ac[i] - 0.164423 * std::pow(comp_data.ac[i], 2) + 0.016667 * std::pow(comp_data.ac[i], 3);
    		}
	    }
    }
    // Soave-Redlich-Kwong EoS
    else if (type == CubicEoS::SRK)
    {
        this->d1 = 1.;
        this->d2 = 0.;
        omegaA = 1. / (9. * (std::cbrt(2.) - 1.));  // ~= 0.42748
        omegaB = (std::cbrt(2.) - 1.) / 3.;  // ~= 0.08664

        // Calculate SRK kappa
        for (int i = 0; i < comp_data.nc; i++)
    	{
            kappa[i] = 0.48 + 1.574 * comp_data.ac[i] - 0.176 * std::pow(comp_data.ac[i], 2);
	    }
    }

    // Initialize Mixing Rule object
	this->mix = std::make_unique<Mix>(Mix(comp_data, omegaA, omegaB, kappa, volume_shift, (type == CubicEoS::SRK)));
}

CubicEoS::CubicEoS(CompData& comp_data, CubicParams& cubic_params) : HelmholtzEoS(comp_data)
{
    d1 = cubic_params.d1;
    d2 = cubic_params.d2;
    mix = cubic_params.mix->getCopy();
}

CubicParams::CubicParams(double d1_, double d2_, double omegaA, double omegaB, std::vector<double>& kappa, CompData& comp_data, bool volume_shift)
{
    this->d1 = d1_;
    this->d2 = d2_;

    // Initialize Mixing Rule object
	this->mix = std::make_unique<Mix>(Mix(comp_data, omegaA, omegaB, kappa, volume_shift, false));
}

void CubicEoS::init_VT(double V_, double T_)
{
    this->v = V_;

    // Only recalculate if T is different
    if (T_ != T)
    {
        this->T = T_;

        // Calculate ai, aij, bi and bij mixing rule parameters, ignore output
        (void) mix->aij(T);
        (void) mix->bij();
    }
    return;
}
void CubicEoS::init_PT(double p_, double T_)
{
    // Only recalculate if T is different
    if (T_ != T)
    {
        // Calculate ai, aij, bi and bij mixing rule parameters, ignore output
        (void) mix->aij(T_);
        (void) mix->bij();
    }
    if (p_ != p || T_ != T)
    {
        this->p = p_;
        // this->p = this->units.input_to_pa(p_);
        this->T = T_;
    }
    return;
}

std::vector<double> CubicEoS::lnphi0(double X, double T_, bool pt)
{
    (void) pt;
    this->init_PT(X, T_);
    std::vector<double> lnphi0_(nc, NAN);

    // Loop over components
    for (int i = 0; i < nc; i++)
    {
        std::vector<double> n_(nc, 0.);
        n_[i] = 1.;

        if (this->eos_in_range(n_.begin()))
        {
            this->root_flag = EoS::RootFlag::STABLE;
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
    return lnphi0_;
}

void CubicEoS::calc_coefficients()
{
    // Calculate coefficients of cubic polynomial
    double N_inv = 1./N;
    double a_mix = mix->D() * std::pow(N_inv, 2);
    double b_mix = mix->B() * N_inv;
    double T_inv = 1./T;
    this->A = a_mix * p * std::pow(T_inv, 2);
    this->B = b_mix * p * T_inv;

    // Find roots of cubic EoS: f(Z) = Z^3 + a2 Z^2 + a1 Z + a0 = 0
    this->a2 = (d1 + d2 - 1.) * B - 1.;
    this->a1 = A + d1 * d2 * std::pow(B, 2) - (d1 + d2) * B * (B + 1.);
    this->a0 = - (A * B + d1 * d2 * std::pow(B, 2) * (B + 1.));
}
std::vector<double> CubicEoS::calc_coefficients(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Calculate coefficients of cubic polynomial
    (void) pt;
    this->init_PT(p_, T_);
    this->N = std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);

    // Mixing rules
    mix->zeroth_order(n_.begin() + start_idx);

    // Calculate coefficients
    this->calc_coefficients();
    return {this->a2, this->a1, this->a0};
}

std::vector<std::complex<double>> CubicEoS::Z()
{
    // Calculate coefficients for solving cubic polynomial
    this->calc_coefficients();
    return cubic_roots_analytical(a2, a1, a0);
}
double CubicEoS::dZ_dP()
{
    // Calculate derivative of compressibility factor w.r.t pressure
    // Implicit differentiation of f(Z) = Z^3 + a2 Z^2 + a1 Z + a0 = 0 with respect to pressure
    // df/dP = 3Z^2 dZ/dP + da2/dP Z^2 + 2 a2 Z dZ/dP + da1/dP Z + a1 dZ/dP + da0/dP = 0
    // dZ/dP = -(da2/dP Z^2 + da1/dP Z + da0/dP) / [3Z^2 + 2 a2 Z + a1]

    // Calculate A and B coefficients for solving cubic polynomial
    this->calc_coefficients();

    // Calculate dA/dP and dB/dP
    double dAdP = this->A/p;
    double dBdP = this->B/p;

    // Find derivatives of a0, a1 and a2 w.r.t. pressure
    this->da2dP = (d1 + d2 - 1.) * dBdP;
    this->da1dP = dAdP + 2 * d1 * d2 * B * dBdP - (d1 + d2) * (2 * B + 1.) * dBdP;
    this->da0dP = -(dAdP * B + A * dBdP + d1 * d2 * (3 * std::pow(B, 2) + 2 * B) * dBdP);

    return -(da2dP * std::pow(z, 2) + da1dP * z + da0dP) / (3 * std::pow(z, 2) + 2 * a2 * z + a1);
}
double CubicEoS::d2Z_dP2()
{
    // Calculate second derivative of compressibility factor w.r.t pressure
    // dZ/dP = -(da2/dP Z^2 + da1/dP Z + da0/dP) / [3Z^2 + 2 a2 Z + a1]
    // d2Z/dP2 = [-(2 da2/dP Z dZ/dP + d2a1/dP2 Z + da1/dP dZ/dP + d2a0/dP2) * (3Z^2 + 2 a2 Z + a1)
    //            + (da2/dP Z^2 + da1/dP Z + da0/dP) * (6 Z dZ/dP + 2 da2/dP Z + 2 a2 dZ/dP + da1/dP)] / [3Z^2 + 2 a2 Z + a1]^2;

    // Calculate dZ/dP
    double dZdP = this->dZ_dP();

    // Calculate dA/dP and dB/dP
    double dAdP = this->A/p;
    double dBdP = this->B/p;

    // Find second derivatives of a0 and a1 w.r.t. pressure (d2a2/dP2 = 0)
    this->d2a1dP2 = 2 * d1 * d2 * std::pow(dBdP, 2) - (d1 + d2) * 2 * std::pow(dBdP, 2);
    this->d2a0dP2 = -(2 * dAdP * dBdP + d1 * d2 * (6 * B + 2.) * std::pow(dBdP, 2));

    // Calculate d2Z/dP2
    double denom = 1./(3 * std::pow(z, 2) + 2 * a2 * z + a1);
    return -(2 * z * da2dP * dZdP + d2a1dP2 * z + da1dP * dZdP + d2a0dP2) * denom 
            + (da2dP * std::pow(z, 2) + da1dP * z + da0dP) * (6 * z * dZdP + 2 * da2dP * z + 2 * a2 * dZdP + da1dP) * std::pow(denom, 2);
}
double CubicEoS::d2f_dZ2(double Z_)
{
    // Calculate curvature of f(Z)
    return 6.*Z_ + 2.*this->a2;
}
EoS::RootFlag CubicEoS::identify_root(bool& is_below_spinodal)
{
    // Identify root as being vapour(-like) or liquid(-like)
    // Store P and T
    double p_ = p, v_ = v, T_ = T, z_ = z;

    // If temperature is subcritical, f(Zinfl) > 0 for which f'(Zinfl) = 0
    if (this->f_at_zero_slope_inflection_point_at_T(T) > 0.)
    {
        is_below_spinodal = true;
        // Restore P and T
        this->p = p_; this->v = v_;

        // If dZ/dP > 0, pressure must be above the two-phase region or inflection point of Z w.r.t. P --> phase is liquid(-like)
        if (this->dZ_dP() > 0.)
        {
            return EoS::RootFlag::MIN;
        }

        // For pressures below mechanical spinodal, curvature is negative and phase is a vapour, else liquid
        return (this->d2Z_dP2() < 0.) ? EoS::RootFlag::MAX : EoS::RootFlag::MIN;
    }
    // Else, temperature is supercritical: find if volume is smaller or larger than critical volume
    else
    {
        is_below_spinodal = false;
        // Find critical point
        HelmholtzEoS::CriticalPoint cp = this->critical_point(n);

        // Set p, v and T back to original values
        this->init_VT(v_, T_);
        this->p = p_;
        this->z = z_;
        (void) mix->zeroth_order(n.begin());

        // If V > Vc, vapour-like; else, liquid-like
        return (this->v > cp.Vc) ? EoS::RootFlag::MAX : EoS::RootFlag::MIN;
    }
}

// Methods to calculate critical point for pure component and mixtures
double CubicEoS::dfdZ_at_inflection_point(double p_, double T_)
{
    // Calculate Z_infl from f''(Z) = 6Z + 2 a2
    if (T_ != T)
    {
        // Calculate ai, aij, bi and bij mixing rule parameters, ignore output
        (void) mix->aij(T_);
        (void) mix->bij();
        (void) mix->zeroth_order(n.begin());
        this->T = T_;
    }
    this->p = p_;
    this->calc_coefficients();

    // Find slope at Z = 0. If negative, pressure is too high
    if (this->a1 < 0.) { return INFINITY; }

    // Calculate the slope of the cubic polynomial at the inflection point
    double Z_infl = -this->a2 / 3.;
    return 3.*std::pow(Z_infl, 2) + 2 * this->a2 * Z_infl + this->a1;
}
double CubicEoS::f_at_zero_slope_inflection_point_at_T(double T_)
{
    // Find slope of f w.r.t. Z at inflection point
    if (T_ != T)
    {
        // Calculate ai, aij, bi and bij mixing rule parameters, ignore output
        (void) mix->aij(T_);
        (void) mix->bij();
        (void) mix->zeroth_order(n.begin());
        this->T = T_;
    }

    // Find analytically the pressure where df/dZ = d2f/dZ^2 = 0
    // Calculate coefficients of cubic polynomial
    double N_inv = 1./N;
    double a_mix = mix->D() * std::pow(N_inv, 2);
    double b_mix = mix->B() * N_inv;
    double T_inv = 1./T;

    // Find p and Z where f''(Z) = f'(Z) = 0
    // f''(Zinfl) = 0 -> Zinfl = -a2/3
    // Substituting Zinfl into f': f'(Zinfl) = 3 Zinfl^2 + 2a2 Zinfl + a1 = 0 -> 1/3 a2^2 = a1  // or 1/3 [(d1 + d2 - 1)^2 B^2 + 2 (d1 + d2 - 1) B + 1] = 0
    // Find p that satisfies f'(Zinfl) = 0:
    // a p^2 + b p + c = 0 with a = [1/3 (d1 + d2 - 1)^2 - d1d2 + d1 + d2] (bmix/T)^2, b = [-2/3 (d1 + d2 - 1) + d1 + d2] bmix/T - amix/T^2 and c = 1/3)
    // A = amix p / T^2, B = bmix p / T
    double a = (1./3 * std::pow(d1 + d2 - 1, 2) - d1 * d2 + d1 + d2) * std::pow(b_mix * T_inv, 2);
    double b = (-2./3 * (d1 + d2 - 1.) + d1 + d2) * b_mix * T_inv - a_mix * std::pow(T_inv, 2);
    double c = 1./3.;
    double discr = std::pow(b, 2) - 4.*a*c;
    if (discr < 0.)
    {
        // In case discriminant D < 0, temperature is above critical temperature: return -INF
        return -INFINITY;
    }

    this->p = (-b - std::sqrt(discr)) / (2*a);
    this->calc_coefficients();
    double Z_infl = -this->a2 / 3.;
    this->v = Z_infl * (this->N * this->units.R * T) / p;

    // Calculate the value of f(Zinfl) at p where f'(Zinfl) = 0
    return std::pow(Z_infl, 3) + this->a2 * std::pow(Z_infl, 2) + this->a1 * Z_infl + this->a0;
}
bool CubicEoS::is_critical(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Determine if temperature is subcritical or supercritical
    (void) p_;
    (void) pt;

    // Calculate mixing rules
    std::copy(n_.begin() + start_idx, n_.begin() + start_idx + nc, this->n.begin());
    this->N = std::accumulate(n.begin(), n.end(), 0.);

    // Determine slope of f(Z) at P where inflection point is a root of the cubic polynomial
    return (this->f_at_zero_slope_inflection_point_at_T(T_) < 0.);
}
HelmholtzEoS::CriticalPoint CubicEoS::critical_point(std::vector<double>& n_)
{
    // Critical point of mixture
    // dP/dV = 0 or dZ/dP = -inf and d2P/dV2 = 0 or d2Z/dP2 = 0
    this->n = n_;
    this->N = std::accumulate(n.begin(), n.end(), 0.);

    // Start with mixture average Tc
    double nT_inv = 1./N;    
    std::vector<double> x = n;
    std::transform(n.begin(), n.end(), x.begin(), [&nT_inv](double element) { return element *= nT_inv; });
    double Tinit = std::inner_product(x.begin(), x.begin() + nc, this->compdata.Tc.begin(), 0.);
    
    // Find lower bound where f(Zinfl) > 0
    double Tmin = Tinit - 10.;
    int iter = 0;
    while (iter < 10 && Tmin > 100.)
    {
        if (f_at_zero_slope_inflection_point_at_T(Tmin) > 0)
        {
            break;
        }
        Tmin -= 100.;
        iter++;
    }
    // Find upper bound where f(Zinfl) < 0
    double Tmax = Tinit + 10.;
    iter = 0;
    while (iter < 10 && Tmax < 1200.)
    {
        if (f_at_zero_slope_inflection_point_at_T(Tmax) < 0)
        {
            break;
        }
        Tmax += 100.;
        iter++;
    }

    // Use Brent's method to find Tc for which dP/dV = 0 at inflection point d2Z/dP2 = 0
    RootFinding rf;
    auto fn = std::bind(&CubicEoS::f_at_zero_slope_inflection_point_at_T, this, std::placeholders::_1);
    double tol_T = 1e-12;
    double tol_f = 1e-13;
    int error_output = rf.brent(fn, Tinit, Tmin, Tmax, tol_f, tol_T);
    if (error_output)
    {
        print("CP for composition n not found", n);
        return HelmholtzEoS::CriticalPoint();  // CP initialized with nans
    }

    // Get Pc, Tc, Vc and Zc
    double Pc = this->p;
    double Tc = rf.getx();
    double Vc = this->v;
    double Zc = Pc * Vc / (this->N * this->units.R * Tc);
    return HelmholtzEoS::CriticalPoint(Pc, Tc, Vc, Zc);
}

double CubicEoS::V() {
    // Check if three real roots
    this->is_stable_root = true;
    this->is_preferred_root = RootSelect::ACCEPT;
    if (Z_roots[2].imag() != 0. || Z_roots[2].real() < 0.)
    {
        // only one real root
        this->z = Z_roots[0].real();
        this->v = z * N * this->units.R * T / p;

        if (root_flag > EoS::RootFlag::STABLE)
        {
            bool is_below_spinodal;
            this->root_type = this->identify_root(is_below_spinodal);
            if (root_flag != this->root_type)
            {
                this->is_preferred_root = RootSelect::REJECT;
            }
        }
        else
        {
            this->root_type = EoS::RootFlag::NONE;
        }
    }
    else
    {
        // Find zmin and zmax
        std::sort(Z_roots.begin(), Z_roots.end(), [](std::complex<double> a, std::complex<double> b) { return a.real() > b.real(); });
        double z_min = Z_roots[2].real();
        double z_max = Z_roots[0].real();

        if ((mix->B() > 0. && z_min * N * T / p <= mix->B()) || (mix->B() < 0. && z_min * N * T / p >= mix->B()))
        {
            // Volume of smallest root smaller than co-volume parameter B
            this->z = z_max;
            this->root_type = EoS::RootFlag::MAX;
        }
        else 
        {
            // Calculate Gibbs energy of min (L) and max (V) roots
            double gE_l = (z_min - 1.) - std::log(z_min - B) - A / (B * (d1 - d2)) * std::log((z_min + d1 * B) / (z_min + d2 * B));
            double gE_v = (z_max - 1.) - std::log(z_max - B) - A / (B * (d1 - d2)) * std::log((z_max + d1 * B) / (z_max + d2 * B));
            if (root_flag == EoS::RootFlag::STABLE)
            {
                if (gE_v < gE_l)
                {
                    this->z = z_max;
                    this->root_type = EoS::RootFlag::MAX;
                }
                else
                {
                    this->z = z_min;
                    this->root_type = EoS::RootFlag::MIN;
                }
            }
            else
            {
                this->z = (root_flag == EoS::RootFlag::MIN) ? z_min : z_max;  // min (L) or max (V) root
                this->root_type = root_flag;
                this->is_stable_root = ((root_flag == EoS::RootFlag::MAX && gE_v < gE_l) || (root_flag == EoS::RootFlag::MIN && gE_l < gE_v)) ? true : false;
            }
        }
    }
    this->v = z * N * this->units.R * T / p;
    this->n_iterations = 1;

    return v;
}

// Calculate derivatives of Helmholtz function
double CubicEoS::F() 
{
    return -N * g - mix->D()/T * f;
}
double CubicEoS::dF_dV() 
{
    return this->F_V() * this->units.R_inv;
}
double CubicEoS::dF_dT() 
{
    return this->F_T() + this->F_D() * mix->DT();
}
double CubicEoS::dF_dni(int i) 
{
    return this->F_n() + this->F_B() * mix->Bi(i) + this->F_D() * mix->Di(i);
}
double CubicEoS::d2F_dnidnj(int i, int j) 
{
    return this->F_nB() * (mix->Bi(i) + mix->Bi(j)) + this->F_BD()*(mix->Bi(i) * mix->Di(j) + mix->Bi(j) * mix->Di(i))
        + this->F_B() * mix->Bij(i, j) + this->F_BB() * mix->Bi(i) * mix->Bi(j) + this->F_D() * mix->Dij(i, j);
}
double CubicEoS::d2F_dTdni(int i) 
{
    return (this->F_BT() + this->F_BD() * mix->DT()) * mix->Bi(i) + this->F_DT() * mix->Di(i) + this->F_D() * mix->DiT(i);
}
double CubicEoS::d2F_dVdni(int i) 
{
    return (this->F_nV() + this->F_BV() * mix->Bi(i) + this->F_DV() * mix->Di(i)) * this->units.R_inv;
}
double CubicEoS::d2F_dTdV() 
{
    return (this->F_TV() + this->F_DV() * mix->DT()) * this->units.R_inv;
}
double CubicEoS::d2F_dV2() 
{
    return this->F_VV() * std::pow(this->units.R_inv, 2);
}
double CubicEoS::d2F_dT2() 
{
    return this->F_TT() + 2*this->F_DT() * mix->DT() + this->F_D() * mix->DTT();
}
double CubicEoS::d3F_dV3()
{
    return this->F_VVV() * std::pow(this->units.R_inv, 3);
}

// Elements of Helmholtz function and derivatives
// Zero'th order: g, f, B, D
void CubicEoS::zeroth_order(std::vector<double>::iterator n_it)
{
    // Zero'th order parameters for P, T, n specification
    // Mixing rules
    mix->zeroth_order(n_it);

    // Calculate compressibility roots
    this->Z_roots = this->Z();

    // Calculate V(P, T, n) and zero'th order parameters
    this->zeroth_order(this->V());
    return;
}
void CubicEoS::zeroth_order(std::vector<double>::iterator n_it, double V_)
{
    // Zero'th order parameters for (T, V, n) specification
    // Mixing rules
    mix->zeroth_order(n_it);
    this->zeroth_order(V_);
    return;
}
void CubicEoS::zeroth_order(double V_)
{
    // Elements of Helmholtz function and derivatives
    // Zero'th order: g, f, B, D
    this->Vr = V_/this->units.R;

    g = std::log(1. - mix->B()/Vr);
    f = 1./(mix->B() * (d1-d2)) * std::log((Vr + d1*mix->B())/(Vr + d2*mix->B()));
    return;
}

// First order: g_V, g_B, f_V, f_B, B_i, D_i, D_T
void CubicEoS::first_order(std::vector<double>::iterator n_it)
{
    mix->first_order(n_it);

    g_V = mix->B() / (Vr * (Vr - mix->B()));
    g_B = -1. / (Vr - mix->B());
    f_V = -1. / ((Vr + d1 * mix->B()) * (Vr + d2 * mix->B()));
    f_B = -(f + Vr * f_V) / mix->B();
    return;
}
double CubicEoS::F_n() 
{
    return -g;
}
double CubicEoS::F_T() 
{
    return mix->D()/std::pow(T, 2) * f;
}
double CubicEoS::F_V() 
{
    return -N * g_V - mix->D()/T * f_V;
}
double CubicEoS::F_B() 
{
    return -N * g_B - mix->D()/T * f_B;
}
double CubicEoS::F_D() 
{
    return -f/T;
}

// Second order:
void CubicEoS::second_order(std::vector<double>::iterator n_it)
{
    this->first_order(n_it);
    mix->second_order(T, n_it);

    g_VV = -1. / std::pow(Vr - mix->B(), 2) + 1. / std::pow(Vr, 2);
    g_BV = 1. / std::pow(Vr - mix->B(), 2);
    g_BB = -1. / std::pow(Vr - mix->B(), 2);

    double Vd1B_inv = 1./(Vr + d1 * mix->B());
    double Vd2B_inv = 1./(Vr + d2 * mix->B());
    f_VV = 1. / (mix->B() * (d1 - d2)) * (-std::pow(Vd1B_inv, 2) + std::pow(Vd2B_inv, 2));
    f_BV = -(2. * f_V + Vr * f_VV) / mix->B();
    f_BB = -(2. * f_B + Vr * f_BV) / mix->B();

    double V_B_inv = 1./(Vr - mix->B());
    double V_inv = 1./Vr;
    g_VVV = 2. * std::pow(V_B_inv, 3) - 2. * std::pow(V_inv, 3);
    f_VVV = 1. / (mix->B() * (d1 - d2)) * (2. * std::pow(Vd1B_inv, 3) - 2. * std::pow(Vd2B_inv, 3));
    return;
}
double CubicEoS::F_nV() 
{
    return -g_V;
}
double CubicEoS::F_nB() 
{
    return -g_B;
}
double CubicEoS::F_TT() 
{
    return -2. * this->F_T()/T;
}
double CubicEoS::F_BT() 
{
    return mix->D() * f_B / std::pow(T, 2);
}
double CubicEoS::F_DT() 
{
    return f / std::pow(T, 2);
}
double CubicEoS::F_BV() 
{
    return -N * g_BV - mix->D()/T * f_BV;
}
double CubicEoS::F_BB() 
{
    return -N * g_BB - mix->D()/T * f_BB;
}
double CubicEoS::F_DV() 
{
    return -f_V / T;
}
double CubicEoS::F_BD() 
{
    return -f_B / T;
}
double CubicEoS::F_TV() 
{
    return mix->D() / std::pow(T, 2) * f_V;
}
double CubicEoS::F_VV() 
{
    return -N * g_VV - mix->D()/T * f_VV;
}
double CubicEoS::F_VVV()
{
    return -N * g_VVV - mix->D()/T * f_VVV;
}

int CubicEoS::mix_dT_test(double T_, std::vector<double>& n_, double tol)
{
    // Analytical derivatives of mixing rule w.r.t. temperature
    int error_output = 0;
    this->T = T_;
    double dT = 2e-3;

    // Calculate mixing rule parameters at p, T-dT, n
    std::vector<double> ai_ = mix->ai(T-dT);
    std::vector<double> aij_ = mix->aij(T-dT);

    // Calculate mixing rule parameters at p, T+dT, n
    std::vector<double> ai1 = mix->ai(T+dT);
    std::vector<double> aij1 = mix->aij(T+dT);

    // Calculate mixing rule parameters at p, T, n
    std::vector<double> ai0 = mix->ai(T);
    std::vector<double> aij0 = mix->aij(T);
    mix->second_order(T, n_.begin());

    // Calculate derivatives of ai w.r.t. T
    for (int i = 0; i < nc; i++)
    {
        double dai_num = (ai1[i]-ai_[i])/(2*dT);
        double d = (dai_num - mix->dai_dT(T, i))/dai_num;
        if (!(std::fabs(d) < tol))
        {
            print("dai/dT", {dai_num, mix->dai_dT(T, i), d});
            error_output++;
        }

        double d2ai_num = (ai1[i] - 2*ai0[i] + ai_[i])/std::pow(dT, 2);
        d = (d2ai_num - mix->d2ai_dT2(T, i))/d2ai_num;
        if (!(std::fabs(d) < tol))
        {
            print("d2ai/dT2", {d2ai_num, mix->d2ai_dT2(T, i), d});
            error_output++;
        }
    }

    // Calculate derivatives of aij w.r.t. T
    for (int i = 0; i < nc; i++)
    {
        for (int j = 0; j < nc; j++)
        {
            int idx = i*nc + j;
            double daij_num = (aij1[idx]-aij_[idx])/(2*dT);
            double d = (daij_num - mix->daij_dT(T, i, j))/daij_num;
            if (!(std::fabs(d) < tol))
            {
                print("daij/dT", {daij_num, mix->daij_dT(T, i, j), d});
                error_output++;
            }

            double d2aij_num = (aij1[idx] - 2*aij0[idx] + aij_[idx])/std::pow(dT, 2);
            d = (d2aij_num - mix->d2aij_dT2(T, i, j))/d2aij_num;
            if (!(std::fabs(d) < tol))
            {
                print("d2aij/dT2", {d2aij_num, mix->d2aij_dT2(T, i, j), d});
                error_output++;
            }
        }
    }

    return error_output;
}
