//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_IAPWS_IAPWS95_H
#define OPENDARTS_FLASH_EOS_IAPWS_IAPWS95_H
//--------------------------------------------------------------------------

#include <unordered_map>
#include <string>
#include <vector>

#include "dartsflash/eos/helmholtz/helmholtz.hpp"
#include "dartsflash/global/global.hpp"

namespace iapws95 {
    extern double Mw, Tc, rhoc, Pc, Vc, Zc, R;

    extern std::vector<double> ni0, ji0;
    extern std::vector<int> ci, di, ei, Ci, Di;
    extern std::vector<double> ti, ni, ai, bi, ji, Bi, Ai, betai;
}

namespace iapws95ref {
    extern double phi0, phi0_d, phi0_dd, phi0_t, phi0_tt, phi0_dt;
    extern double phir, phir_d, phir_dd, phir_t, phir_tt, phir_dt;
    extern std::vector<double> T, rho, p, cv, w, s;
}

class IAPWS95 : public HelmholtzEoS
{
protected:
    double vm, dv, tau, T_for_ig{ NAN };  // molar volume, reduced volume, reduced temperature, T for initial guesses for volume solver
    double phir, phir_d, phir_dd, phir_ddd, phir_t, phir_tt, phir_dt;
    double p_minL, d_minL, p_maxV, d_maxV;  // Minimum/maximum pressure and reduced volume for L and V roots
    bool iapws_ideal;

public:
    IAPWS95(CompData& comp_data, bool iapws_ideal_);

    virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<IAPWS95>( *this ); }

    virtual void init_VT(double V_, double T_) override;
    virtual void init_PT(double p_, double T_) override;

    // Volume, pressure function, compressibility factor and derivatives
    virtual double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return HelmholtzEoS::V(p_, T_, n_, start_idx, pt); }
    virtual double P(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false) override { return HelmholtzEoS::P(V_, T_, n_, start_idx, pt); }
    virtual double V() override;
    virtual double P() override;
    double Pd(double d_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);  // evaluate pressure as a function of d
    double Zd(double d_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);  // evaluate Z-factor as a function of d

    virtual std::vector<std::complex<double>> Z() override { return {NAN}; }
	virtual double dZ_dP() override { return NAN; }
	virtual double d2Z_dP2() override { return NAN; }
    virtual EoS::RootFlag identify_root(bool& is_below_spinodal) override { (void) is_below_spinodal; return EoS::RootFlag::MIN; }

    // Critical point calculations
    virtual CriticalPoint critical_point(std::vector<double>& n_) override { (void) n_; return HelmholtzEoS::CriticalPoint(iapws95::Pc, iapws95::Tc, iapws95::Vc, iapws95::Zc); }
	virtual bool is_critical(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override 
        { (void) p_; (void) n_; (void) start_idx; (void) pt; return (T_ > iapws95::Tc); }

protected:
    // Zero'th, first and second order parameters for Helmholtz formulation
	virtual void zeroth_order(std::vector<double>::iterator n_it) override { (void) n_it; return; };
	virtual void zeroth_order(std::vector<double>::iterator n_it, double V_) override { (void) n_it; return this->zeroth_order(V_); }
	virtual void zeroth_order(double V_) override;
	virtual void first_order(std::vector<double>::iterator n_it) override;
	virtual void second_order(std::vector<double>::iterator n_it) override;

    // Find minimum P and d for liquid root to exist and maximum P and d for vapour root to exist
    void calc_initial_guesses();

    // Helmholtz function and derivatives
    virtual double F() override;
    virtual double dF_dV() override;
    virtual double dF_dT() override;
    virtual double dF_dni(int i) override;
    virtual double d2F_dnidnj(int i, int j) override;
    virtual double d2F_dTdni(int i) override;
    virtual double d2F_dVdni(int i) override;
    virtual double d2F_dTdV() override;
    virtual double d2F_dV2() override;
    virtual double d2F_dT2() override;
    virtual double d3F_dV3() override;

public:
    // Translate from dimensionless density and temperature to volume and temperature
    void set_d(double V_);
    void set_tau(double T_);
    double get_tau() { return this->tau; }
    double get_dv() { return this->dv; }
    double get_v(double d_, double N_);
    double dP_dd();
    double d2P_dd2();

    double dd_dV() { return -iapws95::Mw * 1e-3 / (std::pow(this->vm, 2) * iapws95::rhoc) * 1./N; }
    double d2d_dV2() { return 2. * iapws95::Mw * 1e-3 / (std::pow(this->vm, 3) * iapws95::rhoc) * std::pow(1./N, 2); }
    double d2d_dVdni() { return -2. * iapws95::Mw * 1e-3 / (std::pow(this->vm, 3) * iapws95::rhoc) * this->v/std::pow(N, 3) + iapws95::Mw * 1e-3 / (std::pow(this->vm, 2) * iapws95::rhoc) * 1./std::pow(N, 2); }

    double dd_dni() { return iapws95::Mw * 1e-3 / (std::pow(this->vm, 2) * iapws95::rhoc) * (this->v/std::pow(N, 2)); }
    double d2d_dni2() { return iapws95::Mw * 1e-3 / iapws95::rhoc * (2. / std::pow(this->vm, 3) * std::pow(this->v/std::pow(N, 2), 2) - 2. / std::pow(this->vm, 2) * this->v/std::pow(N, 3)); }
    
    double dtau_dT() { return -iapws95::Tc / std::pow(this->T, 2); }
    double d2tau_dT2() { return 2. * iapws95::Tc / std::pow(this->T, 3); }

    // Objective functions for Brent's method root finding
    double P_obj(double d_);
    double dP_obj(double d_);

    // phir, phi0 and derivatives
    void set_phir();
    void set_dphir();

    double phi0();
    double phi0_d();
    double phi0_dd();
    double phi0_t();
    double phi0_tt();
    double phi0_dt() { return 0.; }

    // Ideal heat capacities, enthalpy and entropy
	virtual double cpi(double T_, int i) override;  // Cpi/R: heat capacity at constant pressure of component i
	virtual double hi(double T_, int i) override;  // Hi/R: Ideal gas enthalpy
	virtual double si(double X, double T_, int i, bool pt=true) override;  // Si/R: ideal gas entropy at constant pressure/constant volume
    // virtual double dsi_dP(double X, double T_, int i, bool pt=true) override;
    // virtual double dsi_dT(double X, double T_, int i, bool pt=true) override;
    virtual std::vector<double> dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;

	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true) override;

    virtual int derivatives_test(double V_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
    int references_test(double tol, bool verbose=false);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_IAPWS_IAPWS95_H
//--------------------------------------------------------------------------
