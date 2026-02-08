//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_EOS_H
#define OPENDARTS_FLASH_EOS_EOS_H
//--------------------------------------------------------------------------

#include <vector>
#include <string>
#include <map>
#include <memory>
#include <Eigen/Dense>
#include "dartsflash/global/components.hpp"
#include "dartsflash/global/units.hpp"

class EoS
{
public:
	enum RootFlag : int { NONE = -2, STABLE = -1, MIN = 0, MAX };
	enum RootSelect : int { REJECT = 0, ACCEPT, PREFER };
	enum class Property : int { ENTROPY = 0, GIBBS, ENTHALPY, HELMHOLTZ, INTERNAL_ENERGY };

protected:
	int nc, ni = 0, ns;
	double v{ -1. }, p{ -1. }, T{ -1. }, z{ -1. }, N{ -1. };
	std::vector<double> n;
	std::map<int, std::vector<double>> eos_range, stationary_point_range;
	std::vector<double> dlnphidn, dlnphidP, dlnphidT, d2lnphidPdT, d2lnphidT2, d2lnphidTdn;
	RootFlag root_flag{ STABLE }, root_type{ STABLE };
	RootSelect is_preferred_root = RootSelect::ACCEPT;
	bool is_stable_root = true;
	bool multiple_minima = true;
	CompData compdata;
	Units units;

public:
	EoS(CompData& comp_data);
	virtual ~EoS() = default;

	EoS(const EoS&) = default;
	virtual std::unique_ptr<EoS> getCopy() = 0;

	// Getters and setters for EoS range, roots, comp_data, etc.
	void set_eos_range(int i, const std::vector<double>& range);
	bool eos_in_range(std::vector<double>::iterator n_it, bool in_stationary_point_range = false);
	
	CompData &get_comp_data() { return this->compdata; }
	void set_root_flag(EoS::RootFlag root_flag_) { this->root_flag = root_flag_; }
	virtual RootFlag is_root_type(bool& is_below_spinodal) { (void) is_below_spinodal; return EoS::RootFlag::STABLE; }
	RootFlag is_root_type(std::vector<double>::iterator n_it, bool& is_below_spinodal, bool pt=true);
	RootFlag is_root_type(double X, double T_, std::vector<double>& n_, bool& is_below_spinodal, int start_idx=0, bool pt=true); 
	
	bool is_stable() { return this->is_stable_root; }
	virtual RootSelect select_root(std::vector<double>::iterator n_it) { (void) n_it; return RootSelect::ACCEPT; }
	bool has_multiple_minima() { return this->multiple_minima; }

	// Solve EoS
	virtual void init_PT(double p_, double T_) = 0;
	virtual void solve_PT(std::vector<double>::iterator n_it, bool second_order=true) = 0;
	virtual void solve_PT(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true);
	virtual void init_VT(double V_, double T_) = 0;
	virtual void solve_VT(std::vector<double>::iterator n_it, bool second_order=true) = 0;
	virtual void solve_VT(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true);

	virtual double lnphii(int i) = 0;
	virtual double dlnphii_dP(int i) { return dlnphidP[i]; }
	virtual double dlnphii_dT(int i) { return dlnphidT[i]; }
	virtual double dlnphii_dnj(int i, int j) { return dlnphidn[i*nc + j]; }
	virtual double d2lnphii_dPdT(int i) { return d2lnphidPdT[i]; }
	virtual double d2lnphii_dT2(int i) { return d2lnphidT2[i]; }
	virtual double d2lnphii_dTdnj(int i, int j) { return d2lnphidTdn[i*nc + j]; }

	std::vector<double> lnphi();
	virtual std::vector<double> dlnphi_dn();
	virtual std::vector<double> dlnphi_dP();
	virtual std::vector<double> dlnphi_dT();
	virtual std::vector<double> d2lnphi_dPdT();
	virtual std::vector<double> d2lnphi_dT2();
	virtual std::vector<double> d2lnphi_dTdn();
	std::vector<double> lnphi(double p_, double T_, std::vector<double>& n_);
	std::vector<double> fugacity(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	// Ideal heat capacities, enthalpy and entropy
	virtual double cpi(double T_, int i);  // cpi/R: molar ideal gas heat capacity at constant pressure of component i
	double cvi(double T_, int i);  // cvi/R: molar ideal gas heat capacity at constant volume of component i
	virtual double hi(double T_, int i);  // hi/R: molar ideal gas enthalpy
	virtual double si(double X, double T_, int i, bool pt=true);  // Si/R: molar ideal gas entropy at constant pressure/constant volume
	virtual double dsi_dP(double X, double T_, int i, bool pt=true);  // 1/R dsi/dP: derivative of molar ideal gas entropy at constant pressure/constant volume w.r.t. pressure
	virtual double dsi_dT(double X, double T_, int i, bool pt=true);  // 1/R dsi/dT: derivative of molar ideal gas entropy at constant pressure/constant volume w.r.t. temperature

	double Cpi(double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Cpi/R: heat capacity at constant pressure of mixture n
	double Cvi(double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Cvi/R: heat capacity at constant volume of mixture n
	double Hi(double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Hi/R: Ideal gas enthalpy of mixture n
	double Si(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Si/R: ideal gas entropy at constant pressure/constant volume of mixture n
	double Gi(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Gi/R: ideal gas Gibbs free energy of mixture n
	// double Ai(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);  // Ai/R: ideal gas Helmholtz free energy of mixture n
	// double Ui(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);  // Ui/R: ideal gas internal energy of mixture n

	// dHi/dP = 0
	double dSi_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double dGi_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// double dAi_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// double dUi_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	double dHi_dT(double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Hi/R: Ideal gas enthalpy of mixture n
	double dSi_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Si/R: ideal gas entropy at constant pressure/constant volume of mixture n
	double dGi_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // Gi/R: ideal gas Gibbs free energy of mixture n
	// double dAi_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);  // Ai/R: ideal gas Helmholtz free energy of mixture n
	// double dUi_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);  // Ui/R: ideal gas internal energy of mixture n

	std::vector<double> dHi_dni(double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual std::vector<double> dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	std::vector<double> dGi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// std::vector<double> dAi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// std::vector<double> dUi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	std::vector<double> d2Hi_dTdni(double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual std::vector<double> d2Si_dTdni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	std::vector<double> d2Gi_dTdni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// // std::vector<double> d2Ai_dTdni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// // std::vector<double> d2Ui_dTdni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	// Residual properties
	virtual double Sr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double Gr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double Hr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double Ar(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double Ur(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double dSr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dGr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dHr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double dAr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double dUr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double dSr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dGr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dHr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double dAr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double dUr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual std::vector<double> dSr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual std::vector<double> dGr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual std::vector<double> dHr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual std::vector<double> dAr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) = 0;
	// virtual std::vector<double> dUr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) = 0;

	virtual double Cpr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double Cvr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) 
	{ (void) X; (void) T_; (void) n_; (void) start_idx; (void) pt; return NAN; }  // heat capacity at constant volume: Cv = (dU/dT)_V

	// "Total" properties (reference conditions arbitrary)
	double S(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double G(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double H(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// double A(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// double U(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double dS_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dG_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dH_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double dA_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// virtual double dU_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double dS_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dG_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double dH_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// double dA_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// double dU_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	std::vector<double> dS_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	std::vector<double> dG_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	std::vector<double> dH_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	// std::vector<double> dA_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) = 0;
	// std::vector<double> dU_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) = 0;

	double Cp(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // heat capacity at constant pressure: Cp = (dH/dT)_P
	double Cv(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // heat capacity at constant volume: Cv = (dU/dT)_V
	
	// Pure component properties
	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true);
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true);
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true);
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true);
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true);
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true);
	
	// Method to find if GE/TPD surface at composition is convex
	Eigen::MatrixXd calc_hessian(std::vector<double>::iterator n_it);
	bool is_convex(std::vector<double>::iterator n_it);
	bool is_convex(double p_, double T_, std::vector<double>& n_, int start_idx=0);
	double calc_condition_number(double p_, double T_, std::vector<double>& n_, int start_idx=0);

	// Translation of derivatives w.r.t. mole fractions to mole numbers
	double dxi_dnj(std::vector<double>& n_, int i, int j);
	double dxj_to_dnk(std::vector<double>& dlnphiidxj, std::vector<double>::iterator n_it, int k);
	std::vector<double> dxj_to_dnk(std::vector<double>& dlnphiidxj, std::vector<double>::iterator n_it);

	// Test derivatives of fugacity coefficients
	int dlnphi_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	virtual int properties_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_EOS_H
//--------------------------------------------------------------------------
