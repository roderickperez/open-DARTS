//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_HELMHOLTZ_HELMHOLTZ_H
#define OPENDARTS_FLASH_EOS_HELMHOLTZ_HELMHOLTZ_H
//--------------------------------------------------------------------------

#include <complex>
#include "dartsflash/eos/eos.hpp"
#include "dartsflash/global/global.hpp"

class HelmholtzEoS : public EoS
{
protected:
	std::unordered_map<int, std::pair<double, EoS::RootFlag>> preferred_roots = {};
	std::vector<std::complex<double>> Z_roots;
	int n_iterations;

public:
	HelmholtzEoS(CompData& comp_data);
	
	// Volume, pressure function, compressibility factor and derivatives
	virtual double V() = 0;
	virtual double dZ_dP() = 0;
	virtual double d2Z_dP2() = 0;
	virtual double P();
	virtual double dP_dV();
	virtual double dP_dT();
	virtual double dP_dni(int i);
	double dV_dni(int i);
	double dV_dT();
	virtual double d2P_dV2();

protected:
	// Zero'th, first and second order parameters for Helmholtz formulation
	virtual void zeroth_order(std::vector<double>::iterator n_it) = 0;
	virtual void zeroth_order(std::vector<double>::iterator n_it, double V_) = 0;
	virtual void zeroth_order(double V_) = 0;
	virtual void first_order(std::vector<double>::iterator n_it) = 0;
	virtual void second_order(std::vector<double>::iterator n_it) = 0;

	// Reduced Helmholtz function and derivatives
	virtual double F() = 0;
	virtual double dF_dV() = 0;
	virtual double dF_dT() = 0;
	virtual double dF_dni(int i) = 0;
    virtual double d2F_dnidnj(int i, int j) = 0;
    virtual double d2F_dTdni(int i) = 0;
    virtual double d2F_dVdni(int i) = 0;
    virtual double d2F_dTdV() = 0;
    virtual double d2F_dV2() = 0;
    virtual double d2F_dT2() = 0;
	virtual double d3F_dV3() = 0;

public:
	// Calculate compressibility factor Z
	virtual std::vector<std::complex<double>> Z() = 0;
	void set_preferred_roots(int i, double x, EoS::RootFlag root_flag_) { this->preferred_roots[i] = std::pair<double, EoS::RootFlag>{x, root_flag_}; }
	virtual EoS::RootSelect select_root(std::vector<double>::iterator n_it) override;
	virtual EoS::RootFlag is_root_type(bool& is_below_spinodal) override 
		{ return (this->root_type > EoS::RootFlag::NONE) ? this->root_type : this->identify_root(is_below_spinodal); }
	virtual EoS::RootFlag identify_root(bool& is_below_spinodal) = 0;

	// Overloaded function for calculation of P(V, T, n) and V(p, T, n)
	virtual double P(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);
	virtual double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double Z(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double rho(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	int volume_iterations(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	// Critical point calculations
	struct CriticalPoint
	{
		double Pc, Tc, Vc, Zc;
		bool is_nan{ false };

		CriticalPoint() : Pc(NAN), Tc(NAN), Vc(NAN), Zc(NAN), is_nan(true) {}
		CriticalPoint(double pc, double tc, double vc, double zc) : Pc(pc), Tc(tc), Vc(vc), Zc(zc) {}
		void print_point() { print("Pc, Tc, Vc, Zc", {Pc, Tc, Vc, Zc}); }
	};
	virtual CriticalPoint critical_point(std::vector<double>& n_) = 0;
	virtual bool is_critical(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) = 0;

	// Evaluation of EoS at (P, T, n) or (T, V, n)
	virtual void init_PT(double p_, double T_) override = 0;
	virtual void solve_PT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_PT(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override { return EoS::solve_PT(p_, T_, n_, start_idx, second_order); }
	virtual void init_VT(double V_, double T_) override = 0;
	virtual void solve_VT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_VT(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override { return EoS::solve_VT(V_, T_, n_, start_idx, second_order); }
	
	// Fugacity coefficient and derivatives
	double lnphii(int i) override;
	double dlnphii_dP(int i) override;
	double dlnphii_dT(int i) override;
	double dlnphii_dnj(int i, int j) override;
	double d2lnphii_dPdT(int i) override { return not_implemented_double(i); }
	double d2lnphii_dT2(int i) override { return not_implemented_double(i); }
	double d2lnphii_dTdnj(int i, int j) override { return not_implemented_double(i, j); }

	// Ideal gas properties
	virtual double dsi_dP(double X, double T_, int i, bool pt) override;
	virtual double dsi_dT(double X, double T_, int i, bool pt) override { return EoS::dsi_dT(X, T_, i, pt); }

	// Residual bulk properties
	virtual double Gr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double Hr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double Sr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double Ar(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // override;
	virtual double Ur(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);  // override;

	virtual double dSr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dGr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dHr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dAr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dUr_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) { return not_implemented_double(X, T_, n_, start_idx, pt); }

	virtual double dSr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dGr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dHr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dAr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) { return not_implemented_double(X, T_, n_, start_idx, pt); }
	virtual double dUr_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) { return not_implemented_double(X, T_, n_, start_idx, pt); }

	virtual std::vector<double> dSr_dni(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual std::vector<double> dGr_dni(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual std::vector<double> dHr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual std::vector<double> dAr_dni(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual std::vector<double> dUr_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double Cvr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double Cpr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;

	virtual double dS_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double dG_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double dH_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	double dA_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double dU_dP(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double dS_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double dG_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	virtual double dH_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;
	double dA_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double dU_dT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual double vs(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	virtual double JT(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	// Consistency tests
	virtual int derivatives_test(double V_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
	int lnphi_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	int pressure_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	int temperature_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	int composition_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	int pvt_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	int critical_point_test(std::vector<double>& n_, double tol, bool verbose=false);
	virtual int properties_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_HELMHOLTZ_HELMHOLTZ_H
//--------------------------------------------------------------------------
