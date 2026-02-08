//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_SOLID_SOLID_H
#define OPENDARTS_FLASH_EOS_SOLID_SOLID_H
//--------------------------------------------------------------------------

#include <unordered_map>
#include <string>
#include <vector>

#include "dartsflash/eos/eos.hpp"
#include "dartsflash/global/global.hpp"

namespace solid_par {
	extern double T_0, P_0;

	extern std::unordered_map<std::string, std::string> pure_comp;

    extern std::unordered_map<std::string, double> gi0, gp0, hi0, hp0, v0;

    extern std::unordered_map<std::string, std::vector<double>> cp, alpha;

	class Integral
	{
	protected:
    	double pp, TT;
		std::string phase;
    
	public:
		Integral(std::string phase_) { phase = phase_; }
	};

	class IG : public Integral
	{
	private:
		double gi0, hi0;
		std::vector<double> cpi;
	
	public:
		IG(std::string component_);

		// Jager/Ballard integrals
		double G() { return this->gi0 / (M_R * solid_par::T_0); }  // gi0/RT0: ideal gas Gibbs energy at p0 = 1 bar, T0 = 298.15 K
		double H(double T);  // Integral of H(T)/RT^2 dT from T_0 to T
		double dHdT(double T);  // Derivative of integral w.r.t. temperature
		double d2HdT2(double T);  // Second derivative of integral w.r.t. temperature

		int test_derivatives(double T, double tol, bool verbose=false);
	};

	class H : public Integral
	{
	public:
		H(std::string phase_) : Integral(phase_) {}

		double f(double T);
		double F(double T);

		double dfdT(double T);
		double dFdT(double T);
		double d2FdT2(double T);

		int test_derivatives(double T, double tol, bool verbose=false);
	};

	class V : public Integral
	{
	public:
		V(std::string phase_) : Integral(phase_) {}
		
		double f(double p, double T);
		double F(double p, double T);

		double dfdP(double p, double T);
		double dfdT(double p, double T);
		double dFdP(double p, double T);
		double dFdT(double p, double T);
		double d2FdPdT(double p, double T);
		double d2FdT2(double p, double T);

		int test_derivatives(double p, double T, double tol, bool verbose=false);
	};
}

class PureSolid : public EoS
{
protected:
	std::string phase;
	int pure_comp_idx;
	double lnfS;

public:
	PureSolid(CompData& comp_data, std::string phase);

	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<PureSolid>( *this ); }

	// Overloaded function for calculation of P(V, T, n) and V(p, T, n)
	double P(double V_, double T_, std::vector<double>& n_);
	double V(double p_, double T_, std::vector<double>& n_);
	double dV_dP(double p_, double T_, std::vector<double>& n_);
	double dV_dT(double p_, double T_, std::vector<double>& n_);
	double dV_dni(double p_, double T_, std::vector<double>& n_, int i);

	virtual void init_PT(double p_, double T_) override;
	virtual void solve_PT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_PT(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override { return EoS::solve_PT(p_, T_, n_, start_idx, second_order); }
	virtual void init_VT(double V_, double T_) override;
	virtual void solve_VT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_VT(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override { return EoS::solve_VT(V_, T_, n_, start_idx, second_order); }

	virtual double lnphii(int i) override;
	virtual std::vector<double> dlnphi_dP() override;
	virtual std::vector<double> dlnphi_dT() override;
	virtual std::vector<double> dlnphi_dn() override;
	virtual std::vector<double> d2lnphi_dPdT() override;
	virtual std::vector<double> d2lnphi_dT2() override;
	virtual std::vector<double> d2lnphi_dTdn() override;

	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true) override;

	virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
	int pvt_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);

};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_SOLID_SOLID_H
//--------------------------------------------------------------------------
