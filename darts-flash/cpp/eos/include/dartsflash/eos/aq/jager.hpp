//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_AQ_JAGER_H
#define OPENDARTS_FLASH_EOS_AQ_JAGER_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/ideal.hpp"
#include "dartsflash/eos/aq/aq.hpp"

namespace jager {
	extern double R, T_0, P_0, m_s0;
    extern std::unordered_map<std::string, double> gi0, hi0, gi_0, hi_0;
    
    extern std::unordered_map<std::string, double> omega, cp1, cp2;
    extern std::unordered_map<std::string, std::vector<double>> vp;
    extern std::vector<double> eps;

    extern std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Bca, Cca, Dca, B;

    extern double e, eps0;

	class Integral
	{
	protected:
    	double pp, TT;
		std::string component;
    
	public:
		Integral() { }
		Integral(std::string component_) { component = component_; }

		virtual double f(double X) { (void) X; return 0.; }

	protected:
		double simpson(double x0, double x1, int steps);
	};

	class IG : public Integral
	{
	private:
		double gi0, hi0;
		std::vector<double> cpi;
	
	public:
		IG(std::string component_);

		// Jager/Ballard integrals
		double G() { return this->gi0 / (M_R * jager::T_0); }  // gi0/RT0: ideal gas Gibbs energy at p0 = 1 bar, T0 = 298.15 K
		double H(double T);  // Integral of H(T)/RT^2 dT from T_0 to T
		double dHdT(double T);  // Derivative of integral w.r.t. temperature
		double d2HdT2(double T);  // Second derivative of integral w.r.t. temperature

		int test_derivatives(double T, double tol, bool verbose=false);
	};

	class H : public Integral
	{
	public:
		H(std::string component_) : Integral(component_) {}
		
		double f(double T) override;
		double F(double T);

		double dFdT(double T);
		double d2FdT2(double T);

		int test_derivatives(double T, double tol, bool verbose=false);
	};

	class V : public Integral
	{
	public:
		V(std::string component_) : Integral(component_) {}
		
		double f(double p) override;
		double f(double p, double T);
		double F(double p, double T);

		double dfdT(double p, double T);
		double dFdP(double p, double T);
		double dFdT(double p, double T);
		double d2FdPdT(double p, double T);
		double d2FdT2(double p, double T);

		int test_derivatives(double p, double T, double tol, bool verbose=false);
	};

	class PX : public Integral
	{
	public:
		PX() : Integral() {}
		
		double f(double p) override;
		double f(double p, double T);
		double dfdT(double p, double T);
		double F(double p, double T);
	};
}

class Jager2003 : public AQBase
{
private:
	// parameters for H2O fugacity
	std::vector<double> gi, hi, vi, gi0; // gibbs energy, enthalpy, volume, gibbs energy of ideal gas
	double I, A_DH, dA_DHdP, dA_DHdT, d2A_DHdPdT, d2A_DHdT2; // ionic contribution coefficients
	std::vector<double> dIdxj, B0, B1, dB0dP, dB0dT, B_ca, C_ca, D_ca, dBcadT, dCcadT, dDcadT, d2BcadT2, d2CcadT2, d2DcadT2;
	std::vector<double> lna, dlnadxj;

public:
	Jager2003(CompData& comp_data);

	virtual std::shared_ptr<AQBase> getCopy() override { return std::make_shared<Jager2003>( *this ); }

	virtual void init_PT(double p_, double T_, AQEoS::CompType comp_type) override;
	virtual void solve_PT(std::vector<double>& x_, bool second_order, AQEoS::CompType comp_type) override;

	virtual double lnphii(int i) override;
	virtual double dlnphii_dP(int i) override;
	virtual double dlnphii_dT(int i) override;
	virtual double dlnphii_dxj(int i, int j) override;
	virtual double d2lnphii_dPdT(int i) override;
	virtual double d2lnphii_dT2(int i) override;
	virtual std::vector<double> d2lnphii_dTdxj(int i) override;

	virtual double lnphi0(double X, double T_, bool pt=true) override;
	// virtual double dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual double dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual double d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual double d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual double d2lnphi0_dPdT(double X, double T_, bool pt=true) override;

private:
	// Activity of water component
	double lnaw();
	double dlnaw_dP();
	double dlnaw_dT();
	double d2lnaw_dPdT();
	double d2lnaw_dT2();
	std::vector<double> dlnaw_dxj();
	std::vector<double> d2lnaw_dTdxj();

	// Activity of molecular species
	double lnam(int i);
	double dlnam_dP(int i);
	double dlnam_dT(int i);
	double d2lnam_dPdT(int i);
	double d2lnam_dT2(int i);
	std::vector<double> dlnam_dxj(int i);
	std::vector<double> d2lnam_dTdxj(int i);

	// Activity of ionic species
	double lnai(int i);
	double dlnai_dP(int i);
	double dlnai_dT(int i);
	double d2lnai_dPdT(int i);
	double d2lnai_dT2(int i);
	std::vector<double> dlnai_dxj(int i);
	std::vector<double> d2lnai_dTdxj(int i);

public:
	virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_AQ_JAGER_H
//--------------------------------------------------------------------------
