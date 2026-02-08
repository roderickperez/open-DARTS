//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_VDWP_BALLARD_H
#define OPENDARTS_FLASH_EOS_VDWP_BALLARD_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/vdwp/vdwp.hpp"

namespace ballard {
	extern double R, T_0, P_0;
	extern std::unordered_map<std::string, double> gi0, hi0;

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
		double G() { return this->gi0 / (M_R * ballard::T_0); }  // gi0/RT0: ideal gas Gibbs energy at p0 = 1 bar, T0 = 298.15 K
		double H(double T);  // Integral of H(T)/RT^2 dT from T_0 to T
		double dHdT(double T);  // Derivative of integral w.r.t. temperature
		double d2HdT2(double T);  // Second derivative of integral w.r.t. temperature

		int test_derivatives(double T, double tol, bool verbose=false);
	};

	class HB : public Integral
	{
	public:
		HB(std::string phase_) : Integral(phase_) {}

		double f(double T);
		double F(double T);

		double dFdT(double T);
		double d2FdT2(double T);

		int test_derivatives(double T, double tol, bool verbose=false);
	};

	class VB : public Integral
	{
	public:
		VB(std::string phase_) : Integral(phase_) {}

		double f(double p, double T);
		double F(double p, double T);

		double dfdT(double p, double T);
		double dFdP(double p, double T);
		double dFdT(double p, double T);
		double d2FdPdT(double p, double T);
		double d2FdT2(double p, double T);

		int test_derivatives(double p, double T, double tol, bool verbose=false);
	};

	class VH : public Integral
	{
	private:
		std::vector<std::string> components;
		int nc, water_index;

	public:
		VH(std::string phase_, std::vector<std::string> components_);

		double f(double p, double T, std::vector<double> theta);
		/*
		double F(double p, double T, std::vector<double> theta);

		double dFdP(double p, double T, std::vector<double> theta);
		double dFdT(double p, double T, std::vector<double> theta, std::vector<double> dthetadT);
		std::vector<double> dFdxj(double p, double T, std::vector<double> theta, std::vector<double> dthetadxj);

		double dfdT(double p, double T, std::vector<double> theta, std::vector<double> dthetadT);
		std::vector<double> dfdxj(double p, double T, std::vector<double> theta, std::vector<double> dthetadxj);

		int test_derivatives(double p, double T, double tol);
		*/
	};
	
	class Kihara : public Integral
	{
	public:
		double R0, R1;
		int cage_index, R1_index;

	public:
		Kihara(double r0, double r1, std::string phase_) : Integral(phase_) { R0 = r0; R1 = r1; }

		double w(double r, std::string component);

		double f(double r, double T, std::string component);
		double F(double T, std::string component);

		double dfdT(double r, double T, std::string component);
		double d2fdT2(double r, double T, std::string component);
		double dFdT(double T, std::string component);
		double d2FdT2(double T, std::string component);

		int test_derivatives(double T, std::string component, double tol, bool verbose=false);
	};
}

class Ballard : public VdWP
{
private:
	double g_w0, g_B0, h_B0;
	double g_B, h_B;

	// member variables needed for integrals
	int R1_index; // index of innermost shell in cage, needed for integrals
	std::vector<int> zn, n_shells; // #waters in cage, #waters in shell, #shells in cage
	std::vector<double> Rn; // radius of shells, #guests per unit cell, #cages per H2O per unit cell

public:
	Ballard(CompData& comp_data, std::string hydrate_type);
	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<Ballard>( *this ); }

	virtual void init_PT(double p_, double T_) override;
	virtual void init_VT(double V_, double T_) override;

	virtual double V(double p_, double T_, std::vector<double>& n, int start_idx=0, bool pt=true) override;
	virtual double fw(std::vector<double>& fi) override;

private:
	// Fugacity of water
	virtual double dfw_dP(std::vector<double>& dfidP) override;
	virtual double dfw_dT(std::vector<double>& dfidT) override;
	virtual double d2fw_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) override;
	virtual double d2fw_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2) override;
	virtual std::vector<double> dfw_dxj(std::vector<double>& dfidxj) override;
	virtual std::vector<double> d2fw_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj) override;

	// Langmuir constant
	virtual std::vector<double> calc_Ckm() override;
	virtual std::vector<double> dCkm_dP() override;
	virtual std::vector<double> dCkm_dT() override;
	virtual std::vector<double> d2Ckm_dPdT() override;
	virtual std::vector<double> d2Ckm_dT2() override;

public:
	// Test derivatives and integrals
	virtual int derivatives_test(double p_, double T_, std::vector<double>& x_, double tol, bool verbose=false) override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_VDWP_BALLARD_H
//--------------------------------------------------------------------------
