//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_VDWP_MUNCK_H
#define OPENDARTS_FLASH_EOS_VDWP_MUNCK_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/vdwp/vdwp.hpp"

namespace munck {
	extern double R, T_0;
	
	extern std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> A_km, B_km;
	extern std::unordered_map<std::string, std::unordered_map<std::string, double>> dmu0, dH0, dV0, dCp;

	class Integral
	{
	protected:
    	double pp, TT;
		std::string phase, ref_phase;
    
	public:
		Integral(std::string phase_, std::string ref_phase_) { phase = phase_; ref_phase = ref_phase_; }
	};

	class HB : public Integral
	{
	public:
		HB(std::string phase_, std::string ref_phase_) : Integral(phase_, ref_phase_) { }

		double f(double T);
		double F(double T);

		double dFdT(double T);
		double d2FdT2(double T);

		int test_derivatives(double T, double tol);
	};

	class VB : public Integral
	{
	public:
		VB(std::string phase_, std::string ref_phase_) : Integral(phase_, ref_phase_) { }

		double f(double p, double T);
		double F(double p, double T);

		double dfdT(double p, double T);
		double dFdP(double p, double T);
		double dFdT(double p, double T);
		double d2FdPdT(double p, double T);
		double d2FdT2(double p, double T);

		int test_derivatives(double p, double T, double tol);
	};
	
} // namespace munck

class Munck : public VdWP
{
private:
	std::string ref_phase;
	double dmu, dH, dV;

public:
	Munck(CompData& comp_data, std::string hydrate_type);
	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<Munck>( *this ); }

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
#endif // OPENDARTS_FLASH_EOS_VDWP_MUNCK_H
//--------------------------------------------------------------------------
