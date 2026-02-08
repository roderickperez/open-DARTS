//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_VDWP_KLAUDA_H
#define OPENDARTS_FLASH_EOS_VDWP_KLAUDA_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/vdwp/vdwp.hpp"

namespace klauda {
	extern std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Ps_ABC;

	extern std::unordered_map<std::string, std::vector<int>> zn, n_shells;
	extern std::unordered_map<std::string, std::unordered_map<char, std::vector<double>>> Rn;

	extern std::unordered_map<std::string, double> ai, sigma, eik;

	class Kihara : public Integral
	{
	public:
		std::string phase;

	public:
		Kihara(std::string phase_) { phase = phase_; }
		
		double f(double x, std::string component) override;
		double F(double p, double T, std::string component) override;

		double dFdP(double p, double T, std::string component) override;
		double dFdT(double p, double T, std::string component) override;
	};
}

class Klauda : public VdWP
{
public:
	Klauda(CompData& comp_data, std::string hydrate_type);
	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<Klauda>( *this ); }

	virtual void init_PT(double p_, double T_) override;

	virtual double V(double p_, double T_, std::vector<double>& n) override;
	virtual double fw(std::vector<double>& fi) override;

private:
	// Fugacity of water
	virtual double dfw_dP(std::vector<double>& dfidP) override;
	virtual double dfw_dT(std::vector<double>& dfidT) override;
	virtual std::vector<double> dfw_dxj(std::vector<double>& dfidxj) override;

	// Langmuir constant
	virtual std::vector<double> calc_Ckm() override;
	virtual std::vector<double> dCkm_dP() override;
	virtual std::vector<double> dCkm_dT() override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_VDWP_KLAUDA_H
//--------------------------------------------------------------------------

