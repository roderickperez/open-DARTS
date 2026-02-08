//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_VDWP_VDWP_H
#define OPENDARTS_FLASH_EOS_VDWP_VDWP_H
//--------------------------------------------------------------------------

#include <unordered_map>
#include <vector>
#include <string>

#include "dartsflash/eos/eos.hpp"
#include "dartsflash/global/global.hpp"

namespace vdwp {
	extern double R;
	extern std::unordered_map<std::string, int> n_cages;
	extern std::unordered_map<std::string, std::vector<double>> Nm, vm;
    extern std::unordered_map<std::string, double> nH2O, xwH_full;
	extern std::unordered_map<std::string, std::vector<int>> zm;

} // namespace vdwp

class VdWP : public EoS
{
protected:
	std::string phase;
	int water_index;
	std::vector<double> theta_km, C_km, dCkmdP, dCkmdT, d2CkmdPdT, d2CkmdT2;
	std::vector<double> f, x;
	std::vector<double>::iterator n_iterator;

	// Structure-specific parameters
	std::vector<int> zm; // #waters in cage
	std::vector<double> Nm, vm; // #guests per unit cell, #cages per H2O per unit cell
	int n_cages; // #cages in hydrate structure
	double nH2O; // #H2O molecules in hydrate structure

	// Michelsen (1990) parameters
	double eta, N0;
	std::vector<double> alpha, Nk;

public:
	VdWP(CompData& comp_data, std::string hydrate_type);

	virtual void solve_PT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_VT(std::vector<double>::iterator n_it, bool second_order=true) override;

	virtual double lnphii(int i) override;
	virtual double dlnphii_dnj(int i, int k) override;

	virtual std::vector<double> dlnphi_dP() override;
	virtual std::vector<double> dlnphi_dT() override;
	virtual std::vector<double> dlnphi_dn() override;
	virtual std::vector<double> d2lnphi_dPdT() override;
	virtual std::vector<double> d2lnphi_dT2() override;
	virtual std::vector<double> d2lnphi_dTdn() override;
	std::vector<double> dlnphi_dx();
	std::vector<double> d2lnphi_dTdx();
	
	std::vector<double> xH();
	virtual double V(double p_, double T_, std::vector<double>& n, int start_idx=0, bool pt=true) = 0;
	virtual double fw(std::vector<double>& fi) = 0;
	double fw(double p_, double T_, std::vector<double>& fi);

	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override
	{ (void) X; (void) T_; (void) pt; return std::vector<double>(nc, NAN); }
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true) override;
	
protected:
	// Fugacity of water (specific for each model)
	virtual double dfw_dP(std::vector<double>& dfidP) = 0;
	virtual double dfw_dT(std::vector<double>& dfidT) = 0;
	virtual double d2fw_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) = 0;
	virtual double d2fw_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2) = 0;
	virtual std::vector<double> dfw_dxj(std::vector<double>& dfidxj) = 0;
	virtual std::vector<double> d2fw_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj) = 0;

	// Change in chemical potential upon cage filling
	double calc_dmuH();
	double ddmuH_dP(std::vector<double>& dfidP);
	double ddmuH_dT(std::vector<double>& dfidT);
	double d2dmuH_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT);
	double d2dmuH_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2);
	std::vector<double> ddmuH_dxj(std::vector<double>& dfidxj);
	std::vector<double> d2dmuH_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj);

	// Fugacity of guest molecules (Michelsen 1990)
	std::vector<double> fi();
	std::vector<double> dfi_dP();
	std::vector<double> dfi_dT();
	std::vector<double> dfi_dxj();
	std::vector<double> d2fi_dPdT();
	std::vector<double> d2fi_dT2();
	std::vector<double> d2fi_dTdxj();

	// Langmuir constants C_km (specific for each model)
	virtual std::vector<double> calc_Ckm() = 0;
	virtual std::vector<double> dCkm_dP() = 0;
	virtual std::vector<double> dCkm_dT() = 0;
	virtual std::vector<double> d2Ckm_dPdT() = 0;
	virtual std::vector<double> d2Ckm_dT2() = 0;
	double alphai(int i);
	double dalphai_dP(int i);
	double dalphai_dT(int i);
	double d2alphai_dPdT(int i);
	double d2alphai_dT2(int i);

	// Cage occupancy theta_km (generic VdWP)
	std::vector<double> calc_theta();
	std::vector<double> dtheta_dP(std::vector<double>& dfidP);
	std::vector<double> dtheta_dT(std::vector<double>& dfidT);
	std::vector<double> dtheta_dxj(std::vector<double>& dfidxj);
	std::vector<double> d2theta_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidPdT, std::vector<double>& d2fidPdT);
	std::vector<double> d2theta_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2);
	std::vector<double> d2theta_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj);

	// Eta and N parameters for guest fugacities (generic VdWP)
	double calc_eta();
	double deta_dP();
	double deta_dT();
	double d2eta_dPdT(double detadT);
	double d2eta_dT2(double detadT);
	std::vector<double> deta_dxj();
	std::vector<double> d2eta_dTdxj(double detadT);

	double dN0dxk(int k);
	double dNjdxk(int j, int k);

public:
	// Test derivatives and integrals
	virtual int derivatives_test(double p_, double T_, std::vector<double>& x_, double tol, bool verbose=false) override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_VDWP_VDWP_H
//--------------------------------------------------------------------------
