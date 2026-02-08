//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_IDEAL_H
#define OPENDARTS_FLASH_EOS_IDEAL_H
//--------------------------------------------------------------------------

#include <unordered_map>
#include <vector>
#include <string>
#include "dartsflash/global/global.hpp"
#include "dartsflash/eos/eos.hpp"

class IdealGas : public EoS
{
private:
	double v, N;

public:
	IdealGas(CompData& comp_data);
	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<IdealGas>( *this ); }

	// PV = nRT
	double P(double v_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false);
	double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double rho(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	// Solve EoS
	virtual void init_PT(double p_, double T_) override;
	virtual void solve_PT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_PT(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override { return EoS::solve_PT(p_, T_, n_, start_idx, second_order); }
	virtual void init_VT(double V_, double T_) override;
	virtual void solve_VT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void solve_VT(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override { return EoS::solve_VT(V_, T_, n_, start_idx, second_order); }

	virtual double lnphii(int i) override { (void) i; return 0.; }
	virtual double dlnphii_dP(int i) override { (void) i; return 0.; }
	virtual double dlnphii_dT(int i) override { (void) i; return 0.; }
	virtual double dlnphii_dnj(int i, int j) override { (void) i; (void) j; return 0.; }
	virtual double d2lnphii_dPdT(int i) override { (void) i; return 0.; }
	virtual double d2lnphii_dT2(int i) override { (void) i; return 0.; }
	virtual double d2lnphii_dTdnj(int i, int j) override { (void) i; (void) j; return 0.; }

	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_IDEAL_H
//--------------------------------------------------------------------------
