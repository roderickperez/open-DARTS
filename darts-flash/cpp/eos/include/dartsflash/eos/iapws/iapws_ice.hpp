//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_IAPWS_IAPWS_ICE_H
#define OPENDARTS_FLASH_EOS_IAPWS_IAPWS_ICE_H
//--------------------------------------------------------------------------

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

#include "dartsflash/global/global.hpp"
#include "dartsflash/eos/eos.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"

class IAPWSIce : public EoS
{
protected:
	bool iapws_ideal;
	double g, g_p, g_T, g_pp, g_TT, g_TP;
	std::shared_ptr<IAPWS95> iapws;

public:
	IAPWSIce(CompData& comp_data, bool iapws_ideal_);

	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<IAPWSIce>( *this ); }

	// Overloaded function for calculation of P(T, V, n) and V(p, T, n)
	double P(double T_, double V_, std::vector<double>& n_, int start_idx=0, bool pt=false);
	double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);
	double dV_dP();
	double dV_dT();
	double dV_dni(int i);
	double rho(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

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

	// Ideal heat capacities, enthalpy and entropy
	virtual double cpi(double T_, int i) override;  // Cpi/R: heat capacity at constant pressure of component i
	virtual double hi(double T_, int i) override;  // Hi/R: Ideal gas enthalpy
	virtual double si(double X, double T_, int i, bool pt=true) override;  // Si/R: ideal gas entropy at constant pressure/constant volume
	virtual double dsi_dP(double X, double T_, int i, bool pt=true) override;
	virtual double dsi_dT(double X, double T_, int i, bool pt=true) override;
    virtual std::vector<double> dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;

	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true) override;

	// Consistency tests
	virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
	int pvt_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
	int references_test(double tol, bool verbose=false);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_IAPWS_IAPWS_ICE_H
//--------------------------------------------------------------------------
