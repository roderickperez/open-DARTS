//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_AQ_ZIABAKHSH_H
#define OPENDARTS_FLASH_EOS_AQ_ZIABAKHSH_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/aq/aq.hpp"

namespace ziabakhsh {
	extern std::vector<double> Psw;
	extern std::unordered_map<std::string, std::vector<double>> labda, ksi;
	extern std::unordered_map<std::string, double> eta, tau, beta, Gamma;
	extern double R;
}

class Ziabakhsh2012 : public AQBase
{
private:
	double m_c, m_ac;
	double V_H2O{ 18.1 }, Mw{ 18.0152 };
	double K0_H2O, lnKw;
	double rho0_H2O, drho0_H2OdP, drho0_H2OdT, d2rho0_H2OdPdT, d2rho0_H2OdT2, f0_H2O, df0_H2OdP, df0_H2OdT, d2f0_H2OdPdT, d2f0_H2OdT2;
	std::vector<double> lnk_H, labda, ksi, dmcdxj, dmacdxj;

public:
	Ziabakhsh2012(CompData& comp_data);

	virtual std::shared_ptr<AQBase> getCopy() override { return std::make_shared<Ziabakhsh2012>( *this ); }
	
	virtual void init_PT(double p_, double T_, AQEoS::CompType component) override;
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

	// virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_AQ_ZIABAKHSH_H
//--------------------------------------------------------------------------
