//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_HELMHOLTZ_CUBIC_H
#define OPENDARTS_FLASH_EOS_HELMHOLTZ_CUBIC_H
//--------------------------------------------------------------------------

#include <complex>
#include <memory>
#include "dartsflash/eos/helmholtz/helmholtz.hpp"
#include "dartsflash/eos/helmholtz/mix.hpp"
#include "dartsflash/global/components.hpp"

struct CubicParams;

class CubicEoS : public HelmholtzEoS
{
protected:
	double d1, d2, Vr;
	std::shared_ptr<Mix> mix;

	// Cubic parameters
	double A, B, a0, a1, a2;
	double da0dP, d2a0dP2, da1dP, d2a1dP2, da2dP;

	// 0th order
    double g, f;
    // 1st order
    double g_V, g_B, f_V, f_B;
    std::vector<double> B_i, D_i;
    // 2nd order
    double g_VV, g_BV, g_BB, g_VVV, f_VV, f_BV, f_BB, f_VVV;
    std::vector<double> B_ij, D_ij, D_iT;
    double D_T, D_TT;

public:
	enum CubicType { PR = 0, SRK };

	CubicEoS(CompData& comp_data, CubicEoS::CubicType type, bool volume_shift = false);
	CubicEoS(CompData& comp_data, CubicParams& cubic_params);

	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<CubicEoS>( *this ); }

	// Pure component and mixture parameters
	void init_VT(double V_, double T_) override;
	void init_PT(double p_, double T_) override;
	std::vector<double> calc_coefficients(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true);

	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true) override;

	// Overloaded function for calculation of P(V, T, n) and V(p, T, n)
	virtual double P(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false) override { return HelmholtzEoS::P(V_, T_, n_, start_idx, pt); }
	virtual double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override { return HelmholtzEoS::V(p_, T_, n_, start_idx, pt); }

	// Pressure function and derivatives
	virtual std::vector<std::complex<double>> Z() override;
	virtual EoS::RootFlag identify_root(bool& is_below_spinodal) override;
	virtual HelmholtzEoS::CriticalPoint critical_point(std::vector<double>& n_) override;
	virtual bool is_critical(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override;

protected:
	// Volume
	virtual double V() override;
	virtual double dZ_dP() override;
	virtual double d2Z_dP2() override;
	double d2f_dZ2(double Z_);
	void calc_coefficients();  // A, B, a0, a1, a2, da/dP, d2a/dP2

	// Methods to calculate critical point
	double dfdZ_at_inflection_point(double p_, double T_);
	double f_at_zero_slope_inflection_point_at_T(double T_);
	// double f_at_zero_slope_inflection_point_at_p(double p_);
	
	// Reduced Helmholtz function and derivatives
	double F() override;
	double dF_dV() override;
	double dF_dT() override;
	double dF_dni(int i) override;
    double d2F_dnidnj(int i, int j) override;
    double d2F_dTdni(int i) override;
    double d2F_dVdni(int i) override;
    double d2F_dTdV() override;
    double d2F_dV2() override;
    double d2F_dT2() override;
	double d3F_dV3() override;

	// Elements of Helmholtz function
	void zeroth_order(std::vector<double>::iterator n_it) override;
	void zeroth_order(std::vector<double>::iterator n_it, double V_) override;
	void zeroth_order(double V_) override;
	void first_order(std::vector<double>::iterator n_it) override;
	void second_order(std::vector<double>::iterator n_it) override;

	double F_n(), F_T(), F_V(), F_B(), F_D();
	double F_nV(), F_nB(), F_TT(), F_BT(), F_DT(), F_BV(), F_BB(), F_DV(), F_BD(), F_TV(), F_VV(), F_VVV();

public:
	int mix_dT_test(double T_, std::vector<double>& n_, double tol);
};

struct CubicParams
{
	double d1, d2;
	std::shared_ptr<Mix> mix;

    CubicParams() {}
	CubicParams(double d1_, double d2_, double omegaA, double omegaB, std::vector<double>& kappa, CompData& comp_data, bool volume_shift = false);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_HELMHOLTZ_CUBIC_H
//--------------------------------------------------------------------------
