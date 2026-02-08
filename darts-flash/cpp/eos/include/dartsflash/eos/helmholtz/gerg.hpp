//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_HELMHOLTZ_GERG_H
#define OPENDARTS_FLASH_EOS_HELMHOLTZ_GERG_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/helmholtz/helmholtz.hpp"

/*
class GERG : public HelmholtzEoS
{
protected:

public:
	GERG(CompData& comp_data) : HelmholtzEoS(comp_data) {}

    virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<GERG>( *this ); }

	// Pure component and mixture parameters
	void component_init_PT(double p_, double T_) override { P = p_; T = T_; }
	void mixture_solve_PT(std::vector<double> n_, int order) override;

	// Pressure function and derivatives
    // Table B2
	double P() override;
    double dP_dT() override;
    double dP_dV() override;
    double dP_dni(int i) override;

    double dP_drho();
    double ndP_dV();
    double ndP_dni(int i);

    // Helmholtz function and derivatives
    // Table B4
    double dna0_dni(int i);
    double dnar_dni(int i);
    double ndar_dni(int i);
    double ndard_dni(int i);
    double ndrhor_dni(int i);
    double ndTr_dni(int i);

    // Table B5
    double ao();
    double ao_d();
    double ao_dd();
    double ao_dt();
    double ao_t();
    double ao_tt();

    double ar();
    double ar_d();
    double ar_dd();
    double ar_dt();
    double ar_t();
    double ar_tt();
    double ar_x(int i);
    double ar_xx(int i, int j);
    double ar_dx(int i);
    double ar_tx(int i);

    // Table B6
    double ao_oi(int i);
    double dao_oi_drhoci(int i);
    double d2ao_oi_drhoci2(int i);
    double d2ao_oi_drhoci_dTci(int i);
    double dao_oi_dTci(int i);
    double d2ao_oi_dTci2(int i);

    // Table B7
    double ar_oi(int i);
    double dar_oi_dd(int i);
    double d2ar_oi_dd2(int i);
    double d2ar_oi_ddt(int i);
    double dar_oi_dt(int i);
    double d2ar_oi_dt2(int i);

    // Table B8
    double ar_ij(int i, int j);
    double dar_ij_dd(int i, int j);
    double d2ar_ij_dd2(int i, int j);
    double d2ar_ij_ddt(int i, int j);
    double dar_ij_dt(int i, int j);
    double d2ar_ij_dt2(int i, int j);

    // Table B9
    double Y(char y);
    double Y_x(char y, int i);
    double Y_xx(char y, int i, int j);
    double fki_x(char y, int i, int k);
    double fik_x(char y, int i, int k);
    double fki_xx(char y, int i, int k);
    double fik_xx(char y, int i, int k);
    double fij_xx(char y, int i, int j);

};
*/

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_HELMHOLTZ_GERG_H
//--------------------------------------------------------------------------
