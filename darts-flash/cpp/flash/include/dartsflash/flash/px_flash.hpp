//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_FLASH_PXFLASH_H
#define OPENDARTS_FLASH_FLASH_PXFLASH_H
//--------------------------------------------------------------------------

#include <vector>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/trial_phase.hpp"

class PXFlash : public Flash
{
protected:
	int error = 0;
	double X_a, X_b, X_c, X_spec;  // enthalpy/entropy at temperature b and c and specified state specification
	double T_a, T_b, T_tol, F_tol;  // a, b and c values for Brent's method root finding, T tolerance and Function tolerance for Brent loop
	bool is_a, at_transition_temperature;  // bool to keep track of last temperature (a or c) and bool to switch derivatives calculation
	bool set_temperature, located_phase_boundary;  // track if temperature guess has been set and if phase boundary has been located
	std::vector<double> norm_path, T_path;  // track solution path of outer loop: norm and T
	std::vector<TrialPhase> ref_compositions_a, ref_compositions_c;  // keep track of compositions at temperatures a and c
	std::shared_ptr<FlashResults> pt_results_a, pt_results_c;  // keep track of results at temperatures a and c

public:	
	PXFlash(FlashParams& flashparams, StateSpecification state_spec_);
	~PXFlash() = default;

	virtual int evaluate(double p_, double X_spec_) override;
	virtual int evaluate(double p_, double X_spec_, std::vector<double>& z_, bool start_from_feed = true) override;
	virtual int evaluate(double p_, double X_spec_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results) override;  // Set initial guess from previous flash results

	int evaluate_PT(double p_, double T_) { this->at_transition_temperature = false; return Flash::evaluate(p_, T_); }
	int evaluate_PT(double p_, double T_, std::vector<double>& z_) { this->at_transition_temperature = false; return Flash::evaluate(p_, T_, z_); }

protected:
	virtual void init(double p_, double X_spec_) override;
	virtual void init(double p_, double X_spec_, std::vector<double>& z_) override;
	
	double obj_fun(double T_);
	double obj_fun2(double T_, std::shared_ptr<FlashResults> flash_results);
	double gradient(double T_);
	int locate_transition_temperature();

public:
	std::shared_ptr<FlashResults> get_pt_flash_results(bool derivs = false) { return Flash::get_flash_results(derivs); }
	std::shared_ptr<PXFlashResults> get_flash_results(bool derivs = false);
	std::shared_ptr<PXFlashResults> extrapolate_flash_results(double p_, double X_spec_, std::vector<double>& z_, std::shared_ptr<PXFlashResults> flash_results);

};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_FLASH_PXFLASH_H
//--------------------------------------------------------------------------
