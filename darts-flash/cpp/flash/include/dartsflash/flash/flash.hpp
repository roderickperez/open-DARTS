//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_FLASH_FLASH_H
#define OPENDARTS_FLASH_FLASH_FLASH_H
//--------------------------------------------------------------------------

#include <vector>
#include <memory>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/trial_phase.hpp"

class Flash
{
protected:
	StateSpecification state_spec = StateSpecification::TEMPERATURE;
    int np, ns_nonzero;
	int total_ssi_flash_iter, total_ssi_stability_iter, total_newton_flash_iter, total_newton_stability_iter;

	double p, T;
	double gibbs;
	std::vector<double> z, nu, X, lnphi0, z_mid;
	std::vector<std::string> eos;
	std::vector<TrialPhase> ref_compositions, stationary_points;
	std::vector<int> sp_idxs, ref_idxs;
	std::vector<bool> nonzero_comp;
	FlashParams flash_params;

public:
	Flash(FlashParams& flashparams);
	virtual ~Flash() = default;

	virtual int evaluate(double p_, double T_);  // single-component "flash"
	virtual int evaluate(double p_, double T_, std::vector<double>& z_, bool start_from_feed = true);  // multicomponent flash
	virtual int evaluate(double p_, double T_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results);  // Set initial guess from previous flash results
	
	double locate_phase_boundary(std::shared_ptr<FlashResults> results_a, std::shared_ptr<FlashResults> results_b);

protected:
	virtual void init(double p_, double T_);
	virtual void init(double p_, double T_, std::vector<double>& z_);
	
	int run_stability();
	int run_split(std::vector<int>& sp_idxs_, std::vector<double>& z_, bool negative_flash = false, bool set_root_flags = false);
	int run_loop();

	virtual std::vector<double> generate_lnK(std::vector<int>& sp_idxs_);
	bool compare_stationary_points(TrialPhase& stationary_point);
	void sort_stationary_points();

public:
	std::shared_ptr<FlashResults> get_flash_results(bool derivs = false);
	std::shared_ptr<FlashResults> extrapolate_flash_results(double p_, double T_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results);

    int get_flash_total_ssi_iter(){return total_ssi_flash_iter;}
    int get_flash_total_newton_iter(){return total_newton_flash_iter;}
    int get_stability_total_ssi_iter(){return total_ssi_stability_iter;}
    int get_stability_total_newton_iter(){return total_ssi_stability_iter;}

	std::vector<TrialPhase> find_stationary_points(double p_, double T_, std::vector<double>& X_);
	void identify_vl_phases();
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_FLASH_FLASH_H
//--------------------------------------------------------------------------
