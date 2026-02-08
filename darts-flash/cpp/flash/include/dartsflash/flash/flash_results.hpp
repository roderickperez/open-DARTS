//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_FLASH_FLASHRESULTS_H
#define OPENDARTS_FLASH_FLASH_FLASHRESULTS_H
//--------------------------------------------------------------------------

#include <vector>
#include <memory>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/trial_phase.hpp"

struct FlashResults
{
	FlashParams flash_params;
	StateSpecification state_spec;

	bool set_results, set_derivs;
	int np, np_tot, phase_state_id;
	double pressure, temperature, X_spec, dQ;
	bool at_transition_temperature{ false };

	// Store PT-flash results and derivatives
	std::vector<double> zi, nuj, Xij, nij;
	std::vector<double> dnudP, dnudT, dnudzk, dnudX, dxdP, dxdT, dxdzk, dxdX;
	std::vector<int> eos_idx, phase_idxs, ref_idxs;
	std::vector<EoS::RootFlag> root_type;
	std::vector<TrialPhase> stationary_points;
	Eigen::PartialPivLU<Eigen::MatrixXd> LUofA;

	int total_ssi_stability_iter, total_newton_flash_iter, total_ssi_flash_iter, total_newton_stability_iter;
	std::vector<double> norm_path, T_path;

	// Constructors
	FlashResults() {}
	FlashResults(FlashParams& flashparams, StateSpecification state_spec_);
	virtual ~FlashResults() = default;
	
	// Setters for PT flash results and derivatives
	void set_flash_results(double p_, double T_, std::vector<double>& z_, std::vector<TrialPhase>& sps, std::vector<int>& ref_idxs_);
	virtual void set_flash_derivs();

	// Getters for partial derivatives
	virtual void get_derivs(std::vector<double>& dnudP_, std::vector<double>& dnudX_, std::vector<double>& dnudzk_, 
                    		std::vector<double>& dxdP_, std::vector<double>& dxdX_, std::vector<double>& dxdzk_);

	// Phase and total properties
	double phase_prop(EoS::Property prop, int phase_idx);  // phase property
	std::vector<double> phase_prop(EoS::Property prop);  // phase properties
	double total_prop(EoS::Property prop);  // total property

	// Partial derivatives of primary flash variables: P, T, z
	void calc_matrix_and_inverse();
	void calc_dP_derivs();
	void calc_dT_derivs();
	void calc_dz_derivs();

	// Secondary flash variables and derivatives of flash with respect to other specifications: G, H, S
	double dX_dP(StateSpecification state_spec_);
	double dX_dT(StateSpecification state_spec_);
	std::vector<double> dX_dz(StateSpecification state_spec_);

	// Print results and derivatives
	virtual void print_results(bool derivs = false)
	{
		print("p, T", {pressure, temperature});
		print("nu", nuj);
		print("X", Xij, np);
		print("eos_idx", eos_idx);
		print("roots", root_type);
		if (derivs) { FlashResults::print_derivs(); }
	}
	virtual void print_derivs()
	{
		print("dnu/dP", dnudP);
		print("dnu/dT", dnudT);
		print("dnu/dzk", dnudzk);

		print("dx/dP", dxdP);
		print("dx/dT", dxdT);
		print("dx/dzk", dxdzk);
	}
};

struct PXFlashResults : public FlashResults
{
	// Store FlashResults objects to keep track of PT-flashes on both sides of the transition temperature
	std::shared_ptr<FlashResults> pt_results_a, pt_results_c;
	int phase_idx_c{ -1 };  // index of phase that is in c but not in a
	double a;  // weight of flash results a at transition temperature

	// Partial derivatives of temperature with respect to P, X, z
	double dTdP, dTdX;
	std::vector<double> dTdzk;

	// Performance data
	std::vector<double> norm_path, T_path;

	// Constructors
	PXFlashResults() : FlashResults() {}
	PXFlashResults(FlashParams& flashparams, StateSpecification state_spec_) : FlashResults(flashparams, state_spec_) {}

	// Setters for PX flash results and derivatives
	void set_flash_results(std::shared_ptr<FlashResults> results_a, double X_spec_);
	void set_flash_results(std::shared_ptr<FlashResults> results_a, std::shared_ptr<FlashResults> results_c, double X_spec_);
	virtual void set_flash_derivs() override;

	// Getters for partial derivatives of flash and temperature with respect to P, X, z
	std::shared_ptr<FlashResults>& get_pt_results(bool is_a)
	{ 
		// Get FlashResults object of one of the PT-flashes on the two sides of the transition temperature
		return (is_a) ? this->pt_results_a : this->pt_results_c; 
	}
	virtual void get_derivs(std::vector<double>& dnudP_, std::vector<double>& dnudX_, std::vector<double>& dnudzk_, 
                    		std::vector<double>& dxdP_, std::vector<double>& dxdX_, std::vector<double>& dxdzk_) override;
	void get_dT_derivs(double& dTdP_, double& dTdX_, std::vector<double>& dTdzk_);

	// Print results and derivatives
	virtual void print_results(bool derivs = false) override
	{
		print("p, T, X", {pressure, temperature, X_spec});
		print("nu", nuj);
		print("X", Xij, np);
		print("eos_idx", eos_idx);
		print("roots", root_type);
		if (derivs) { PXFlashResults::print_derivs(); }
	}
	virtual void print_derivs() override
	{
		print("dnu/dP", dnudP);
		print("dnu/dT", dnudT);
		print("dnu/dX", dnudX);
		print("dnu/dzk", dnudzk);

		print("dx/dP", dxdP);
		print("dx/dT", dxdT);
		print("dx/dX", dxdX);
		print("dx/dzk", dxdzk);
	}

};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_FLASH_FLASHRESULTS_H
//--------------------------------------------------------------------------
