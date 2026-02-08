//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_INITIALGUESS_H
#define OPENDARTS_FLASH_EOS_INITIALGUESS_H
//--------------------------------------------------------------------------

#include <map>
#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/global/units.hpp"
#include "dartsflash/flash/trial_phase.hpp"

struct InitialGuess
{
	enum Ki : int { Wilson_VL, Wilson_LV, Henry_VA, Henry_AV };
	enum Yi : int { Wilson = -8, Wilson13, Henry, sI, sII, sH, Free, Min, Pure };

	int nc, ni, ns, water_index, rich_idx;
	double p, T;
	double zero{ 1e-12 }, rich_comp{ NAN };
	std::vector<std::string> components, ions;
	std::vector<double> ypure;
	CompData comp_data;
	Units units;
	
	InitialGuess() {}
	InitialGuess(CompData& comp_data_);
	virtual ~InitialGuess() = default;

	// Initialize at p, T and define set of initial guesses
	void init(double p_, double T_);

	// evaluate() for PhaseSplit
	std::vector<double> evaluate(std::vector<int>& initial_guesses);
	// evaluate(x) for Stability
	std::vector<TrialPhase> evaluate(int eos_idx, std::string eos_name, std::vector<int>& initial_guesses, std::vector<TrialPhase>& ref_comps);

	// Correlations
	std::vector<double> k_wilson(bool inverse);
	std::vector<double> y_henry();
	std::vector<double> k_henry(bool inverse);
	std::vector<double> k_vapour_sI(bool inverse);
	std::vector<double> k_vapour_sII(bool inverse);
	std::vector<double> k_aq_ice(bool inverse);
	// std::vector<double> k_aq_salt(bool inverse);
	std::vector<double> y_pure(int pure_idx, double pure_comp = NAN);
	std::vector<double> y_min(int rich_idx_, double rich_comp_, std::vector<double>& composition);
	void set_ypure(int j, double pure) { this->ypure[j] = pure; }
	void set_ymin(int rich_idx_, double rich_comp_) { this->rich_idx = rich_idx_; this->rich_comp = rich_comp_; }
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_INITIALGUESS_H
//--------------------------------------------------------------------------
