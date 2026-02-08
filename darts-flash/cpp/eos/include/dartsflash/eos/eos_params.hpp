//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_EOSPARAMS_H
#define OPENDARTS_FLASH_EOS_EOSPARAMS_H
//--------------------------------------------------------------------------

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "dartsflash/global/global.hpp"
#include "dartsflash/eos/eos.hpp"

struct EoSParams
{
    EoSParams() : eos(nullptr) {}
	// EoSParams() {}
	EoSParams(EoS* eos_ptr, CompData& comp_data);

	// EoS object
	std::unique_ptr<EoS> eos;

	// Root selection and phase ordering
	std::vector<int> trial_comps = {};
	std::vector<EoS::RootFlag> root_order = {EoS::RootFlag::STABLE};
	std::vector<int> rich_phase_order = {};
	double rich_phase_composition = 0.5;
	
	// EoS related parameters
	double stability_tol{ 1e-20 };
	double stability_switch_tol = stability_tol;
	double stability_switch_diff{ 2. };  // If decrease in log(norm) between two SSI iterations is below this number (and tol < switch_tol), switch to Newton - make use of effectiveness of SSI
	double stability_loose_tol_multiplier{ 1e3 };  // Multiply actual tolerance by this number to not waste many Newton iterations close to the solution without improving the norm
	double stability_line_tol{ 1e-8 };
	int stability_max_iter{ 500 };
	int stability_line_iter{ 10 };
	int stability_loose_iter{ 3 };
	bool use_gmix{ false };

	// In-/exclude components from flash with constant salinity or (augmented) free-water flash
	int nc, ni, ns;
	std::vector<int> comp_idxs;
	void set_active_components(std::vector<int>& idxs);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_EOSPARAMS_H
//--------------------------------------------------------------------------
