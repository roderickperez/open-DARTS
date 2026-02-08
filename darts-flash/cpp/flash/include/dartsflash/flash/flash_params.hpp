//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_FLASH_FLASHPARAMS_H
#define OPENDARTS_FLASH_FLASH_FLASHPARAMS_H
//--------------------------------------------------------------------------

#include <vector>
#include <string>
#include <map>
#include <memory>
#include "dartsflash/global/global.hpp"
#include "dartsflash/global/units.hpp"
#include "dartsflash/global/timer.hpp"
#include "dartsflash/eos/eos_params.hpp"
#include "dartsflash/eos/eos.hpp"
#include "dartsflash/flash/initial_guess.hpp"
#include "dartsflash/flash/trial_phase.hpp"

struct FlashParams
{
	// Timers
	Timer timer;

	// Input/output units
	Units units; // default BAR, KELVIN, M3, KJ

	// Flash-related parameters
	double min_z{ 1e-13 };
	double y_pure{ 0.9 };
	double tpd_tol{ 1e-8 };
	double tpd_1p_tol{ 1e-4 };
	double tpd_close_to_boundary{ 1e-3 };
	double comp_tol{ 1e-4 };

	double rr2_tol{ 1e-12 };
	double rrn_tol{ 1e-14 };
	double rr_loose_tol_multiplier{ 1e3 };  // Multiply actual tolerance by this number to not waste many Newton iterations close to the solution without improving the norm
	int rr_max_iter{ 100 };
	int rr_line_iter{ 10 };
	int rr_loose_iter{ 10 };

	double split_tol{ 1e-15 };
	double split_switch_tol = split_tol;
	double split_switch_diff{ 2. };  // If decrease in log(norm) between two SSI iterations is below this number (and tol < switch_tol), switch to Newton - make use of effectiveness of SSI
	double split_loose_tol_multiplier{ 1e3 };  // Multiply actual tolerance by this number to not waste many Newton iterations close to the solution without improving the norm
	double split_line_tol{ 1e-8 };
	int split_max_iter{ 500 };
	int split_line_iter{ 100 };
	int split_negative_flash_iter = split_max_iter;
	int split_loose_iter{ 10 };
	double split_negative_flash_tol{ 1e-4 };
	enum SplitVars { nik = 0, lnK, lnK_chol };
	SplitVars split_variables = SplitVars::lnK;
	bool modChol_split = true;

	enum StabilityVars { Y = 0, lnY, alpha };
	StabilityVars stability_variables = StabilityVars::alpha;
	bool modChol_stability = true;

	// Ph-flash parameters. Temperature (K)
	double T_min{100};
	double T_max{1000};
	double T_init{300};

	enum PXFlashType { BRENT = 0, BRENT_NEWTON = 1 };
	PXFlashType pxflash_type = PXFlashType::BRENT;
	double pxflash_Ftol{ 1e-6 };  // Function tolerance for specification equation in PXFlash: |X-Xspec| < Ftol
	double pxflash_Ttol{ 1e-1 };  // Temperature tolerance for switch to locate_phase_boundary in PXFlash: Tmax-Tmin < Ttol
	double phase_boundary_Gtol{ 1e-6 };  // Function tolerance for locating equal Gibbs energies: |Ga-Gb| < Gtol
	double phase_boundary_Ttol{ 1e-8 };  // Temperature tolerance for locating equal Gibbs energies: Tmax-Tmin < Ttol

	// verbose = flashpar::verbose::NONE;
	bool save_performance_data = false;
	bool verbose = false;

	// EoS-related parameters
	int nc, ni, ns;
	CompData comp_data;
	std::unordered_map<std::string, std::shared_ptr<EoSParams>> eos_params = {};
	std::vector<std::string> eos_order;
	std::string vl_eos_name;
	int light_comp_idx{ -1 };

	// Initial guesses
	InitialGuess initial_guess;
	std::vector<TrialPhase> get_trial_comps(int eos_idx, std::string eos_name, std::vector<TrialPhase>& ref_comps);

	// Constructor
	FlashParams() {}
	FlashParams(CompData& comp_data);

	// Add EoS to EoSMap
	void add_eos(std::string name, EoS* eos);
	
	// Set EoS order and store map of unique phase states
	void set_eos_order(std::vector<std::string> eos_order_);  // 
	int get_phase_state(std::vector<int>& phase_idxs);
	std::string get_phase_state(int phase_state);

	int np_max;
	std::map<int, int> phase_states_map;  // stores an id for each unique combination of phases
	std::vector<std::string> phase_states_str;  // stores strings of phase states
	
	// EoS computations
	void init_eos(double p, double T);
	TrialPhase find_ref_comp(double p, double T, std::vector<double>& z);
	std::vector<std::string> find_pure_phase(double p, double T, std::vector<double>& Gpure);
	std::vector<double> G_pure(double p, double T);
	std::vector<double> H_pure(double p, double T);

	// EoS property computations with flash results
	std::vector<double> prop_pure(EoS::Property prop, double p, double T);
	std::vector<double> prop_1p(EoS::Property prop, double p, double T, std::vector<double>& X, std::vector<int>& eos_idxs, std::vector<int>& roots);
	std::vector<double> prop_np(EoS::Property prop, double p, double T, std::vector<double>& X, std::vector<int>& eos_idxs, std::vector<int>& roots);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_FLASH_FLASHPARAMS_H
//--------------------------------------------------------------------------
