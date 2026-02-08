//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_TRIALPHASE_H
#define OPENDARTS_FLASH_EOS_TRIALPHASE_H
//--------------------------------------------------------------------------

#include <vector>
#include <string>
#include "dartsflash/eos/eos.hpp"

struct TrialPhase
{
	double tpd, gmin, nu;
	int eos_idx;
	std::string eos_name;
	std::vector<double> X, Y, y, ymin;
	EoS::RootFlag root = EoS::RootFlag::STABLE;
	bool is_stable_root = true;
	EoS::RootSelect is_preferred_root = EoS::RootSelect::ACCEPT;
	bool is_in_range = true;
	bool has_converged = true;
	bool gmin_converged = false;

	TrialPhase() {}
	TrialPhase(int eos_idx_, std::string eos_name_, std::vector<double>& Y_, EoS::RootFlag root_ = EoS::RootFlag::STABLE);

	void set_stationary_point(std::vector<double>::iterator Y_it, double tpd_);
	void set_gmix_min(std::vector<double>::iterator X_it, double gmix_);
	void set_equilibrium_phase(std::vector<double>::iterator X_it, double nu_);
	void print_point(std::string text="TrialPhase");
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_TRIALPHASE_H
//--------------------------------------------------------------------------
