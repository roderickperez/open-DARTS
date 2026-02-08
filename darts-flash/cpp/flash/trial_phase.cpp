#include <iostream>
#include <numeric>
#include <algorithm>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/trial_phase.hpp"
#include "dartsflash/eos/eos.hpp"

TrialPhase::TrialPhase(int eos_idx_, std::string eos_name_, std::vector<double>& Y_, EoS::RootFlag root_)
{
	this->Y = Y_;
	this->y = Y_;
	this->eos_idx = eos_idx_;
	this->eos_name = eos_name_;
	this->tpd = 0.;
	this->gmin = 0.;
	this->nu = 0.;
	this->X = Y;
	// this->x = y;
	this->ymin = y;
	this->root = root_;
	this->is_stable_root = true;
	this->is_preferred_root = EoS::RootSelect::ACCEPT;
	this->is_in_range = true;
}

void TrialPhase::set_stationary_point(std::vector<double>::iterator Y_it, double tpd_)
{
	// Set stationary point value and tpd
	std::copy(Y_it, Y_it + this->Y.size(), this->Y.begin());
	this->tpd = tpd_;
	this->has_converged = true;

	// Calculate mole fractions
	double Y_tot_inv = 1./std::accumulate(Y.begin(), Y.end(), 0.);
	std::transform(Y.begin(), Y.end(), y.begin(), [&Y_tot_inv](double element) { return element *= Y_tot_inv; });
}

void TrialPhase::set_gmix_min(std::vector<double>::iterator X_it, double gmin_)
{
	// Set gmin composition and value
	this->gmin = gmin_;
	this->gmin_converged = true;

	// Calculate mole fractions
	double X_tot_inv = 1./std::accumulate(X_it, X_it + ymin.size(), 0.);
	std::transform(X_it, X_it + ymin.size(), ymin.begin(), [&X_tot_inv](double element) { return element *= X_tot_inv; });
}

void TrialPhase::set_equilibrium_phase(std::vector<double>::iterator X_it, double nu_)
{
	// Set phase fraction nu and phase composition X
	std::copy(X_it, X_it + this->X.size(), this->X.begin());
	this->nu = nu_;
	this->tpd = 0.;
}

void TrialPhase::print_point(std::string text)
{
	std::cout << text << ":\n";
	print("Y", this->Y);
	print("y", this->y);
	print("local minimum of Gmix", this->ymin);
	print("tpd", this->tpd);
	print("gmin", this->gmin);
	print("nu", this->nu);
	print("X", this->X);
	print("eos", this->eos_name);
	print("root", (this->root == EoS::STABLE) ? "STABLE" : ((this->root == EoS::MIN) ? "MIN" : "MAX"));
	print("has converged?", this->has_converged);
	print("within EoS-range?", this->is_in_range);
	print("is stable root?", this->is_stable_root);
	print("is preferred root?", this->is_preferred_root);
}
