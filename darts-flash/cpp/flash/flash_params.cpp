#include <cmath>
#include <numeric>
#include <memory>
// #include <bits/stdc++.h>
#include "dartsflash/global/timer.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/maths/maths.hpp"

FlashParams::FlashParams(CompData& compdata) : FlashParams()
{
	this->comp_data = compdata;
	this->nc = compdata.nc;
	this->ni = compdata.ni;
	this->ns = nc + ni;

	this->initial_guess = InitialGuess(compdata);
	this->eos_order = {};
	this->phase_states_map = {};
}

void FlashParams::add_eos(std::string name, EoS* eos)
{
	// Create instance of EoSParams object and add pointer to map
	eos_params.insert({name, std::make_shared<EoSParams>(EoSParams(eos, this->comp_data))});
	
	eos_order.push_back(name);
	return;
}

void FlashParams::set_eos_order(std::vector<std::string> eos_order_)
{
	// Set EoS order
	this->eos_order = eos_order_;

	// Determine maximum number of phases
    this->np_max = 0;
	for (std::string eosname: this->eos_order)
	{
		std::shared_ptr<EoSParams> params{ this->eos_params[eosname] };
		this->np_max += params->root_order.size();  // V/L
		if (params->rich_phase_order.size())
		{
			this->np_max += params->rich_phase_order.size() - 1;  // Rich liquids
		}
	}

	// Create map of unique combinations of phases
	int phase_state = 0;
	this->phase_states_str = {};
	for (int np = 1; np <= this->np_max; np++)
	{
		// Find all unique combinations of length np
		Combinations combinations(this->np_max, np);

		for (std::vector<int>& combination: combinations.combinations)
		{
			// binary_id is the binary representation of the combination of phases
			double binary_id = 0;
			for (int& i : combination)
			{
				binary_id += std::pow(2, i);
			}
			this->phase_states_map.insert({static_cast<int>(binary_id), phase_state});

			// Create string of phase state
			std::string phase_state_str = "";
			for (int j: combination)
			{
				phase_state_str += std::to_string(j) + "-";
			}
			phase_state_str.pop_back();
			this->phase_states_str.push_back(phase_state_str);

			phase_state++;
		}
	}

	return;
}
int FlashParams::get_phase_state(std::vector<int>& phase_idxs)
{
	double binary_id = 0;
	for (int& i : phase_idxs)
	{
		binary_id += static_cast<int>(std::pow(2, i));
	}
	return this->phase_states_map[static_cast<int>(binary_id)];
}
std::string FlashParams::get_phase_state(int phase_state)
{
	// Return string of phase names at phase state
	return this->phase_states_str[phase_state];
}

void FlashParams::init_eos(double p, double T)
{
	// Initialise EoS component parameters at p, T
    this->timer.start(Timer::timer::EOS);
	for (auto& it: this->eos_params) 
	{
		it.second->eos->set_root_flag(EoS::RootFlag::STABLE);
		it.second->eos->init_PT(p, T);
	}
	this->initial_guess.init(p, T);
    this->timer.stop(Timer::timer::EOS);
}

TrialPhase FlashParams::find_ref_comp(double p, double T, std::vector<double>& z)
{
    // Find reference compositions - hypothetical single phase
    double gmin = NAN;
    int ref_eos_idx = 0;
    EoS::RootFlag ref_root = EoS::RootFlag::STABLE;
	std::vector<double> gr(this->eos_order.size(), NAN);
    for (size_t j = 0; j < this->eos_order.size(); j++)
    {
		std::shared_ptr<EoSParams> eosparams{ this->eos_params[this->eos_order[j]] };
        if (eosparams->eos->eos_in_range(z.begin()))
        {
			eosparams->eos->set_root_flag(EoS::RootFlag::STABLE);
            double grj = eosparams->eos->Gr(p, T, z, 0, true);
			
            if (eosparams->eos->select_root(z.begin()) >= EoS::RootSelect::ACCEPT)  // root should be selected or not
			{
				gr[j] = grj;
				if ((std::isnan(gmin) // gmin not initialized
                    || (grj < gmin))) // Gibbs energy of EoS is lower
            	{
					bool is_below_spinodal;
					ref_root = eosparams->eos->is_root_type(is_below_spinodal);
                	ref_eos_idx = static_cast<int>(j);
                	gmin = grj;
            	}
			}
        }
    }
    TrialPhase ref_comp = TrialPhase(ref_eos_idx, this->eos_order[ref_eos_idx], z, ref_root);

    if (this->verbose)
    {
        ref_comp.print_point("Reference phase");
    }
    return ref_comp;
}

std::vector<std::string> FlashParams::find_pure_phase(double p, double T, std::vector<double>& Gpure)
{
	std::vector<std::string> eos_pure(nc);

	for (std::string eosname: eos_order)
	{
		std::shared_ptr<EoSParams> params{ this->eos_params[eosname] };
        params->eos->set_root_flag(EoS::RootFlag::STABLE);

		std::vector<double> lnphi0 = params->eos->lnphi0(p, T, true);

		for (int i = 0; i < nc; i++)
		{
			// if (!std::isnan(gpure[i]) && (std::isnan(Gpure[i]) || gpure[i] < Gpure[i]))
			if (!std::isnan(lnphi0[i]) && (std::isnan(Gpure[i]) || lnphi0[i] < Gpure[i]))
			{
				Gpure[i] = lnphi0[i];
				eos_pure[i] = eosname;
			}
		}
	}
	return eos_pure;
}

std::vector<double> FlashParams::G_pure(double p, double T)
{
	std::vector<double> Gpure(nc, NAN);
	std::vector<std::string> eos_pure = this->find_pure_phase(p, T, Gpure);

	return Gpure;
}

std::vector<double> FlashParams::H_pure(double p, double T)
{
	std::vector<double> Gpure(nc, NAN);
	std::vector<std::string> eos_pure = this->find_pure_phase(p, T, Gpure);
	std::vector<double> Hpure(nc, NAN);
	std::vector<double> n(nc, 0.);

	for (int i = 0; i < nc; i++)
	{
		std::shared_ptr<EoSParams> eosparams{ this->eos_params[eos_pure[i]] };
        eosparams->eos->set_root_flag(EoS::RootFlag::STABLE);

		n[i] = 1.;
		Hpure[i] = eosparams->eos->H(p, T, n, 0, true);

		n[i] = 0.;
	}

	return Hpure;
}

std::vector<TrialPhase> FlashParams::get_trial_comps(int eos_idx, std::string eos_name, std::vector<TrialPhase>& ref_comps)
{
	std::vector<TrialPhase> trial_comps = this->initial_guess.evaluate(eos_idx, eos_name, this->eos_params[eos_name]->trial_comps, ref_comps);

	for (TrialPhase trial_comp: trial_comps)
	{
		trial_comp.root = EoS::RootFlag::STABLE;
	}
	return trial_comps;
}

std::vector<double> FlashParams::prop_pure(EoS::Property prop, double p, double T)
{
	switch (prop)
	{
		case EoS::Property::GIBBS:
		{
			std::vector<double> Gpure = this->G_pure(p, T);

			std::vector<double> n(nc, 0.);
			for (int i = 0; i < nc; i++)
			{
				// Multiply Gr/RT by T to obtain Gr/R
				Gpure[i] *= T;

				// Add ideal contribution to Gpure
				std::shared_ptr<EoSParams> eosparams{ this->eos_params[eos_order[0]] };
				eosparams->eos->set_root_flag(EoS::RootFlag::STABLE);
				
				n[i] = 1.;
				Gpure[i] += eosparams->eos->Gi(p, T, n, 0, true);
				n[i] = 0.;
			}
			return Gpure;
		}
		default:
		// case EoS::Property::ENTHALPY:
		{
			return this->H_pure(p, T);
		}
	}
}

std::vector<double> FlashParams::prop_1p(EoS::Property prop, double p, double T, std::vector<double>& X, std::vector<int>& eos_idxs, std::vector<int>& roots)
{
	std::vector<double> result(eos_idxs.size(), NAN);
	
	int j = 0;
	for (int eos_idx: eos_idxs)
	{
		std::shared_ptr<EoSParams> eosparams{ this->eos_params[this->eos_order[eos_idx]] };
		EoS::RootFlag root = static_cast<EoS::RootFlag>(roots[j]);
		eosparams->eos->set_root_flag(root);
		switch (prop)
		{
			case EoS::Property::GIBBS:
			{
				double resj = eosparams->eos->G(p, T, X, 0, true);
				bool is_below_spinodal;
				result[j] = (eosparams->eos->is_root_type(is_below_spinodal) == root) ? resj : NAN;
				break;
			}
			case EoS::Property::ENTHALPY:
			{
				result[j] = eosparams->eos->H(p, T, X, 0, true);
				break;
			}
			default:
			{
				result[j] = eosparams->eos->S(p, T, X, 0, true);
			}
		}
		
		j++;
	}
	return result;
}

std::vector<double> FlashParams::prop_np(EoS::Property prop, double p, double T, std::vector<double>& X, std::vector<int>& eos_idxs, std::vector<int>& roots)
{
	std::vector<double> result(eos_idxs.size(), NAN);
	
	int j = 0;
	for (int eos_idx: eos_idxs)
	{
		if (X[j*ns])
		{
			std::shared_ptr<EoSParams> eosparams{ this->eos_params[this->eos_order[eos_idx]] };
			EoS::RootFlag root = static_cast<EoS::RootFlag>(roots[j]);
			eosparams->eos->set_root_flag(root);

			switch (prop)
			{
				case EoS::Property::GIBBS:
				{
					result[j] = eosparams->eos->G(p, T, X, j*ns, true);
					break;
				}
				case EoS::Property::ENTHALPY:
				{
					result[j] = eosparams->eos->H(p, T, X, j*ns, true);
					break;
				}
				default:
				{
					result[j] = eosparams->eos->S(p, T, X, j*ns, true);
				}
			}
		}
		
		j++;
	}
	return result;
}
