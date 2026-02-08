#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <map>

#include "dartsflash/eos/aq/aq.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/aq/jager.hpp"

AQEoS::AQEoS(CompData& comp_data) : EoS(comp_data) 
{
	this->evaluators = {};
	this->evaluator_map = {};
	
	this->m_s.resize(comp_data.ns);
	this->comp_types.resize(comp_data.ns);
	for (int i = 0; i < ns; i++)
	{
		if (i == comp_data.water_index)
		{
			this->comp_types[i] = AQEoS::CompType::water;
		}
		else if (i < nc)
		{
			this->comp_types[i] = AQEoS::CompType::solute;
		}
		else
		{
			this->comp_types[i] = AQEoS::CompType::ion;
		}
	}

	this->eos_range[comp_data.water_index] = {0.6, 1.};
	this->stationary_point_range[comp_data.water_index] = {0.8, 1.};
	this->constant_salinity = false;
	this->multiple_minima = false;
}

AQEoS::AQEoS(CompData& comp_data, AQEoS::Model model) : AQEoS(comp_data)
{
	// Set evaluator map for each CompType to Model
	this->evaluator_map = {
		{AQEoS::CompType::water, model},
		{AQEoS::CompType::solute, model},
	};
	// If ions present, add ion to evaluator_map
	if (ni > 0)	{ this->evaluator_map.insert({AQEoS::CompType::ion, model}); }

	// Add instance of AQBase to map of evaluators
	if (model == AQEoS::Model::Ziabakhsh2012)
	{
		this->evaluators.insert({model, std::make_shared<Ziabakhsh2012>(Ziabakhsh2012(comp_data))});
	}
	else
	{
		this->evaluators.insert({model, std::make_shared<Jager2003>(Jager2003(comp_data))});
	}
}

AQEoS::AQEoS(CompData& comp_data, std::map<CompType, Model>& evaluator_map_)
	: AQEoS(comp_data)
{
	// Set evaluator_map with keys (water, solute, ion) that point to a model
	this->evaluator_map = evaluator_map_;

	// Add AQBase objects to evaluators
	for (auto& it: evaluator_map) 
	{
		AQEoS::Model model = it.second;
		if (evaluators.count(model) == 0)
		{
			if (model == AQEoS::Model::Ziabakhsh2012)
			{
				this->evaluators.insert({model, std::make_unique<Ziabakhsh2012>(Ziabakhsh2012(comp_data))});
			}
			else
			{
				this->evaluators.insert({model, std::make_unique<Jager2003>(Jager2003(comp_data))});
			}
		}
	}
}

/*
AQEoS::AQEoS(CompData& comp_data, std::map<CompType, Model>& evaluator_map_, std::map<Model, std::shared_ptr<AQBase>>& evaluators_)
	: AQEoS(comp_data)
{
	// Define evaluator_map with keys (water, solute, ion) that point to a model
	this->evaluator_map = evaluator_map_;

	// Add copies of AQBase objects to evaluators
	for (auto& it: evaluators_) 
	{
		evaluators[it.first] = it.second->getCopy();
	}
}
*/

void AQEoS::init_PT(double p_, double T_)
{
	// Evaluate water, solute and ion evaluator parameters
	if (p_ != p || T_ != T)
	{
		this->p = p_;
		this->T = T_;

		this->evaluators[this->evaluator_map[CompType::water]]->init_PT(p_, T_, CompType::water);
		this->evaluators[this->evaluator_map[CompType::solute]]->init_PT(p_, T_, CompType::solute);
		if (ni > 0)	{ this->evaluators[this->evaluator_map[CompType::ion]]->init_PT(p_, T_, CompType::ion); }
	}
}
void AQEoS::solve_PT(std::vector<double>::iterator n_it, bool second_order)
{
	// Calculate mole fractions and molality of molecular and ionic species
	std::vector<double> x(ns);
	this->n_iterator = n_it;
	this->N = std::accumulate(n_it, n_it + ns, 0.);
	double nT_inv = 1./this->N;
    std::transform(n_it, n_it + ns, x.begin(), [&nT_inv](double element) { return element *= nT_inv; });
	
	this->species_molality(x);
	this->evaluators[this->evaluator_map[CompType::water]]->solve_PT(x, second_order, CompType::water);
	this->evaluators[this->evaluator_map[CompType::solute]]->solve_PT(x, second_order, CompType::solute);
	if (ni > 0)	{ this->evaluators[this->evaluator_map[CompType::ion]]->solve_PT(x, second_order, CompType::ion); }
	return;
}

void AQEoS::init_VT(double, double)
{
	std::cout << "No implementation of volume-based calculations exists for AQEoS, aborting.\n";
	exit(1);
}
void AQEoS::solve_VT(std::vector<double>::iterator, bool)
{
	std::cout << "No implementation of volume-based calculations exists for AQEoS, aborting.\n";
	exit(1);
}

double AQEoS::lnphii(int i)
{
	// Evaluate fugacity coefficient of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	return this->evaluators[model]->lnphii(i);
}
double AQEoS::dlnphii_dP(int i)
{
	// Evaluate derivative of fugacity coefficient w.r.t. pressure of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	return this->evaluators[model]->dlnphii_dP(i);
}
double AQEoS::dlnphii_dT(int i)
{
	// Evaluate derivative of fugacity coefficient w.r.t. temperature of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	return this->evaluators[model]->dlnphii_dT(i);
}
double AQEoS::d2lnphii_dPdT(int i)
{
	// Evaluate derivative of fugacity coefficient w.r.t. temperature of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	return this->evaluators[model]->d2lnphii_dPdT(i);
}
double AQEoS::d2lnphii_dT2(int i)
{
	// Evaluate derivative of fugacity coefficient w.r.t. temperature of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	return this->evaluators[model]->d2lnphii_dT2(i);
}
double AQEoS::dlnphii_dnj(int i, int k) 
{
	// Evaluate derivative of fugacity coefficient w.r.t. n_k of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	std::vector<double> dlnphiidxj = this->evaluators[model]->dlnphii_dxj(i);

	// Translate from dxj to dnj
	return this->dxj_to_dnk(dlnphiidxj, this->n_iterator, k);
}
double AQEoS::d2lnphii_dTdnj(int i, int k) 
{
	// Evaluate derivative of fugacity coefficient w.r.t. n_k of component i
	Model model = this->evaluator_map[this->comp_types[i]];
	std::vector<double> d2lnphiidTdxj = this->evaluators[model]->d2lnphii_dTdxj(i);

	// Translate from dxj to dnj
	return this->dxj_to_dnk(d2lnphiidTdxj, this->n_iterator, k);
}

std::vector<double> AQEoS::dlnphi_dn() 
{
	// Evaluate derivative of fugacity coefficients w.r.t. composition
	for (int i = 0; i < ns; i++)
	{
		Model model = this->evaluator_map[this->comp_types[i]];
		std::vector<double> dlnphiidxj = this->evaluators[model]->dlnphii_dxj(i);

		// Translate from dxj to dnj
		for (int k = 0; k < ns; k++)
		{
			dlnphidn[i*ns + k] = this->dxj_to_dnk(dlnphiidxj, this->n_iterator, k);
		}
	}
	return dlnphidn;
}
std::vector<double> AQEoS::d2lnphi_dTdn() 
{
	// Evaluate derivative of fugacity coefficients w.r.t. composition
	for (int i = 0; i < ns; i++)
	{
		Model model = this->evaluator_map[this->comp_types[i]];
		std::vector<double> d2lnphiidTdxj = this->evaluators[model]->d2lnphii_dTdxj(i);

		// Translate from dxj to dnj
		for (int k = 0; k < ns; k++)
		{
			d2lnphidTdn[i*ns + k] = this->dxj_to_dnk(d2lnphiidTdxj, this->n_iterator, k);
		}
	}
	return d2lnphidTdn;
}

int AQEoS::derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
	// Test derivatives for AQBase classes
	int error_output = 0;

	error_output += this->evaluators[this->evaluator_map[CompType::water]]->derivatives_test(p_, T_, n_, tol, verbose);
	error_output += this->evaluators[this->evaluator_map[CompType::solute]]->derivatives_test(p_, T_, n_, tol, verbose);
	if (ni > 0) { error_output += this->evaluators[this->evaluator_map[CompType::ion]]->derivatives_test(p_, T_, n_, tol, verbose); }
	return error_output;
}

void AQEoS::species_molality(std::vector<double>& x) {
	// molality of dissolved species
	for (int ii = 0; ii < ns; ii++)
	{
		if (ii == this->compdata.water_index)
		{
			m_s[ii] = 0.;
		}
		else
		{
			m_s[ii] = 55.509 * x[ii] / x[this->compdata.water_index];
			if (ii >= nc)
			{
				if (m_s[ii] > 6.)
				{
					m_s[ii] = 6.;
				}
			}
		}
	}

	// Molality of ions
	if (this->compdata.constant_salinity)
	{
		for (int ii = 0; ii < ni; ii++)
		{
			m_s[nc + ii] = this->compdata.m_i[ii];
		}
	}

	// Set
	this->evaluators[this->evaluator_map[CompType::water]]->set_species_molality(m_s);
	this->evaluators[this->evaluator_map[CompType::solute]]->set_species_molality(m_s);
	if (ni > 0)	{ this->evaluators[this->evaluator_map[CompType::ion]]->set_species_molality(m_s); }

	return;
}

std::vector<double> AQEoS::lnphi0(double X, double T_, bool pt)
{
    // Calculate pure component Gibbs free energy
    std::vector<double> lnphi0_(nc, NAN);

	// Only H2O component is not NAN
	Model H2O_model = this->evaluator_map[CompType::water];
	this->evaluators[H2O_model]->init_PT(X, T_, CompType::water);
	lnphi0_[this->compdata.water_index] = this->evaluators[H2O_model]->lnphi0(X, T_, pt);

	return lnphi0_;
}

AQBase::AQBase(CompData& comp_data)
{
	this->compdata = comp_data;
	this->nc = comp_data.nc;
	this->ni = comp_data.ni;
	this->ns = comp_data.ns;

	this->x.resize(ns);
	this->m_s.resize(ns);
	this->dmidxj.resize(ns*ns);
	this->set_molality = false;

	this->species.resize(ns);
	std::copy(comp_data.components.begin(), comp_data.components.end(), this->species.begin());
	std::copy(comp_data.ions.begin(), comp_data.ions.end(), this->species.begin()+nc);
	
	this->charge = comp_data.charge;

	this->water_index = comp_data.water_index;
}

void AQBase::set_species_molality()
{
	for (int i = 0; i < ns; i++)
	{
		if (i != water_index)
		{
			m_s[i] = this->mi(i);
		}
		else
		{
			m_s[i] = 0.;
		}
	}
}

double AQBase::mi(int i)
{
	// Molality of species i
	return 55.509 * x[i] / x[water_index];
}

double AQBase::dmi_dxi() 
{
	// Derivative of mi(xi) w.r.t. xi
	return 55.509 / x[water_index];
}

double AQBase::dmi_dxw(int i) 
{
	// Derivative of mi(xi) w.r.t. xw
	return -55.509 * x[i] / std::pow(x[water_index], 2);
}

std::vector<double> AQBase::dlnphii_dxj(int i)
{
    std::vector<double> dlnphiidxj(ns);
    for (int j = 0; j < ns; j++)
    {
        dlnphiidxj[j] = this->dlnphii_dxj(i, j);
    }
    return dlnphiidxj;
}

int AQBase::derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose)
{
    // Default implementation of derivatives_test() tests nothing and returns 0
    (void) p_;
    (void) T_;
    (void) n_;
    (void) tol;
    (void) verbose;
    return 0;
}
