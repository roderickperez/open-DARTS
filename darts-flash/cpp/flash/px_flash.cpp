#include <iostream>
#include <vector>
#include <cassert>
#include <numeric>
#include <functional>
#include <memory>

#include "dartsflash/maths/root_finding.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/px_flash.hpp"

#include <Eigen/Dense>

PXFlash::PXFlash(FlashParams& flashparams, StateSpecification state_spec_) : Flash(flashparams)
{
	this->state_spec = state_spec_;
	this->is_a = false;

	this->at_transition_temperature = false;
	this->set_temperature = false;

	this->norm_path = {};
	this->T_path = {};
}

void PXFlash::init(double p_, double X_spec_)
{
	// Initialize PH-flash algorithm at P and Hspec
	p = p_;
	X_spec = X_spec_;
	this->X_b = NAN;
	this->X_c = NAN;
	this->T_tol = this->flash_params.pxflash_Ttol;
	this->F_tol = this->flash_params.pxflash_Ftol;

	this->pt_results_a = std::make_shared<FlashResults>(FlashResults(this->flash_params, this->state_spec));
	this->pt_results_c = std::make_shared<FlashResults>(FlashResults(this->flash_params, this->state_spec));
	this->at_transition_temperature = false;
	this->located_phase_boundary = false;

	this->norm_path.clear();
	this->T_path.clear();
	return;
}

void PXFlash::init(double p_, double X_spec_, std::vector<double>& z_)
{
	// Initialize PH-flash algorithm at P, Hspec and z
	this->init(p_, X_spec_);

    // Check if feed composition needs to be corrected for 0 values
    z.resize(flash_params.ns);
    for (int i = 0; i < flash_params.ns; i++)
    {
        z[i] = (z_[i] > flash_params.min_z) ? z_[i] : flash_params.min_z;
    }
    return;
}

int PXFlash::evaluate(double p_, double X_spec_)
{
	this->z = {1.};
	return this->evaluate(p_, X_spec_, z);
}

int PXFlash::evaluate(double p_, double X_spec_, std::vector<double>& z_, bool start_from_feed)
{
	// Initialize PX-flash
	// If start from feed, find reference composition at feed
    if (start_from_feed || ref_compositions.empty())
    {
        if (z_.size() == 1)
    	{
			PXFlash::init(p_, X_spec_);
		}
		else
		{
			PXFlash::init(p_, X_spec_, z_);
		}
	}
	
	// Set initial temperature if no guess has been provided
	if (!this->set_temperature)
	{
		this->T_a = this->flash_params.T_min;
		this->T_b = this->flash_params.T_max;
		this->T = ((this->flash_params.T_init > this->T_a) && (this->flash_params.T_init < this->T_b)) 
						? this->flash_params.T_init : (this->T_a + this->T_b) * 0.5;
	}

	// Run Brent or Brent-Newton root finding algorithm to locate T
	int output = 0;
	RootFinding root;
   	auto f = std::bind(&PXFlash::obj_fun, this, std::placeholders::_1);

	if (flash_params.pxflash_type == FlashParams::PXFlashType::BRENT)
	{
		output = root.brent(f, this->T, T_a, T_b, this->F_tol, this->T_tol);
	}
	else
	{
		auto df = std::bind(&PXFlash::gradient, this, std::placeholders::_1);
		output = root.brent_newton(f, df, this->T, T_a, T_b, this->F_tol, this->T_tol);
	}
	this->T = root.getx();
	this->set_temperature = false;

	if (output == 0)
	{
		// Brent method has converged to within specification function tolerance (|X-Xspec| < ftol)
		this->at_transition_temperature = false;
		if (flash_params.verbose)
        {
            print("PXFlash", "===============");
            print("p, X", {p, X_spec});
			print("z", z_);
            PXFlash::get_flash_results()->print_results();
        }
		return 0;
	}
	else if (output == -1)
	{
		// Brent method temperature bounds have converged to within tolerance. Could indicate transition temperature
		// T_max - T_min < eps, H-Hspec > eps
		// Indicates that solution is oscillating between two flashes
		// Find which are the phases that split in the phase transition, handled in FlashResults
		if (!this->located_phase_boundary)
		{
			if (flash_params.verbose)
			{
				print("Locating transition temperature", {T_a, T_b, T});
				print("p, X", {p, X_spec});
				print("z", z_);
			}
			return this->locate_transition_temperature();
		}
		else
		{
			if (this->pt_results_a->dQ * this->pt_results_c->dQ < 0)
			{
				// Flash specification is at transition temperature
				this->at_transition_temperature = true;
			}
			return 0;
		}
	}
   	else
   	{
		if (error == 1)
		{
			std::cout << "ERROR in PXFlash\n";
			print("Error in PT-flash", {p, T_b, X_spec});
			print("z", z_);
		}
      	else  // error == 2
		{
			T = (X_b > X_spec) ? flash_params.T_min : flash_params.T_max;
			(void) this->obj_fun(T);
			error = 0;
			if (flash_params.verbose)
			{
				std::cout << "Temperature root out of bounds\n";
        		print("p, X", {p, X_spec});
				print("z", z_);
			}
		}
      	return error;
   	}
}

int PXFlash::evaluate(double p_, double X_spec_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results)
{
	// This method takes the flash results and temperature from a previous PXFlash evaluation and initializes current Tmid with it
	if (z_.size() == 1)
    {
		PXFlash::init(p_, X_spec_);
    }
	else
	{
		PXFlash::init(p_, X_spec_, z_);
	}

	// Initialize first flash with previous flash results
	this->T = flash_results->temperature;

	// If previous flash was at transition temperature,
	if (flash_results->at_transition_temperature)
	{
		// Find which are the phases that split in the phase transition, handled in FlashResults
		bool found_a{ false }, found_c{ false };
		int iter = 0;
		while (iter < 20)
		{
			iter++;
			double dQ = this->obj_fun(T);
			if (dQ > 0.) 
			{ 
				// Temperature a located, increment to locate c
				T_a = T; T += flash_params.pxflash_Ttol; found_a = true;
				if (found_c) { break; }
			}
			else 
			{ 
				// Temperature c located, decrement to locate a
				T_b = T; T -= flash_params.pxflash_Ttol; found_c = true; 
				if (found_a) { break; }
			}
		}

		if (iter < 20)
		{
			if (flash_params.verbose)
			{
				print("Locating transition temperature", {T_a, T_b, T});
				print("p, X", {p, X_spec});
				print("z", z_);
			}
			T = (T_a + T_b) * 0.5;
			return this->locate_transition_temperature();
		}
	}

	// Set temperature to temperature of (extrapolated) previous flash results
	this->T_a = this->flash_params.T_min;
	this->T_b = this->flash_params.T_max;
	this->T = flash_results->temperature;
	this->set_temperature = true;
	
	return this->evaluate(p_, X_spec_, z_, true);
}

double PXFlash::obj_fun(double T_)
{
	// Perform PT-flash at P,T,z
	if (flash_params.verbose)
	{
		print("PT evaluation at P, T, X", {p, T_, X_spec});
		print((X_b-X_spec > 0) ? "X-Xspec > 0; evaluating at lower temperature" : "X-Xspec < 0, evaluating at higher temperature ", X_b-X_spec);
	}
	if (1==0)
	{
		std::shared_ptr<FlashResults> flash_results = Flash::extrapolate_flash_results(p, T_, z, 
																					(X_b-X_spec) > 0 ? this->pt_results_c : this->pt_results_a);
		error = (flash_params.nc == 1) ? Flash::evaluate(p, T_) : 
										 Flash::evaluate(p, T_, z, flash_results);
	}
	else
	{
		error = (flash_params.nc == 1) ? Flash::evaluate(p, T_) : Flash::evaluate(p, T_, z);
	}

    // In case of error, return 1
    if (error)
    {
		return NAN;
    }

	// Get data from PT-flash
    std::shared_ptr<FlashResults> results = Flash::get_flash_results();

    // Calculate total enthalpy/entropy of mixture
	X_b = 0.;
	for (int j: results->phase_idxs)
	{
		// Ideal and residual enthalpy/entropy
		// X = Σj Xj
		std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[results->eos_idx[j]]] };
		params->eos->set_root_flag(results->root_type[j]);
		if (state_spec == StateSpecification::ENTHALPY)
		{
			X_b += params->eos->H(p, T_, results->nij, j*flash_params.ns, true) * M_R;
		}
 		else  // ENTROPY
		{
			X_b += params->eos->S(p, T_, results->nij, j*flash_params.ns, true) * M_R;
		}
	}

	// Return dQ/dT
	double dQ = 0.;
	if (state_spec == StateSpecification::ENTHALPY)
	{
		// QH = (G-Hspec)/T
		// dQH/dT = -(H-Hspec)/T^2 
		dQ = -1./std::pow(T_, 2) * (X_b - X_spec);
	}
	else
	{
		// QS = G + T Sspec
		// dQS/dT = -S + Sspec
		dQ = -(X_b - X_spec);
	}

	// Store lower and upper bound results, depending on which one is evaluated currently
	this->is_a = (dQ > 0.);
	if (this->is_a)
	{
		this->pt_results_a = std::shared_ptr<FlashResults>(results);
		pt_results_a->X_spec = X_b;
		pt_results_a->dQ = dQ;
		this->ref_compositions_a = this->ref_compositions;
		X_a = X_b;
	}
	else
	{
		this->pt_results_c = std::shared_ptr<FlashResults>(results);
		pt_results_c->X_spec = X_b;
		pt_results_c->dQ = dQ;
		this->ref_compositions_c = this->ref_compositions;
		X_c = X_b;
	}

	if (flash_params.save_performance_data)
	{
		this->norm_path.push_back({std::fabs(dQ)});
		this->T_path.push_back({T_});	
	}
	return dQ;
}
double PXFlash::obj_fun2(double T_, std::shared_ptr<FlashResults> results)
{
	// Perform PT-flash at P,T*,z with specific FlashResults
	if (1==0)
	{
		results = Flash::extrapolate_flash_results(p, T_, z, results);
	}
	
	// Evaluate split
	this->stationary_points = results->stationary_points;
    if (results->ref_idxs.size() == 1)
    {
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[this->stationary_points[results->ref_idxs[0]].eos_name] };
        params->eos->set_root_flag(this->stationary_points[results->ref_idxs[0]].root);
    }
    else
    {
		bool set_root_flags = true;
        this->run_split(results->ref_idxs, this->z, true, set_root_flags);
    }

    // In case of error, return 1
    if (error)
    {
		return NAN;
    }

    // Calculate total enthalpy/entropy of mixture
	results->temperature = T_;
	results->X_spec = 0.;
	for (int j: results->phase_idxs)
	{
		// Ideal and residual enthalpy/entropy
		// X = Σj Xj
		std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[results->eos_idx[j]]] };
		params->eos->set_root_flag(results->root_type[j]);
		if (state_spec == StateSpecification::ENTHALPY)
		{
			results->X_spec += params->eos->H(p, T_, results->nij, j*flash_params.ns, true) * M_R;
		}
 		else  // ENTROPY
		{
			results->X_spec += params->eos->S(p, T_, results->nij, j*flash_params.ns, true) * M_R;
		}
	}

	double dQ = 0.;
	if (state_spec == StateSpecification::ENTHALPY)
	{
		// QH = (G-Hspec)/T
		// dQH/dT = -(H-Hspec)/T^2 
		dQ = -1./std::pow(T_, 2) * (results->X_spec - X_spec);
	}
	else
	{
		// QS = G + T Sspec
		// dQS/dT = -S + Sspec
		dQ = -(results->X_spec - X_spec);
	}
	return dQ;
}
double PXFlash::gradient(double T_)
{
	// Calculate gradient of specified variable with respect to temperature to apply gradient update of objective function

	// Calculate derivative of specified property (H, S) at solution with P, T(H_spec), z with respect to temperature
	double dXdT;
	if (this->is_a)
	{
		this->pt_results_a->set_flash_derivs();
		dXdT = this->pt_results_a->dX_dT(this->state_spec);
	}
	else
	{
		this->pt_results_c->set_flash_derivs();
		dXdT = this->pt_results_c->dX_dT(this->state_spec);
	}
	
	// Calculate derivative of modified objective function for specified property (H, S) with respect to temperature
	if (state_spec == StateSpecification::ENTHALPY)
	{
		return 1./std::pow(T_, 2) * (2./T_ * (this->X_b-this->X_spec) - dXdT);
	}
	else
	{
		return -dXdT;
	}
}

int PXFlash::locate_transition_temperature()
{
	// Temperature bounds have converged to within specified tolerance: check if there is indeed a transition temperature
	bool transition = true;
	this->at_transition_temperature = false;

	// Compare flash results at points a and c
	// If unequal amount of phases, there is some transition
	// If equal amount of phases, check if a and c have different phases 
	if (this->pt_results_a->np == this->pt_results_c->np)
	{
		// Phase idxs are sorted
		// Find the first phase idx that is different
		// int phase_idx_c = -1;
		auto p1 = std::mismatch(this->pt_results_a->phase_idxs.begin(), this->pt_results_a->phase_idxs.end(), this->pt_results_c->phase_idxs.begin());

		// There should be differences in phase states
		// std::mismatch iterates over phase_idxs a and phase_idxs c and returns elements at first index where they mismatch
		if (p1.first == this->pt_results_a->phase_idxs.end())
		{
			transition = false;
		}
	}

	// Locate transition temperature: G^a(T) = G^c(T)
	if (transition)
	{
		// Solve T for which G^a - G^c = 0
		double Ta = pt_results_a->temperature, Tc = pt_results_c->temperature;
		double T_trans = Flash::locate_phase_boundary(this->pt_results_a, this->pt_results_c);
		this->located_phase_boundary = true;

		// Check if transition temperature is within bounds of T_a and T_c
		if ((Ta - T_trans) * (Tc - T_trans) <= 1e-14)
		{
			// Evaluate objective function at transition temperature
			double dQ1 = this->obj_fun(T_trans);
			if (std::fabs(dQ1) < this->flash_params.pxflash_Ftol)
			{
				// dQ close enough
				return 0;
			}

			// Calculate flash at the other side of the phase transition
			double dQ2;
			if (dQ1 > 0) { dQ2 = this->obj_fun2(T_trans, this->pt_results_c); }
			else { dQ2 = this->obj_fun2(T_trans, this->pt_results_a); }
			if (std::fabs(dQ2) < this->flash_params.pxflash_Ftol)
			{
				// dQ close enough
				return 0;
			}

			// Determine if specification is at phase transition or phase transition is above or below specification
			if (dQ1 * dQ2 < 0)
			{
				// Flash specification is at transition temperature
				this->at_transition_temperature = true;
				return 0;
			}
		
			// Transition temperature located, but specification is not at transition temperature
			// Use transition temperature as min or max bound for new Brent evaluation
			if (dQ1 > 0)
			{
				// At transition temperature, dQ > 0 -> transition temperature is below solution temperature
				this->T_a = T_trans;
			}
			else
			{
				// At transition temperature, dQ < 0 -> transition temperature is above solution temperature
				this->T_b = T_trans;
			}
		}
		else
		{
			// Transition temperature located, but specification is not at transition temperature/located temperature out of bounds
			// Continue Brent evaluation
			T_a = Ta; T_b = Tc; T = (T_a + T_b) * 0.5;
		}
	}

	// If not at a transition temperature, continue Brent method to find |X-Xspec| < ftol
	if (!this->at_transition_temperature)
	{
		this->set_temperature = true;
		this->F_tol = 1e-4;
		this->T_tol = 1e-12;
		return this->evaluate(p, X_spec, z, false);
	}
	return 0;
}

std::shared_ptr<PXFlashResults> PXFlash::get_flash_results(bool derivs)
{
    // Return flash results and performance data
	std::shared_ptr<PXFlashResults> px_results = std::make_shared<PXFlashResults>(this->flash_params, this->state_spec);

	if (at_transition_temperature)
	{
		// Add FlashResults objects from PT-flashes to PXFlashResults object
		px_results->set_flash_results(this->pt_results_a, this->pt_results_c, this->X_spec);
	}
	else
	{
		// Add FlashResults object from PT-flash to PXFlashResults object
		px_results->set_flash_results((this->is_a) ? this->pt_results_a : this->pt_results_c, this->X_spec);
	}

	// If needed, get derivatives at temperature a and c
	if (derivs)
    {
    	// Solve linear system to obtain derivatives of flash for simulation
    	// Simulation requires derivatives of phase fractions and phase compositions w.r.t. primary variables X
	    // Need to apply chain rule on derivatives w.r.t. pressure, temperature and composition
		px_results->set_flash_derivs();
    }

	if (flash_params.save_performance_data)
	{
		px_results->norm_path = this->norm_path;
		px_results->T_path = this->T_path;
	}
    return px_results;
}

std::shared_ptr<PXFlashResults> PXFlash::extrapolate_flash_results(double p_, double X_spec_, std::vector<double>& z_, std::shared_ptr<PXFlashResults> flashresults)
{
	// Method to extrapolate flash results to a (nearby) state to obtain an initial guess of flash results and temperature
	std::shared_ptr<PXFlashResults> flash_results = std::make_shared<PXFlashResults>(*flashresults);
	if (!flash_results->set_results)
    {
        return flash_results;
    }

    // If derivatives have not yet been calculated, do so
    if (!flash_results->set_derivs)
    {
        flash_results->set_flash_derivs();
    }

    // Extrapolate: calculate dot product of directional derivatives of phase compositions with difference vector
    std::vector<double> dX(flash_params.ns + 2);  // difference vector of 2 + nc-1 primary variables P,T,z
    dX[0] = p_ - flashresults->pressure;
    dX[1] = X_spec_ - flashresults->X_spec;
    for (int k = 0; k < flash_params.ns; k++)
    {
        dX[2 + k] = z_[k] - flashresults->zi[k];
    }

    // Loop over all phases to extrapolate compositions
    for (int j : flash_results->phase_idxs)
    {
        for (int i = 0; i < flash_params.ns; i++)
        {
            int idx = j * flash_params.ns + i;
            
            // P and X derivatives: dx/dP * dP and dx/dX * dX
            double dXij = dX[0] * flash_results->dxdP[idx] + dX[1] * flash_results->dxdX[idx];

            // Compositional derivatives: dx/dzk * dzk
            for (int k = 0; k < flash_params.ns; k++)
            {
                int idxk = k * flash_results->np_tot * flash_params.ns + j * flash_params.ns + i;
                dXij += dX[2 + k] * flash_results->dxdzk[idxk];
            }

            flash_results->Xij[idx] += dXij;
        }
    }

	// Extrapolate temperature
	double dTdP, dTdX;
	std::vector<double> dTdzk;
	
	flash_results->get_dT_derivs(dTdP, dTdX, dTdzk);
	double dT = dX[0] * dTdP + dX[1] * dTdX;
	for (int k = 0; k < flash_params.ns; k++)
	{
		dT += dX[2 + k] * dTdzk[k];
	}
	flash_results->temperature += dT;

	flash_results->set_results = false;
    flash_results->set_derivs = false;
    return flash_results;
}
