#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>

#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/flash.hpp"

FlashResults::FlashResults(FlashParams& flashparams, StateSpecification state_spec_) 
: flash_params(flashparams), state_spec(state_spec_), set_results(false), set_derivs(false) { }

// Setters
void FlashResults::set_flash_results(double p_, double T_, std::vector<double>& z_, std::vector<TrialPhase>& sps, std::vector<int>& ref_idxs_)
{
    // Return p and T values
    this->pressure = p_;
    this->temperature = T_;
    this->zi = z_;
    
    // Clear vectors
    this->stationary_points = sps;
    this->ref_idxs = ref_idxs_;
    this->nuj.clear();
    this->Xij.clear();
    this->nij.clear();
    this->eos_idx.clear();
    this->root_type.clear();
    this->phase_idxs.clear();

    // Phase fractions, phase compositions, eos and roots
    np = 0;
    int j = 0;
    for (std::string eos_name: flash_params.eos_order)
    {
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[eos_name] };
        for (EoS::RootFlag root: params->root_order)
        {
            // Find compositions of particular EoS and root type
            bool phase_present = false;
            std::vector<int> comp_order = {};
            for (int jj: ref_idxs)
            {
                // If eos_name and root type correspond, add jj'th comp to idxs
                bool is_below_spinodal = true;
                if (sps[jj].eos_name == eos_name && (root == EoS::RootFlag::STABLE || sps[jj].root == root ||
                                                    (sps[jj].root == EoS::RootFlag::STABLE &&
                                                    params->eos->is_root_type(sps[jj].X.begin(), is_below_spinodal) == root)))
                {
                    phase_present = true;
                    comp_order.push_back(jj);
                }
            }
            // If phase is not present, add -1 idx of length of comp order or of length 1 if comp order has not been specified
            if (!phase_present)
            {
                comp_order = std::vector<int>{-1};
            }

            // If root type is liquid and rich phase order has been specified, put rich-phases in the right order
            if (root == EoS::RootFlag::MIN && !params->rich_phase_order.empty())
            {
                // If no liquid phase is present, comp_order is set to {-1}; clear it
                if (comp_order.size() == 1 && comp_order[0] == -1)
                {
                    comp_order.clear();
                }

                // Loop over rich phases to find the right order
                size_t ii = 0;  // ii'th rich phase
                for (int i: params->rich_phase_order)
                {
                    size_t idx = (i == -1) ? ii : -1;
                    double Xi = NAN;

                    // If i'th rich phase is specified
                    if (i > -1)
                    {
                        // Find if there is a i'th-component rich phase
                        for (size_t jj = ii; jj < comp_order.size(); jj++)
                        {
                            double Xj = sps[comp_order[jj]].X[i];
                            // If Xj is rich phase, set idx and continue loop to find max
                            if (Xj >= params->rich_phase_composition && (std::isnan(Xi) || Xj > Xi))
                            {
                                idx = jj;
                                Xi = Xj;
                            }
                        }
                    }
                    // Else, if non-rich phase is specified, check if there are phases left in the set of compositions
                    else
                    {
                        if (ii < comp_order.size())
                        {
                            idx = ii;
                            Xi = 1.;
                        }
                    }

                    // If the (rich) phase exists, put it to the right location in comp_order
                    if (!std::isnan(Xi))
                    {
                        std::rotate(comp_order.begin() + ii, comp_order.begin() + idx, comp_order.end());
                    }
                    // Else, rich phase is not present, add -1 index in the right place
                    else
                    {
                        comp_order.insert(comp_order.begin() + ii, -1);
                    }

                    ii++;
                }
            }

            // Add jj compositions to the results vectors
            for (int jj: comp_order)
            {
                // If (rich) phase is present, it has been assigned a non-negative index
                if (jj >= 0)
                {
                    double nu_j = sps[jj].nu;
                    nuj.push_back(nu_j);
                    Xij.insert(Xij.end(), sps[jj].X.begin(), sps[jj].X.end());
                    nij.resize(Xij.size());
                    std::transform(Xij.end() - flash_params.ns, Xij.end(), nij.end() - flash_params.ns, 
                                        [&nu_j](double element) { return element *= nu_j; });
                    eos_idx.push_back(j);
                    root_type.push_back(root);
                    phase_idxs.push_back(static_cast<int>(nuj.size()-1));  // global index of jj'th phase type);
                    np++;
                }
                // If not present, they have been assigned a -1 index
                else
                {
                    nuj.push_back(0.);
                    std::vector<double> x(flash_params.ns, 0.);
                    Xij.insert(Xij.end(), x.begin(), x.end());
                    nij.insert(nij.end(), x.begin(), x.end());
                    eos_idx.push_back(j);
                    root_type.push_back(root);
                }
            }
        }
        j++;
    }
    this->np_tot = static_cast<int>(this->nuj.size());
    this->phase_state_id = this->flash_params.get_phase_state(this->phase_idxs);
    this->set_results = true;
}
void FlashResults::set_flash_derivs()
{
    // Set derivatives of PT-flash
    // Calculate matrix of dlnphiij/dnkp and material balance
    this->calc_matrix_and_inverse();

    // Calculate derivatives of PT-flash
    this->calc_dP_derivs();
	this->calc_dT_derivs();
	this->calc_dz_derivs();

    this->set_derivs = true;

    return;
}
void PXFlashResults::set_flash_results(std::shared_ptr<FlashResults> results_a, double X_spec_)
{
    // Set PX-flash results from PT-flash results object
    this->at_transition_temperature = false;
    this->pt_results_a = results_a;
    this->X_spec = X_spec_;

    // Copy the basic fields
    this->pressure = pt_results_a->pressure;
    this->temperature = pt_results_a->temperature;
    this->zi = pt_results_a->zi;
    this->nuj = pt_results_a->nuj;
    this->Xij = pt_results_a->Xij;
    this->nij = pt_results_a->nij;
    this->np = pt_results_a->np;
    this->np_tot = pt_results_a->np_tot;
    this->phase_idxs = pt_results_a->phase_idxs;
    this->phase_state_id = pt_results_a->phase_state_id;
    this->eos_idx = pt_results_a->eos_idx;
    this->root_type = pt_results_a->root_type;

    this->set_results = true;

    return;
}
void PXFlashResults::set_flash_results(std::shared_ptr<FlashResults> results_a, std::shared_ptr<FlashResults> results_c, double X_spec_)
{
    // Flash results at transition temperature
    this->set_flash_results(results_a, X_spec_);  // call other method to copy basic fields
    this->pt_results_c = results_c;

    // T_max - T_min < eps, H-Hspec > eps
	// Indicates that solution is oscillating between two flashes; check if there is indeed a transition temperature
	// Find which are the phases that split across the transition temperature, handled in FlashResults
    this->phase_idx_c = -1;  // differing phase index (global index)
    int distance_to_insert_phase_idx_c = 0;

    // Defensive: ensure both flashes have the same total slot count
    if (this->pt_results_a->np_tot != this->pt_results_c->np_tot)
    {
        std::cout << "Mismatch in np_tot between flashes (a vs c)\n";
        exit(1);
    }

    // Check if there is indeed a phase transition (np equal, but different idxs)
	// If equal amount of phases, check if a and c have different phases
    this->at_transition_temperature = false;  // set to true once all conditions satisfied
	if (this->pt_results_a->np == this->pt_results_c->np)
	{
		// Phase idxs are sorted
		// Find the first phase idx that is different
		auto p1 = std::mismatch(this->pt_results_a->phase_idxs.begin(), this->pt_results_a->phase_idxs.end(), this->pt_results_c->phase_idxs.begin());

		// There should be differences in phase states
		// std::mismatch iterates over phase_idxs a and phase_idxs c and returns elements at first index where they mismatch
		if (p1.first < this->pt_results_a->phase_idxs.end())
		{
            this->at_transition_temperature = true;

            // Find the second phase idx that is different
            if (*p1.first < *p1.second)
            {
                // Phase idxs at c skips one of those at a, for instance a = [0, 1, 3] and c = [0, 2, 3] -> p1 gives (1, 2)
                // Start search for phase idx at T_c that is not in the remaining phase idxs at T_a
                auto p2 = std::mismatch(p1.first + 1, this->pt_results_a->phase_idxs.end(), p1.second);

                // If a = [0, 1, 3] and c = [0, 2, 3], new search starts from third element in a and second element in c
                // p2 gives (3, 2) -> phase idx c is 2, distance at which phase idx should be inserted is 2
                phase_idx_c = *p2.second;
                distance_to_insert_phase_idx_c = std::distance(this->pt_results_c->phase_idxs.begin(), p2.second) + 1;
            }
            else
            {
                // Else, other way around: phase idxs at a skips one of those at c, for instance a = [0, 2, 3] and c = [0, 1, 3] -> p1 gives (2, 1)
                // phase idx c is 1, distance at which phase idx should be inserted is 1
                phase_idx_c = *p1.second;
                distance_to_insert_phase_idx_c = std::distance(this->pt_results_c->phase_idxs.begin(), p1.second);
            }
		}   
	}
    
    // If no transition temperature found
    if (!this->at_transition_temperature)
    {
        std::cout << "Not a transition temperature, invalid call of combine_flash_results()\n";
		print("p, X, T", {pressure, X_spec, temperature});
		print("z", zi);
        print("phase idxs a", this->pt_results_a->phase_idxs);
        print("phase idxs c", this->pt_results_c->phase_idxs);
        exit(1);
    }

	// Validate computed index before using it for pointer arithmetic
    if (phase_idx_c < 0 || phase_idx_c >= this->np_tot)
	{
		std::cout << "Phase transition temperature not found or invalid phase index\n";
		print("p, X, T", {pressure, X_spec, temperature});
		print("z", zi);
        print("phase idxs a", this->pt_results_a->phase_idxs);
        print("phase idxs c", this->pt_results_c->phase_idxs);
        exit(1);
	}

	// Find weights X of linear combination at solution of obj(T) = Xspec
	// X = nu_a * obj_a + (1-nu_a) * obj_b;
    double X_a = this->pt_results_a->X_spec;
    double X_c = this->pt_results_c->X_spec;

    double denom = (X_a - X_c);
    if (std::fabs(denom) <= 1e-14)
    {
        std::cout << "Invalid linear weights: X_a == X_c in combine_flash_results\n";
        print("Xa, Xc, Xspec", {X_a, X_c, X_spec});
        exit(1);
    }
    this->a = (X_spec - X_c) / denom;
    if (a < 0. || a > 1.)
    {
        std::cout << "Wrong flashes at T_a and T_c found, exit\n";
        print("Xa, Xc, Xspec", {X_a, X_c, X_spec});
        exit(1);
    }

    // Fill nu and X vectors
    this->nuj.resize(this->np_tot);
	for (int j = 0; j < this->np_tot; j++)
	{
        this->nuj[j] = a * this->pt_results_a->nuj[j] + (1. - a) * this->pt_results_c->nuj[j];
	}

	// Copy phase composition of phase a into correct index
    this->Xij = this->pt_results_a->Xij;

    // Bounds check to prevent invalid iterator arithmetic
    int start_idx = phase_idx_c * flash_params.ns;
    if (start_idx < 0
        || start_idx + flash_params.ns > static_cast<int>(this->pt_results_c->Xij.size())
        || start_idx + flash_params.ns > static_cast<int>(this->Xij.size()))
    {
        std::cout << "Invalid composition copy bounds in PXFlashResults::set_flash_results()\n";
        exit(1);
    }
    std::vector<double>::iterator begin = this->pt_results_c->Xij.begin() + start_idx;
    std::copy(begin, begin + flash_params.ns, this->Xij.begin() + start_idx);

    this->np = this->pt_results_a->np + 1;
    this->phase_idxs.insert(this->phase_idxs.begin() + distance_to_insert_phase_idx_c, phase_idx_c);
    this->phase_state_id = this->flash_params.get_phase_state(this->phase_idxs);

    this->set_results = true;

    return;
}
void PXFlashResults::set_flash_derivs()
{
    // Calculate derivatives of PX-flash
    
    // First, calculate derivatives of PT-flash
    this->pt_results_a->set_flash_derivs();
    
    // Set derivatives of flash at transition temperature
    if (this->at_transition_temperature)
    {
        // Calculate derivatives of PT-flash
        this->pt_results_c->set_flash_derivs();

        // Partial derivatives of transition temperature with respect to the primary variables
        // Calculate derivatives of Gibbs free energy with respect to P, T, zk
        double dGdP_a = this->pt_results_a->dX_dP(StateSpecification::TEMPERATURE);
        double dGdP_c = this->pt_results_c->dX_dP(StateSpecification::TEMPERATURE);

        double dGdT_a = this->pt_results_a->dX_dT(StateSpecification::TEMPERATURE);
        double dGdT_c = this->pt_results_c->dX_dT(StateSpecification::TEMPERATURE);
        
        std::vector<double> dGdzk_a = this->pt_results_a->dX_dz(StateSpecification::TEMPERATURE);
        std::vector<double> dGdzk_c = this->pt_results_c->dX_dz(StateSpecification::TEMPERATURE);

        // Derivative of a with respect to temperature
        double denom = 1./(this->pt_results_a->X_spec - this->pt_results_c->X_spec);
        double dXdT_a = this->pt_results_a->dX_dT(this->state_spec);
        double dXdT_c = this->pt_results_c->dX_dT(this->state_spec);
        double dadT = -denom * (a * dXdT_a + (1.-a) * dXdT_c);

        // Derivative of a with respect to pressure
        double dXdP_a = this->pt_results_a->dX_dP(this->state_spec);
        double dXdP_c = this->pt_results_c->dX_dP(this->state_spec);
        double dadP = -denom * (a * dXdP_a + (1.-a) * dXdP_c);
        this->dTdP = -(dGdP_a-dGdP_c)/(dGdT_a-dGdT_c);
        dadP += dTdP * dadT;  // da/dT dT/dP
        
        // Derivative of a with respect to enthalpy/entropy
        double dadX = denom;

        // Derivatives of a with respect to composition
        std::vector<double> dXdzk_a = this->pt_results_a->dX_dz(this->state_spec);
        std::vector<double> dXdzk_c = this->pt_results_c->dX_dz(this->state_spec);
        
        std::vector<double> dadzk(flash_params.ns);
        this->dTdzk = std::vector<double>(flash_params.ns);
        for (int k = 0; k < flash_params.ns; k++)
        {
            dadzk[k] = -denom * (a * dXdzk_a[k] + (1.-a) * dXdzk_c[k]);
            this->dTdzk[k] = -(dGdzk_a[k]-dGdzk_c[k])/(dGdT_a-dGdT_c);
            dadzk[k] += dTdzk[k] * dadT;
        }

        // Apply chain rule to linear combination of two flashes
        // For any property Y = a Y_a + (1-a) Y_b -> dY/dX = a dY_a/dX + (1-a) dY_b/dX + (Y_a - Y_b) da/dX
        this->dnudP = std::vector<double>(np_tot);
        this->dnudX = std::vector<double>(np_tot);
        this->dnudzk = std::vector<double>(np_tot * flash_params.ns);
        for (int j = 0; j < this->np_tot; j++)
	    {
            this->dnudP[j] = a * (this->pt_results_a->dnudP[j] + this->pt_results_a->dnudT[j] * dTdP) 
                            + (1. - a) * (this->pt_results_c->dnudP[j] + this->pt_results_c->dnudT[j] * dTdP)
                            + (this->pt_results_a->nuj[j]-this->pt_results_c->nuj[j]) * dadP;
            this->dnudX[j] = (this->pt_results_a->nuj[j]-this->pt_results_c->nuj[j]) * dadX;
            for (int k = 0; k < flash_params.ns; k++)
            {
                this->dnudzk[k * np_tot + j] = a * (this->pt_results_a->dnudzk[k * np_tot + j] + this->pt_results_a->dnudT[j] * dTdzk[k])
                                            + (1. - a) * (this->pt_results_c->dnudzk[k * np_tot + j] + this->pt_results_c->dnudT[j] * dTdzk[k])
                                            + (this->pt_results_a->nuj[j]-this->pt_results_c->nuj[j]) * dadzk[k];
            }
	    }

        // Copy derivatives only if the differing phase comes from c
        int start_idx = this->phase_idx_c * flash_params.ns;

        this->dxdT = this->pt_results_a->dxdT;
        auto begin = this->pt_results_c->dxdT.begin() + start_idx;
        std::copy(begin, begin + flash_params.ns, this->dxdT.begin() + start_idx);

        this->dxdP = this->pt_results_a->dxdP;
        begin = this->pt_results_c->dxdP.begin() + start_idx;
        std::copy(begin, begin + flash_params.ns, this->dxdP.begin() + start_idx);
        
        for (int j = 0; j < np_tot; j++)
        {
            for (int k = 0; k < flash_params.ns; k++)
            {
                int idx = j * flash_params.ns + k;
                this->dxdP[idx] += this->dxdT[idx] * this->dTdP;
            }
        }

        this->dxdX = std::vector<double>(np_tot * flash_params.ns, 0.);
        
        this->dxdzk = this->pt_results_a->dxdzk;
        for (int k = 0; k < flash_params.ns; k++)
        {
            int start_idxj = k * np_tot * flash_params.ns;
            start_idx = start_idxj + phase_idx_c * flash_params.ns;
            
            begin = this->pt_results_c->dxdzk.begin() + start_idx;
            std::copy(begin, begin + flash_params.ns, this->pt_results_a->dxdzk.begin() + start_idx);
            
            for (int j = 0; j < np_tot; j++)
            {
                int start_idxi = start_idxj + j*flash_params.ns;
                for (int i = 0; i < flash_params.ns; i++)
                {
                    this->dxdzk[start_idxi + i] += this->pt_results_a->dxdT[j*flash_params.ns + i] * dTdzk[k];
                }
            }
        }
    //     std::vector<double> nuj_X(np_tot, 0.);
    //     for (int j = 0; j < this->np_tot; j++)
    //     {
    //         nuj_X[j] = a * this->nuj[j] + (1. - a) * this->pt_results_c->nuj[j];
    //     }

    //     // Copy phase composition of phase a into correct index
    //     std::vector<double> Xij_X = this->Xij;
    //     int start_idx = phase_idx_c * flash_params.ns;
    //     // Bounds check to prevent invalid iterator arithmetic
    //     if (start_idx < 0
    //         || start_idx + flash_params.ns > static_cast<int>(this->pt_results_c->Xij.size())
    //         || start_idx + flash_params.ns > static_cast<int>(Xij_X.size()))
    //     {
    //         std::cout << "Invalid composition copy bounds in combine_flash_results\n";
    //         exit(1);
    //     }
    //     std::vector<double>::iterator begin = this->pt_results_c->Xij.begin() + start_idx;
    //     std::copy(begin, begin + flash_params.ns, Xij_X.begin() + start_idx);

    //     // Apply chain rule to linear combination of two flashes
    //     // For any property Y = a Y_a + (1-a) Y_b -> dY/dX = a dY_a/dX + (1-a) dY_b/dX + (Y_a - Y_b) da/dX
    //     this->dnudX = std::vector<double>(np_tot);
    //     for (int j = 0; j < this->np_tot; j++)
	//     {
    //         this->dnudP[j] = a * (this->dnudP_X[j] + this->dnudT[j] * dTdP) 
    //                         + (1. - a) * (this->pt_results_c->dnudP_X[j] + this->pt_results_c->dnudT[j] * dTdP)
    //                         + (this->nuj[j]-this->pt_results_c->nuj[j]) * dadP;
    //         this->dnudX[j] = (this->nuj[j]-this->pt_results_c->nuj[j]) * dadX;
    //         for (int k = 0; k < flash_params.ns; k++)
    //         {
    //             this->dnudzk[k * np_tot + j] = a * (this->dnudzk[k * np_tot + j] + this->dnudT[j] * dTdzk[k])
    //                                         + (1. - a) * (this->pt_results_c->dnudzk[k * np_tot + j] + this->pt_results_c->dnudT[j] * dTdzk[k])
    //                                         + (this->nuj[j]-this->pt_results_c->nuj[j]) * dadzk[k];
    //         }
	//     }

    //     // Copy derivatives only if the differing phase comes from c
    //     this->dxdX = std::vector<double>(np_tot * flash_params.ns, 0.);
    //     for (int j = 0; j < this->np_tot; j++)
	//     {
    //         for (int i = 0; i < flash_params.ns; i++)
    //         {
    //             int idx = j * flash_params.ns + i;
    //             this->dxdP[idx] = ((dadP * (this->nuj[j] * this->Xij[idx] - this->pt_results_c->nuj[j] * this->pt_results_c->Xij[idx]) 
    //                             + a * (this->dnudP_X[j] * this->Xij[idx] + this->nuj[j] * this->dxdP_X[idx]) 
    //                             + (1.-a) * (this->pt_results_c->dnudP_X[j] * this->pt_results_c->Xij[idx] + this->pt_results_c->nuj[j] * this->pt_results_c->dxdP_X[idx])) 
    //                             - this->dnudP[j] * Xij_X[idx]) / nuj_X[j];
    //             // this->dxdP[idx] = a * (this->dxdP[idx] + this->dxdT[idx] * dTdP) 
    //             //                 + (1. - a) * (this->pt_results_c->dxdP[idx] + this->pt_results_c->dxdT[idx] * dTdP);
    //             //                 + (this->Xij[idx]-this->pt_results_c->Xij[idx]) * dadP;
    //             // this->dxdP[idx] = a * this->dxdP[idx]
    //             //                 + (1. - a) * this->pt_results_c->dxdP[idx];
    //                             // + (this->Xij[idx]-this->pt_results_c->Xij[idx]) * dadP;
    //             // this->dxdX[j] = (this->nuj[j]-this->pt_results_c->nuj[j]) * dadX;
    //             // for (int k = 0; k < flash_params.ns; k++)
    //             // {
    //             //     this->dxdzk[k * np_tot + j] = a * (this->dxdzk[k * np_tot + j] + this->dxdT[j] * dTdzk[k])
    //             //                                 + (1. - a) * (this->pt_results_c->dxdzk[k * np_tot + j] + this->pt_results_c->dxdT[j] * dTdzk[k])
    //             //                                 + (this->nuj[j]-this->pt_results_c->nuj[j]) * dadzk[k];
    //             // }
    //         }
	//     }

    //     // int start_idx = phase_idx_c * flash_params.ns;
    //     // std::vector<double>::iterator begin = this->pt_results_c->dxdT.begin() + start_idx;
    //     // std::copy(begin, begin + flash_params.ns, this->dxdT.begin() + start_idx);

    //     // begin = this->pt_results_c->dxdP.begin() + start_idx;
    //     // std::copy(begin, begin + flash_params.ns, this->dxdP.begin() + start_idx);
        
    //     // for (int j = 0; j < np_tot; j++)
    //     // {
    //     //     for (int k = 0; k < flash_params.ns; k++)
    //     //     {
    //     //         int idx = j * flash_params.ns + k;
    //     //         dxdP[idx] += dxdT[idx] * this->dTdP;
    //     //     }
    //     // }
        
    //     // for (int k = 0; k < flash_params.ns; k++)
    //     // {
    //     //     int start_idxj = k * np_tot * flash_params.ns;
    //     //     start_idx = start_idxj + phase_idx_c * flash_params.ns;
            
    //     //     begin = this->pt_results_c->dxdzk.begin() + start_idx;
    //     //     std::copy(begin, begin + flash_params.ns, this->dxdzk.begin() + start_idx);
            
    //     //     for (int j = 0; j < np_tot; j++)
    //     //     {
    //     //         int start_idxi = start_idxj + j*flash_params.ns;
    //     //         for (int i = 0; i < flash_params.ns; i++)
    //     //         {
    //     //             dxdzk[start_idxi + i] += dxdT[j*flash_params.ns + i] * dTdzk[k];
    //     //         }
    //     //     }
    //     // }
    // }
    
    // for (int j = 0; j < this->np_tot; j++)
    // {
    //     nuj[j] = a * this->nuj[j] + (1. - a) * this->pt_results_c->nuj[j];
    // }

    // // Copy phase composition of phase a into correct index
    // int start_idx = phase_idx_c * flash_params.ns;
    // // Bounds check to prevent invalid iterator arithmetic
    // if (start_idx < 0
    //     || start_idx + flash_params.ns > static_cast<int>(this->pt_results_c->Xij.size())
    //     || start_idx + flash_params.ns > static_cast<int>(Xij.size()))
    // {
    //     std::cout << "Invalid composition copy bounds in combine_flash_results\n";
    //     exit(1);
    // }
    // std::vector<double>::iterator begin = this->pt_results_c->Xij.begin() + start_idx;
    // std::copy(begin, begin + flash_params.ns, Xij.begin() + start_idx);
    }
    // Away from transition temperature
    else
    {   
        // Calculate derivatives of specification X at current T with respect to P, T and zk for temperature derivative
        double dXdP = this->pt_results_a->dX_dP(this->state_spec);
        double dXdT = this->pt_results_a->dX_dT(this->state_spec);
        std::vector<double> dXdzk = this->pt_results_a->dX_dz(this->state_spec);

        // Calculate derivative of temperature w.r.t. P, X, zk
        this->dTdP = -dXdP/dXdT;
        this->dTdX = 1./dXdT;
        this->dTdzk = std::vector<double>(flash_params.ns);
        for (int i = 0; i < flash_params.ns; i++)
        {
            this->dTdzk[i] = -dXdzk[i]/dXdT;
        }

        // If PH- or PS-flash, use chain rule to calculate derivatives
        this->dnudP.resize(np_tot);
        this->dnudX.resize(np_tot);
        this->dnudzk.resize(np_tot * flash_params.ns);
        this->dxdP.resize(np_tot * flash_params.ns);
        this->dxdX.resize(np_tot * flash_params.ns);
        this->dxdzk.resize(np_tot * flash_params.ns * flash_params.ns);

        for (int j: phase_idxs)
        {
            // Derivatives of flash with respect to pressure at constant second state specifications: (dnik/dP)_X,n with X = {T, H, S}
            // Enthalpy or entropy: (dnu/dP)_X,n = (dnu/dP)_T,n + (dnu/dT)_P,n (dT/dP)_X,n
            //                      dx/dX = (dx/dP)_T,n + (dx/dT)_P,n (dT/dP)_X,n
            //                      (dT/dP)_X,n = - (dX/dP)_T,n / (dX/dT)_P,n
            this->dnudP[j] = this->pt_results_a->dnudP[j] + this->pt_results_a->dnudT[j] * this->dTdP;

            // Derivatives of flash with respect to other state specifications: H, S
            // Enthalpy or entropy: dnu/dX = dnu/dT dT/dX
            //                      dx/dX = dx/dT dT/dX
            this->dnudX[j] = this->pt_results_a->dnudT[j] * this->dTdX;
            
            int start_idx = j*flash_params.ns;
            for (int i = 0; i < flash_params.nc; i++)
            {
                dxdP[start_idx + i] = this->pt_results_a->dxdP[start_idx + i] + this->pt_results_a->dxdT[start_idx + i] * this->dTdP;
                dxdX[start_idx + i] = this->pt_results_a->dxdT[start_idx + i] * this->dTdX;
            }
        }

        // Derivatives of flash with respect to composition at constant state specifications: (dnij/dzk)_P,X with X = {T, H, S}
        // Enthalpy or entropy: (dnu/dP)_X,n = (dnu/dP)_T,n + (dnu/dT)_P,n (dT/dP)_X,n
        //                      dx/dX = (dx/dP)_T,n + (dx/dT)_P,n (dT/dP)_X,n
        //                      (dT/dP)_X,n = - (dX/dP)_T,n / (dX/dT)_P,n
        for (int k = 0; k < flash_params.nc; k++)
        {
            int start_idxj = k * np_tot;
            for (int j: phase_idxs)
            {
                this->dnudzk[start_idxj + j] = this->pt_results_a->dnudzk[start_idxj + j] + this->pt_results_a->dnudT[j] * this->dTdzk[k];
                
                int start_idxi = start_idxj*flash_params.nc + j*flash_params.nc;
                for (int i = 0; i < flash_params.nc; i++)
                {
                    this->dxdzk[start_idxi + i] += this->pt_results_a->dxdzk[start_idxi + i] + this->pt_results_a->dxdT[j*flash_params.nc + i] * this->dTdzk[k];
                }
            }
        }
    }
    
    this->set_derivs = true;
    return;
}

// Getters
void FlashResults::get_derivs(std::vector<double>& dnudP_, std::vector<double>& dnudT_, std::vector<double>& dnudzk_, 
                              std::vector<double>& dxdP_, std::vector<double>& dxdT_, std::vector<double>& dxdzk_)
{
    // Get partial derivatives of PT-flash
    dnudP_ = this->dnudP;
    dnudT_ = this->dnudT;
    dnudzk_ = this->dnudzk;
    dxdP_ = this->dxdP;
    dxdT_ = this->dxdT;
    dxdzk_ = this->dxdzk;
}
void PXFlashResults::get_derivs(std::vector<double>& dnudP_, std::vector<double>& dnudX_, std::vector<double>& dnudzk_, 
                                std::vector<double>& dxdP_, std::vector<double>& dxdX_, std::vector<double>& dxdzk_)
{
    // Get partial derivatives of PX-flash with respect to primary variables P,X,z
    dnudP_ = this->dnudP;
    dnudX_ = this->dnudX;
    dnudzk_ = this->dnudzk;
    dxdP_ = this->dxdP;
    dxdX_ = this->dxdX;
    dxdzk_ = this->dxdzk;
}
void PXFlashResults::get_dT_derivs(double& dTdP_, double& dTdX_, std::vector<double>& dTdzk_)
{
    // Get partial derivatives of temperature with respect to primary variables P,X,z
    dTdP_ = this->dTdP;
    dTdX_ = this->at_transition_temperature ? 0. : this->dTdX;
    dTdzk_ = this->dTdzk;
}

double FlashResults::phase_prop(EoS::Property prop, int phase_idx)
{
    // Calculate phase molar property
    std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idx]]] };
    params->eos->set_root_flag(root_type[phase_idx]);
    if (prop == EoS::Property::ENTROPY)
    {
        return params->eos->S(this->pressure, this->temperature, this->Xij, phase_idx * flash_params.ns, true) * M_R;
    }
    else if (prop == EoS::Property::GIBBS)
    {
        return params->eos->G(this->pressure, this->temperature, this->Xij, phase_idx * flash_params.ns, true) * M_R;
    }
    else // if (prop == EoS::Property::ENTHALPY)
    {
        return params->eos->H(this->pressure, this->temperature, this->Xij, phase_idx * flash_params.ns, true) * M_R;
    }
}
std::vector<double> FlashResults::phase_prop(EoS::Property prop)
{
    // Calculate phase molar property for all phases
    std::vector<double> result(np_tot, NAN);
    for (int j = 0; j < np_tot; j++)
    {
        result[j] = this->phase_prop(prop, j);
    }
    return result;
}
double FlashResults::total_prop(EoS::Property prop)
{
    // Calculate total molar property
    double result = 0.;
    for (int j: phase_idxs)
    {
        result += this->nuj[j] * this->phase_prop(prop, j);
    }
    return result;
}

void FlashResults::calc_matrix_and_inverse()
{
    // Calculate generic matrix and inverse to calculate derivatives of flash equations w.r.t. state variables and composition

    // Mass balance equations: 1 - Σj nu_j = 0 [1]; lnfi0 - lnfij = 0 [(NP-1)*NC]; zi - Σj xij nu_j = 0 [NC], Σi (xi0-xij) = 0 [NP-1] -> total [(NC+1)*NP]
    // Unknowns: dnuj/dX [NP], dxij/dX [NP*NC], dnuj/dzk [NP * NC], dxij/dzk [(NP*NC) * NC]
    int nc = flash_params.nc;
    int n_eq = np * (nc+1);
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_eq, n_eq);

    // Solve reference phase fugacity coefficients
    std::vector<std::vector<double>> dlnphiijdn(this->np);
    for (int j = 0; j < this->np; j++)
    {
        // Solve phase j fugacity coefficients
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idxs[j]]]] };
        params->eos->set_root_flag(root_type[phase_idxs[j]]);
        params->eos->solve_PT(pressure, temperature, nij, phase_idxs[j]*nc, true);
        dlnphiijdn[j] = params->eos->dlnphi_dn();
    }

    // Construct A and B

    // [1 eq] [0] 1 - Σj nu_j = 0
    // d/dX: dnu0/dX + Σj dnuj/dX = 0
    for (int j = 0; j < this->np; j++)
    {
        A(0, j) += 1.;
    }

    // [NP-1 eq] [1, NP] Σi (xi0-xij) = 0
    // Σi (dx_i0/dX - dx_ij/dX) = 0
    for (int j = 1; j < this->np; j++)
    {
        for (int i = 0; i < nc; i++)
        {
            A(j, np + i) += 1.;
            A(j, np + j*nc + i) -= 1.;
        }
    }

    // [NC eq] [NP, NP+NC] zi - Σj nu_j xij = 0
    // Σj x_ij dnu_j/dzk + nu_j dx_ij/dzk = dzi/dX {1-zi if i == k else -zi}
    for (int i = 0; i < nc; i++)
    {
        int idxi = np + i;
        for (int j = 0; j < this->np; j++)
        {
            // nu_j contribution: Σj x_ij
            A(idxi, j) += Xij[phase_idxs[j]*nc + i];

            // xij contribution: Σj nu_j
            A(idxi, np + j*nc + i) += nuj[phase_idxs[j]];
        }
    }

    // [(NP-1)*NC eq] [NP+NC, (NC+1)*NP] lnfi0 - lnfij = 0
    // dlnphi_i0/dX + Σk dlnphi_i0/dn_k0 dn_k0/dX + 1/xi0 dx_i0/dX - dlnphi_ij/dX - Σk dlnphi_ij/dn_kj dn_kj/dX - 1/xij dx_ij/dX  = 0
    // dn_ij/dX = dnu_j/dX x_ij + nu_j dx_ij/dX
    for (int j = 1; j < this->np; j++)
    {
        for (int i = 0; i < nc; i++)
        {
            int idxi = np + j*nc + i;

            // dnuj/dX contribution: Σk dlnphi_i0/dn_k0 x_k0 dnu_0/dX - Σk dlnphi_ij/dn_kj x_kj dnu_j/dX
            for (int k = 0; k < nc; k++)
            {
                A(idxi, 0) += dlnphiijdn[0][i*nc + k] * Xij[phase_idxs[0]*nc + k];
                A(idxi, j) -= dlnphiijdn[j][i*nc + k] * Xij[phase_idxs[j]*nc + k];
            }

            // dxi0/dX contribution: Σi dlnphi_i0/dn_k0 nu_0 dx_k0/dX + 1/xi0 dx_i0/dX
            A(idxi, this->np + i) += 1./Xij[phase_idxs[0]*nc + i];
            for (int k = 0; k < nc; k++)
            {
                A(idxi, this->np + k) += dlnphiijdn[0][i*nc + k] * nuj[phase_idxs[0]];
            }

            // dxij/dX contribution: - Σk dlnphi_ij/dn_kj nu_j dx_kj/dX - 1/xij dx_ij/dX
            A(idxi, this->np + j*nc + i) -= 1./Xij[phase_idxs[j]*nc + i];
            for (int k = 0; k < nc; k++)
            {
                A(idxi, this->np + j*nc + k) -= dlnphiijdn[j][i*nc + k] * nuj[phase_idxs[j]];
            }
        }
    }

    // Solve Ay = b for dP and dT and AY = B for dzk
    this->LUofA.compute(A);
}

void FlashResults::calc_dP_derivs()
{
    // Derivatives of flash w.r.t. pressure, temperature and composition zk (mole fractions)

    // Mass balance equations: 1 - Σj nu_j = 0 [1]; lnfi0 - lnfij = 0 [(NP-1)*NC]; zi - Σj xij nu_j = 0 [NC], Σi (xi0-xij) = 0 [NP-1] -> total [(NC+1)*NP]
    // Unknowns: dnuj/dX [NP], dxij/dX [NP*NC], dnuj/dzk [NP * NC], dxij/dzk [(NP*NC) * NC]
    int nc = flash_params.nc;
    int n_eq = np * (nc+1);
    this->dnudP = std::vector<double>(np_tot, 0.);
    this->dxdP = std::vector<double>(np_tot * nc, 0.);
    Eigen::VectorXd bP = Eigen::VectorXd::Zero(n_eq);

    // Solve reference phase fugacity coefficients
    std::vector<std::vector<double>> dlnphiijdP(this->np);
    for (int j = 0; j < this->np; j++)
    {
        // Solve phase j fugacity coefficients
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idxs[j]]]] };
        params->eos->set_root_flag(root_type[phase_idxs[j]]);
        params->eos->solve_PT(pressure, temperature, nij, phase_idxs[j]*nc, true);
        dlnphiijdP[j] = params->eos->dlnphi_dP();
    }

    // Construct matrix B

    // [1 eq] [0] 1 - Σj nu_j = 0
    // d/dX: dnu0/dX + Σj dnuj/dX = 0

    // [NP-1 eq] [1, NP] Σi (xi0-xij) = 0
    // Σi (dx_i0/dX - dx_ij/dX) = 0

    // [NC eq] [NP, NP+NC] zi - Σj nu_j xij = 0
    // Σj x_ij dnu_j/dzk + nu_j dx_ij/dzk = dzi/dX {1 if i == k else -1}

    // [(NP-1)*NC eq] [NP+NC, (NC+1)*NP] lnfi0 - lnfij = 0
    // dlnphi_i0/dX + Σk dlnphi_i0/dn_k0 dn_k0/dX + 1/xi0 dx_i0/dX - dlnphi_ij/dX - Σk dlnphi_ij/dn_kj dn_kj/dX - 1/xij dx_ij/dX  = 0
    // dn_ij/dX = dnu_j/dX x_ij + nu_j dx_ij/dX
    for (int j = 1; j < this->np; j++)
    {
        for (int i = 0; i < nc; i++)
        {
            int idxi = np + j*nc + i;

            // bi contribution: -(dlnphi_i0/dX - dlnphi_ij/dX)
            bP(idxi) -= dlnphiijdP[0][i] - dlnphiijdP[j][i];
        }
    }

    // Solve Ay = b for dP and dT and AY = B for dzk
    Eigen::VectorXd yP = this->LUofA.solve(bP);

    for (int j = 0; j < np; j++)
    {
        // P and T derivatives
        dnudP[phase_idxs[j]] = yP(j);
        for (int i = 0; i < nc; i++)
        {
            dxdP[phase_idxs[j]*nc + i] = yP(np + j*nc + i);
        }
    }
    return;
}
void FlashResults::calc_dT_derivs()
{
    // Derivatives of flash w.r.t. pressure, temperature and composition zk (mole fractions)

    // Mass balance equations: 1 - Σj nu_j = 0 [1]; lnfi0 - lnfij = 0 [(NP-1)*NC]; zi - Σj xij nu_j = 0 [NC], Σi (xi0-xij) = 0 [NP-1] -> total [(NC+1)*NP]
    // Unknowns: dnuj/dX [NP], dxij/dX [NP*NC], dnuj/dzk [NP * NC], dxij/dzk [(NP*NC) * NC]
    int nc = flash_params.nc;
    int n_eq = np * (nc+1);
    this->dnudT = std::vector<double>(np_tot, 0.);
    this->dxdT = std::vector<double>(np_tot * nc, 0.);
    Eigen::VectorXd bT = Eigen::VectorXd::Zero(n_eq);

    // Solve reference phase fugacity coefficients
    std::vector<std::vector<double>> dlnphiijdT(this->np);
    for (int j = 0; j < this->np; j++)
    {
        // Solve phase j fugacity coefficients
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idxs[j]]]] };
        params->eos->set_root_flag(root_type[phase_idxs[j]]);
        params->eos->solve_PT(pressure, temperature, nij, phase_idxs[j]*nc, true);
        dlnphiijdT[j] = params->eos->dlnphi_dT();
    }

    // Construct matrix B

    // [1 eq] [0] 1 - Σj nu_j = 0
    // d/dX: dnu0/dX + Σj dnuj/dX = 0

    // [NP-1 eq] [1, NP] Σi (xi0-xij) = 0
    // Σi (dx_i0/dX - dx_ij/dX) = 0

    // [NC eq] [NP, NP+NC] zi - Σj nu_j xij = 0
    // Σj x_ij dnu_j/dzk + nu_j dx_ij/dzk = dzi/dX {1 if i == k else -1}

    // [(NP-1)*NC eq] [NP+NC, (NC+1)*NP] lnfi0 - lnfij = 0
    // dlnphi_i0/dX + Σk dlnphi_i0/dn_k0 dn_k0/dX + 1/xi0 dx_i0/dX - dlnphi_ij/dX - Σk dlnphi_ij/dn_kj dn_kj/dX - 1/xij dx_ij/dX  = 0
    // dn_ij/dX = dnu_j/dX x_ij + nu_j dx_ij/dX
    for (int j = 1; j < this->np; j++)
    {
        for (int i = 0; i < nc; i++)
        {
            int idxi = np + j*nc + i;

            // bi contribution: -(dlnphi_i0/dX - dlnphi_ij/dX)
            bT(idxi) -= dlnphiijdT[0][i] - dlnphiijdT[j][i];
        }
    }

    // Solve Ay = b for dP and dT and AY = B for dzk
    Eigen::VectorXd yT = this->LUofA.solve(bT);
    
    for (int j = 0; j < np; j++)
    {
        // P and T derivatives
        dnudT[phase_idxs[j]] = yT(j);
        for (int i = 0; i < nc; i++)
        {
            dxdT[phase_idxs[j]*nc + i] = yT(np + j*nc + i);
        }
    }
    return;
}
void FlashResults::calc_dz_derivs()
{
    // Derivatives of flash w.r.t. pressure, temperature and composition zk (mole fractions)

    // Mass balance equations: 1 - Σj nu_j = 0 [1]; lnfi0 - lnfij = 0 [(NP-1)*NC]; zi - Σj xij nu_j = 0 [NC], Σi (xi0-xij) = 0 [NP-1] -> total [(NC+1)*NP]
    // Unknowns: dnuj/dX [NP], dxij/dX [NP*NC], dnuj/dzk [NP * NC], dxij/dzk [(NP*NC) * NC]
    int nc = flash_params.nc;
    int n_eq = np * (nc+1);
    this->dnudzk = std::vector<double>(np_tot * nc, 0.);
    this->dxdzk = std::vector<double>(np_tot * nc * nc, 0.);
    Eigen::MatrixXd Bz = Eigen::MatrixXd::Zero(n_eq, nc);

    // Construct matrix B

    // [1 eq] [0] 1 - Σj nu_j = 0
    // d/dX: dnu0/dX + Σj dnuj/dX = 0

    // [NP-1 eq] [1, NP] Σi (xi0-xij) = 0
    // Σi (dx_i0/dX - dx_ij/dX) = 0

    // [NC eq] [NP, NP+NC] zi - Σj nu_j xij = 0
    // Σj x_ij dnu_j/dzk + nu_j dx_ij/dzk = dzi/dX {1-zi if i == k else -zk}
    for (int i = 0; i < nc; i++)
    {
        int idxi = np + i;

        // Bik contribution: {1-zi if i == k else -zk}
        for (int k = 0; k < nc; k++)
        {
            Bz(idxi, k) = -zi[i];
        }
        Bz(idxi, i) = 1.-zi[i];
    }

    // [(NP-1)*NC eq] [NP+NC, (NC+1)*NP] lnfi0 - lnfij = 0
    // dlnphi_i0/dX + Σk dlnphi_i0/dn_k0 dn_k0/dX + 1/xi0 dx_i0/dX - dlnphi_ij/dX - Σk dlnphi_ij/dn_kj dn_kj/dX - 1/xij dx_ij/dX  = 0
    // dn_ij/dX = dnu_j/dX x_ij + nu_j dx_ij/dX

    // Solve Ay = b for dP and dT and AY = B for dzk
    Eigen::MatrixXd Yz = LUofA.solve(Bz);

    for (int j = 0; j < np; j++)
    {
        // Composition derivatives
        for (int k = 0; k < nc; k++)
        {
            dnudzk[k * np_tot + phase_idxs[j]] = Yz(j, k);
            for (int i = 0; i < nc; i++)
            {
                dxdzk[k * np_tot * nc + phase_idxs[j]*nc + i] = Yz(np + j*nc + i, k);
            }
        }
    }
    return;
}

double FlashResults::dX_dP(StateSpecification state_spec_)
{
    // Calculate derivative of total Gibbs free energy, enthalpy or entropy with respect to pressure

    // Gibbs free energy: dG/dP = d/dP [Σj G_j] = Σj dGj/dP
    //                          = Σj [(dGj/dP)_T,n + Σk (dGj/dnk)_P,T dnk/dP] = Σj [(dGj/dP)_T,n + Σk dGj/dnk (nuj dxk/dT + xk dnuj/dT]
    // Enthalpy: dH/dP = d/dP [Σj H_j] = Σj dHj/dP
    //                 = Σj [(dHj/dP)_T,n + Σk (dHj/dnk)_P,T dnk/dP] = Σj [(dHj/dP)_T,n + Σk dHj/dnk (nuj dxk/dT + xk dnuj/dT]
    // Entropy: dS/dP = d/dP [Σj Sj] = Σj dSj/dP
    //                = Σj [(dSj/dP)_T,n + Σk (dSj/dnk)_P,T dnk/dP] = Σj [(dSj/dP)_T,n + Σk dSj/dnk (nuj dxk/dT + xk dnuj/dT]
    double dXdP = 0.;
    for (int j = 0; j < np; j++)
    {
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idxs[j]]]] };
        params->eos->set_root_flag(root_type[phase_idxs[j]]);
        int start_idx = phase_idxs[j]*flash_params.ns;

        // Derivative of Gibbs free energy/enthalpy/entropy with respect to pressure (dXj/dP)_T,n
        if (state_spec_ == StateSpecification::TEMPERATURE) { dXdP += params->eos->dG_dP(pressure, temperature, nij, start_idx, true) * M_R; }
        else if (state_spec_ == StateSpecification::ENTHALPY) { dXdP += params->eos->dH_dP(pressure, temperature, nij, start_idx, true) * M_R; }
        else { dXdP += params->eos->dS_dP(pressure, temperature, nij, start_idx, true) * M_R; }
        
        // Derivative of Gibbs free energy/enthalpy/entropy with respect to composition (dXj/dnk)_P,T
        std::vector<double> dXj_dnk;
        if (state_spec_ == StateSpecification::TEMPERATURE) { dXj_dnk = params->eos->dG_dni(pressure, temperature, nij, start_idx, true); }
        else if (state_spec_ == StateSpecification::ENTHALPY) { dXj_dnk = params->eos->dH_dni(pressure, temperature, nij, start_idx, true); }
        else { dXj_dnk = params->eos->dS_dni(pressure, temperature, nij, start_idx, true); }
        
        for (int i = 0; i < flash_params.nc; i++)
        {
            dXdP += dXj_dnk[i] * (this->dnudP[phase_idxs[j]] * Xij[start_idx + i] 
                                 + nuj[phase_idxs[j]] * this->dxdP[start_idx + i]) * M_R;
        }
    }
    return dXdP;
}
double FlashResults::dX_dT(StateSpecification state_spec_)
{
    // Calculate derivative of total Gibbs free energy, enthalpy or entropy with respect to temperature

    // Gibbs free energy: dG/dT = d/dT [Σj G_j] = Σj dGj/dT
    //                          = Σj [(dGj/dT)_P,n + Σk (dGj/dnk)_P,T dnk/dT] = Σj [-Sj + Σk dHj/dnk (nuj dxk/dT + xk dnuj/dT]
    // Enthalpy: dH/dT = d/dT [Σj H_j] = Σj dHj/dT
    //                 = Σj [(dHj/dT)_P,n + Σk (dHj/dnk)_P,T dnk/dT] = Σj [Cpj + Σk dHj/dnk (nuj dxk/dT + xk dnuj/dT]
    // Entropy: dS/dT = d/dT [Σj Sj] = Σj dSj/dT
    //                = Σj [(dSj/dT)_P,n + Σk (dSj/dnk)_P,T dnk/dT] = Σj [Cpj/T + Σk dSj/dnk (nuj dxk/dT + xk dnuj/dT]
    double dXdT = 0.;
    for (int j = 0; j < np; j++)
    {
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idxs[j]]]] };
        params->eos->set_root_flag(root_type[phase_idxs[j]]);
        int start_idx = phase_idxs[j]*flash_params.ns;

        // Derivative of Gibbs free energy/enthalpy/entropy with respect to temperature (dXj/dT)_P,n: -S, Cpr or Cpr/T
        if (state_spec_ == StateSpecification::TEMPERATURE) { dXdT += params->eos->dG_dT(pressure, temperature, nij, start_idx, true) * M_R; }
        else if (state_spec_ == StateSpecification::ENTHALPY) { dXdT += params->eos->dH_dT(pressure, temperature, nij, start_idx, true) * M_R; }
        else { dXdT += params->eos->dS_dT(pressure, temperature, nij, start_idx, true) * M_R; }
        
        // Derivative of Gibbs free energy/enthalpy/entropy with respect to composition (dXj/dnk)_P,T
        std::vector<double> dXj_dnk;
        if (state_spec_ == StateSpecification::TEMPERATURE) { dXj_dnk = params->eos->dG_dni(pressure, temperature, nij, start_idx, true); }
        else if (state_spec_ == StateSpecification::ENTHALPY) { dXj_dnk = params->eos->dH_dni(pressure, temperature, nij, start_idx, true); }
        else { dXj_dnk = params->eos->dS_dni(pressure, temperature, nij, start_idx, true); }
        
        for (int i = 0; i < flash_params.nc; i++)
        {
            dXdT += dXj_dnk[i] * (this->dnudT[phase_idxs[j]] * Xij[start_idx + i] 
                                 + nuj[phase_idxs[j]] * this->dxdT[start_idx + i]) * M_R;
        }
    }
    return dXdT;
}
std::vector<double> FlashResults::dX_dz(StateSpecification state_spec_)
{
    // Calculate derivative of total Gibbs free energy, enthalpy or entropy with respect to temperature

    // Gibbs free energy: dG/dzk = d/dzk [Σj G_j] = Σj dGj/dzk
    //                           = Σj [(dGj/dzk)_P,T + Σk (dGj/dnk)_P,T dnk/dzl] = Σj [Σl dGj/dnl (nuj dxl/dzk + xl dnuj/dzk]
    // Enthalpy: dH/dzk = d/dzk [Σj H_j] = Σj dHj/dzk
    //                  = Σj [(dHj/dzk)_P,T + Σk (dHj/dnk)_P,T dnk/dzl] = Σj [Σl dHj/dnl (nuj dxl/dzk + xl dnuj/dzk]
    // Entropy: dH/dzk = d/dzk [Σj S_j] = Σj dSj/dzk
    //                 = Σj [(dSj/dzk)_P,T + Σk (dSj/dnk)_P,T dnk/dzl] = Σj [Σl dSj/dnl (nuj dxl/dzk + xl dnuj/dzk]
    std::vector<double> dXdzk(flash_params.nc, 0.);
    
    for (int j = 0; j < np; j++)
    {
        std::shared_ptr<EoSParams> params{ flash_params.eos_params[flash_params.eos_order[eos_idx[phase_idxs[j]]]] };
        params->eos->set_root_flag(root_type[phase_idxs[j]]);
        int start_idx = phase_idxs[j]*flash_params.nc;

        // Derivative of Gibbs free eneryg/enthalpy/entropy with respect to composition (dXj/dnk)_P,T
        std::vector<double> dXj_dnk;
        if (state_spec_ == StateSpecification::TEMPERATURE) { dXj_dnk = params->eos->dG_dni(pressure, temperature, nij, start_idx, true); }
        else if (state_spec_ == StateSpecification::ENTHALPY) { dXj_dnk = params->eos->dH_dni(pressure, temperature, nij, start_idx, true); }
        else { dXj_dnk = params->eos->dS_dni(pressure, temperature, nij, start_idx, true); }
        
        for (int k = 0; k < flash_params.nc; k++)
        {
            for (int i = 0; i < flash_params.nc; i++)
            {
                dXdzk[k] += dXj_dnk[i] * (this->dnudzk[k * np_tot + phase_idxs[j]] * Xij[start_idx + i] 
                                         + nuj[phase_idxs[j]] * this->dxdzk[k * np_tot * flash_params.nc + start_idx + i]) * M_R;
            }
        }
    }
    return dXdzk;
}
