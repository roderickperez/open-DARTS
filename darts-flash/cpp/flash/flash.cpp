#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <map>

#include "dartsflash/maths/maths.hpp"
#include "dartsflash/maths/geometry.hpp"
#include "dartsflash/maths/root_finding.hpp"
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/trial_phase.hpp"
#include "dartsflash/stability/stability.hpp"
#include "dartsflash/phase-split/twophasesplit.hpp"
#include "dartsflash/phase-split/multiphasesplit.hpp"

Flash::Flash(FlashParams& flashparams) {
	this->flash_params = flashparams;

    this->z.resize(flashparams.ns);
    this->lnphi0.resize(flashparams.ns);
    this->nonzero_comp.resize(flash_params.ns);
    // this->eos.reserve(NP_MAX);
    this->nu.reserve(NP_MAX);
    this->X.reserve(NP_MAX*flashparams.ns);

    this->stationary_points.reserve(NP_MAX);
    this->ref_compositions.reserve(NP_MAX);
    this->sp_idxs.reserve(NP_MAX + 1);
    this->ref_idxs.reserve(NP_MAX);
}

void Flash::init(double p_, double T_)
{
    // Set iterations to zero
    total_ssi_flash_iter = total_ssi_stability_iter = total_newton_flash_iter = total_newton_stability_iter = 0;
    
    // Initialize EoS and InitialGuess at p, T
    this->p = p_; this->T = T_;
    this->flash_params.init_eos(p, T);

    return;
}

void Flash::init(double p_, double T_, std::vector<double>& z_)
{
    Flash::init(p_, T_);

    // Calculate pure component Gibbs free energies
    this->lnphi0 = flash_params.G_pure(p, T);

    // Check if feed composition needs to be corrected for 0 values
    z = z_;
    this->nonzero_comp = std::vector<bool>(flash_params.ns, true);
    this->ns_nonzero = flash_params.ns;
    for (int i = 0; i < flash_params.ns; i++)
    {
        if (z_[i] < flash_params.min_z)
        {
            z[i] = flash_params.min_z;
            this->nonzero_comp[i] = false;
            this->ns_nonzero--;
        }
    }
    return;
}

int Flash::evaluate(double p_, double T_)
{
    // Find reference compositions - hypothetical single phase
    Flash::init(p_, T_);
    this->np = 1;
    z = {1.};
    this->ref_compositions = {this->flash_params.find_ref_comp(p_, T_, z)};
    this->ref_idxs = {0};

    // Set nu and X for Results
    this->ref_compositions[0].nu = 1.;
    this->ref_compositions[0].X = z;
    stationary_points = ref_compositions;
    return 0;
}

int Flash::evaluate(double p_, double T_, std::vector<double>& z_, bool start_from_feed)
{
    // Evaluate sequential stability + flash algorithm
    
    // Initialize flash
    // Initialize EoS at p, T and check if feed composition needs to be corrected
    if (z_.size() == 1)
    {
        return Flash::evaluate(p_, T_);
    }

    // If start from feed, find reference composition at feed
    if (start_from_feed || ref_compositions.empty())
    {
        Flash::init(p_, T_, z_);

        // Perform stability and phase split loop starting from np = 1
        ref_compositions = {flash_params.find_ref_comp(p, T, z)};
        ref_compositions[0].nu = 1.;
        ref_compositions[0].X = z;
        stationary_points = ref_compositions;
        ref_idxs = {0};
    }

    this->np = static_cast<int>(ref_idxs.size());
    int it = 1;
    while (true)
    {
        int output = this->run_loop();
        if (output == -1)
        {
            // Output -1, all phases stable
            // Gather equilibrium phases
            this->ref_compositions = {};
            for (int sp_idx: ref_idxs) { ref_compositions.push_back(stationary_points[sp_idx]); }

            if (flash_params.verbose)
            {
                print("StabilityFlash", "===============");
                print("p, T", {p, T});
			    print("z", z_);
                Flash::get_flash_results()->print_results();
            }
            return 0;
        }
        else if (output > 0 || it > 10)
        {
            // Else, error occurred in split
            if (flash_params.verbose)
            {
                print("ERROR in StabilityFlash", output);
    		    print("p, T", {p, T});
			    print("z", z_);
            }
            return output;
        }
        it++;
    }

    return 0;
}

int Flash::evaluate(double p_, double T_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results)
{
    // Evaluate sequential stability + flash algorithm, but start from previous flash results
    // Initialize flash
    // Initialize EoS at p, T and check if feed composition needs to be corrected
    if (z_.size() == 1)
    {
        return Flash::evaluate(p_, T_);
    }
    Flash::init(p_, T_, z_);

    // If previous result is a single-phase state, start from feed. Else, try to run a phase split with (extrapolated) results
    bool start_from_feed = true;
    if (flash_results->phase_idxs.size() > 1)
    {
        // If previous result is a multiphase equilibrium, run phase split with initial guess from previous flash results
        start_from_feed = false;
        this->ref_compositions.clear();
        for (int j : flash_results->phase_idxs)
        {
            std::vector<double> y(flash_params.ns);
            auto begin = flash_results->Xij.begin() + j * flash_params.ns;
            std::copy(begin, begin + flash_params.ns, y.begin());

            TrialPhase ref_comp = TrialPhase(flash_results->eos_idx[j], flash_params.eos_order[flash_results->eos_idx[j]], y);
            ref_compositions.push_back(ref_comp);
        }
        stationary_points = ref_compositions;
        this->np = flash_results->np;
        ref_idxs.resize(np);
        std::iota(ref_idxs.begin(), ref_idxs.end(), 0);

        // Run split (accept negative flash, it is just an initial guess)
        int split_output = this->run_split(ref_idxs, z, true);
        if (split_output == -1)
        {
            // If split output is -1, one or more negative phases present
            std::vector<int> ref_idxs_new;
            for (int j: ref_idxs)
            {
                if (stationary_points[j].nu > 0.)
                {
                    ref_idxs_new.push_back(j);
                }
            }
            ref_idxs = ref_idxs_new;
            if (ref_idxs.size() == 1)
            {
                start_from_feed = true;
            }
        }
        ref_compositions = stationary_points;
    }
    
    // Perform stability and phase split loop starting from split output
    return Flash::evaluate(p_, T_, z_, start_from_feed);
}

double Flash::locate_phase_boundary(std::shared_ptr<FlashResults> results_a, std::shared_ptr<FlashResults> results_b)
{
    // Method to find exact conditions where two flashes have equal Gibbs energies
    // Define lambda to compute difference between Gibbs energies of flashes a and b
    std::vector<int> sp_idxs_a = results_a->ref_idxs;  // idxs of reference compositions in set of stationary_points in FlashResults object
    std::vector<int> sp_idxs_b = results_b->ref_idxs;
    auto dG = [this, &sp_idxs_a, &sp_idxs_b](std::shared_ptr<FlashResults> results_a_, std::shared_ptr<FlashResults> results_b_, double T_, double& grad) 
    {
        // Find difference of Gibbs energies between flashes a and b
        double G_a, G_b, dG_a = 1, dG_b = 1;
        bool run_negative_flash{ true }, set_root_flags{ true };
        Flash::init(this->p, T_);

        // Evaluate split with stationary points of flash at a
        this->stationary_points = results_a_->stationary_points;
        if (sp_idxs_a.size() == 1)
        {
            std::shared_ptr<EoSParams> params{ flash_params.eos_params[this->stationary_points[sp_idxs_a[0]].eos_name] };
            params->eos->set_root_flag(this->stationary_points[sp_idxs_a[0]].root);
            G_a = params->eos->G(this->p, T_, this->z, 0, true) * M_R;
            if (!std::isnan(grad))
            {
                dG_a = params->eos->dG_dT(this->p, T_, this->z, 0, true) * M_R;
            }
        }
        else
        {
            this->run_split(sp_idxs_a, this->z, run_negative_flash, set_root_flags);
            results_a_->set_flash_results(this->p, T_, this->z, this->stationary_points, sp_idxs_a);
            G_a = results_a_->total_prop(EoS::Property::GIBBS);

            // Calculate derivative if required
            if (!std::isnan(grad))
            {
                results_a_->set_flash_derivs();
                dG_a = results_a_->dX_dT(StateSpecification::TEMPERATURE);
            }
        }

        // Evaluate split with stationary points of flash at b
        this->stationary_points = results_b_->stationary_points;
        if (sp_idxs_b.size() == 1)
        {
            std::shared_ptr<EoSParams> params{ flash_params.eos_params[this->stationary_points[sp_idxs_b[0]].eos_name] };
            params->eos->set_root_flag(this->stationary_points[sp_idxs_b[0]].root);
            G_b = params->eos->G(this->p, T_, this->z, 0, true) * M_R;
            if (!std::isnan(grad))
            {
                dG_b = params->eos->dG_dT(this->p, T_, this->z, 0, true) * M_R;
            }
        }
        else
        {
            this->run_split(sp_idxs_b, this->z, run_negative_flash, set_root_flags);
            results_b_->set_flash_results(this->p, T_, this->z, this->stationary_points, sp_idxs_b);
            G_b = results_b_->total_prop(EoS::Property::GIBBS);
            
            // Calculate derivative if required
            if (!std::isnan(grad))
            {
                results_b_->set_flash_derivs();
                dG_b = results_b_->dX_dT(StateSpecification::TEMPERATURE);
            }
        }

        // Return difference in Gibbs energies
        if (!std::isnan(grad)) { grad = dG_a - dG_b; }
        return G_a - G_b;
    };

    // Apply Brent method to find P/T/z of phase boundary
    double Ta = results_a->temperature;
    double Tb = results_b->temperature;
    this->T = (Ta+Tb)*0.5;

    RootFinding root;
    double gradient = NAN;
    auto f = std::bind(dG, results_a, results_b, std::placeholders::_1, gradient);
	int output = root.brent(f, this->T, Ta, Tb, flash_params.phase_boundary_Gtol, flash_params.phase_boundary_Ttol);
    this->T = root.getx();

    // Perform Newton loop if Brent, for some reason, has not converged
    if (output != 0)
    {
        auto df = std::bind([](double T_, double& gradient_) { (void) T_; return gradient_; }, std::placeholders::_1, gradient);
        output = root.brent_newton(f, df, this->T, Ta, Tb, flash_params.phase_boundary_Gtol, 1e-14);
        this->T = root.getx();
    }
    return T;
}

std::shared_ptr<FlashResults> Flash::get_flash_results(bool derivs)
{
    // Identify vapour and liquid phases
    if (this->flash_params.light_comp_idx >= 0)
    {
        this->identify_vl_phases();
    }

    // Return flash results and performance data
    std::shared_ptr<FlashResults> flash_results = std::make_shared<FlashResults>(this->flash_params, this->state_spec);
    // FlashResults flash_results(this->flash_params, this->state_spec);
    flash_results->set_flash_results(this->p, this->T, this->z, this->stationary_points, this->ref_idxs);

    // If needed, get derivatives
    if (derivs)
    {
        // Solve linear system to obtain derivatives of flash for simulation
        // Simulation requires derivatives of phase fractions and phase compositions w.r.t. primary variables X
        // Need to apply chain rule on derivatives w.r.t. pressure, temperature and composition
        flash_results->set_flash_derivs();
    }
    return flash_results;
}

std::shared_ptr<FlashResults> Flash::extrapolate_flash_results(double p_, double T_, std::vector<double>& z_, std::shared_ptr<FlashResults> flashresults)
{
    // Method to extrapolate flash results to a (nearby) state to obtain an initial guess
    std::shared_ptr<FlashResults> flash_results = std::make_shared<FlashResults>(*flashresults);
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
    dX[1] = T_ - flashresults->temperature;
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
            
            // P and T derivatives: dx/dP * dP and dx/dT * dT
            double dXij = dX[0] * flash_results->dxdP[idx] + dX[1] * flash_results->dxdT[idx];

            // Compositional derivatives: dx/dzk * dzk
            for (int k = 0; k < flash_params.ns; k++)
            {
                int idxk = k * flash_results->np_tot * flash_params.ns + j * flash_params.ns + i;
                dXij += dX[2 + k] * flash_results->dxdzk[idxk];
            }

            flash_results->Xij[idx] += dXij;
        }
    }

    flash_results->set_results = false;
    flash_results->set_derivs = false;
    return flash_results;
}

int Flash::run_stability()
{
    // Run stability test on current ref compositions
    // This is either feed composition (when no information is available) or from reference compositions of previous flash loop
    std::vector<TrialPhase> ref_comps = {};
    std::vector<bool> is_ref(stationary_points.size(), false);
    for (int sp_idx: ref_idxs)
    {
        ref_comps.push_back(stationary_points[sp_idx]);
        is_ref[sp_idx] = true;
    }
    Stability stab(flash_params);

    // If starting from a feed composition that is far from minimum of gmix (and use_gmix=true), run stability test from minimum of gmix
    // Have to recompute tpd from actual feed composition later on
    bool use_gmix_min_as_reference = false;
    std::vector<TrialPhase> original_comp;

    // Add each of the minima trivial stationary points (for np > 1, this corresponds to all phase compositions)
    std::shared_ptr<EoSParams> params0{ flash_params.eos_params[ref_comps[0].eos_name] };
    if (ref_comps.size() == 1)
    {
        if (!params0->eos->is_convex(ref_comps[0].Y.begin()) && params0->eos->has_multiple_minima())
        {
            // If feed composition is not a minimum, remove stationary point from list
            stationary_points.clear();
            ref_idxs.clear();
            is_ref.clear();
            if (flash_params.verbose)
            {
                std::cout << "Ref composition is not a minimum\n";
            }
        }

        // For some feed compositions, local minimum of gmix is better for stability testing, compute it
        if (params0->use_gmix && !params0->eos->eos_in_range(ref_comps[0].Y.begin(), true))
        {
            std::vector<TrialPhase> trial_comps = this->flash_params.get_trial_comps(ref_comps[0].eos_idx, ref_comps[0].eos_name, ref_comps);
            stab.init_gmix(trial_comps[0], lnphi0);
            stab.run(trial_comps[0], true);

            original_comp = ref_comps;
            ref_comps[0] = trial_comps[0];
            use_gmix_min_as_reference = true;
        }
    }

    // Run stability tests starting from each stationary point
    // Keep track of EoSs in stationary points: if EoS has only a single minimum, no need to evaluate additional initial guesses
    stab.init(ref_comps);

    std::vector<bool> eos_to_test(flash_params.eos_order.size(), true);
    size_t jj = 0;
    for (size_t j = 0; j < is_ref.size(); j++)
    {
        // Test only stationary points that are not part of the reference compositions
        int eos_idx = stationary_points[jj].eos_idx;
        if (!is_ref[j])
        {
            TrialPhase trial_comp = stationary_points[jj];
            int error = stab.run(trial_comp);

            if (error > 0)
            {
                return error;
            }

            // Get TPD value and check if it is already in the list
            if (trial_comp.is_preferred_root >= EoS::RootSelect::ACCEPT && !this->compare_stationary_points(trial_comp))
            {
                // No duplicate found: add stationary point to vector of stationary points
                stationary_points[jj] = trial_comp;
            }
            else
            {
                // Duplicate found: stability test converged to one of the reference compositions
                stationary_points.erase(stationary_points.begin() + jj);
                for (int& ref_idx: ref_idxs)
                {
                    if (ref_idx > static_cast<int>(jj))
                    {
                        ref_idx--;
                    }
                }
                jj--;
            }

            // Get number of iterations from stability
            this->total_ssi_stability_iter += stab.get_ssi_iter();
            this->total_newton_stability_iter += stab.get_newton_iter();
        }
        // If EoS has only a single minimum, no need to evaluate additional initial guesses
        if (!flash_params.eos_params[flash_params.eos_order[eos_idx]]->eos->has_multiple_minima())
        {
            eos_to_test[eos_idx] = false;
        }
        jj++;
    }

    // Iterate over initial guesses in Y to run stability tests
    for (size_t j = 0; j < this->flash_params.eos_order.size(); j++)
    {
        if (eos_to_test[j])
        {
            std::vector<TrialPhase> trial_comps = this->flash_params.get_trial_comps(j, this->flash_params.eos_order[j], ref_comps);
            for (TrialPhase& trial_comp: trial_comps)
            {
                int error = stab.run(trial_comp);

                if (error > 0)
                {
                    return error;
                }

                // Get TPD value and check if it is already in the list
                if (trial_comp.is_preferred_root >= EoS::RootSelect::ACCEPT && !this->compare_stationary_points(trial_comp))
                {
                    // No duplicate found: add stationary point to vector of stationary points
                    stationary_points.push_back(trial_comp);
                }

                // Get number of iterations from stability
                this->total_ssi_stability_iter += stab.get_ssi_iter();
                this->total_newton_stability_iter += stab.get_newton_iter();
            }
        }
    }

    // If gmix has been used as a reference composition, Recompute tpd
    if (use_gmix_min_as_reference)
    {
        stab.init(original_comp);
        for (size_t j = 1; j < stationary_points.size(); j++)
        {
            stationary_points[j].tpd = stab.calc_modtpd(stationary_points[j]);
        }
    }

    // If for EoS gmix is preferred over stationary point, evaluate local minimum
    for (TrialPhase& sp: stationary_points)
    {
        if (flash_params.eos_params[sp.eos_name]->use_gmix)
        {
            if (!sp.gmin_converged)
            {
                stab.init_gmix(sp, lnphi0);
                (void) stab.run(sp, true);
            }
        }
        else
        {
            sp.ymin = sp.y;
        }
    }

    // Sort stationary points according to tpd value
    this->sort_stationary_points();

    return 0;
}

int Flash::run_split(std::vector<int>& sp_idxs_, std::vector<double>& z_, bool negative_flash, bool set_root_flags)
{
    // Choose proper set of lnK
    std::vector<double> lnK = this->generate_lnK(sp_idxs_);
    if (std::isnan(lnK[0])) { return -1; }

    std::vector<TrialPhase> trial_comps{};
    for (int sp_idx: sp_idxs_)
    {
        trial_comps.push_back(stationary_points[sp_idx]);
    }

    // Initialize split object
    int split_output = 0;
    np = static_cast<int>(sp_idxs_.size());
    if (np == 2)
    {
        // Initialize PhaseSplit object for two phases
        TwoPhaseSplit split(flash_params);
        
        // Find a composition that is in the "middle" of the stationary points for easier solution
        split_output = split.run(z_, lnK, trial_comps);

        // Use converged solution as initial guess for actual composition and return nu and x
        std::vector<double> lnk = split.get_lnk();
        if (!std::isnan(lnk[0]))
        {
            split_output = split.run(this->z, lnk, trial_comps, set_root_flags);
        }

        gibbs = split.get_gibbs();
        total_ssi_flash_iter += split.get_ssi_iter();
        total_newton_flash_iter += split.get_newton_iter();
    }
    else
    {
        // Initialize MultiPhaseSplit object for 3 or more phases
        MultiPhaseSplit split(flash_params, np);

        // Find a composition that is in the "middle" of the stationary points for easier solution
        split_output = split.run(z_, lnK, trial_comps);

        // Use converged solution as initial guess for actual composition and return nu and x
        std::vector<double> lnk = split.get_lnk();
        if (!std::isnan(lnk[0]))
        {
            split_output = split.run(this->z, lnk, trial_comps, set_root_flags);
        }

        gibbs = split.get_gibbs();
        total_ssi_flash_iter += split.get_ssi_iter();
        total_newton_flash_iter += split.get_newton_iter();
    }

    // Determine output value
    if (split_output == 0 || (negative_flash && split_output <= 0))
    {
        for (size_t i = 0; i < sp_idxs_.size(); i++)
        {
            stationary_points[sp_idxs_[i]] = trial_comps[i];
        }
    }
    return split_output;
}

int Flash::run_loop()
{
    // Perform stability test starting from each of the initial guesses
    int stability_output = this->run_stability();
    if (stability_output > 0)
    {
        // Error occurred in stability test
        return stability_output;
    }

    // Count number of stationary points with negative TPD and sort stationary points according to tpd
    int negative_tpds = 0;
    for (int i: sp_idxs)
    {
        if (stationary_points[i].tpd < -flash_params.tpd_tol)  // negative TPD
        {
            negative_tpds++;
        }
        else
        {
            break;
        }
    }

    // If no negative TPDs, feed is stable, return -1
    if (negative_tpds == 0 || stationary_points[sp_idxs[0]].tpd > -std::fabs(this->flash_params.tpd_1p_tol))
    {
        if (ref_idxs.size() == 0) { ref_idxs = sp_idxs; stationary_points[0].nu = 1.; stationary_points[0].X = z; }
        return -1;
    }
    else
    {
        // Else not stable
        if (flash_params.verbose)
        {
            print("Unstable feed", "============");
            for (TrialPhase stationary_point : stationary_points) { stationary_point.print_point(); }
        }

        // Determine lnK-values from set of stationary points for phase split
        int tot_sp = static_cast<int>(sp_idxs.size());
        int split_output = 0;

        if (tot_sp == 1)
        {
            if (flash_params.verbose)
            {
                std::cout << "Unstable feed, but only 1 stationary point in ref_compositions!\n";
                print("p, T", {p, T});
                print("z", z);
            }
            // Add trivial solution, even though it is non-convex
            stationary_points.push_back(flash_params.find_ref_comp(p, T, z));
            sp_idxs.push_back(tot_sp);
            tot_sp++;
        }
        if (tot_sp == 2)  // TWO STATIONARY POINTS <= 0 --> 2 PHASES: Select both stationary points
        {
            // Find if composition is within simplex of stationary point compositions
            std::vector<std::vector<double>> coords = {};
            for (TrialPhase& sp: stationary_points)
            {
                coords.push_back(sp.ymin);
            }
            this->z_mid = (this->ns_nonzero == 2) ? find_midpoint(coords) : this->z;

            // Run split with two stationary points
            split_output = this->run_split(sp_idxs, this->z_mid);
            if (split_output == 0) { ref_idxs = sp_idxs; }
            return split_output;
        }
        else  // In case of 3 or more stationary points <= 0, find possible combinations and determine which one has lowest Gibbs energy
        {
            // Maximum number of phases is equal to number of stationary points, but not larger than NC
            int NP_max = std::min(tot_sp, this->ns_nonzero);

            for (int NP = NP_max; NP >= 2; NP--)
            {
                // Find combinations of stationary points of maximum length np
                Combinations c(tot_sp, NP);
                
                // Find for each combination of stationary points if simplex with stationary points as vertices contains the feed composition
                // If it does, combination may be a solution to the multiphase equilibrium
                for (int jj = 0; jj < c.n_combinations; jj++)
                {
                    bool run_combination = false;

                    std::vector<int> idxs = {};
                    bool close_to_boundary = true;  // if minimum tpd is too close to zero, use of lnK becomes problematic
                    for (int idx: c.combinations[jj])
                    {
                        int sp_idx = sp_idxs[idx];
                        idxs.push_back(sp_idx);
                        TrialPhase* sp = &stationary_points[sp_idx];

                        // Check if combination contains negative tpd
                        if (sp->tpd < 0.)
                        {
                            run_combination = true;
                        }
                        // Check if combination contains positive gmin, if so do not run combination and break from loop
                        if ((flash_params.eos_params[sp->eos_name]->use_gmix && sp->gmin > 0.) || (NP > 2 && !sp->has_converged))
                        {
                            run_combination = false;
                            break;
                        }
                        if (sp->tpd < -flash_params.tpd_close_to_boundary)
                        {
                            close_to_boundary = false;
                        }
                    }

                    if (run_combination && NP == flash_params.nc)
                    {
                        // Find if composition is within simplex of stationary point compositions
                        std::vector<std::vector<double>> coords = {};
                        for (int sp_idx: idxs)
                        {
                            coords.push_back(stationary_points[sp_idx].y);
                        }
                        this->z_mid = find_midpoint(coords);
                        if (!is_in_simplex(z, coords))
                        {
                            // Or if it is within minimum Gibbs energy compositions
                            coords = {};
                            for (int sp_idx: idxs)
                            {
                                coords.push_back(stationary_points[sp_idx].ymin);
                            }
                            if (!is_in_simplex(z, coords))
                            {
                                run_combination = false;
                            }
                        }
                    }
                    else
                    {
                        this->z_mid = z;
                    }
                    
                    if (run_combination)
                    {
                        // If too close to phase boundary, change split variables to nik
                        FlashParams::SplitVars split_vars = this->flash_params.split_variables;
                        this->flash_params.split_variables = (close_to_boundary) ? FlashParams::SplitVars::nik : split_vars;

                        // Run phase split
                        split_output = this->run_split(idxs, this->z_mid);

                        // Change back the variables
                        this->flash_params.split_variables = split_vars;
                        
                        if (split_output == 0)
                        {
                            ref_idxs = idxs;
                            return 0;
                        }
                    }
                }
            }
            std::cout << "Error occurred in Flash::run_loop(), no stable flashes have been found\n";
            print("p, T", {p, T});
			print("z", z);
            return 1;
        }
    }
}

std::vector<double> Flash::generate_lnK(std::vector<int>& sp_idxs_)
{
    // Determine lnK initialization for phase split
    np = static_cast<int>(sp_idxs_.size());
    int nc = flash_params.nc;

    std::vector<double> lnY0(nc);
    int sp_idx = sp_idxs_[0];
    TrialPhase* sp = &stationary_points[sp_idxs_[0]];
    if (flash_params.eos_params[sp->eos_name]->use_gmix)
    {
        for (int i = 0; i < nc; i++)
        {
            lnY0[i] = std::log(sp->ymin[i]);
        }
    }
    else
    {
        for (int i = 0; i < nc; i++)
        {
            lnY0[i] = std::log(sp->y[i]);
        }
    }
    // eos = {stationary_points[sp_idx].eos_name};
    std::vector<TrialPhase> trial_comps = {stationary_points[sp_idx]};

    std::vector<double> lnK((np-1)*nc);
    for (size_t j = 1; j < sp_idxs_.size(); j++)
    {
        sp = &stationary_points[sp_idxs_[j]];

        // Determine composition to initialize K-values with
        if (flash_params.eos_params[sp->eos_name]->use_gmix)
        {
            for (int i = 0; i < nc; i++)
            {
                lnK[(j-1) * nc + i] = std::log(sp->ymin[i]) - lnY0[i];
            }
        }
        else
        {
            for (int i = 0; i < nc; i++)
            {
                lnK[(j-1) * nc + i] = std::log(sp->y[i]) - lnY0[i];
            }
        }
        // eos.push_back(sp->eos_name);
        trial_comps.push_back(*sp);
    }
    return lnK;
}

bool Flash::compare_stationary_points(TrialPhase& stationary_point)
{
    // Compare stationary point with entries in vector of stationary points to check if it is unique
    // Returns true if point is already in the list
    double tpd0 = stationary_point.tpd;
    double lntpd0 = std::log(std::fabs(tpd0));
    for (size_t j = 0; j < stationary_points.size(); j++)
    {
        double tpdj = stationary_points[j].tpd;
        // For small tpd difference (tpd < 1), compare absolute difference; for large tpd values, logarithmic scale is used to compare
        double tpd_diff = lntpd0 < 0. ? tpdj-tpd0 : lntpd0 - std::log(std::fabs(tpdj) + 1e-15);
        if (stationary_points[j].eos_name == stationary_point.eos_name // eos is the same
             && (std::fabs(tpd_diff) < flash_params.tpd_tol || // tpd is within tolerance
                (std::fabs(tpd0) < flash_params.tpd_tol && std::fabs(tpdj) < flash_params.tpd_tol))) // both tpds are within absolute tpd tolerance
        {
            // Similar TPD found; Check if composition is also the same
            if (compare_compositions(stationary_point.y, stationary_points[j].y, flash_params.comp_tol))
            {
                if (stationary_point.root != stationary_points[j].root)
                {
                    stationary_points[j].root = EoS::RootFlag::STABLE;
                }
                return true;
            }
        }
    }
    return false;
}

void Flash::sort_stationary_points()
{
    sp_idxs = {};
    for (size_t j = 0; j < this->stationary_points.size(); j++)
    {
        // Find location of tpd in sorted idxs
        std::vector<int>::iterator it;
        for (it = sp_idxs.begin(); it != sp_idxs.end(); it++)
        {
            if (this->stationary_points[*it].tpd > this->stationary_points[j].tpd)
            {
                sp_idxs.insert(it, static_cast<int>(j));
                break;
            }
        }
        if (it == sp_idxs.end())
        {
            sp_idxs.push_back(static_cast<int>(j));
        }
    }
    return;
}

std::vector<TrialPhase> Flash::find_stationary_points(double p_, double T_, std::vector<double>& X_)
{
    // Initialize EoS and InitialGuess at p, T
    Flash::init(p_, T_);
    
    // Perform stability test starting from each of the initial guesses
    this->stationary_points = {this->flash_params.find_ref_comp(p, T, X_)};
    this->ref_idxs = {0};
    int stability_output = this->run_stability();
    if (stability_output > 0)
    {
        // Error occurred in stability test
        if (this->flash_params.verbose)
		{
            print("ERROR in find_stationary_points()", stability_output);
		    print("p", p);
            print("T", T);
        }
        return stationary_points;
    }
    return this->stationary_points;
}

void Flash::identify_vl_phases()
{
    // Find vapour and liquid phases
    // Check if vapour phase can be identified from mechanical spinodal
    std::vector<TrialPhase> sps = this->stationary_points;  // store set of stationary points
    std::string vl_eos_name = this->flash_params.vl_eos_name;
    std::shared_ptr<EoSParams> params{ this->flash_params.eos_params[vl_eos_name] };
    int eos_idx = std::distance(flash_params.eos_order.begin(), std::find(flash_params.eos_order.begin(), flash_params.eos_order.end(), vl_eos_name));

    // Find compositions of particular EoS
    for (size_t j = 0; j < sps.size(); j++)
    {
        TrialPhase* sp = &sps[j];
        if (sp->eos_name == vl_eos_name)
        {
            // Use mechanical spinodal to identify vapour or liquid phases
            bool is_below_spinodal;
            EoS::RootFlag root = params->eos->is_root_type(sp->Y.begin(), is_below_spinodal);
            if (root > EoS::RootFlag::STABLE && is_below_spinodal)
            {
                if (root == EoS::RootFlag::MAX)
                {
                    sps[j].root = EoS::RootFlag::MAX;
                    for (size_t jj = j; jj < sps.size(); jj++)
                    {
                        if (jj != j && sps[jj].eos_name == vl_eos_name)
                        {
                            sps[jj].root = EoS::RootFlag::MIN;
                        }
                    }
                    break;
                }
                else
                {
                    sps[j].root = EoS::RootFlag::MIN;
                }
            }
            else
            {
                // Use stability test to identify vapour or liquid phases
                // Set tested phase as stationary points
                this->stationary_points = {TrialPhase(sp->eos_idx, sp->eos_name, sp->Y)};
                Stability stab(flash_params);
                stab.init(this->stationary_points);

                // Test initial guess
                std::vector<int> initial_guess = {flash_params.light_comp_idx};
                int idx = 0;
                // std::vector<int> initial_guess = {InitialGuess::Yi::Wilson};
                // int idx = 1;
                std::vector<TrialPhase> trial_comps = flash_params.initial_guess.evaluate(eos_idx, vl_eos_name, initial_guess, this->stationary_points);
                (void) stab.run(trial_comps[idx]);

                // If stability test using lightest component converges tested phase, it is identified as vapour; else, liquid
                if (this->compare_stationary_points(trial_comps[idx]))
                {
                    // Vapour phase located
                    sps[j].root = EoS::RootFlag::MAX;
                    for (size_t jj = 0; jj < sps.size(); jj++)
                    {
                        if (jj != j && sps[jj].eos_name == vl_eos_name)
                        {
                            sps[jj].root = EoS::RootFlag::MIN;
                        }
                    }
                    break;
                }
                else
                {
                    sps[j].root = EoS::RootFlag::MIN;
                }
            }
        }
    }

    this->stationary_points = sps;
    this->ref_compositions = {};
    for (int sp_idx: ref_idxs) { ref_compositions.push_back(stationary_points[sp_idx]); }

    return;
}
