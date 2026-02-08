#include <chrono>
#include <iterator>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/initial_guess.hpp"
#include "dartsflash/phase-split/twophasesplit.hpp"
#include "dartsflash/phase-split/multiphasesplit.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"
#include "dartsflash/eos/aq/jager.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"

int test_split2_cubics();
int test_splitN_cubics();
int test_split2_aq_cubics();
int test_split2_aq_cubics_si();

struct Reference
{
	double pressure, temperature, tol{ 1e-5 };
	std::vector<int> eos_idxs;
	std::vector<double> composition, lnK, nu, nu_ref, X, X_ref;
	int np, nc;

	Reference(const double p, const double T, const std::vector<double>& z, const std::vector<double>& lnk, const std::vector<int>& eos_idxs_, const std::vector<double>& nu_, const std::vector<double>& x) 
	{
		pressure = p;
		temperature = T;
		composition = z;
		lnK = lnk;
		eos_idxs = eos_idxs_;
		nu_ref = nu_;
		X_ref = x;
		np = static_cast<int>(nu_.size());
		nc = static_cast<int>(z.size());
		nu.resize(np);
		X.resize(np*nc);
	}

	int test(bool multi, FlashParams& flash_params, bool verbose)
	{
		if (verbose)
		{
			std::cout << "==================================\n";
			print("p, T", {pressure, temperature});
			print("z", composition);
		}

		// Initialize EOS at P, T
		flash_params.init_eos(pressure, temperature);
		flash_params.initial_guess.init(pressure, temperature);
		std::unique_ptr<BaseSplit> split;
		if (multi)
		{
			split = std::make_unique<MultiPhaseSplit>(MultiPhaseSplit(flash_params, np));
		}
		else
		{
			split = std::make_unique<TwoPhaseSplit>(TwoPhaseSplit(flash_params));
		}
		
		// Run split and return nu and x
		std::vector<TrialPhase> trial_comps = {};
    	for (int eos_idx: eos_idxs)
    	{
	        trial_comps.push_back(TrialPhase{eos_idx, flash_params.eos_order[eos_idx], composition});
    	}
		int error = split->run(composition, lnK, trial_comps);  // Run multiphase split algorithm at P, T, z with initial guess lnK
		nu = split->getnu();
		X = split->getx();

		if (error > 0)
		{
			if (!this->negativeflash())
			{
				error = 0;
			}
			else
			{
				print("Error in Flash", error);
				this->print_conditions(flash_params);				
				return error;
			}
		}
		
		// Compare results
		if (verbose)
		{
			std::cout << "\nResults:\n";
			print("nu", nu);
			print("x", X, np);
			std::cout << " Number of flash iterations ssi = " << split->get_ssi_iter() << std::endl;
			std::cout << " Number of flash iterations newton = " << split->get_newton_iter() << std::endl;
			std::cout << " Number of flash total iterations = " << split->get_ssi_iter() + split->get_newton_iter() << std::endl;
		}

		for (int j = 0; j < np; j++)
		{
			if (std::sqrt(std::pow(nu_ref[j]-nu[j], 2)) > tol)
			{
				std::cout << "Different values for nu\n";
				this->print_conditions(flash_params);
				return 1;
			}
			if (X_ref.size() > 0)
			{
				for (int i = 0; i < nc; i++)
				{
					if (std::sqrt(std::pow(X_ref[j*nc + i]-X[j*nc + i], 2)) > tol)
					{
						std::cout << "Different values for X\n";
						this->print_conditions(flash_params);
						return 1;
					}
				}
			}
		}
		if (verbose)
		{
			std::cout << "Output is correct!\n";
		}

		// Check if matrix inverse is correct
		if (split->test_matrices())
		{
			return 1;
		}
		return 0;
	}

	bool negativeflash()
	{
		// Check if there are any negative phase fractions
		for (double nu_j: nu)
		{
			if (nu_j < 0.) { return true; }
		}
		return false;
	}

	void print_conditions(FlashParams& flash_params)
	{
		print("p, T", {pressure, temperature});
		print("z", composition);
		print("nu", nu);
		print("nu_ref", nu_ref);
		print("X", X);
		print("vars", flash_params.split_variables);
	}

	void write_ref(std::string& ref_string)
	{
		std::string p_str = std::to_string(pressure);
		std::string t_str = std::to_string(temperature);

		std::ostringstream z_str_, lnk_str_, eos_str_, nu_str_, x_str_;

    	// Convert all but the last element to avoid a trailing ", "
	    std::copy(composition.begin(), composition.end()-1, std::ostream_iterator<double>(z_str_, ", "));
		std::copy(lnK.begin(), lnK.end()-1, std::ostream_iterator<double>(lnk_str_, ", "));
		std::copy(eos_idxs.begin(), eos_idxs.end()-1, std::ostream_iterator<int>(eos_str_, "\", \""));
		std::copy(nu.begin(), nu.end()-1, std::ostream_iterator<double>(nu_str_, ", "));
		std::copy(X.begin(), X.end()-1, std::ostream_iterator<double>(x_str_, ", "));

    	// Now add the last element with no delimiter
    	z_str_ << composition.back();
		lnk_str_ << lnK.back();
		eos_str_ << eos_idxs.back();
		nu_str_ << nu.back();
		x_str_ << X.back();

		// Add curly brackets front and back
		std::string z_str = "{" + z_str_.str() + "}";
		std::string lnk_str = "{" + lnk_str_.str() + "}";
		std::string eos_str = "{\"" + eos_str_.str() + "\"}";
		std::string nu_str = "{" + nu_str_.str() + "}";
		std::string x_str = "{" + x_str_.str() + "}";
		ref_string += "\t\t";
		ref_string += ("Reference(" + p_str + ", " + t_str + ", " + z_str + ", " + lnk_str + ", " + eos_str + ", " + nu_str + ", " + x_str + "),");
		ref_string += "\n";
		return;
	}
};

int main()
{
	int error_output = 0;

	error_output += test_split2_cubics();
	error_output += test_splitN_cubics();
	// error_output += test_split2_aq_cubics();
	error_output += test_split2_aq_cubics_si();

    return error_output;
}

int test_split2_cubics()
{
	// Test phase split for Vapour-Liquid mixture with PR
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
    int error_output = 0;

   	std::vector<std::string> comp{"C1", "C2", "C3", "nC5", "nC7", "nC10"};
	CompData comp_data(comp);
	comp_data.Pc = {45.99, 48.72, 42.48, 33.70, 27.40, 21.10};
	comp_data.Tc = {190.56, 305.32, 369.83, 469.70, 540.20, 617.70};
	comp_data.ac = {0.011, 0.099, 0.152, 0.252, 0.350, 0.490};
	comp_data.kij = std::vector<double>(6*6, 0.);

	std::vector<double> z = {0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244};

	CubicEoS pr(comp_data, CubicEoS::PR);

	FlashParams flash_params(comp_data);
    flash_params.split_switch_tol = 1e-3;
	flash_params.split_loose_tol_multiplier = 1.;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	std::vector<int> eos{0, 0};

	std::vector<Reference> references = {
		// Wilson K-values
		Reference(220., 335., z, {-0.776924666, 0.9843790174, 2.288137786, 4.580984519, 6.526154563, 9.100268473}, eos, {0.8884390842, 0.1115609158},
								 {0.8221423089, 0.05614761493, 0.02979141223, 0.04262333649, 0.0293377959, 0.01995753152, 
								  0.7106130261, 0.06020266471, 0.0370393607, 0.07020166435, 0.06216474137, 0.05977854273}),
		// Stationary point K-values
		Reference(220., 335., z, {-0.1534524683, 0.05750830524, 0.2004291414, 0.4707291258, 0.7106977244, 1.037107362}, eos, {0.8884390842, 0.1115609158},
								 {0.8221423089, 0.05614761493, 0.02979141223, 0.04262333649, 0.0293377959, 0.01995753152, 
								  0.7106130261, 0.06020266471, 0.0370393607, 0.07020166435, 0.06216474137, 0.05977854273})
	};

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			for (Reference condition: references)
			{
				error_output += condition.test(false, flash_params, verbose);
			}
		}

		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			for (Reference condition: references)
			{
				error_output += condition.test(true, flash_params, verbose);
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_split2_cubics(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_split2_cubics(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_splitN_cubics()
{
	// Test Maljamar separator mixture (Orr, 1981), data from (Li, 2012)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH\n" : "");
	int error_output = 0;

    std::vector<std::string> comp = {"CO2", "C5-7", "C8-10", "C11-14", "C15-20", "C21-28", "C29+"};
	int nc = 7;

    CompData comp_data(comp);
    comp_data.Pc = {73.9, 28.8, 23.7, 18.6, 14.8, 12.0, 8.5};
    comp_data.Tc = {304.2, 516.7, 590.0, 668.6, 745.8, 812.7, 914.9};
    comp_data.ac = {0.225, 0.265, 0.364, 0.499, 0.661, 0.877, 1.279};
    comp_data.kij = std::vector<double>(7*7, 0.);
    comp_data.set_binary_coefficients(0, {0.0, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115});

	std::vector<double> z_init = {0.0, 0.2354, 0.3295, 0.1713, 0.1099, 0.0574, 0.0965};

	FlashParams flash_params(comp_data);
    flash_params.split_switch_tol = 1e-5;
	flash_params.split_loose_tol_multiplier = 1.;
	flash_params.stability_variables = FlashParams::alpha;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson, 
										  		  InitialGuess::Yi::Wilson13,
												  0};  // pure CO2 initial guess
	std::vector<int> eos = {0, 0, 0};

	std::vector<std::pair<Reference, double>> references = {
		// Stationary point K-values
		{Reference(68.5, 305.35, z_init, {0.3834965448, -2.785241313, -4.104610373, -5.900809482, -8.210331241, -11.22148172, -17.70290258,
										  0.3396862351, -1.174919545, -1.652213599, -2.301283461, -3.135738997, -4.22438433, -6.566479715}, 
										  eos, {0.2738025267, 0.5856075977, 0.1405898757}, {}), 0.90},
	};
    
	for (size_t j = 0; j < references.size(); j++)
	{
		references[j].first.composition[0] = references[j].second;
    	for (int i = 1; i < nc; i++)
    	{
        	references[j].first.composition[i] = z_init[i]*(1.0-references[j].second);
    	}
	}

	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (FlashParams::SplitVars var: vars)
	{
		flash_params.split_variables = var;
		for (std::pair<Reference, double> condition: references)
		{
			error_output += condition.first.test(true, flash_params, verbose);
		}
	}
	
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_splitN_cubics(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_splitN_cubics(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_split2_aq_cubics()
{
	// Test phase split for Brine-Vapour mixture with PR and AQ
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	int error_output = 0;

	std::vector<std::string> comp = {"H2O", "CO2", "C1"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 73.75, 46.04};
	comp_data.Tc = {647.14, 304.10, 190.58};
	comp_data.ac = {0.328, 0.239, 0.012};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.47893});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.0936});
	comp_data.Mw = {18.015, 44.01, 16.043};

	CubicEoS pr(comp_data, CubicEoS::PR);

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.9, 1.});

	FlashParams flash_params(comp_data);
	flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-10;
	flash_params.split_loose_tol_multiplier = 1.;
	flash_params.rr2_tol = 1e-15;
	flash_params.min_z = 1e-13;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	flash_params.add_eos("AQ", &aq);
	flash_params.set_eos_order({"AQ", "PR"});
	std::vector<int> eos = {0, 1};

	std::vector<std::vector<double>> zero = {{1.-1e-11, 1e-11, 0.}, {1e-11, 1.-1e-11, 0.}};
	std::vector<double> min_z = {1e-10, 1e-5};
	std::vector<Reference> references = {
		// Henry's law K-values
		// Freezing point
		Reference(1.000000, 273.150000, {1e-10, 0.5, 0.5}, {-5.10131, 6.78637, 10.0999}, eos, {-0.0065869, 1.00659}, {0.999319, 0.000658101, 2.23938e-05, 0.00653934, 0.496732, 0.496728}),
		Reference(1.000000, 273.150000, {1e-05, 0.5, 0.49999}, {-5.10131, 6.78637, 10.0999}, eos, {-0.00657683, 1.00658}, {0.999319, 0.000658108, 2.23936e-05, 0.00653934, 0.496737, 0.496723}),
		Reference(1.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {-5.10131, 6.78637, 10.0999}, eos, {1.00005, -4.50631e-05}, {0.999955, 5.17557e-11, 4.47674e-05, 0.0065176, 3.9013e-08, 0.993482}),
		Reference(1.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {-5.10131, 6.78637, 10.0999}, eos, {1.00004, -4.00103e-05}, {0.99995, 5.15528e-06, 4.45923e-05, 0.00651776, 0.00388605, 0.989596}),
		Reference(1.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {-5.10131, 6.78637, 10.0999}, eos, {-0.00660744, 1.00661}, {0.998684, 0.0013156, 2.24245e-15, 0.00655543, 0.993445, 4.96718e-11}),
		Reference(1.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {-5.10131, 6.78637, 10.0999}, eos, {-0.0066024, 1.0066}, {0.998684, 0.0013156, 2.24246e-10, 0.00655543, 0.99344, 4.96721e-06}),
		Reference(1.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {-5.10131, 6.78637, 10.0999}, eos, {-0.00656066, 1.00656}, {0.999955, 6.58995e-14, 4.47674e-05, 0.0065176, 4.96745e-11, 0.993482}),
		Reference(1.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {-5.10131, 6.78637, 10.0999}, eos, {-0.00655562, 1.00656}, {0.999955, 6.58999e-09, 4.47672e-05, 0.0065176, 4.96748e-06, 0.993477}),
		Reference(1.000000, 273.150000, {0.5, 0.25, 0.25}, {-5.10131, 6.78637, 10.0999}, eos, {0.497049, 0.502951}, {0.99932, 0.000657683, 2.24081e-05, 0.00653933, 0.496416, 0.497044}),
		Reference(1.000000, 273.150000, {0.8, 0.1, 0.1}, {-5.10131, 6.78637, 10.0999}, eos, {0.79923, 0.20077}, {0.999321, 0.000656428, 2.24508e-05, 0.0065393, 0.495468, 0.497992}),
		Reference(1.000000, 273.150000, {0.9, 0.05, 0.05}, {-5.10131, 6.78637, 10.0999}, eos, {0.899955, 0.100045}, {0.999323, 0.000654337, 2.25219e-05, 0.00653925, 0.493889, 0.499572}),
		// Reference(1., 273.15, zero[0], {-5.101313553, 6.786366766, 10.09988994}, eos, {1., 0.}, {}),
		// Reference(1., 273.15, zero[1], {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		Reference(10.000000, 273.150000, {1e-10, 0.5, 0.5}, {-7.4039, 4.48378, 7.7973}, eos, {-0.00071291, 1.00071}, {0.993638, 0.00614723, 0.000217267, 0.000707869, 0.499648, 0.499644}),
		Reference(10.000000, 273.150000, {1e-05, 0.5, 0.49999}, {-7.4039, 4.48378, 7.7973}, eos, {-0.000702839, 1.0007}, {0.993635, 0.00614729, 0.000217265, 0.000707868, 0.499653, 0.499639}),
		Reference(10.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {-7.4039, 4.48378, 7.7973}, eos, {1.00043, -0.000432917}, {0.999568, 5.17734e-11, 0.000432434, 0.00068301, 4.14827e-09, 0.999317}),
		Reference(10.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {-7.4039, 4.48378, 7.7973}, eos, {1.00043, -0.000427909}, {0.999563, 5.17522e-06, 0.000432254, 0.000683032, 0.000414663, 0.998902}),
		Reference(10.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {-7.4039, 4.48378, 7.7973}, eos, {-0.000737612, 1.00074}, {0.987765, 0.0122322, 2.20674e-14, 0.000728051, 0.999272, 4.99632e-11}),
		Reference(10.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {-7.4039, 4.48378, 7.7973}, eos, {-0.000732546, 1.00073}, {0.987768, 0.0122321, 2.20675e-09, 0.000728052, 0.999267, 4.99634e-06}),
		Reference(10.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {-7.4039, 4.48378, 7.7973}, eos, {-0.000683772, 1.00068}, {0.999568, 6.23616e-13, 0.000432434, 0.00068301, 4.99663e-11, 0.999317}),
		Reference(10.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {-7.4039, 4.48378, 7.7973}, eos, {-0.000678767, 1.00068}, {0.999568, 6.23619e-08, 0.000432432, 0.00068301, 4.99665e-06, 0.999312}),
		Reference(10.000000, 273.150000, {0.5, 0.25, 0.25}, {-7.4039, 4.48378, 7.7973}, eos, {0.502831, 0.497169}, {0.993671, 0.0061109, 0.000218551, 0.000707729, 0.496666, 0.502626}),
		Reference(10.000000, 273.150000, {0.8, 0.1, 0.1}, {-7.4039, 4.48378, 7.7973}, eos, {0.804872, 0.195128}, {0.993776, 0.00600199, 0.000222399, 0.000707327, 0.487726, 0.511566}),
		Reference(10.000000, 273.150000, {0.9, 0.05, 0.05}, {-7.4039, 4.48378, 7.7973}, eos, {0.905411, 0.0945892}, {0.99395, 0.00582108, 0.000228789, 0.00070665, 0.472882, 0.526411}),
		// Reference(10., 273.15, zero[0], {-7.403898646, 4.483781673, 7.797304845}, eos, {1., 0.}, {}),
		// Reference(10., 273.15, zero[1], {-7.403898646, 4.483781673, 7.797304845}, eos, {0., 1.}, {}),
		Reference(100.000000, 273.150000, {1e-10, 0.5, 0.5}, {-9.70648, 2.1812, 5.49472}, eos, {-0.000236799, 1.00024}, {0.973452, 0.0248677, 0.00167975, 0.000230458, 0.499888, 0.499882}),
		Reference(100.000000, 273.150000, {1e-05, 0.5, 0.49999}, {-9.70648, 2.1812, 5.49472}, eos, {-0.000226527, 1.00023}, {0.973452, 0.0248679, 0.00167974, 0.000230461, 0.499892, 0.499877}),
		Reference(100.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {-9.70648, 2.1812, 5.49472}, eos, {1.00296, -0.00295863}, {0.99705, 5.21427e-11, 0.00294957, 0.000112014, 7.76356e-10, 0.999888}),
		Reference(100.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {-9.70648, 2.1812, 5.49472}, eos, {1.00295, -0.00295361}, {0.997045, 5.2139e-06, 0.00294934, 0.000112023, 7.76332e-05, 0.99981}),
		Reference(100.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {-9.70648, 2.1812, 5.49472}, eos, {-0.00186987, 1.00187}, {0.965308, 0.0346947, 4.92577e-13, 0.00180163, 0.998198, 4.99076e-11}),
		Reference(100.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {-9.70648, 2.1812, 5.49472}, eos, {-0.00186464, 1.00186}, {0.965305, 0.0346946, 4.92574e-08, 0.00180159, 0.998193, 4.99079e-06}),
		Reference(100.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {-9.70648, 2.1812, 5.49472}, eos, {-0.000112358, 1.00011}, {0.997055, 3.35782e-12, 0.00294957, 0.000112015, 4.99948e-11, 0.999888}),
		Reference(100.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {-9.70648, 2.1812, 5.49472}, eos, {-0.000107343, 1.00011}, {0.99705, 3.35782e-07, 0.00294956, 0.000112015, 4.9995e-06, 0.999883}),
		Reference(100.000000, 273.150000, {0.5, 0.25, 0.25}, {-9.70648, 2.1812, 5.49472}, eos, {0.513358, 0.486642}, {0.973767, 0.0245309, 0.00170225, 0.000223582, 0.487847, 0.511929}),
		Reference(100.000000, 273.150000, {0.8, 0.1, 0.1}, {-9.70648, 2.1812, 5.49472}, eos, {0.820591, 0.179409}, {0.974862, 0.0233606, 0.00177732, 0.000205516, 0.450538, 0.549257}),
		Reference(100.000000, 273.150000, {0.9, 0.05, 0.05}, {-9.70648, 2.1812, 5.49472}, eos, {0.921139, 0.0788615}, {0.977036, 0.0210481, 0.00191582, 0.00018275, 0.388172, 0.611645}),
		// Reference(100., 273.15, zero[0], {-9.706483739, 2.18119658, 5.494719752}, eos, {1., 0.}, {}),
		// Reference(100., 273.15, zero[1], {-9.706483739, 2.18119658, 5.494719752}, eos, {0., 1.}, {}),

		// Boiling point
		// Reference(1., 373.15, {min_z[0], 0.5, 0.5-min_z[0]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {min_z[1], 0.5, 0.5-min_z[1]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {1.-min_z[0], 0.5*min_z[0], 0.5*min_z[0]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {1.-min_z[1], 0.5*min_z[1], 0.5*min_z[1]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.5*min_z[0], 1.-min_z[0], 0.5*min_z[0]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.5*min_z[1], 1.-min_z[1], 0.5*min_z[1]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.5*min_z[0], 0.5*min_z[0], 1.-min_z[0]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.5*min_z[1], 0.5*min_z[1], 1.-min_z[1]}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.5, 0.25, 0.25}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.8, 0.1, 0.1}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, {0.9, 0.05, 0.05}, {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, zero[0], {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		// Reference(1., 373.15, zero[1], {-5.101313553, 6.786366766, 10.09988994}, eos, {0., 1.}, {}),
		Reference(10.000000, 373.150000, {1e-10, 0.5, 0.5}, {-2.28458, 6.83843, 9.6614}, eos, {-0.122557, 1.12256}, {0.999022, 0.000907458, 7.02041e-05, 0.10907, 0.445511, 0.445419}),
		Reference(10.000000, 373.150000, {1e-05, 0.5, 0.49999}, {-2.28458, 6.83843, 9.6614}, eos, {-0.122546, 1.12255}, {0.999022, 0.000907467, 7.02034e-05, 0.10907, 0.445515, 0.445415}),
		Reference(10.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {-2.28458, 6.83843, 9.6614}, eos, {1.00016, -0.000157089}, {0.99986, 5.41541e-11, 0.000140169, 0.107573, 2.64983e-08, 0.892427}),
		Reference(10.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {-2.28458, 6.83843, 9.6614}, eos, {1.00015, -0.000151471}, {0.999855, 5.39938e-06, 0.000139754, 0.107583, 0.00264206, 0.889775}),
		Reference(10.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {-2.28458, 6.83843, 9.6614}, eos, {-0.124199, 1.1242}, {0.998186, 0.00181177, 7.05727e-15, 0.110277, 0.889723, 4.44769e-11}),
		Reference(10.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {-2.28458, 6.83843, 9.6614}, eos, {-0.124193, 1.12419}, {0.998188, 0.00181176, 7.05731e-10, 0.110277, 0.889718, 4.44771e-06}),
		Reference(10.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {-2.28458, 6.83843, 9.6614}, eos, {-0.120559, 1.12056}, {0.999862, 9.12103e-14, 0.000140169, 0.107573, 4.46304e-11, 0.892427}),
		Reference(10.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {-2.28458, 6.83843, 9.6614}, eos, {-0.120553, 1.12055}, {0.99986, 9.12107e-09, 0.000140168, 0.107573, 4.46306e-06, 0.892423}),
		Reference(10.000000, 373.150000, {0.5, 0.25, 0.25}, {-2.28458, 6.83843, 9.6614}, eos, {0.439271, 0.560729}, {0.999023, 0.000906699, 7.02628e-05, 0.10907, 0.445138, 0.445793}),
		Reference(10.000000, 373.150000, {0.8, 0.1, 0.1}, {-2.28458, 6.83843, 9.6614}, eos, {0.776366, 0.223634}, {0.999025, 0.000904426, 7.04387e-05, 0.109066, 0.444019, 0.446915}),
		Reference(10.000000, 373.150000, {0.9, 0.05, 0.05}, {-2.28458, 6.83843, 9.6614}, eos, {0.888728, 0.111272}, {0.999029, 0.000900636, 7.07318e-05, 0.10906, 0.442156, 0.448784}),
		// Reference(10., 373.15, zero[0], {-2.284581466, 6.838432915, 9.661403745}, eos, {0., 1.}, {}),
		// Reference(10., 373.15, zero[1], {-2.284581466, 6.838432915, 9.661403745}, eos, {0., 1.}, {}),
		Reference(100.000000, 373.150000, {1e-10, 0.5, 0.5}, {-4.58717, 4.53585, 7.35882}, eos, {-0.0164221, 1.01642}, {0.991978, 0.0073583, 0.000662918, 0.0160272, 0.49204, 0.491932}),
		Reference(100.000000, 373.150000, {1e-05, 0.5, 0.49999}, {-4.58717, 4.53585, 7.35882}, eos, {-0.0164119, 1.01641}, {0.991979, 0.00735837, 0.000662912, 0.0160272, 0.492045, 0.491927}),
		Reference(100.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {-4.58717, 4.53585, 7.35882}, eos, {1.00132, -0.00131624}, {0.998703, 5.44409e-11, 0.00129705, 0.0132877, 3.42833e-09, 0.986712}),
		Reference(100.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {-4.58717, 4.53585, 7.35882}, eos, {1.00131, -0.00131117}, {0.998698, 5.44225e-06, 0.00129659, 0.0132895, 0.000342735, 0.986368}),
		Reference(100.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {-4.58717, 4.53585, 7.35882}, eos, {-0.0202157, 1.02022}, {0.985681, 0.0143203, 7.26423e-14, 0.0195314, 0.980469, 4.90107e-11}),
		Reference(100.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {-4.58717, 4.53585, 7.35882}, eos, {-0.0202105, 1.02021}, {0.98568, 0.0143203, 7.26426e-09, 0.0195313, 0.980464, 4.90109e-06}),
		Reference(100.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {-4.58717, 4.53585, 7.35882}, eos, {-0.0134844, 1.01348}, {0.998701, 7.83587e-13, 0.00129705, 0.0132877, 4.93452e-11, 0.986712}),
		Reference(100.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {-4.58717, 4.53585, 7.35882}, eos, {-0.0134793, 1.01348}, {0.998703, 7.8359e-08, 0.00129704, 0.0132877, 4.93454e-06, 0.986707}),
		Reference(100.000000, 373.150000, {0.5, 0.25, 0.25}, {-4.58717, 4.53585, 7.35882}, eos, {0.495887, 0.504113}, {0.992022, 0.00731098, 0.00066718, 0.0160071, 0.488729, 0.495264}),
		Reference(100.000000, 373.150000, {0.8, 0.1, 0.1}, {-4.58717, 4.53585, 7.35882}, eos, {0.803165, 0.196835}, {0.992151, 0.00716883, 0.000679965, 0.0159467, 0.478788, 0.505265}),
		Reference(100.000000, 373.150000, {0.9, 0.05, 0.05}, {-4.58717, 4.53585, 7.35882}, eos, {0.905412, 0.0945877}, {0.992367, 0.00693205, 0.000701209, 0.0158467, 0.462255, 0.521898}),
		// Reference(100., 373.15, zero[0], {-4.587166559, 4.535847822, 7.358818652}, eos, {0., 1.}, {}),
		// Reference(100., 373.15, zero[1], {-4.587166559, 4.535847822, 7.358818652}, eos, {0., 1.}, {}),

		// Difficult conditions
		// Reference(109.29, 300.1, {0.999, 0.001, 0.}, {-8.030917791, 2.881408754, 6.030547157}, eos, {0., 1.}, {}), // difficult conditions
		Reference(30.000000, 330.000000, {0.6, 0.39, 0.01}, {-5.16189, 4.89882, 7.897}, eos, {0.60191, 0.39809}, {0.991825, 0.00816218, 1.29962e-05, 0.00756364, 0.967336, 0.0251003}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
	
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			for (Reference condition: references)
			{
				error_output += condition.test(false, flash_params, verbose);
				if (write) { condition.write_ref(ref_string); }
			}
			write = false;
		}
	}
	ref_string += "\t};\n";

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_split2_aq_cubics(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_split2_aq_cubics(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_split2_aq_cubics_si()
{
	// Test phase split for Brine-Vapour-Hydrate mixture with SRK-AQ-sI
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	int error_output = 0;

	std::vector<std::string> comp = {"H2O", "CO2"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 73.75};
	comp_data.Tc = {647.14, 304.10};
	comp_data.ac = {0.328, 0.239};
	comp_data.kij = std::vector<double>(2*2, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014});
	comp_data.Mw = {18.015, 44.01};

	FlashParams flash_params(comp_data);
	
	CubicEoS ceos(comp_data, CubicEoS::SRK);
	ceos.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("CEOS", &ceos);

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});
	flash_params.add_eos("AQ", &aq);

	Ballard si(comp_data, "sI");
	flash_params.add_eos("sI", &si);

	flash_params.split_tol = 1e-20;
	flash_params.split_switch_tol = 1e1;
	flash_params.verbose = verbose;

	flash_params.set_eos_order({"AQ", "sI", "CEOS"});

	std::vector<Reference> references = {
		// Reference(1.000000, 273.150000, {0.01, 0.99}, {4., -2}, {2, 1}, {1.00690561, -0.006905610333}, {0.00873859, 0.991261, 0.914933, 0.0850673}),
		Reference(1.000000, 273.150000, {0.1, 0.9}, {4, -2}, {2, 1}, {0.899287, 0.100713}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.2, 0.8}, {4, -2}, {2, 1}, {0.788934, 0.211066}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.3, 0.7}, {4, -2}, {2, 1}, {0.678581, 0.321419}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.4, 0.6}, {4, -2}, {2, 1}, {0.568228, 0.431772}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.5, 0.5}, {4, -2}, {2, 1}, {0.457874, 0.542126}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.6, 0.4}, {4, -2}, {2, 1}, {0.347521, 0.652479}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.7, 0.3}, {4, -2}, {2, 1}, {0.237168, 0.762832}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.8, 0.2}, {4, -2}, {2, 1}, {0.126815, 0.873185}, {0.00873624, 0.991264, 0.914917, 0.0850831}),
		Reference(1.000000, 273.150000, {0.9, 0.1}, {-0.1, 1.5}, {0, 1}, {0.261029, 0.738971}, {0.985345, 0.0146554, 0.869853, 0.130147}),
		Reference(1.000000, 273.150000, {0.95, 0.05}, {-0.1, 1.5}, {0, 1}, {0.693962, 0.306038}, {0.985345, 0.0146554, 0.869853, 0.130147}),
		// Reference(1.000000, 273.150000, {0.98, 0.02}, {-0.1, 1.5}, {0, 1}, {0.953711, 0.0462892}, {0.985346, 0.0146539, 0.869854, 0.130146}),
		// Reference(10.000000, 273.150000, {0.01, 0.99}, {4, -2}, {2, 1}, {0.989378, 0.0106225}, {0.000745899, 0.999254, 0.871925, 0.128075}),
		Reference(10.000000, 273.150000, {0.1, 0.9}, {4, -2}, {2, 1}, {0.886057, 0.113943}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.2, 0.8}, {4, -2}, {2, 1}, {0.771261, 0.228739}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.3, 0.7}, {4, -2}, {2, 1}, {0.656465, 0.343535}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.4, 0.6}, {4, -2}, {2, 1}, {0.541669, 0.458331}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.5, 0.5}, {4, -2}, {2, 1}, {0.426872, 0.573128}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.6, 0.4}, {4, -2}, {2, 1}, {0.312076, 0.687924}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.7, 0.3}, {4, -2}, {2, 1}, {0.19728, 0.80272}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.8, 0.2}, {4, -2}, {2, 1}, {0.0824833, 0.917517}, {0.00074373, 0.999256, 0.871852, 0.128148}),
		Reference(10.000000, 273.150000, {0.9, 0.1}, {-0.1, 1.5}, {0, 1}, {0.261887, 0.738113}, {0.985364, 0.0146364, 0.869713, 0.130287}),
		Reference(10.000000, 273.150000, {0.95, 0.05}, {-0.1, 1.5}, {0, 1}, {0.694222, 0.305778}, {0.985364, 0.0146364, 0.869713, 0.130287}),
		// Reference(10.000000, 273.150000, {0.98, 0.02}, {-0.1, 1.5}, {0, 1}, {0.953506, 0.0464942}, {0.985378, 0.0146223, 0.869714, 0.130286}),
		// Reference(100.000000, 273.150000, {0.01, 0.99}, {4, -2}, {2, 1}, {0.990218, 0.00978235}, {0.00159938, 0.998401, 0.860352, 0.139648}),
		Reference(100.000000, 273.150000, {0.1, 0.9}, {4, -2}, {2, 1}, {0.885421, 0.114579}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.2, 0.8}, {4, -2}, {2, 1}, {0.768926, 0.231074}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.3, 0.7}, {4, -2}, {2, 1}, {0.652431, 0.347569}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.4, 0.6}, {4, -2}, {2, 1}, {0.535936, 0.464064}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.5, 0.5}, {4, -2}, {2, 1}, {0.419442, 0.580558}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.6, 0.4}, {4, -2}, {2, 1}, {0.302947, 0.697053}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.7, 0.3}, {4, -2}, {2, 1}, {0.186452, 0.813548}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.8, 0.2}, {4, -2}, {2, 1}, {0.0699571, 0.930043}, {0.00164454, 0.998355, 0.860052, 0.139948}),
		Reference(100.000000, 273.150000, {0.9, 0.1}, {-0.1, 1.5}, {0, 1}, {0.270083, 0.729917}, {0.985546, 0.0144542, 0.868346, 0.131654}),
		Reference(100.000000, 273.150000, {0.95, 0.05}, {-0.1, 1.5}, {0, 1}, {0.696707, 0.303293}, {0.985546, 0.0144542, 0.868346, 0.131654}),
		// Reference(100.000000, 273.150000, {0.98, 0.02}, {-0.1, 1.5}, {0, 1}, {0.951574, 0.0484256}, {0.985682, 0.0143185, 0.868357, 0.131643}),

		// Reference(40.000000, 273.150000, {0.444445, 0.555555}, {-7.88278, 1.9801}, {"sI", "CEOS"}, {0.515319, 0.484681}, {0.861027, 0.138973, 0.00152909, 0.998471}),
		Reference(100.000000, 273.150000, {0.2, 0.8}, {-5.78272, 3.30926}, {0, 2}, {0.205986, 0.794014}, {0.963759, 0.0362411, 0.00186309, 0.998137}),
		Reference(100.000000, 273.150000, {0.2, 0.8}, {-6.19304, 3.38109}, {0, 2}, {0.205986, 0.794014}, {0.963759, 0.0362411, 0.00186309, 0.998137}),
		// Reference(40.000000, 273.150000, {0.7499995, 0.2500005}, {6.322482873, -3.383576654}, {"CEOS", "AQ"}, {0.2237277372, 0.7762722628}, {0.001717606562, 0.9982823934, 0.9656601939, 0.03433980615}),
		// Reference(34.35897436, 273.150000, {0.869345995, 0.130654005}, {-0.1254624623, 2.196417905}, {0, 1}, {}, {}),
	};

	// references = {
	// 	Reference(34.35897436, 273.150000, {0.869345995, 0.130654005}, {-0.1254624623, 2.196417905}, {0, 1}, {}, {}),
	// 	Reference(73.33333, 273.150000, {0.2160809698, 0.7839190302}, {-6.147787584 1.975140779}, {"sI", "CEOS"}, {}, {})
	// };

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
	
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			for (Reference condition: references)
			{
				error_output += condition.test(false, flash_params, verbose);
				if (write) { condition.write_ref(ref_string); }
			}
			write = false;
		}
	}
	ref_string += "\t};\n";

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_split2_aq_cubics_si(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_split2_aq_cubics_si(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}
