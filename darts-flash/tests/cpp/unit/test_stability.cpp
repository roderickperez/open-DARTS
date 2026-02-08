#include <chrono>
#include <cassert>
#include <iterator>
#include "dartsflash/global/global.hpp"
#include "dartsflash/stability/stability.hpp"
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/initial_guess.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"
#include "dartsflash/eos/aq/aq.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"

int test_stability_vapour_liquid();
int test_stability_vapour_liquid_water();
int test_stability_vapour_brine();
int test_stability_vapour_brine_hydrate();
int test_stationary_points_vapour_liquid();

struct Reference
{
	double pressure, temperature, tpd_tol{ 1e-3 };
	std::vector<double> X, tpd_ref, tpd;
	std::vector<std::string> eos_ref;

	Reference(const double p, const double T, const std::vector<double>& x, const std::vector<double>& tpd_ref_)
	: pressure(p), temperature(T), X(x), tpd_ref(tpd_ref_) {}

	int test(FlashParams& flash_params, bool verbose)
	{
		if (verbose) { print_conditions(); }
		flash_params.init_eos(pressure, temperature);
		Stability stability(flash_params);

		// Find reference compositions - hypothetical single phase
		std::vector<TrialPhase> ref_comps = {flash_params.find_ref_comp(pressure, temperature, X)};
		stability.init(ref_comps);

		tpd = {};
		for (size_t j = 0; j < flash_params.eos_order.size(); j++)
		{
			std::string eosname = flash_params.eos_order[j];
        	std::vector<TrialPhase> trial_comps = flash_params.get_trial_comps(j, eosname, ref_comps);
			for (TrialPhase trial_comp: trial_comps)
			{
				if (verbose)
				{
					trial_comp.print_point("\nTrialPhase");
				}

				int error = stability.run(trial_comp);
				if (error > 0)
				{
					print("Error in Stability", error);
					return error;
				}
			
				// Output and compare results
				tpd.push_back(trial_comp.tpd);
				if (verbose)
				{
					std::cout << "\nResults:\n";
					trial_comp.print_point();
					std::cout << " Number of stability iterations ssi = " << stability.get_ssi_iter() << std::endl;
					std::cout << " Number of stability iterations newton = " << stability.get_newton_iter() << std::endl;
					std::cout << " Number of stability total iterations = " << stability.get_ssi_iter() + stability.get_newton_iter() << std::endl;
				}
			}
		}

		if (tpd.size() != tpd_ref.size())
		{
			print_conditions();
			std::cout << "tpd and tpd_ref are not the same size\n";
			print("tpd", tpd);
			print("tpd_ref", tpd_ref);
			return 1;
		}
		if (this->compare_tpd())
		{
			std::cout << "Different values for tpd\n";
			print("p, T", {pressure, temperature});
			print("X", X);
			print("tpd", tpd);
			print("tpd_ref", tpd_ref);
			return 1;
		}
		if (verbose)
		{
			std::cout << "Output is correct!\n";
		}
		
		// Check if matrix inverse is correct
		if (stability.test_matrices())
		{
			return 1;
		}
		return 0;
	}

	int compare_tpd()
	{
		std::sort(tpd.begin(), tpd.end());
		std::sort(tpd_ref.begin(), tpd_ref.end());

		for (size_t j = 0; j < tpd.size(); j++)
		{
			double lntpdj = std::log(std::abs(tpd[j]) + 1e-15);
			// For small tpd difference (|tpd| < 1), compare absolute difference; for large tpd values, logarithmic scale is used to compare
			double tpd_diff = lntpdj < 0. ? std::fabs(tpd[j]-tpd_ref[j]) : std::fabs(std::log(std::fabs(tpd_ref[j]) + 1e-15) - lntpdj);
			if (tpd_diff > tpd_tol)
			{
				print("tpd diff", tpd_diff);
				return 1;
			}
		}
		return 0;
	}

	void print_conditions()
	{
		std::cout << "==================================\n";
		print("p, T", {pressure, temperature});
		print("X", X);
	}

	void write_ref(std::string& ref_string)
	{
		std::string p_str = std::to_string(pressure);
		std::string t_str = std::to_string(temperature);

		for (size_t i = 0; i < tpd.size(); i++)
		{
			tpd[i] = std::fabs(tpd[i]) < 1e-12 ? 0. : tpd[i];
		}

		std::ostringstream z_str_, tpd_str_;

    	// Convert all but the last element to avoid a trailing ", "
	    std::copy(X.begin(), X.end()-1, std::ostream_iterator<double>(z_str_, ", "));
		std::copy(tpd.begin(), tpd.end()-1, std::ostream_iterator<double>(tpd_str_, ", "));

    	// Now add the last element with no delimiter
    	z_str_ << X.back();
		tpd_str_ << tpd.back();

		// Add curly brackets front and back
		std::string z_str = "{" + z_str_.str() + "}";
		std::string tpd_str = "{" + tpd_str_.str() + "}";

		ref_string += "\t\t";
		ref_string += ("Reference(" + p_str + ", " + t_str + ", " + z_str + ", " + tpd_str + "),");
		ref_string += "\n";
		return;
	}
};

int main()
{
	int error_output = 0;

	error_output += test_stability_vapour_liquid();
	error_output += test_stability_vapour_liquid_water();
	error_output += test_stability_vapour_brine();
	error_output += test_stability_vapour_brine_hydrate();
	error_output += test_stationary_points_vapour_liquid();

    return error_output;
}

int test_stability_vapour_liquid()
{
	// Test stationary points for Y8 mixture with PR
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING STATIONARY POINTS FOR Y8\n" : "");
    int error_output = 0;

	std::vector<std::string> comp{"C1", "C2", "C3", "C4", "C5", "C6"};
	std::vector<double> x = {0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244};

	CompData comp_data(comp);
	comp_data.Pc = {45.99, 48.72, 42.48, 33.70, 27.40, 21.10};
	comp_data.Tc = {190.56, 305.32, 369.83, 469.70, 540.20, 617.70};
	comp_data.ac = {0.011, 0.099, 0.152, 0.252, 0.350, 0.490};
	comp_data.kij = std::vector<double>(6*6, 0.);

	FlashParams flash_params(comp_data);
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;

	std::vector<Reference> references = {
		Reference(220., 335., x, {-0.0010297, 0.}),
		Reference(10., 210., x, {-168623.034, 0.}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::StabilityVars> vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;
	
		for (FlashParams::StabilityVars var: vars)
		{
			flash_params.stability_variables = var;

			for (Reference condition: references)
			{
				error_output += condition.test(flash_params, verbose);
				if (write) { condition.write_ref(ref_string); }
			}
			write = false;
		}
	}
	ref_string += "\t};\n";

	// Test stationary points for Maljamar separator mixture (Orr, 1981), data from (Li, 2012) with PR
	std::cout << (verbose ? "TESTING STATIONARY POINTS FOR MALJAMAR SEPARATOR MIXTURE\n" : "");
	comp = {"CO2", "C5-7", "C8-10", "C11-14", "C15-20", "C21-28", "C29+"};
	int nc = 7;

    comp_data = CompData(comp);
    comp_data.Pc = {73.9, 28.8, 23.7, 18.6, 14.8, 12.0, 8.5};
    comp_data.Tc = {304.2, 516.7, 590.0, 668.6, 745.8, 812.7, 914.9};
    comp_data.ac = {0.225, 0.265, 0.364, 0.499, 0.661, 0.877, 1.279};
    comp_data.kij = std::vector<double>(7*7, 0.);
    comp_data.set_binary_coefficients(0, {0.0, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115});

	std::vector<double> z_init = {0.0, 0.2354, 0.3295, 0.1713, 0.1099, 0.0574, 0.0965};

	flash_params = FlashParams(comp_data);
	flash_params.stability_variables = FlashParams::alpha;
	// flash_params.verbose = true;

	pr = CubicEoS(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson,
										  		  InitialGuess::Yi::Wilson13,
												  0};  // pure CO2 initial guess
	
	references = {
		Reference(64., 305.35, z_init, {0., -0.003364121438, 0.,  0.01355346099, -0.003364121438}),
		Reference(68.5, 305.35, z_init, {-0.2834472399, -0.02788030935, -0.2834472399, -0.02788030935, -0.02788030935})
	};

	std::vector<double> zCO2 = {0.65, 0.9};
	for (size_t j = 0; j < zCO2.size(); j++)
	{
		references[j].X[0] = zCO2[j];
    	for (int i = 1; i < nc; i++)
    	{
        	references[j].X[i] = z_init[i]*(1.0-zCO2[j]);
    	}
	}

	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;
	
		for (FlashParams::StabilityVars var: vars)
		{
			flash_params.stability_variables = var;

			for (Reference condition: references)
			{
				error_output += condition.test(flash_params, verbose);
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_stability_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stability_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stability_vapour_liquid_water()
{
	// Test stationary points for NWE OIL mixture with PR
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING STATIONARY POINTS FOR NWE OIL\n" : "");
    int error_output = 0;
	std::vector<std::string> comp = {"H2O","CO2", "C1", "C2-3", "C4-6", "C7-14", "C15-24", "C25+"};
	CompData comp_data = CompData(comp);
	comp_data.Pc = {220.48, 73.76, 46.0, 45.05, 33.50,24.24, 18.03, 17.26};
	comp_data.Tc = {647.3, 304.2, 190.6, 343.64, 466.41, 603.07, 733.79, 923.2};
	comp_data.ac = {0.344, 0.225, 0.008, 0.130, 0.244, 0.6, 0.903, 1.229};
	comp_data.kij = std::vector<double>(8*8, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.1896, 0.4850, 0.5, 0.5, 0.5, 0.5, 0.5});
	comp_data.set_binary_coefficients(1, {0.1896, 0., 0.12, 0.12, 0.12, 0.09, 0.09, 0.09});

	std::vector<double> x = {0.5, 0.251925, 0.050625, 0.02950, 0.0371, 0.071575, 0.03725, 0.022025};

	FlashParams flash_params(comp_data);
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-3;

	std::vector<Reference> references = {
		Reference(400.000000, 600.000000, {0.5, 0.251925, 0.050625, 0.0295, 0.0371, 0.071575, 0.03725, 0.022025}, {-0.0443709, -0.0888223}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::StabilityVars> vars = {FlashParams::Y, FlashParams::alpha};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;
	
		for (FlashParams::StabilityVars var: vars)
		{
			flash_params.stability_variables = var;
			for (Reference condition: references)
			{
				error_output += condition.test(flash_params, verbose);
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
		std::cout << "Errors occurred in test_stability_vapour_liquid_water(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stability_vapour_liquid_water(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stability_vapour_brine()
{
	// Test stationary points for Brine-Vapour mixture with PR and AQ
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING STATIONARY POINTS FOR PR-AQ\n" : "");
	int error_output = 0;

	// Test ternary mixture H2O-CO2-C1
	std::vector<std::string> comp = {"H2O", "CO2", "C1"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 73.75, 46.04};
	comp_data.Tc = {647.14, 304.10, 190.58};
	comp_data.ac = {0.328, 0.239, 0.012};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.47893});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.0936});
	comp_data.Mw = {18.015, 44.01, 16.043};

	FlashParams flash_params(comp_data);
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	pr.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {0, 1, 2};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-2;
	flash_params.eos_params["PR"]->stability_max_iter = 500;

	std::map<AQEoS::CompType, AQEoS::Model> aq_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
    AQEoS aq(comp_data, aq_map);
	aq.set_eos_range(0, {0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {0};
	flash_params.eos_params["AQ"]->stability_switch_tol = 1e-10;
	flash_params.eos_params["AQ"]->stability_max_iter = 10;

	std::vector<Reference> references = {
		Reference(1.000000, 273.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(1.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 0.99795}),
		Reference(1.000000, 273.150000, {1, 5e-16, 5e-16}, {0, 0.993388}),
		Reference(1.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {0, 0.878747, 0.878747, 0.878747}),
		Reference(1.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 0.999767}),
		Reference(1.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 0.999186}),
		Reference(1.000000, 273.150000, {0.58, 0.21, 0.21}, {-86.8843, 0, 0, 0}),
		Reference(1.000000, 273.150000, {0.6, 0.2, 0.2}, {-89.8989, 0, 0, 0}),
		Reference(10.000000, 273.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(10.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 0.981133}),
		Reference(10.000000, 273.150000, {1, 5e-16, 5e-16}, {0, 0.999194}),
		Reference(10.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {0, 0.987354, 0.987354, 0.987354}),
		Reference(10.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 0.989081}),
		Reference(10.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 0.992236}),
		Reference(10.000000, 273.150000, {0.58, 0.21, 0.21}, {-222.499, -222.499, -222.499, -4.32584}),
		Reference(10.000000, 273.150000, {0.6, 0.2, 0.2}, {-477.217, -477.217, -477.217, 0}),
		Reference(100.000000, 273.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(100.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0.938737, 0.952345}),
		Reference(100.000000, 273.150000, {1, 5e-16, 5e-16}, {-0.125364, 0}),
		Reference(100.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {-0.125353, -0.125353, 0, 0.998095}),
		Reference(100.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 1.00754}),
		Reference(100.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0.949897, 0.952435}),
		Reference(100.000000, 273.150000, {0.58, 0.21, 0.21}, {-33.4303, -33.4303, -4.82642, -4.3151}),
		Reference(100.000000, 273.150000, {0.6, 0.2, 0.2}, {-69.8093, -69.8093, 0, 0.36216}),
		Reference(1.000000, 373.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(1.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 1.00433}),
		Reference(1.000000, 373.150000, {1, 5e-16, 5e-16}, {0, 0.043646}),
		Reference(1.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0, 0, 0, 0.0436556}),
		Reference(1.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 1.00879}),
		Reference(1.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 1.00021}),
		Reference(1.000000, 373.150000, {0.58, 0.21, 0.21}, {0, 0, 0, 0.444785}),
		Reference(1.000000, 373.150000, {0.6, 0.2, 0.2}, {0, 0, 0, 0.42569}),
		Reference(10.000000, 373.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(10.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 1.0118}),
		Reference(10.000000, 373.150000, {1, 5e-16, 5e-16}, {0, 0.886187}),
		Reference(10.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0, 0.853735, 0.853735, 0.853735}),
		Reference(10.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 1.02136}),
		Reference(10.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 1.00067}),
		Reference(10.000000, 373.150000, {0.58, 0.21, 0.21}, {-4.14765, 0, 0, 0}),
		Reference(10.000000, 373.150000, {0.6, 0.2, 0.2}, {-4.32039, 0, 0, 0}),
		Reference(100.000000, 373.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(100.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 1.0249}),
		Reference(100.000000, 373.150000, {1, 5e-16, 5e-16}, {0, 0.000258183}),
		Reference(100.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0, 0.000267661, 0.000267661, 0.000267661}),
		Reference(100.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 1.04283}),
		Reference(100.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 1.00102}),
		Reference(100.000000, 373.150000, {0.58, 0.21, 0.21}, {-4.45439, -4.45439, -1.98222, -1.96542}),
		Reference(100.000000, 373.150000, {0.6, 0.2, 0.2}, {-5.68804, -5.68804, -1.67046, -1.65118}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::StabilityVars> vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;
	
		for (FlashParams::StabilityVars var: vars)
		{
			flash_params.stability_variables = var;

			for (Reference condition: references)
			{
				error_output += condition.test(flash_params, verbose);
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
		std::cout << "Errors occurred in test_stability_vapour_brine(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stability_vapour_brine(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stability_vapour_brine_hydrate()
{
	// Test stationary points for Brine-Vapour mixture with PR, AQ and VDWP
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING STATIONARY POINTS FOR PR-AQ-VDWP\n" : "");
	int error_output = 0;

	// Binary mixture H2O-CO2
	std::vector<std::string> comp = {"H2O", "CO2"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 73.75};
	comp_data.Tc = {647.14, 304.10};
	comp_data.ac = {0.328, 0.239};
	comp_data.kij = std::vector<double>(2*2, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014});
	comp_data.Mw = {18.015, 44.01};

	FlashParams flash_params(comp_data);
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	pr.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {0, 1};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-2;

	std::map<AQEoS::CompType, AQEoS::Model> aq_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
    AQEoS aq(comp_data, aq_map);
	aq.set_eos_range(0, {0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {0};
	flash_params.eos_params["AQ"]->stability_switch_tol = 1e-10;
	flash_params.eos_params["AQ"]->stability_max_iter = 10;

	Ballard si(comp_data, "sI");
	flash_params.add_eos("sI", &si);
	flash_params.eos_params["sI"]->trial_comps = {0};
	flash_params.eos_params["sI"]->stability_max_iter = 20;

	Ballard sii(comp_data, "sII");
	flash_params.add_eos("sII", &sii);
	flash_params.eos_params["sII"]->trial_comps = {0};
	flash_params.eos_params["sII"]->stability_max_iter = 20;
	std::vector<Reference> references = {
		Reference(1.000000, 273.150000, {1e-15, 1}, {0}),
		Reference(1.000000, 273.150000, {1e-05, 0.99999}, {0.997695, 0.997348, 0.997898, 0, 0}),
		Reference(1.000000, 273.150000, {1, 1e-15}, {0.32434, 0.355175, 0, 0.993388}),
		Reference(1.000000, 273.150000, {1, 1e-10}, {0.32434, 0.355175, 0, 0.993388, 0.993388}),
		Reference(1.000000, 273.150000, {0.99999, 1e-05}, {0.322811, 0.353077, 0, 0.985863, 0.985863}),
		Reference(1.000000, 273.150000, {0.58, 0.42}, {-58.4324, -55.7557, -86.8349, 0, 0}),
		Reference(1.000000, 273.150000, {0.6, 0.4}, {-0.0873765, -0.0536933, 0, -301.055, -301.055}),
		Reference(1.000000, 273.150000, {0.2, 0.8}, {-19.6901, -18.813, -29.3925, 0, 0}),
		Reference(10.000000, 273.150000, {1e-15, 1}, {0}),
		Reference(10.000000, 273.150000, {1e-05, 0.99999}, {0.97847, 0.978056, 0.979139, 0, 0}),
		Reference(10.000000, 273.150000, {1, 1e-15}, {0.32563, 0.356328, 0, 0.999194}),
		Reference(10.000000, 273.150000, {0.99999, 1e-05}, {0.324081, 0.354203, 0, 1.55322, 1.55322}),
		Reference(10.000000, 273.150000, {0.58, 0.42}, {-1.93471, -0.831958, -2.0146, -10.2443, -10.2443}),
		Reference(10.000000, 273.150000, {0.6, 0.4}, {-0.0815588, -0.273656, 0, -31.6775, -31.6775}),
		Reference(10.000000, 273.150000, {0.2, 0.8}, {-178.245, -170.716, -262.647, 0, 0}),
		Reference(100.000000, 273.150000, {1e-15, 1}, {0}),
		Reference(100.000000, 273.150000, {1e-05, 0.99999}, {0.989288, 0.988623, 1.00332, 0.993784, 0}),
		Reference(100.000000, 273.150000, {1, 1e-15}, {0.338496, 0.367765, 0, -0.125364}),
		Reference(100.000000, 273.150000, {0.99999, 1e-05}, {0.336736, 0.365349, 0, -0.125353, -0.125353}),
		Reference(100.000000, 273.150000, {0.58, 0.42}, {-2.04384, -2.45398, -2.03195, -2.25761, -3.04009}),
		Reference(100.000000, 273.150000, {0.6, 0.4}, {-0.112989, -0.282515, 0, 0.143417, -10.5099}),
		Reference(100.000000, 273.150000, {0.2, 0.8}, {-14.8607, -15.4535, -20.1532, -22.7672, -0.0179633}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<bool> modcholesky = {false, true};
	std::vector<FlashParams::StabilityVars> vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;

		for (FlashParams::StabilityVars var: vars)
		{
			flash_params.stability_variables = var;

			for (Reference condition: references)
			{
				error_output += condition.test(flash_params, verbose);
				if (write) { condition.write_ref(ref_string); }
			}
			write = false;
		}
	}
	ref_string += "\t};\n";

	if (error_output > 0)
	{
		std::cout << ref_string;
	}

	// Test ternary mixture H2O-CO2-C1
	comp = {"H2O", "CO2", "C1"};
	comp_data = CompData(comp);
	comp_data.Pc = {220.50, 73.75, 46.04};
	comp_data.Tc = {647.14, 304.10, 190.58};
	comp_data.ac = {0.328, 0.239, 0.012};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.47893});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.0936});
	comp_data.Mw = {18.015, 44.01, 16.043};

	flash_params = FlashParams(comp_data);
	flash_params.verbose = verbose;

	pr = CubicEoS(comp_data, CubicEoS::PR);
	pr.set_preferred_roots(0, 0.61, EoS::RootFlag::MAX);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {0, 1, 2};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-2;

    aq = AQEoS(comp_data, aq_map);
	aq.set_eos_range(0, {0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {0};
	flash_params.eos_params["AQ"]->stability_switch_tol = 1e-10;
	flash_params.eos_params["AQ"]->stability_max_iter = 10;

	si = Ballard(comp_data, "sI");
	flash_params.add_eos("sI", &si);
	flash_params.eos_params["sI"]->trial_comps = {0};
	flash_params.eos_params["sI"]->stability_max_iter = 20;

	sii = Ballard(comp_data, "sII");
	flash_params.add_eos("sII", &sii);
	flash_params.eos_params["sII"]->trial_comps = {0};
	flash_params.eos_params["sII"]->stability_max_iter = 20;

	references = {
		Reference(1.000000, 273.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(1.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 0.99795, 0.997335, 0.998656}),
		Reference(1.000000, 273.150000, {1, 5e-16, 5e-16}, {0.993388, 0, 0.355175, 0.32434}),
		Reference(1.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {0.878747, 0.878747, 0.878747, 0, 0.344638, 0.315902}),
		Reference(1.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 0.999767, 0.998475, 0.998681}),
		Reference(1.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 0.999186, 0.999013, 0.999351}),
		Reference(1.000000, 273.150000, {0.58, 0.21, 0.21}, {0, 0, 0, -86.8843, -55.7481, -58.4385}),
		Reference(1.000000, 273.150000, {0.6, 0.2, 0.2}, {-89.8989253183336, -60.472680150312, -57.6883449588182, 0, 0, 0}),
		Reference(1.000000, 273.150000, {0.2, 0.4, 0.4}, {0, 0, 0, -29.4533, -18.7819, -19.6841}),
		Reference(10.000000, 273.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(10.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0, 0, 0.981133, 0.975435, 0.998727}),
		Reference(10.000000, 273.150000, {1, 5e-16, 5e-16}, {0.999194, 0, 0.356328, 0.32563}),
		Reference(10.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {0.987354, 0.987354, 0.987354, 0, 0.345651, 0.31708}),
		Reference(10.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 0.989081, 0.986828, 0.988051}),
		Reference(10.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 0, 0.992236, 0.986926, 0.98709}),
		Reference(10.000000, 273.150000, {0.58, 0.21, 0.21}, {-222.499, -222.499, -222.499, -4.32584, -6.81771, -5.81658}),
		// Reference(10.000000, 273.150000, {0.6, 0.2, 0.2}, {-477.217, -477.217, -477.217, -477.217, -477.217, -477.217, -0.318348, 0, 2.06115}),
		Reference(10.000000, 273.150000, {0.2, 0.4, 0.4}, {0, 0, 0, -268.86, -174.075, -182.011}),
		Reference(100.000000, 273.150000, {1e-15, 0.5, 0.5}, {0, 0}),
		Reference(100.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0.952345, 0, 0, 0.938737, 0.919238, 0.945247}),
		Reference(100.000000, 273.150000, {1, 5e-16, 5e-16}, {-0.125364, 0, 0.367765, 0.338496}),
		Reference(100.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {-0.125353, -0.125353, 0.998095, 0, 0.355604, 0.328763}),
		Reference(100.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 0, 1.00754, 0.992654, 0.994059}),
		Reference(100.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0.949897, 0, 0, 0.952435, 0.923925, 0.919784}),
		Reference(100.000000, 273.150000, {0.58, 0.21, 0.21}, {-4.82642, -33.4303, -33.4303, -4.3151, -6.92721, -6.69614}),
		Reference(100.000000, 273.150000, {0.6, 0.2, 0.2}, {0.36216, -69.8093, -69.8093, 0, -0.343087, -0.31661}),
		Reference(100.000000, 273.150000, {0.2, 0.4, 0.4}, {-135.257, -135.257, -0.277918, -120.096, -82.8835, -84.8243}),
	};

	ref_string = "\treferences = {\n";
	write = true;

	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;

		for (FlashParams::StabilityVars var: vars)
		{
			flash_params.stability_variables = var;

			for (Reference condition: references)
			{
				error_output += condition.test(flash_params, verbose);
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
		std::cout << "Errors occurred in test_stability_vapour_brine_hydrates(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stability_vapour_brine_hydrates(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stationary_points_vapour_liquid()
{
	// Test stationary points on Y8 mixture
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	const bool verbose = false;
	std::cout << (verbose ? "TESTING STATIONARY POINTS FOR Y8\n" : "");
	int error_output = 0;
	double tol = 1e-7;

	std::vector<std::string> comp{"C1", "C2", "C3", "nC5", "nC7", "nC10"};
	std::vector<double> x = {0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244};

	CompData comp_data(comp);
	comp_data.Pc = {45.99, 48.72, 42.48, 33.70, 27.40, 21.10};
	comp_data.Tc = {190.56, 305.32, 369.83, 469.70, 540.20, 617.70};
	comp_data.ac = {0.011, 0.099, 0.152, 0.252, 0.350, 0.490};
	comp_data.kij = std::vector<double>(6*6, 0.);

	FlashParams flash_params(comp_data);
	flash_params.stability_variables = FlashParams::alpha;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson, InitialGuess::Yi::Wilson13, 0, 1, 2, 3, 4, 5};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-2;

	Flash flash(flash_params);

	std::vector<double> pres = {220., };
	std::vector<double> temp = {335., };
	std::vector<std::vector<double>> tpd_ref{{-0.001034175302, 0.}, };

	std::vector<bool>modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_stability = modchol;
		for (size_t j = 0; j < pres.size(); j++)
		{
			std::vector<TrialPhase> stationary_points = flash.find_stationary_points(pres[j], temp[j], x);
			if (stationary_points.size() != tpd_ref[j].size())
			{
				print("tpd_ref", tpd_ref[j]);
				for (TrialPhase sp: stationary_points)
				{
					sp.print_point();
				}
				error_output++;
			}
			else
			{
				for (TrialPhase sp: stationary_points)
				{
					if (verbose) { sp.print_point(); }
					if (std::find_if(tpd_ref[j].begin(), tpd_ref[j].end(), [&sp, tol](double ref) { return std::fabs(sp.tpd - ref) < tol; }) == tpd_ref[j].end())
					{
						sp.print_point();
						error_output++;
					}
				}
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_stationary_points_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stationary_points_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}
