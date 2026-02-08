#include <chrono>
#include <memory>
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/px_flash.hpp"
#include "dartsflash/stability/stability.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"
#include "dartsflash/eos/iapws/iapws_ice.hpp"
#include "dartsflash/eos/aq/jager.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"
#include "dartsflash/eos/solid/solid.hpp"
#include "dartsflash/global/global.hpp"

int test_purecomponent_ph();
int test_phflash_vapour_liquid();
int test_phflash_vapour_liquid_water();

struct Reference
{
	double pressure, h_s, T_tol{ 1e-1 }, T_ref;
	std::vector<double> composition;
	std::shared_ptr<PXFlashResults> flash_results;

	Reference(const double p, const double h_spec, const std::vector<double>& z, const double T_)
	: pressure(p), h_s(h_spec), T_ref(T_), composition(z) {}

	void print_conditions(std::string text = "")
	{
		print("====", text);
		print("p, h_spec", {pressure, h_s});
		print("z", composition);
	}
	int test(std::unique_ptr<PXFlash>& flash, bool verbose, bool test_derivs=false)
	{
		if (verbose)
		{
			std::cout << "==================================\n";
			this->print_conditions();
		}
		int error = flash->evaluate(pressure, h_s, composition);
		if (error > 0)
		{
			print("Error in Flash", error);
			return error;
		}

		// Output and compare results
		this->flash_results = flash->get_flash_results(test_derivs);

		if (verbose)
		{
			std::cout << "\nResults:\n";
			flash_results->print_results();
		}

		if (std::fabs(flash_results->temperature-T_ref) > T_tol)
		{
			std::cout << "T and T_ref are not the same \n";
			print("T", flash_results->temperature);
			print("T_ref", T_ref);
			error++;
		}

		if (test_derivs && this->test_derivatives(flash, verbose))
		{
			std::cout << "Error in PXFlash::test_derivatives()\n";
			print_conditions();
			error++;
		}
		if (verbose && !error)
		{
			std::cout << "Output is correct!\n";
		}

		return error;
	}

	int test(std::unique_ptr<PXFlash>& flash, std::shared_ptr<PXFlashResults> flashresults, bool verbose, bool test_derivs=false)
	{
		// Test extrapolation of previous results
		flashresults = flash->extrapolate_flash_results(pressure, h_s, composition, flashresults);

		if (verbose)
		{
			std::cout << "==================================\n";
			this->print_conditions();
		}
		int error = flash->evaluate(pressure, h_s, composition, flashresults);
		if (error > 0)
		{
			print("Error in Flash", error);
			return error;
		}

		// Output and compare results
		this->flash_results = flash->get_flash_results(test_derivs);

		if (verbose)
		{
			std::cout << "\nResults:\n";
			flash_results->print_results();
		}

		if (std::fabs(flash_results->temperature-T_ref) > T_tol)
		{
			std::cout << "T and T_ref are not the same \n";
			print("T", flash_results->temperature);
			print("T_ref", T_ref);
			error++;
		}

		if (test_derivs && this->test_derivatives(flash, verbose))
		{
			std::cout << "Error in PXFlash::test_derivatives()\n";
			print_conditions();
			error++;
		}
		return error;
	}

	int test_derivatives(std::unique_ptr<PXFlash>& flash, bool verbose)
	{
		int nc = static_cast<int>(composition.size());
		int error = 0;
		
		// Output results and derivatives
		std::shared_ptr<PXFlashResults> flash_results0 = this->flash_results;
		int np_tot = flash_results0->np_tot;
		double T0 = flash_results0->temperature;
		double d, dX_tol{ 3e-2 };

		// Derivatives of T with respect to P and z at constant enthalpy
		double dTdP, dTdX;
		std::vector<double> dnudP, dnudzk, dnudH, dxdP, dxdzk, dxdH, dTdzk;
		flash_results0->get_derivs(dnudP, dnudH, dnudzk, dxdP, dxdH, dxdzk);

		// Derivatives of temperature
		flash_results0->get_dT_derivs(dTdP, dTdX, dTdzk);

		// Calculate numerical derivatives w.r.t. pressure
		{
			double dp = 1e-6 * pressure;
			std::shared_ptr<PXFlashResults> flash_results1 = flash->extrapolate_flash_results(pressure+dp, h_s, composition, flash_results0);
			error += flash->evaluate(pressure+dp, h_s, composition, flash_results1);
			flash_results1 = flash->get_flash_results(false);

			// Test derivatives of temperature with respect to pressure at constant enthalpy and composition
			{
				double T1 = flash_results1->temperature;
				double dTdP_num = (T1 - T0)/dp;
				// Use logarithmic scale to compare
				d = std::log(std::fabs(dTdP_num + 1e-15)) - std::log(std::fabs(dTdP + 1e-15));
				if (verbose || std::fabs(d) > dX_tol)
				{
					print("dT/dP", {dTdP, dTdP_num, d});
					error++;
				}
			}
			
			// Compare analytical and numerical
			for (int j = 0; j < np_tot; j++)
			{
				double dnujdP_num = (flash_results1->nuj[j] - flash_results0->nuj[j])/dp;
				// Use logarithmic scale to compare
				d = std::log(std::fabs(dnujdP_num + 1e-15)) - std::log(std::fabs(dnudP[j] + 1e-15));
				if (verbose || std::isnan(dnudP[j]) || (std::fabs(d) > dX_tol && (std::fabs(dnudP[j]) > 1e-8 && std::fabs(dnujdP_num) > 1e-8)))
				{
					print("phase", j);
					print("dnuj/dP", {dnudP[j], dnujdP_num, d});
					error++;
				}

				for (int i = 0; i < nc; i++)
				{
					double dxdP_num = (flash_results1->Xij[j*nc+i] - flash_results0->Xij[j*nc+i])/dp;
					// Use logarithmic scale to compare
					d = std::log(std::fabs(dxdP_num + 1e-15)) - std::log(std::fabs(dxdP[j*nc+i] + 1e-15));
					if (verbose || std::isnan(dxdP[j*nc+i]) || (std::fabs(d) > dX_tol && (std::fabs(dxdP[j*nc+i]) > 1e-8 && std::fabs(dxdP_num) > 1e-8)))
					{
						print("phase, comp", {j, i});
						print("dXij/dP", {dxdP[j*nc+i], dxdP_num, d});
						error++;
					}
				}
			}
		}

		// Calculate numerical derivatives w.r.t. enthalpy
		{
			double dH = 1e-3;
			std::shared_ptr<PXFlashResults> flash_results1 = flash->extrapolate_flash_results(pressure, h_s+dH, composition, flash_results0);
			error += flash->evaluate(pressure, h_s+dH, composition, flash_results1);
			flash_results1 = flash->get_flash_results(false);

			// Test derivatives of temperature with respect to enthalpy at constant pressure and composition
			{
				double T1 = flash_results1->temperature;
				double dTdH_num = (T1 - T0)/dH;
				// Use logarithmic scale to compare
				d = std::log(std::fabs(dTdH_num + 1e-15)) - std::log(std::fabs(dTdX + 1e-15));
				if (verbose || std::isnan(dTdX) || (std::fabs(d) > dX_tol && std::fabs(T1-T0) > 1e-8))
				{
					print("dT/dH", {dTdX, dTdH_num, d});
					error++;
				}
			}

			// Compare analytical and numerical
			for (int j = 0; j < np_tot; j++)
			{
				double dnujdH_num = (flash_results1->nuj[j] - flash_results0->nuj[j])/dH;
				// Use logarithmic scale to compare
				d = std::log(std::fabs(dnujdH_num + 1e-15)) - std::log(std::fabs(dnudH[j] + 1e-15));
				if (verbose || std::isnan(dnudH[j]) || (std::fabs(d) > dX_tol && (std::fabs(dnudH[j]) > 1e-8 && std::fabs(dnujdH_num) > 1e-8)))
				{
					print("phase", j);
					print("dnuj/dH", {dnudH[j], dnujdH_num, d});
					error++;
				}

				for (int i = 0; i < nc; i++)
				{
					double dxdH_num = (flash_results1->Xij[j*nc+i] - flash_results0->Xij[j*nc+i])/dH;
					// Use logarithmic scale to compare
					d = std::log(std::fabs(dxdH_num + 1e-15)) - std::log(std::fabs(dxdH[j*nc+i] + 1e-15));
					if (verbose || std::isnan(dxdH[j*nc+i]) || (std::fabs(d) > dX_tol && (std::fabs(dxdH[j*nc+i]) > 1e-8 && std::fabs(dxdH_num) > 1e-8)))
					{
						print("phase, comp", {j, i});
						print("dXij/dH", {dxdH[j*nc+i], dxdH_num, d});
						error++;
					}
				}
			}
		}

		// Calculate numerical derivatives w.r.t. composition
		{
			std::vector<double> z(nc);
			double nT_inv = 1./std::accumulate(composition.begin(), composition.end(), 0.);
			std::transform(composition.begin(), composition.end(), z.begin(), [&nT_inv](double element) { return element *= nT_inv; });

			double dz = 1e-6;
			for (int k = 0; k < nc; k++)
			{
				// Transform to +dz
				// bool small_composition = false;
				double dzk = dz * z[k];
				z[k] += dzk;
				for (int ii = 0; ii < nc; ii++)
				{
					z[ii] /= (1. + dzk);
					// if (z[ii] < 1e-5) { small_composition = true; }
				}
				
				// Numerical derivative of lnphi w.r.t. zk
				std::shared_ptr<PXFlashResults> flash_results1 = flash->extrapolate_flash_results(pressure, h_s, z, flash_results0);
				error += flash->evaluate(pressure, h_s, z, flash_results1);
				flash_results1 = flash->get_flash_results(false);

				// Test derivatives of temperature with respect to pressure at constant enthalpy
				{
					double T1 = flash_results1->temperature;
					double dTdzk_num = (T1 - T0)/dzk;
					// Use logarithmic scale to compare
					d = std::log(std::fabs(dTdzk_num + 1e-15)) - std::log(std::fabs(dTdzk[k] + 1e-15));
					if (verbose || (std::fabs(d) > dX_tol && std::fabs(dTdzk[k]) > 1e-6))
					{
						print("dT/dzk", {static_cast<double>(k), dTdzk[k], dTdzk_num, d});
						error++;
					}
				}

				// Compare analytical and numerical
				for (int j = 0; j < np_tot; j++)
				{
					double dnujdzk_num = (flash_results1->nuj[j] - flash_results0->nuj[j])/dzk;
					// Use logarithmic scale to compare
					d = std::log(std::fabs(dnujdzk_num + 1e-15)) - std::log(std::fabs(dnudzk[k * np_tot + j] + 1e-15));
					if (verbose || std::isnan(dnudzk[k * np_tot + j]) || (std::fabs(d) > dX_tol && std::fabs(dnudzk[k * np_tot + j]) > 1e-8))
					{
						print("phase, zk", {j, k});
						print("dnuj/dzk", {dnudzk[k * np_tot + j], dnujdzk_num, d});
						error++;
					}

					for (int i = 0; i < nc; i++)
					{
						double dxdzk_num = (flash_results1->Xij[j*nc+i] - flash_results0->Xij[j*nc+i])/dzk;
						// Use logarithmic scale to compare
						d = std::log(std::fabs(dxdzk_num + 1e-15)) - std::log(std::fabs(dxdzk[k * np_tot * nc + j*nc+i] + 1e-15));
						if (verbose || std::isnan(dxdzk[k * np_tot * nc + j*nc+i]) || (std::fabs(d) > dX_tol && std::fabs(dxdzk[k * np_tot * nc + j*nc+i]) > 1e-4))
						{
							print("phase, comp, zk", {j, i, k});
							print("dXij/dzk", {dxdzk[k * np_tot * nc + j*nc+i], dxdzk_num, d});
							error++;
						}
					}	
				}

				// Return to original z
				for (int ii = 0; ii < nc; ii++)
				{
					z[ii] *= (1. + dzk);
				}
				z[k] -= dzk;
			}
		}
		return error;
	}
};

int main()
{
	int error_output = 0;
	
	error_output += test_purecomponent_ph();
	error_output += test_phflash_vapour_liquid();
	error_output += test_phflash_vapour_liquid_water();

    return error_output;
}

int test_purecomponent_ph()
{
	// Test pure component PH-flash
	// Test pure CO2 going from liquid(-like) to vapour(-like)
	// Test pure H2O going from ice to liquid to vapour
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING 1C FLASH VAPOUR-LIQUID AT P-H\n" : "");
    int error_output = 0;

	std::vector<std::string> comp = {"CO2"};
	std::vector<double> z = {1.};
	CompData comp_data(comp);
	comp_data.Pc = {comp_data::Pc["CO2"]};
	comp_data.Tc = {comp_data::Tc["CO2"]};
	comp_data.ac = {comp_data::ac["CO2"]};
	comp_data.Mw = {comp_data::Mw["CO2"]};
	comp_data.kij = std::vector<double>(1, 0.);

	CubicEoS ceos(comp_data, CubicEoS::PR);

	FlashParams flash_params(comp_data);
	flash_params.pxflash_type = FlashParams::BRENT;
	flash_params.save_performance_data = true;
	flash_params.verbose = verbose;
	flash_params.pxflash_Ftol = 1e-8;
	flash_params.pxflash_Ttol = 1e-2;
	flash_params.T_min = 200.;
	flash_params.T_max = 500.;

	flash_params.add_eos("CEOS", &ceos);
	flash_params.eos_params["CEOS"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	PXFlash flash(flash_params, StateSpecification::ENTHALPY);
	std::unique_ptr<PXFlash> flash_ptr = std::make_unique<PXFlash>(flash);

	std::vector<double> pressure = linspace(10., 100., 3);
	std::vector<double> enthalpy = linspace(-15000., 0., 6);

	std::vector<Reference> references = {
		Reference(pressure[0], enthalpy[0], z, 233.886393087369),
		Reference(pressure[0], enthalpy[1], z, 233.886393087369),
		Reference(pressure[0], enthalpy[2], z, 233.886393087369),
		Reference(pressure[0], enthalpy[3], z, 233.886393087369),
		Reference(pressure[0], enthalpy[4], z, 233.886393087369),
		Reference(pressure[0], enthalpy[5], z, 308.98086587075),
		Reference(pressure[1], enthalpy[0], z, 260.366202420562),
		Reference(pressure[1], enthalpy[1], z, 285.439101981081),
		Reference(pressure[1], enthalpy[2], z, 291.442200979283),
		Reference(pressure[1], enthalpy[3], z, 291.442200979283),
		Reference(pressure[1], enthalpy[4], z, 301.310610477766),
		Reference(pressure[1], enthalpy[5], z, 349.801475650221),
		Reference(pressure[2], enthalpy[0], z, 261.327331720971),
		Reference(pressure[2], enthalpy[1], z, 289.750314222817),
		Reference(pressure[2], enthalpy[2], z, 308.934557701995),
		Reference(pressure[2], enthalpy[3], z, 320.203790057289),
		Reference(pressure[2], enthalpy[4], z, 339.518974056816),
		Reference(pressure[2], enthalpy[5], z, 378.70809046145),
	};

	for (Reference condition: references)
	{
		int error = condition.test(flash_ptr, false, test_derivs);
		if (error || verbose) { condition.print_conditions("CO2 V-L"); print("error", error); error_output += error; }
	}

	// Test extrapolation of previous flash results
	error_output += references[6].test(flash_ptr, verbose, false);
	for (auto it = references.begin() + 7; it < references.begin() + 12; it++)
	{
		int error = it->test(flash_ptr, (it-1)->flash_results, verbose, false);
		if (error || verbose) { it->print_conditions("CO2 V-L extrapolation"); error_output += error; }
	}

	// Define H2O
	comp = {"H2O"};
	comp_data = CompData(comp);
	comp_data.Pc = {comp_data::Pc["H2O"]};
	comp_data.Tc = {comp_data::Tc["H2O"]};
	comp_data.ac = {comp_data::ac["H2O"]};
	comp_data.Mw = {comp_data::Mw["H2O"]};
	comp_data.kij = std::vector<double>(1., 0.);

	flash_params = FlashParams(comp_data);
	flash_params.pxflash_type = FlashParams::BRENT_NEWTON;
	flash_params.verbose = verbose;

	ceos = CubicEoS(comp_data, CubicEoS::PR);
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

	PureSolid ice(comp_data, "Ice");
	flash_params.add_eos("I", &ice);

	flash_params.set_eos_order({"AQ", "I", "CEOS"});
	flash_params.pxflash_Ttol = 1e-1;
	flash_params.T_min = 200.;
	flash_params.T_max = 600.;

	flash = PXFlash(flash_params, StateSpecification::ENTHALPY);
	flash_ptr = std::make_unique<PXFlash>(flash);

	pressure = logspace(1.e-5, 10., 3);
	enthalpy = linspace(-40000., 10000., 6);

	references = {
		Reference(pressure[0], enthalpy[0], z, 211.979562973661),
		Reference(pressure[0], enthalpy[1], z, 211.979562973661),
		Reference(pressure[0], enthalpy[2], z, 211.979562973661),
		Reference(pressure[0], enthalpy[3], z, 211.979562973661),
		Reference(pressure[0], enthalpy[4], z, 298.149994915151),
		Reference(pressure[0], enthalpy[5], z, 585.07565943808),
		Reference(pressure[1], enthalpy[0], z, 279.291341140379),
		Reference(pressure[1], enthalpy[1], z, 279.291341140379),
		Reference(pressure[1], enthalpy[2], z, 279.291341140379),
		Reference(pressure[1], enthalpy[3], z, 279.291341140379),
		Reference(pressure[1], enthalpy[4], z, 298.178092293133),
		Reference(pressure[1], enthalpy[5], z, 585.085392884229),
		Reference(pressure[2], enthalpy[0], z, 348.595509630523),
		Reference(pressure[2], enthalpy[1], z, 452.438307085564),
		Reference(pressure[2], enthalpy[2], z, 452.438307085564),
		Reference(pressure[2], enthalpy[3], z, 452.438307085564),
		Reference(pressure[2], enthalpy[4], z, 452.438307085564),
		Reference(pressure[2], enthalpy[5], z, 594.716451178097),
	};

	for (Reference condition: references)
	{
		int error = condition.test(flash_ptr, verbose, test_derivs);
		if (error || verbose) { condition.print_conditions("H2O V-A-I hybrid"); error_output += error; }
	}

	// flash_params = FlashParams(comp_data);
	
	// IAPWS95 iapws(comp_data, true);
	// flash_params.add_eos("IAPWS", &iapws);
	// flash_params.eos_params["IAPWS"].root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	// IAPWSIce ice(comp_data, true);
	// flash_params.add_eos("Ice", &ice);

	// flash_params.set_eos_order({"IAPWS", "Ice"});
	// flash_params.T_min = 200.;
	// flash_params.T_max = 300.;

	// flash = PXFlash(flash_params, StateSpecification::ENTHALPY);

	// pressure = logspace(1e-5, 1e2, 20);
	// hmin = -6.5e6;
	// hmax = 4.55e4;
	// enthalpy = linspace(hmin, hmax, 20);

	// for (double h: enthalpy)
	// {
	// 	for (double p: pressure)
	// 	{
	// 		int error = flash.evaluate(p, h);
	//		FlashResults flash_results = flash.get_flash_results(true);
	// 		if (verbose || error) { print("Error in H2O V-A-I PHFlash", {p, h}); error_output += error; }
	// 	}
	// }

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output != 0)
	{
		std::cout << "Errors occurred in test_purecomponent_ph(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_purecomponent_ph(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_phflash_vapour_liquid()
{
	// Test C1/C4 mixture (Zhu, 2014), data from (Zhu, 2014)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING PH-FLASH WITH NP STABILITYFLASH FOR BINARY MIXTURE\n" : "");
	int error_output = 0;

	std::vector<std::string> comp = {"C1", "C4"};
	CompData comp_data = CompData(comp);
	comp_data.Pc = {46.0, 38.0};
	comp_data.Tc = {190.60, 425.20};
	comp_data.ac = {0.008, 0.193};
	comp_data.kij = std::vector<double>(2*2, 0.);
	comp_data.T_0 = 273.15;

	std::vector<double> z = {0.99, 0.01};

	FlashParams flash_params(comp_data);
	flash_params.split_variables = FlashParams::lnK_chol;
    flash_params.split_switch_tol = 1e-1;
	flash_params.split_tol = 1e-24;
	flash_params.stability_variables = FlashParams::alpha;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-3;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson,
										  		  InitialGuess::Yi::Wilson13
												  };  // pure H2O initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
    flash_params.T_min = {100};
	flash_params.T_max = {900};
	flash_params.T_init = {400};

	PXFlash flash(flash_params, StateSpecification::ENTHALPY);
	std::unique_ptr<PXFlash> flash_ptr = std::make_unique<PXFlash>(flash);

	std::vector<Reference> references = {
		Reference(50., -6500, z, 195.6676379),
	};

	for (Reference condition: references)
	{
		int error = condition.test(flash_ptr, verbose, test_derivs);
		if (error || verbose) { condition.print_conditions(); error_output += error; }
	}

	// Test CO2/C1 mixture
	comp = std::vector<std::string>{"CO2", "C1"};
	comp_data = CompData(comp);
	comp_data.Pc = {73.75, 46.04};
	comp_data.Tc = {304.10, 190.58};
	comp_data.ac = {0.239, 0.012};
	comp_data.kij = std::vector<double>(2*2, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.47893});
	comp_data.T_0 = 273.15;

	flash_params = FlashParams(comp_data);
	flash_params.split_variables = FlashParams::lnK;
    flash_params.split_tol = 1e-22;
	flash_params.split_switch_tol = 1e-3;
	flash_params.stability_variables = FlashParams::alpha;
	flash_params.verbose = verbose;

	pr = CubicEoS(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-3;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Wilson, InitialGuess::Wilson13};  // pure H2O initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.eos_params["PR"]->rich_phase_order = {0, -1};
    flash_params.T_min = {100};
	flash_params.T_max = {500};
	// flash_params.T_init = {300};
	flash_params.verbose = false;

	flash = PXFlash(flash_params, StateSpecification::ENTHALPY);
	flash_ptr = std::make_unique<PXFlash>(flash);

	references = {
		Reference(70., -6.82778577e+03, {0.9, 0.1}, 285.57315),
		Reference(43.63636364, -66.72373531, {0.75, 0.25}, 307.4964235),
	};

	for (Reference condition: references)
	{
		int error = condition.test(flash_ptr, verbose, test_derivs);
		if (error || verbose) { condition.print_conditions(); error_output += error; }
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_phflash_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_phflash_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_phflash_vapour_liquid_water()
{
	// Test H2O/nC5 mixture (Zhu and Okuno, 2015)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING PH-FLASH WITH NP STABILITYFLASH FOR H2O/nC5 MIXTURE\n" : "");
	int error_output = 0;

	std::vector<std::string> comp = {"H2O", "nC5"};
	CompData comp_data = CompData(comp);
	comp_data.Pc = {220.5, 33.701};
	comp_data.Tc = {647.14, 469.7};
	comp_data.ac = {0.328, 0.2515};
	comp_data.kij = std::vector<double>(4, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.526});

	std::vector<double> z = {0.5, 0.5};

	FlashParams flash_params(comp_data);
	flash_params.split_variables = FlashParams::lnK_chol;
    flash_params.split_switch_tol = 1e-1;
	flash_params.split_tol = 1e-28;
	flash_params.stability_variables = FlashParams::alpha;
	// flash_params.verbose = true;

	CubicEoS ceos(comp_data, CubicEoS::PR);
	ceos.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("CEOS", &ceos);
	flash_params.eos_params["CEOS"]->stability_switch_tol = 1e-2;
	flash_params.eos_params["CEOS"]->trial_comps = {0, 1};  // pure H2O initial guess
	flash_params.eos_params["CEOS"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {0};  // pure H2O initial guess

	flash_params.set_eos_order({"AQ", "CEOS"});
    flash_params.T_min = {270};
	flash_params.T_max = {550};
	flash_params.T_init = {300};
	flash_params.pxflash_Ttol = 1e-1;

	PXFlash flash(flash_params, StateSpecification::ENTHALPY);
	std::unique_ptr<PXFlash> flash_ptr = std::make_unique<PXFlash>(flash);

	std::vector<Reference> references = {
		Reference(10., -2.8e4, z, 354.160586251558),  // Aq-L
		Reference(10., -1.8e4, z, 387.7454291),  // Aq-V-L
		Reference(10., 0., z, 411.8054612294),  // Aq-V
		Reference(10., 1.e4, z, 424.07228649014),  // Aq-V
		Reference(10., 2.e4, z, 513.787993188329),  // V
		// Reference(10., 7.61429735e+03, {8.00000000e-01, 2.00000000e-01}, 442.606200250832),
		Reference(10., -1.66726856e+04, {3.00000000e-01, 7.00000000e-01}, 387.833634529772)
	};
	// references = {references[1]};

	for (Reference condition: references)
	{
		error_output += condition.test(flash_ptr, verbose, test_derivs);
	}

	// Test Water/NWE mixture (Khan et al, 1992), data from (Li, 2018)
	std::cout << (verbose ? "TESTING PH-FLASH WITH NP STABILITYFLASH FOR WATER MIXTURE\n" : "");

	comp = std::vector<std::string>{"H2O", "PC1", "PC2", "PC3", "PC4"};
	comp_data = CompData(comp);
	comp_data.Pc = {220.89, 48.82, 19.65, 10.20, 7.72};
	comp_data.Tc = {647.3, 305.556, 638.889, 788.889, 838.889};
	comp_data.ac = {0.344, 0.098, 0.535, 0.891, 1.085};
	comp_data.kij = std::vector<double>(5*5, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.71918, 0.45996, 0.26773, 0.24166});
	
	comp_data.cpi = {comp_data::cpi["H2O"],
					{-3.5 / M_R, 0.005764 / M_R, 5.09E-7 / M_R, 0. },
					{-0.404 / M_R, 0.0006572 / M_R, 5.41E-8 / M_R, 0.},
					{-6.1 / M_R, 0.01093 / M_R, 1.41E-6 / M_R, 0.},
					{-4.5 / M_R, 0.008049 / M_R, 1.04E-6 / M_R, 0.}};
	comp_data.T_0 = 273.15;

	z = std::vector<double>{0.5, 0.15, 0.10, 0.10, 0.15};

	flash_params = FlashParams(comp_data);
	flash_params.split_variables = FlashParams::nik;
    flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-22;
	flash_params.stability_variables = FlashParams::alpha;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson,
										  		  InitialGuess::Yi::Wilson13};  // pure H2O initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
    flash_params.pxflash_Ttol = 1e-1;
	flash_params.T_min = {200};
	flash_params.T_max = {900};
	flash_params.T_init = {400};

	flash = PXFlash(flash_params, StateSpecification::ENTHALPY);
	flash_ptr = std::make_unique<PXFlash>(flash);

	references = {
		// Reference(30., 0, z, 742.7160),
		Reference(60., 0, z, 782.646),
		Reference(90., 0, z, 828.752),
	};

	for (Reference condition: references)
	{
		error_output += condition.test(flash_ptr, verbose, test_derivs);
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_phflash_vapour_liquid_water(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_phflash_vapour_liquid_water(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}