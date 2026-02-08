#include <iterator>
#include <chrono>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/negative_flash.hpp"
#include "dartsflash/stability/stability.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"
#include "dartsflash/eos/aq/jager.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"
#include "dartsflash/eos/iapws/iapws_ice.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"
#include "dartsflash/eos/solid/solid.hpp"

int test_purecomponent_pt();

int test_negativeflash2_vapour_liquid();
int test_stabilityflash2_vapour_liquid_2comp();
int test_stabilityflash2_vapour_liquid_3comp();
int test_stabilityflash2_vapour_liquid();
int test_stabilityflashN_vapour_liquid();
int test_stabilityflashN_vapour_liquid_sour_gas();
int test_stabilityflashN_vapour_liquid_water();

int test_negativeflash2_brine_vapour();
int test_stabilityflash2_brine_vapour();
int test_stabilityflashN_brine_vapour();
int test_stabilityflashN_brine_vapour_h2s();
int test_stabilityflashN_brine_vapour_oil();
int test_negativeflash2_brine_vapour_ions();
int test_stabilityflashN_brine_vapour_si();
int test_stabilityflashN_brine_vapour_ice();

struct Reference
{
	double pressure, temperature, nu_tol{ 1e-5 }, dX_tol;
	std::vector<double> composition, nu_ref, nu;
	std::shared_ptr<FlashResults> flash_results;

	Reference(const double p, const double T, const std::vector<double>& z, const std::vector<double>& nu_ref_, double dX_tol_=1e-2) 
	: pressure(p), temperature(T), dX_tol{dX_tol_}, composition(z), nu_ref(nu_ref_) {}

	void print_conditions(bool verbose)
	{
		if (verbose)
		{
			std::cout << "==================================\n";
			print("p, T", {pressure, temperature});
			print("z", composition);
		}
		return;
	}

	int test(std::unique_ptr<Flash>& flash, bool verbose, bool test_derivs=false)
	{
		int error = flash->evaluate(pressure, temperature, composition);
		if (error > 0)
		{
			print("Error in Flash", error);
			print_conditions(true);
			return error;
		}
		
		// Output and compare results
		this->flash_results = flash->get_flash_results();
		nu = flash_results->nuj;
		if (verbose)
		{
			std::cout << "\nResults:\n";
			print("nu", nu);
			print("x", flash_results->Xij, static_cast<int>(flash_results->nuj.size()));
			std::cout << " Number of flash iterations ssi = " << flash->get_flash_total_ssi_iter() << std::endl;
			std::cout << " Number of flash iterations newton = " << flash->get_flash_total_newton_iter() << std::endl;
			std::cout << " Number of flash total iterations = " << flash->get_flash_total_ssi_iter() + flash->get_flash_total_newton_iter() << std::endl;
		}

		if ( int(std::count_if(nu.begin(), nu.end(), [](double aux){return aux != 0.0;} )) != int(std::count_if(nu_ref.begin(), nu_ref.end(), [](double aux){return aux != 0.0;} )))
		{
			std::cout << "nu and nu_ref are not the same size\n";
			print("p, T", {pressure, temperature});
			print("z", composition);
			print("nu", nu);
			print("nu_ref", nu_ref);
			error++;
		}
		if (this->compare_nu())
		{
			std::cout << "Different values for nu\n";
			print("p, T", {pressure, temperature});
			print("z", composition);
			print("nu", nu);
			print("nu_ref", nu_ref);
			error++;
		}
		if (test_derivs && this->test_derivatives(flash, verbose))
		{
			std::cout << "Error in Flash::test_derivatives()\n";
			this->print_conditions(true);
			error++;
		}
		if (verbose && !error)
		{
			std::cout << "Output is correct!\n";
		}

		return error;
	}

	int test(std::unique_ptr<Flash>& flash, std::shared_ptr<FlashResults> flashresults, bool verbose, bool test_derivs=false)
	{
		flashresults = flash->extrapolate_flash_results(pressure, temperature, composition, flashresults);

		int error = flash->evaluate(pressure, temperature, composition, flashresults);
		if (error > 0)
		{
			print("Error in Flash", error);
			print_conditions(true);
			return error;
		}

		// Output and compare results
		this->flash_results = flash->get_flash_results();
		nu = flash_results->nuj;
		if (verbose)
		{
			std::cout << "\nResults:\n";
			print("nu", nu);
			print("x", flash_results->Xij, static_cast<int>(flash_results->nuj.size()));
			std::cout << " Number of flash iterations ssi = " << flash->get_flash_total_ssi_iter() << std::endl;
			std::cout << " Number of flash iterations newton = " << flash->get_flash_total_newton_iter() << std::endl;
			std::cout << " Number of flash total iterations = " << flash->get_flash_total_ssi_iter() + flash->get_flash_total_newton_iter() << std::endl;
		}

		if ( int(std::count_if(nu.begin(), nu.end(), [](double aux){return aux != 0.0;} )) != int(std::count_if(nu_ref.begin(), nu_ref.end(), [](double aux){return aux != 0.0;} )))
		{
			std::cout << "nu and nu_ref are not the same size\n";
			print("p, T", {pressure, temperature});
			print("z", composition);
			print("nu", nu);
			print("nu_ref", nu_ref);
			error++;
		}
		if (this->compare_nu())
		{
			std::cout << "Different values for nu\n";
			print("p, T", {pressure, temperature});
			print("z", composition);
			print("nu", nu);
			print("nu_ref", nu_ref);
			error++;
		}
		if (test_derivs && this->test_derivatives(flash, verbose))
		{
			std::cout << "Error in Flash::test_derivatives()\n";
			this->print_conditions(true);
			error++;
		}
		if (verbose && !error)
		{
			std::cout << "Output is correct!\n";
		}

		return error;
	}

	int compare_nu()
	{
		std::vector<double> aux_calc = nu;
		std::sort( aux_calc.begin(), aux_calc.end() );
		aux_calc.erase( std::remove_if( aux_calc.begin(), aux_calc.end(), []( double aux ){ return aux == 0.0 ;}), aux_calc.end() );
		std::vector<double> aux_ref  = nu_ref;
		std::sort( aux_ref.begin(), aux_ref.end() );
		aux_ref.erase( std::remove_if( aux_ref.begin(), aux_ref.end(), []( double aux ){ return aux == 0.0 ;} ), aux_ref.end() );

		for (size_t j = 0; j < nu_ref.size(); j++)
		{
			if (std::fabs(aux_calc[j]-aux_ref[j]) > nu_tol)
			{
				print("t", aux_calc);	
				print("c", aux_ref);
				return 1;
			}
		}
		return 0;
	}

	void write_ref(std::string& ref_string)
	{
		std::string p_str = std::to_string(pressure);
		std::string t_str = std::to_string(temperature);

		std::ostringstream z_str_, nu_str_;

    	// Convert all but the last element to avoid a trailing ", "
	    std::copy(composition.begin(), composition.end()-1, std::ostream_iterator<double>(z_str_, ", "));
		std::copy(nu.begin(), nu.end()-1, std::ostream_iterator<double>(nu_str_, ", "));

    	// Now add the last element with no delimiter
    	z_str_ << composition.back();
		nu_str_ << nu.back();

		// Add curly brackets front and back
		std::string z_str = "{" + z_str_.str() + "}";
		std::string nu_str = "{" + nu_str_.str() + "}";

		ref_string += "\t\t";
		ref_string += ("Reference(" + p_str + ", " + t_str + ", " + z_str + ", " + nu_str + "),");
		ref_string += "\n";
		return;
	}

	int test_derivatives(std::unique_ptr<Flash>& flash, bool verbose)
	{
		int error = 0;
		int nc = static_cast<int>(composition.size());
		
		// Output results and derivatives
		std::shared_ptr<FlashResults> flash_results0 = flash->get_flash_results(true);
		
		std::vector<double> dnudP, dnudT, dnudzk, dxdP, dxdT, dxdzk;
		flash_results0->get_derivs(dnudP, dnudT, dnudzk, dxdP, dxdT, dxdzk);

		double H0 = flash_results0->total_prop(EoS::Property::ENTHALPY);
		double S0 = flash_results0->total_prop(EoS::Property::ENTROPY);
		double dSdP = flash_results0->dX_dP(StateSpecification::ENTROPY);
		double dSdT = flash_results0->dX_dT(StateSpecification::ENTROPY);
		std::vector<double> dSdzk = flash_results0->dX_dz(StateSpecification::ENTROPY);
		double dHdP = flash_results0->dX_dP(StateSpecification::ENTHALPY);
		double dHdT = flash_results0->dX_dT(StateSpecification::ENTHALPY);
		std::vector<double> dHdzk = flash_results0->dX_dz(StateSpecification::ENTHALPY);
		double d, H1, S1;
		int np = flash_results0->np;
		int np_tot = flash_results0->np_tot;

		// Calculate numerical derivatives w.r.t. pressure
		{
			double dp = 1e-6 * pressure;
			std::shared_ptr<FlashResults> flash_results1 = flash->extrapolate_flash_results(pressure+dp, temperature, composition, flash_results0);
			error += flash->evaluate(pressure+dp, temperature, composition, flash_results1);
			flash_results1 = flash->get_flash_results(false);

			// Compare analytical and numerical
			for (int jj = 0; jj < np; jj++)
			{
				int j = flash_results0->phase_idxs[jj];

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

			// Test derivatives of total properties with respect to pressure: enthalpy and entropy
			H1 = flash_results1->total_prop(EoS::Property::ENTHALPY);
			double dHdP_num = (H1 - H0)/dp;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dHdP_num + 1e-15)) - std::log(std::fabs(dHdP + 1e-15));
			if (verbose || std::fabs(d) > dX_tol)
			{
				print("dH/dP", {dHdP, dHdP_num, d});
				error++;
			}

			S1 = flash_results1->total_prop(EoS::Property::ENTROPY);
			double dSdP_num = (S1 - S0)/dp;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dSdP_num + 1e-15)) - std::log(std::fabs(dSdP + 1e-15));
			if (verbose || std::fabs(d) > dX_tol)
			{
				print("dS/dP", {dSdP, dSdP_num, d});
				error++;
			}
		}

		// Calculate numerical derivatives w.r.t. temperature
		{
			double dT = 1e-6 * temperature;
			std::shared_ptr<FlashResults> flash_results1 = flash->extrapolate_flash_results(pressure, temperature+dT, composition, flash_results0);
			error += flash->evaluate(pressure, temperature+dT, composition, flash_results1);
			flash_results1 = flash->get_flash_results(false);

			// Compare analytical and numerical
			for (int jj = 0; jj < np; jj++)
			{
				int j = flash_results0->phase_idxs[jj];

				double dnujdT_num = (flash_results1->nuj[j] - flash_results0->nuj[j])/dT;
				// Use logarithmic scale to compare
				d = std::log(std::fabs(dnujdT_num + 1e-15)) - std::log(std::fabs(dnudT[j] + 1e-15));
				if (verbose || std::isnan(dnudT[j]) || (std::fabs(d) > dX_tol && (std::fabs(dnudT[j]) > 1e-8 && std::fabs(dnujdT_num) > 1e-8)))
				{
					print("phase", j);
					print("dnuj/dT", {dnudT[j], dnujdT_num, d});
					error++;
				}

				for (int i = 0; i < nc; i++)
				{
					double dxdT_num = (flash_results1->Xij[j*nc+i] - flash_results0->Xij[j*nc+i])/dT;
					// Use logarithmic scale to compare
					d = std::log(std::fabs(dxdT_num + 1e-15)) - std::log(std::fabs(dxdT[j*nc+i] + 1e-15));
					if (verbose || std::isnan(dxdT[j*nc+i]) || (std::fabs(d) > dX_tol && (std::fabs(dxdT[j*nc+i]) > 1e-8 && std::fabs(dxdT_num) > 1e-8)))
					{
						print("phase, comp", {j, i});
						print("dXij/dT", {dxdT[j*nc+i], dxdT_num, d});
						error++;
					}
				}
			}

			// Test derivatives of total properties with respect to temperature: enthalpy and entropy
			H1 = flash_results1->total_prop(EoS::Property::ENTHALPY);
			double dHdT_num = (H1 - H0)/dT;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dHdT_num + 1e-15)) - std::log(std::fabs(dHdT + 1e-15));
			if (verbose || std::fabs(d) > dX_tol)
			{
				print("dH/dT", {dHdT, dHdT_num, d});
				error++;
			}

			S1 = flash_results1->total_prop(EoS::Property::ENTROPY);
			double dSdT_num = (S1 - S0)/dT;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dSdT_num + 1e-15)) - std::log(std::fabs(dSdT + 1e-15));
			if (verbose || std::fabs(d) > dX_tol)
			{
				print("dS/dT", {dSdT, dSdT_num, d});
				error++;
			}
		}

		// Calculate numerical derivatives w.r.t. composition
		std::vector<double> z(nc);
		double nT_inv = 1./std::accumulate(composition.begin(), composition.end(), 0.);
		std::transform(composition.begin(), composition.end(), z.begin(), [&nT_inv](double element) { return element *= nT_inv; });

		double dz = 1e-6;
		for (int k = 0; k < nc; k++)
		{
			// Transform to +dz
			bool small_composition = false;
			double dzk = dz * z[k];
			z[k] += dzk;
			for (int ii = 0; ii < nc; ii++)
        	{
            	z[ii] /= (1. + dzk);
				if (z[ii] < 1e-5) { small_composition = true; }
        	}
			
			// Numerical derivative of lnphi w.r.t. zk
			std::shared_ptr<FlashResults> flash_results1 = flash->extrapolate_flash_results(pressure, temperature, z, flash_results0);
			error += flash->evaluate(pressure, temperature, z, flash_results1);
			flash_results1 = flash->get_flash_results(false);

			// Compare analytical and numerical
			for (int jj = 0; jj < np; jj++)
			{
				int j = flash_results0->phase_idxs[jj];

				double dnujdzk_num = (flash_results1->nuj[j] - flash_results0->nuj[j])/dzk;
				// Use logarithmic scale to compare
				d = std::log(std::fabs(dnujdzk_num + 1e-15)) - std::log(std::fabs(dnudzk[k * np_tot + j] + 1e-15));
				if (verbose || std::isnan(dnudzk[k * np_tot + j]) || (std::fabs(d) > dX_tol && (std::fabs(dnudzk[k * np_tot + j]) > 1e-8 && std::fabs(dnudzk[k * np_tot + j]) > 1e-8)))
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
					if (verbose || std::isnan(dxdzk[k * np_tot * nc + j*nc+i]) ||
						(z[k] > 1e-8 && std::fabs(d) > dX_tol && (std::fabs(dxdzk[k * np_tot * nc + j*nc+i]) > 1e-8 && std::fabs(dxdzk[k * np_tot * nc + j*nc+i]) > 1e-8)))
					{
						print("phase, comp, zk", {j, i, k});
						print("dXij/dzk", {dxdzk[k * np_tot * nc + j*nc+i], dxdzk_num, d});
						error++;
					}
				}	
			}

			// Test derivatives of total properties with respect to pressure: enthalpy and entropy
			H1 = flash_results1->total_prop(EoS::Property::ENTHALPY);
			double dHdzk_num = (H1 - H0)/dzk;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dHdzk_num + 1e-15)) - std::log(std::fabs(dHdzk[k] + 1e-15));
			if (verbose || std::isnan(dHdzk[k]) || (!small_composition && std::fabs(d) > dX_tol))
			{
				print("dH/dzk", {static_cast<double>(k), dHdzk[k], dHdzk_num, d});
				error++;
			}

			S1 = flash_results1->total_prop(EoS::Property::ENTROPY);
			double dSdzk_num = (S1 - S0)/dzk;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dSdzk_num + 1e-15)) - std::log(std::fabs(dSdzk[k] + 1e-15));
			if (verbose || std::isnan(dSdzk[k]) || (!small_composition && std::fabs(d) > dX_tol))
			{
				print("dS/dzk", {static_cast<double>(k), dSdzk[k], dSdzk_num, d});
				error++;
			}

			// Return to original z
			for (int ii = 0; ii < nc; ii++)
        	{
            	z[ii] *= (1. + dzk);
        	}
			z[k] -= dzk;
		}

		return error;
	}
};

int main()
{
	int error_output = 0;
	
	error_output += test_purecomponent_pt();

	error_output += test_negativeflash2_vapour_liquid();
	// error_output += test_negativeflash2_brine_vapour();

	error_output += test_stabilityflash2_vapour_liquid_2comp();
	error_output += test_stabilityflash2_vapour_liquid_3comp();
	error_output += test_stabilityflash2_vapour_liquid();
	error_output += test_stabilityflashN_vapour_liquid();
	error_output += test_stabilityflashN_vapour_liquid_sour_gas();
	error_output += test_stabilityflashN_vapour_liquid_water();

	error_output += test_stabilityflash2_brine_vapour();
	error_output += test_stabilityflashN_brine_vapour();
	error_output += test_stabilityflashN_brine_vapour_h2s();
	// error_output += test_stabilityflashN_brine_vapour_oil();
	// error_output += test_negativeflash2_brine_vapour_ions();
	error_output += test_stabilityflashN_brine_vapour_si();
	error_output += test_stabilityflashN_brine_vapour_ice();

    return error_output;
}

int test_purecomponent_pt()
{
	// Test pure component PT-flash
	// Test pure CO2 going from liquid(-like) to vapour(-like)
	// Test pure H2O going from ice to liquid to vapour
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING 1C FLASH VAPOUR-LIQUID AT P-T\n" : "");
    int error_output = 0;

	// Define CO2
	std::vector<std::string> comp = {"CO2"};
	CompData comp_data(comp);
	comp_data.Pc = {comp_data::Pc["CO2"]};
	comp_data.Tc = {comp_data::Tc["CO2"]};
	comp_data.ac = {comp_data::ac["CO2"]};
	comp_data.Mw = {comp_data::Mw["CO2"]};
	comp_data.kij = std::vector<double>(1., 0.);

	FlashParams flash_params(comp_data);
	flash_params.verbose = verbose;

	CubicEoS ceos(comp_data, CubicEoS::PR);
	flash_params.add_eos("CEOS", &ceos);

	Flash flash_co2(flash_params);

	// Define H2O
	comp = {"H2O"};
	comp_data = CompData(comp);
	comp_data.Pc = {comp_data::Pc["H2O"]};
	comp_data.Tc = {comp_data::Tc["H2O"]};
	comp_data.ac = {comp_data::ac["H2O"]};
	comp_data.Mw = {comp_data::Mw["H2O"]};
	comp_data.kij = std::vector<double>(1., 0.);

	flash_params = FlashParams(comp_data);
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

	flash_params.set_eos_order({"AQ", "CEOS"});

	Flash flash_h2o(flash_params);

	// IAPWS-95 and IAPWSIce EoS
	flash_params = FlashParams(comp_data);
	flash_params.verbose = verbose;

	IAPWS95 iapws(comp_data, true);
	IAPWSIce ice(comp_data, true);
	flash_params.add_eos("IAPWS", &iapws);
	flash_params.eos_params["IAPWS"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.add_eos("Ice", &ice);

	flash_params.set_eos_order({"IAPWS", "Ice"});

	Flash flash_h2o_ice(flash_params);

	std::vector<double> temperature = linspace(260., 460., 10);
	std::vector<double> pressure = logspace(1e-4, 1e2, 10);

	for (double T: temperature)
	{
		for (double p: pressure)
		{
			error_output += flash_co2.evaluate(p, T);
			error_output += flash_h2o.evaluate(p, T);
			error_output += flash_h2o_ice.evaluate(p, T);
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_purecomponent_pt(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_purecomponent_pt(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_negativeflash2_vapour_liquid()
{
	// Test negative flash on Y8 mixture
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING 2P NEGATIVEFLASH VAPOUR-LIQUID\n" : "");
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
	flash_params.split_tol = 1e-24;
	flash_params.split_switch_tol = 1e-3;
	flash_params.split_max_iter = 50;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);

	std::vector<Reference> references = {
		Reference(1.013250, 298.150000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.961031, 0.0389694}),
		Reference(100.000000, 335.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.854433, 0.145567}),
		Reference(220.000000, 335.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.888439, 0.111561}),
		Reference(10.000000, 210.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.82805, 0.17195}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (FlashParams::SplitVars var: vars)
	{
		flash_params.split_variables = var;
		NegativeFlash flash(flash_params, {"PR", "PR"}, {InitialGuess::Ki::Wilson_VL});
		std::unique_ptr<Flash> flash_ptr = std::make_unique<NegativeFlash>(flash);
		for (Reference condition: references)
		{
			error_output += condition.test(flash_ptr, verbose, test_derivs);
			if (write) { condition.write_ref(ref_string); }
		}
		write = false;
	}
	ref_string += "\t};\n";

	// // Test
	// comp = std::vector<std::string>{"CO2", "N2", "H2S", "H2", "CH4", "C2H6", "C3H8", "C4H10", "C10H22"};
	// comp_data = CompData(comp);
	// comp_data.Pc = {73.77, 33.96, 90.0, 12.96, 45.99, 48.72, 42.51, 37.96, 21.03};
	// comp_data.Tc = {304.128, 126.192, 373.1, 33.145, 190.564, 305.322, 369.89, 425.125, 617.7};
	// comp_data.ac = {0.22394, 0.0372, 0.1005, -0.219, 0.01142, 0.0995, 0.1521, 0.201, 0.4884};
	// comp_data.kij = {
  	// 	      0,      -0.0122,       0.0967,      -0.1622,       0.0978,         0.13,       0.1315,       0.1352,       0.1141,
    //     -0.0122,            0,       0.1652,       0.0711,       0.0289,       0.0533,       0.0878,       0.0711,       0.1122,
    //      0.0967,       0.1652,            0,            0,            0,       0.0952,       0.0878,            0,       0.0333,
    //     -0.1622,       0.0711,            0,            0,      -0.0044,      -0.0781,      -0.1311,       -0.397,            0,
    //      0.0978,       0.0289,            0,      -0.0044,            0,      -0.0059,       0.0119,       0.0185,       0.0411,
    //        0.13,       0.0533,       0.0952,      -0.0781,      -0.0059,            0,       0.0011,       0.0089,       0.0144,
    //      0.1315,       0.0878,       0.0878,      -0.1311,       0.0119,       0.0011,            0,       0.0033,            0,
    //      0.1352,       0.0711,            0,       -0.397,       0.0185,       0.0089,       0.0033,            0,       0.0078,
    //      0.1141,       0.1122,       0.0333,            0,       0.0411,       0.0144,            0,       0.0078,            0
	// };

	// flash_params = FlashParams(comp_data);
	// flash_params.split_switch_tol = 1e-3;
	// flash_params.split_max_iter = 50;
	// flash_params.verbose = true;

	// pr = CubicEoS(comp_data, CubicEoS::PR);
	// flash_params.add_eos("PR", &pr);

	// NegativeFlash flash(flash_params, {"PR", "PR"}, {InitialGuess::Ki::Wilson_VL});

	// {
	// std::cout << "Fluid 04, Sample 0028" << std::endl;
	// double pressure = 4.981434;
	// double temperature = 128.0;
	// std::vector<double> composition = {0.00092, 0.00030, 0.00904, 0.00215, 0.95215, 0.01179, 0.00469, 0.00155, 0.01741};
	// error_output += flash.evaluate(pressure, temperature, composition);
	// }

	// {
	// std::cout << "Fluid 04, Sample 0031" << std::endl;
	// double pressure = 150.4314;
	// double temperature = 302.598;
	// std::vector<double> composition = {0.00092, 0.00030, 0.00904, 0.00215, 0.95215, 0.01179, 0.00469, 0.00155, 0.01741};
	// error_output += flash.evaluate(pressure, temperature, composition);
	// }

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_negativeflash2_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_negativeflash2_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stabilityflash2_vapour_liquid_2comp()
{
	// Test stability flash for a binary mixture
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING BINARY STABILITYFLASH VAPOUR-LIQUID\n" : "");
    int error_output = 0;

	// H2S-C1 mixture (Michelsen, 1982)
	std::vector<std::string> comp{"H2S", "C1"};
	CompData comp_data(comp);
	comp_data.Pc = {89.63, 46.04};
	comp_data.Tc = {373.53, 190.58};
	comp_data.ac = {0.0942, 0.012};
	comp_data.kij = std::vector<double>(2*2, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.1});

	CubicEoS ceos(comp_data, CubicEoS::SRK);

	FlashParams flash_params(comp_data);
	flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-24;
	flash_params.split_max_iter = 50;
	flash_params.verbose = verbose;

	flash_params.add_eos("CEOS", &ceos);
	flash_params.eos_params["CEOS"]->stability_switch_tol = 1e-3;
	flash_params.eos_params["CEOS"]->trial_comps = {InitialGuess::Yi::Wilson, 0, 1};
	flash_params.eos_params["CEOS"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::vector<Reference> references = {
		// 40.1 atm
		Reference(40.631325, 190.000000, {0.01, 0.99}, {1., 0.}),
		Reference(40.631325, 190.000000, {0.05, 0.95}, {0.965267, 0.0347327}),
		Reference(40.631325, 190.000000, {0.1, 0.9}, {0.909455, 0.0905453}),
		Reference(40.631325, 190.000000, {0.2, 0.8}, {0.79783, 0.20217}),
		Reference(40.631325, 190.000000, {0.5, 0.5}, {0.462954, 0.537046}),
		Reference(40.631325, 190.000000, {0.8, 0.2}, {0.128079, 0.871921}),
		Reference(40.631325, 190.000000, {0.9, 0.1}, {0.016454, 0.983546}),
		Reference(40.631325, 190.000000, {0.95, 0.05}, {0., 1.}),
		Reference(40.631325, 190.000000, {0.99, 0.01}, {0., 1.}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::lnK, FlashParams::lnK_chol};
	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, test_derivs);
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

	// Test with (extrapolation of) previous results
	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);

			error_output += references[0].test(flash_ptr, verbose, false);
			for (auto it = references.begin() + 1; it < references.end(); it++)
			{
				error_output += it->test(flash_ptr, (it-1)->flash_results, verbose, test_derivs);
			}
		}
	}

	// Test CO2-C1 mixture
	verbose = false;
	comp = std::vector<std::string>{"CO2", "C1"};
	comp_data = CompData(comp);
	comp_data.Pc = {73.75, 46.04};
	comp_data.Tc = {304.10, 190.58};
	comp_data.ac = {0.239, 0.012};
	comp_data.kij = std::vector<double>(2*2, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.0936});
	comp_data.T_0 = 273.15;

	flash_params = FlashParams(comp_data);
    flash_params.split_switch_tol = 1e-1;
	flash_params.split_tol = 1e-24;
	flash_params.verbose = verbose;

	ceos = CubicEoS(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &ceos);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Wilson};  // pure H2O initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.eos_params["PR"]->rich_phase_order = {0, -1};
	// flash_params.light_comp_idx = 0;

	references = {
		Reference(69.310300, 290.000000, {0.9, 0.1}, {0.389732, 0.610268, 0}),
		Reference(70., 285.6869768, {0.99, 0.01}, {0., 1., 0.}),
	};

	ref_string = "\tstd::vector<Reference> references = {\n";
	write = true;

	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, test_derivs);
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
		std::cout << "Errors occurred in test_stabilityflash2_vapour_liquid_2comp(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflash2_vapour_liquid_2comp(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stabilityflash2_vapour_liquid_3comp()
{
	// Test stability flash for a 3-component mixture
	// It can be used to quickly verify new implementations like Newton modified mole number
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = false;
	std::cout << (verbose ? "TESTING TERNARY STABILITYFLASH VAPOUR-LIQUID\n" : "");
    int error_output = 0;

	//Test 1 ->H2O-C4-C20 mixture
	std::vector<std::string> comp{"H2O", "C4", "C20"};
	CompData comp_data(comp);
	comp_data.Pc = {220.5, 38., 14.6};
	comp_data.Tc = {647., 425.2, 782.};
	comp_data.ac = {0.344, 0.1928, 0.8160};
	comp_data.kij = std::vector<double>(3*3, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.5, 0.5, 0.5, 0.5});

	std::vector<double> z = {0.8, 0.16, 0.04};

	CubicEoS pr(comp_data, CubicEoS::PR);

	FlashParams flash_params(comp_data);
	flash_params.split_switch_tol = 1e-3;
	flash_params.split_max_iter = 50;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-3;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson};
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::vector<Reference> references = {
		Reference(50, 500, z, {0.947413562, 0.0525865263}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, test_derivs);
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
		std::cout << "Errors occurred in test_stabilityflash2_vapour_liquid_3comp(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflash2_vapour_liquid_3comp(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stabilityflash2_vapour_liquid()
{
	// Test stability flash
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING 2P STABILITYFLASH VAPOUR-LIQUID\n" : "");
    int error_output = 0;

	// Test 1 ->Y8 mixture
	std::vector<std::string> comp{"C1", "C2", "C3", "nC5", "nC7", "nC10"};
	CompData comp_data(comp);
	comp_data.Pc = {45.99, 48.72, 42.48, 33.70, 27.40, 21.10};
	comp_data.Tc = {190.56, 305.32, 369.83, 469.70, 540.20, 617.70};
	comp_data.ac = {0.011, 0.099, 0.152, 0.252, 0.350, 0.490};
	comp_data.kij = std::vector<double>(6*6, 0.);

	std::vector<double> z = {0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244};

	CubicEoS pr(comp_data, CubicEoS::PR);

	FlashParams flash_params(comp_data);
	flash_params.split_switch_tol = 1e-2;
	flash_params.split_tol = 1e-22;
	flash_params.split_max_iter = 50;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-2;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson};
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
    flash_params.vl_eos_name = "PR";
	flash_params.light_comp_idx = 0;

	std::vector<Reference> references = {
		Reference(1.013250, 298.150000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.961031, 0.0389694}),
		Reference(100.000000, 335.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.854433, 0.145567}),
		Reference(250.000000, 200.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0, 1}),
		Reference(16.000000, 235.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.84089, 0.15911}),
		Reference(220.000000, 335.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.888439, 0.111561}),
		// Reference(220.000000, 350.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0, 0.0362088, 0.963791}, 1e-1),
		// Reference(1., 150., z, {0.1878472774, 0.8121527226}),
		Reference(191.000000, 275.000000, {0.8097, 0.0566, 0.0306, 0.0457, 0.033, 0.0244}, {0.377731, 0.622269}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, test_derivs);
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

	// Test 2 -> 11 COMP mixture
	comp = std::vector<std::string>{"CO2", "C2", "C3", "C6", "N2+C1", "iC4+nC4", "iC5+nC5", "C7-C15", "C16-C27", "C28-C44", "C45-C80"};
	size_t nc = comp.size();
	comp_data = CompData(comp);
	comp_data.Mw = {44.0098, 30.0704, 44.0968, 86.1759, 16.1696, 58.1232, 72.1517, 138.9024, 287.0269, 481.4092, 798.4030};
    comp_data.Pc = {73.7646, 48.8387, 42.4552, 29.6882, 45.7788, 37.5365, 33.7809, 24.2755, 16.0835, 14.7207, 14.7919};
    comp_data.Tc = {304.200, 305.400, 369.800, 507.400, 189.410, 420.020, 465.993, 607.025, 751.327, 894.601, 1094.780};
    comp_data.ac = {0.22500, 0.09800, 0.15200, 0.29600, 0.00859, 0.18785, 0.24159, 0.60567, 0.94451, 1.21589, 1.08541};
	comp_data.kij = std::vector<double>(nc*nc, 0.);

	comp_data.set_binary_coefficients(0, {0.0000,    0.1200,     0.1200,     0.1200,     0.1184,     0.1200,     0.1200,     0.1000,     0.1000,     0.1000,     0.1000});
    comp_data.set_binary_coefficients(1, {0.1200,    0.0000,     0.0000,     0.0000,     0.0004,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(2, {0.1200,    0.0000,     0.0000,     0.0000,     0.0008,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(3, {0.1200,    0.0000,     0.0000,     0.0000,     0.0009,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(4, {0.1184,    0.0004,     0.0008,     0.0009,     0.0000,     0.0008,     0.0009,     0.0009,     0.0009,     0.0009,     0.0009});
    comp_data.set_binary_coefficients(5, {0.1200,    0.0000,     0.0000,     0.0000,     0.0008,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(6, {0.1200,    0.0000,     0.0000,     0.0000,     0.0009,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(7, {0.1000,    0.0000,     0.0000,     0.0000,     0.0009,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(8, {0.1000,    0.0000,     0.0000,     0.0000,     0.0009,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(9, {0.1000,    0.0000,     0.0000,     0.0000,     0.0009,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});
    comp_data.set_binary_coefficients(10, {0.1000,    0.0000,     0.0000,     0.0000,     0.0009,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000,     0.0000});

	pr = CubicEoS(comp_data, CubicEoS::PR);

	flash_params = FlashParams(comp_data);
	flash_params.split_switch_tol = 1e-2;
	flash_params.split_tol = 1e-22;
	flash_params.split_max_iter = 50;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-0;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson};
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.vl_eos_name = "PR";
	flash_params.light_comp_idx = 4;

	z = {0.112600, 0.073200, 0.046400, 0.011200, 0.500900, 0.027600, 0.015700, 0.111500, 0.046900, 0.030400, 0.023600};

	references = {
		Reference(279.000000, 333.000000, z, {0.272278, 0.727722}),
		Reference(300.000000, 333.000000, z, {0.234811, 0.765189}),
		Reference(558.000000, 333.000000, z, {0, 1}),
	};

	ref_string = "\treferences = {\n";
	write = true;

	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, false);
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
		std::cout << "Errors occurred in test_stabilityflash2_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflash2_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stabilityflashN_vapour_liquid()
{
	// Test Maljamar separator mixture (Orr, 1981), data from (Li, 2012)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = false;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH - MALJAMAR SEPARATOR\n" : "");
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
    flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-18;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson,
										  		  InitialGuess::Yi::Wilson13,
												  0};  // pure CO2 initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.eos_params["PR"]->rich_phase_order = {0, -1};

	std::vector<std::pair<Reference, double>> references = {
		// {Reference(64., 305.35, z_init, {0.98716891, 0.01283109005}), 0.65},
		// {Reference(64., 305.35, z_init, {0.98716891, 0.01283109005}), 0.68},
		{Reference(68.5, 305.35, z_init, {0.5856075976, 0.1405898757, 0.2738025267}), 0.90},
		{Reference(68., 305.35, z_init, {0.04762089785, 0.02586867064, 0.9265104315}), 0.70},
	};

	for (size_t j = 0; j < references.size(); j++)
	{
		references[j].first.composition[0] = references[j].second;
    	for (int i = 1; i < nc; i++)
    	{
        	references[j].first.composition[i] = z_init[i]*(1.0-references[j].second);
    	}
	}

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false, true};

	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;

		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (std::pair<Reference, double> condition: references)
				{
					error_output += condition.first.test(flash_ptr, verbose, test_derivs);
				}
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_stabilityflashN_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_vapour_liquid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stabilityflashN_vapour_liquid_sour_gas()
{
	// Test Sour Gas mixture (Haugen and Firoozabadi, 2011)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH - SOUR GAS\n" : "");
	int error_output = 0;

    std::vector<std::string> comp = {"CO2", "N2", "H2S", "C1", "C2", "C3"};
	int nc = 6;

    CompData comp_data(comp);
	comp_data.Pc = {73.9, 33.5, 89.4, 45.4, 48.2, 41.9};
	comp_data.Tc = {304.2, 126.2, 373.2, 190.6, 305.4, 369.8};
	comp_data.ac = {0.225, 0.04, 0.081, 0.008, 0.098, 0.152};
	comp_data.kij = std::vector<double>(6*6, 0.);
	comp_data.set_binary_coefficients(0, {0., -0.02, 0.12, 0.125, 0.135, 0.150});
	comp_data.set_binary_coefficients(1, {-0.02, 0., 0.2, 0.031, 0.042, 0.091});
	comp_data.set_binary_coefficients(2, {0.12, 0.2, 0., 0.1, 0.08, 0.08});

	std::vector<double> z_init = {0.0, 0.0703, 0.0197, 0.06860, 0.1056, 0.0297};
	double sumz = std::accumulate(z_init.begin(), z_init.end(), 0.);

	FlashParams flash_params(comp_data);
    flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-24;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->trial_comps = {InitialGuess::Yi::Wilson,
										  		  InitialGuess::Yi::Wilson13,
										  		  0};  // pure CO2 initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.eos_params["PR"]->rich_phase_order = {0, 2, -1};  // CO2-rich, H2S-rich and liquid hydrocarbon phase

	std::vector<std::pair<Reference, double>> references = {
		{Reference(25., 178.8, z_init, {0.0599257522068052, 0.588272976549261, 0., 0.351801271243934}), 0.6},
		{Reference(25., 178.8, z_init, {0.0458211071739927, 0.688763869460872, 0., 0.265415023365135}), 0.65},
		{Reference(25., 178.8, z_init, {0.0316945701065235, 0.783201562563939, 0., 0.185103867329538}), 0.7},
	};

	for (size_t j = 0; j < references.size(); j++)
	{
		references[j].first.composition[0] = references[j].second;
    	for (int i = 1; i < nc; i++)
    	{
        	references[j].first.composition[i] = z_init[i] / sumz * (1.0-references[j].second);
    	}
	}

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y};//, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false, true};

	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;

		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (std::pair<Reference, double> condition: references)
				{
					error_output += condition.first.test(flash_ptr, verbose, test_derivs);
				}
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_stabilityflashN_vapour_liquid_sour_gas(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_vapour_liquid_sour_gas(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_stabilityflashN_vapour_liquid_water()
{
	// Test Water/NWE mixture (Khan et al, 1992), data from (Li, 2018)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = false;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH\n" : "");
	int error_output = 0;

	std::vector<std::string> comp = {"H2O","CO2", "C1", "C2-3", "C4-6", "C7-14", "C15-24", "C25+"};
	CompData comp_data = CompData(comp);
	comp_data.Pc = {220.48, 73.76, 46.0, 45.05, 33.50,24.24, 18.03, 17.26};
	comp_data.Tc = {647.3, 304.20, 190.6, 343.64, 466.41, 603.07, 733.79, 923.2};
	comp_data.ac = {0.344, 0.225, 0.008, 0.13, 0.244, 0.6, 0.903, 1.229};
	comp_data.kij = std::vector<double>(8*8, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.1896, 0.4850, 0.5, 0.5, 0.5, 0.5, 0.5});
	comp_data.set_binary_coefficients(1, {0.1896, 0., 0.12, 0.12, 0.12, 0.09, 0.09, 0.09});

	std::vector<double> z = {0.5, 0.251925, 0.050625, 0.02950, 0.0371, 0.071575, 0.03725, 0.022025};

	FlashParams flash_params(comp_data);
    flash_params.split_switch_tol = 1e-1;
	flash_params.split_tol = 1e-18;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
    flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->stability_tol = 1e-15;
	flash_params.eos_params["PR"]->trial_comps = {0,
										  InitialGuess::Yi::Wilson,
										//   InitialGuess::Yi::Wilson13
										 };  // pure H2O initial guess
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.eos_params["PR"]->rich_phase_order = {0, 1, -1};
	flash_params.split_negative_flash_iter = 10;

	std::vector<Reference> references = {
		Reference(200., 500., z, {0.3557326153, 0.2394605522, 0.4048068325}),
	};

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (Reference condition: references)
				{
					error_output += condition.test(flash_ptr, verbose, test_derivs);
				}
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_stabilityflashN_vapour_liquid_water(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_vapour_liquid_water(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_negativeflash2_brine_vapour()
{
	//
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = false;
	std::cout << (verbose ? "TESTING 2P NEGATIVEFLASH VAPOUR-BRINE\n" : "");
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

	FlashParams flash_params(comp_data);

	CubicEoS pr(comp_data, CubicEoS::PR);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.9, 1.});
	flash_params.add_eos("AQ", &aq);

	flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-10;
	flash_params.rr2_tol = 1e-15;
	flash_params.min_z = 1e-13;
	flash_params.verbose = verbose;
	flash_params.set_eos_order({"AQ", "PR"});

	std::vector<std::vector<double>> zero = {{1.-1e-11, 1e-11, 0.}, {1e-11, 1.-1e-11, 0.}};
	std::vector<double> min_z = {1e-10, 1e-5};
	std::vector<Reference> references = {
		// Freezing point
		Reference(1.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {0.5, 0.25, 0.25}, {0.497049, 0.502951, 0}),
		Reference(1.000000, 273.150000, {0.8, 0.1, 0.1}, {0.79923, 0.20077, 0}),
		Reference(1.000000, 273.150000, {0.9, 0.05, 0.05}, {0.899955, 0.100045, 0}),
		Reference(1.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {0.5, 0.25, 0.25}, {0.502831, 0.497169, 0}),
		Reference(10.000000, 273.150000, {0.8, 0.1, 0.1}, {0.804872, 0.195128, 0}),
		Reference(10.000000, 273.150000, {0.9, 0.05, 0.05}, {0.905411, 0.0945892, 0}),
		Reference(10.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {0.5, 0.25, 0.25}, {0.513358, 0.486642, 0}),
		Reference(100.000000, 273.150000, {0.8, 0.1, 0.1}, {0.820591, 0.179409, 0}),
		Reference(100.000000, 273.150000, {0.9, 0.05, 0.05}, {0.921139, 0.0788615, 0}),
		Reference(100.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 0, 1}),

		// Boiling point
		// Reference(1., 373.15, {min_z[0], 0.5, 0.5-min_z[0]}, {1.838212638, 0., 0.}),
		// Reference(1., 373.15, {min_z[1], 0.5, 0.5-min_z[1]}, {1.103157324, 0., 0.}),
		Reference(1.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(1.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		// Reference(1., 373.15, {0.5*min_z[0], 1.-min_z[0], 0.5*min_z[0]}, {0., 1.}),
		// Reference(1., 373.15, {0.5*min_z[1], 1.-min_z[1], 0.5*min_z[1]}, {0., 1.}),
		// Reference(1., 373.15, {0.5*min_z[0], 0.5*min_z[0], 1.-min_z[0]}, {0., 1.}),
		// Reference(1., 373.15, {0.5*min_z[1], 0.5*min_z[1], 1.-min_z[1]}, {0., 1.}),
		// Reference(1., 373.15, {0.5, 0.25, 0.25}, {0.5210220846, 0., 0.}),
		// Reference(1., 373.15, {0.8, 0.1, 0.1}, {0.1795798851, 0.8204201149}),
		// Reference(1., 373.15, {0.9, 0.05, 0.05}, {-74.85585513, 0., 0.}),
		// Reference(1., 373.15, zero[0], {0.1000448299, 0.8999551701}),
		// Reference(1., 373.15, zero[1], {0.1000448299, 0.8999551701}),
		Reference(10.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		// Reference(10., 373.15, {0.5*min_z[0], 1.-min_z[0], 0.5*min_z[0]}, {0., 1.}),
		// Reference(10., 373.15, {0.5*min_z[1], 1.-min_z[1], 0.5*min_z[1]}, {0., 1.}),
		Reference(10.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		// Reference(10., 373.15, {0.5, 0.25, 0.25}, {0.439293082, 0.560706918}),
		// Reference(10., 373.15, {0.8, 0.1, 0.1}, {0.7763746233, 0.2236253767}),
		// Reference(10., 373.15, {0.9, 0.05, 0.05}, {0.8887319884, 0.1112680116}),
		// Reference(10., 373.15, zero[0], {1., 0.}),
		// Reference(10., 373.15, zero[1], {0., 1.}),
		// Reference(100., 373.15, {min_z[0], 0.5, 0.5-min_z[0]}, {0., 1.}),
		// Reference(100., 373.15, {min_z[1], 0.5, 0.5-min_z[1]}, {0., 1.}),
		Reference(100.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		// Reference(100., 373.15, {0.5, 0.25, 0.25}, {0.4958719596, 0.5041280404}),
		// Reference(100., 373.15, {0.8, 0.1, 0.1}, {0.8031386567, 0.1968613433}),
		// Reference(100., 373.15, {0.9, 0.05, 0.05}, {0.9053836364, 0.09461636357}),
		// Reference(100., 373.15, zero[0], {1., 0.}),
		// Reference(100., 373.15, zero[1], {0., 1.}),

		// Difficult conditions
		Reference(50.000000, 293.150000, {0.111112, 0.888888, 0}, {0.112972, 0.887028, 0}),
		Reference(50.000000, 293.150000, {0.222223, 0.777777, 0}, {0.226893, 0.773107, 0}),
		// Reference(109.29, 300.1, {0.999, 0.001, 0.}, {1., 0.}), // difficult conditions
		Reference(30.000000, 330.000000, {0.6, 0.39, 0.01}, {0.60191, 0.39809, 0}), // difficult conditions
	};

	// Test conventional (y/x) phase ordering
	std::cout << (verbose ? "CALCULATING IN ORDER: AQ-V\n" : "");

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			NegativeFlash flash(flash_params, {"AQ", "PR"}, {InitialGuess::Ki::Henry_AV});
			std::unique_ptr<Flash> flash_ptr = std::make_unique<NegativeFlash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, test_derivs);
				if (write) { condition.write_ref(ref_string); }
			}
			write = false;
		}
	}
	ref_string += "\t};\n";

	// Test reverse phase ordering
	std::cout << (verbose ? "CALCULATING IN ORDER: V-AQ\n" : "");
	for (FlashParams::SplitVars var: vars)
	{
		flash_params.split_variables = var;
		NegativeFlash flash(flash_params, {"AQ", "PR"}, {InitialGuess::Ki::Henry_AV});
		std::unique_ptr<Flash> flash_ptr = std::make_unique<NegativeFlash>(flash);
		for (Reference condition: references)
		{
			error_output += condition.test(flash_ptr, verbose, test_derivs);
		}
	}

	for (Reference condition: references)
	{
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			NegativeFlash flash(flash_params, {"PR", "AQ"}, {InitialGuess::Ki::Henry_VA});
			std::unique_ptr<Flash> flash_ptr = std::make_unique<NegativeFlash>(flash);
			error_output += condition.test(flash_ptr, verbose, test_derivs);
		}
	}
	
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_negativeflash2_brine_vapour(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_negativeflash2_brine_vapour(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stabilityflash2_brine_vapour()
{
	//
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = false;
	std::cout << (verbose ? "TESTING 2P STABILITYFLASH VAPOUR-BRINE\n" : "");
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

	CubicEoS pr(comp_data, CubicEoS::PR);
	pr.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {///InitialGuess::Yi::Wilson,
								  		  		  0, 1};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->stability_max_iter = 50;
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.7, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {//InitialGuess::Yi::Henry,
									  	  		  0};
	flash_params.eos_params["AQ"]->stability_max_iter = 10;
	flash_params.eos_params["AQ"]->use_gmix = true;

	flash_params.split_switch_tol = 1e-5;
	flash_params.rr2_tol = 1e-15;
	flash_params.min_z = 1e-13;
	flash_params.verbose = verbose;
	flash_params.set_eos_order({"AQ", "PR"});

	std::vector<std::vector<double>> zero = {{1., 0.}, {0., 1.}};
	std::vector<double> min_z = {1e-10, 1e-5};
	std::vector<Reference> references = {
		Reference(1.000000, 273.150000, {1e-10, 1-1e-10}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {1e-05, 0.99999}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {1-1e-10, 1e-10}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {0.99999, 1e-05}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {0.5, 0.5}, {0.497359, 0.502641, 0}),
		Reference(1.000000, 273.150000, {0.777777, 0.222223}, {0.77734, 0.22266, 0}),
		Reference(1.000000, 273.150000, {0.8, 0.2}, {0.799739, 0.200261, 0}),
		Reference(1.000000, 273.150000, {0.9, 0.1}, {0.900533, 0.0994673, 0}),
		Reference(1.000000, 273.150000, {1, 0}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {0, 1}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-10, 1-1e-10}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-05, 0.99999}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1-1e-10, 1e-10}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {0.99999, 1e-05}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {0.5, 0.5}, {0.505828, 0.494172, 0}),
		Reference(10.000000, 273.150000, {0.777777, 0.222223}, {0.787252, 0.212748, 0}),
		Reference(10.000000, 273.150000, {0.8, 0.2}, {0.809767, 0.190233, 0}),
		Reference(10.000000, 273.150000, {0.9, 0.1}, {0.91108, 0.0889203, 0}),
		Reference(10.000000, 273.150000, {1, 0}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {0, 1}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1e-10, 1-1e-10}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {1e-05, 0.99999}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {1-1e-10, 1e-10}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {0.99999, 1e-05}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {0.5, 0.5}, {0.51707, 0, 0.48293}),
		Reference(100.000000, 273.150000, {0.777777, 0.222223}, {0.805368, 0, 0.194632}),
		Reference(100.000000, 273.150000, {0.8, 0.2}, {0.828433, 0, 0.171567}),
		Reference(100.000000, 273.150000, {0.9, 0.1}, {0.932221, 0, 0.0677789}),
		Reference(100.000000, 273.150000, {1, 0}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {0, 1}, {0, 0, 1}),
		Reference(1.000000, 373.150000, {1e-10, 1-1e-10}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1e-05, 0.99999}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1-1e-10, 1e-10}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.99999, 1e-05}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.5, 0.5}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.777777, 0.222223}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.8, 0.2}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.9, 0.1}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1, 0}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0, 1}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1e-10, 1-1e-10}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1e-05, 0.99999}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1-1e-10, 1e-10}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {0.99999, 1e-05}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {0.5, 0.5}, {0.438921, 0.561079, 0}),
		Reference(10.000000, 373.150000, {0.777777, 0.222223}, {0.751764, 0.248236, 0}),
		Reference(10.000000, 373.150000, {0.8, 0.2}, {0.776793, 0.223207, 0}),
		Reference(10.000000, 373.150000, {0.9, 0.1}, {0.889417, 0.110583, 0}),
		Reference(10.000000, 373.150000, {1, 0}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {0, 1}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1e-10, 1-1e-10}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1e-05, 0.99999}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1-1e-10, 1e-10}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {0.99999, 1e-05}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {0.5, 0.5}, {0.497303, 0.502697, 0}),
		Reference(100.000000, 373.150000, {0.777777, 0.222223}, {0.784813, 0.215187, 0}),
		Reference(100.000000, 373.150000, {0.8, 0.2}, {0.807815, 0.192185, 0}),
		Reference(100.000000, 373.150000, {0.9, 0.1}, {0.911318, 0.0886817, 0}),
		Reference(100.000000, 373.150000, {1, 0}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {0, 1}, {0, 1, 0}),

		// Difficult conditions
		// Reference(50., 293.15, {0.1111118889, 0.8888881111}, {0.1129686354, 0.8870313646}),
		// Reference(50., 293.15, {0.222222778, 0.777777222}, {0.2268868604, 0.7731131396}),
		Reference(100.000000, 474.150000, {0.755102, 0.244898}, {0.698738, 0.301262, 0}),
	};

	// references = {
	// 	// Reference(50., 293.15, {0.1111118889, 0.8888881111}, {0.1129686354, 0.8870313646}),
	// 	// Reference(28.000000, 373.150000, {6.84210158e-01, 1.-6.84210158e-01}, {0.6733729618, 0.3266270382}),
	// 	// Reference(100.000000, 273.150000, {0.9, 0.1}, {0.0680642, 0.931936}),
	// 	Reference(100., 474.15, {0.7551015306, 0.2448984694}, {0.698737, 0.301263, 0}),
	// };

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (Reference condition: references)
				{
					// condition.print_conditions(true);
					error_output += condition.test(flash_ptr, verbose, test_derivs);
					if (write) { condition.write_ref(ref_string); }
				}
				write = false;
			}
		}
	}
	ref_string += "\t};\n";

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_stabilityflash2_brine_vapour(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflash2_brine_vapour(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stabilityflashN_brine_vapour()
{
	//
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH VAPOUR-BRINE\n" : "");
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

	FlashParams flash_params(comp_data);
	flash_params.split_tol = 1e-24;
	flash_params.split_switch_tol = 1e-2;
	flash_params.rr2_tol = 1e-15;
	flash_params.tpd_tol = 1e-10;
	flash_params.tpd_close_to_boundary = 1e-3;
	flash_params.split_negative_flash_iter = 10;
	flash_params.min_z = 1e-13;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	pr.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {//InitialGuess::Yi::Wilson,
										  		  0, 1, 2};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["PR"]->stability_tol = 1e-20;
	flash_params.eos_params["PR"]->stability_max_iter = 50;
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {//InitialGuess::Yi::Henry,
									 	  		  0};
	flash_params.eos_params["AQ"]->stability_switch_tol = 1e-2;
	flash_params.eos_params["AQ"]->stability_tol = 1e-16;
	flash_params.eos_params["AQ"]->stability_max_iter = 10;
	flash_params.eos_params["AQ"]->use_gmix = true;

	flash_params.set_eos_order({"AQ", "PR"});

	std::vector<std::vector<double>> zero = {{1.-1e-11, 1e-11, 0.}, {1e-11, 1.-1e-11, 0.}};
	std::vector<double> min_z = {1e-10, 1e-5};
	std::vector<Reference> references = {
		// Freezing point
		Reference(1.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {0.5, 0.25, 0.25}, {0.497049, 0.502951, 0}),
		Reference(1.000000, 273.150000, {0.8, 0.1, 0.1}, {0.79923, 0.20077, 0}),
		Reference(1.000000, 273.150000, {0.9, 0.05, 0.05}, {0.899955, 0.100045, 0}),
		Reference(1.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1-1e-11, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {0.5, 0.25, 0.25}, {0.502831, 0.497169, 0}),
		Reference(10.000000, 273.150000, {0.8, 0.1, 0.1}, {0.804872, 0.195128, 0}),
		Reference(10.000000, 273.150000, {0.9, 0.05, 0.05}, {0.905411, 0.0945892, 0}),
		Reference(10.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {0.5, 0.25, 0.25}, {0.513358, 0.486642, 0}),
		Reference(100.000000, 273.150000, {0.8, 0.1, 0.1}, {0.820591, 0.179409, 0}),
		Reference(100.000000, 273.150000, {0.9, 0.05, 0.05}, {0.921139, 0.0788615, 0}),
		Reference(100.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 0, 1}),

		// Boiling point
		Reference(1.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.5, 0.25, 0.25}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.8, 0.1, 0.1}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.9, 0.05, 0.05}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1-1e-11, 1e-11, 0}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {0.5, 0.25, 0.25}, {0.439271, 0.560729, 0}),
		Reference(10.000000, 373.150000, {0.8, 0.1, 0.1}, {0.776366, 0.223634, 0}),
		Reference(10.000000, 373.150000, {0.9, 0.05, 0.05}, {0.888728, 0.111272, 0}),
		Reference(10.000000, 373.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {0.5, 0.25, 0.25}, {0.495887, 0.504113, 0}),
		Reference(100.000000, 373.150000, {0.8, 0.1, 0.1}, {0.803165, 0.196835, 0}),
		Reference(100.000000, 373.150000, {0.9, 0.05, 0.05}, {0.905412, 0.0945877, 0}),
		Reference(100.000000, 373.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),

		// Difficult conditions
		// Reference(50., 283.15, zero[1], {1.}),
		// Reference(50., 293.15, {0.1111118889, 0.8888881111, 0.}, {0.1129686354, 0.8870313646}),
		// Reference(50., 293.15, {0.222222778, 0.777777222, 0.}, {0.2268868604, 0.7731131396}),
		Reference(109.290000, 300.100000, {0.999, 0.001, 0}, {1, 0, 0}),
		Reference(30.000000, 330.000000, {0.6+1e-5, 0.39, 0.01-1e-5}, {0.60191, 0.39809, 0}),
		Reference(7.00000000e+01, 2.88353535e+02, {5.26500000e-01, 4.15000000e-01, 5.85000000e-02}, {0.224854774495063, 0.235320684422642, 0.539824541082294}),
		// Reference(70.0, 302.0909, {0.0045, 0.995, 0.0005}, {0.00287959271429661, 0., 0.997120407285703}),
	};

	// references = {
	// 	// Reference(70.0, 280., {0.95, 0.045, 0.005}, {0.977384, 0.00969084, 0.0129254}),
	// 	Reference(100.000000, 373.150000, {0.5, 0.25, 0.25}, {0.495887, 0.504113, 0}),
	// };

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	// std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::alpha};
	// std::vector<FlashParams::SplitVars> split_vars = {FlashParams::lnK};
	std::vector<bool> modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (Reference condition: references)
				{
					// condition.print_conditions(true);
					error_output += condition.test(flash_ptr, verbose, test_derivs);
					if (write) { condition.write_ref(ref_string); }
				}
				write = false;
			}
		}
	}
	ref_string += "\t};\n";

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_stabilityflashN_brine_vapour(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_brine_vapour(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stabilityflashN_brine_vapour_h2s()
{
	// Test experimental mixtures from Huang (1985)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH H2O-C1-CO2-H2S\n" : "");
	int error_output = 0;

	std::vector<std::string> comp = {"H2O", "C1", "CO2", "H2S"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 46.04, 73.75, 89.63};
	comp_data.Tc = {647.14, 190.58, 304.10, 373.53};
	comp_data.ac = {0.328, 0.012, 0.239, 0.0942};
	comp_data.kij = std::vector<double>(4*4, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.47893, 0.19014, 0.105});
	comp_data.set_binary_coefficients(1, {0.47893, 0., 0.0936, 0.0912});
	comp_data.set_binary_coefficients(2, {0.19014, 0.0936, 0., 0.1093});
	comp_data.Mw = {18.015, 16.043, 44.01, 34.10};

	FlashParams flash_params(comp_data);
	flash_params.split_tol = 1e-25;
	flash_params.split_switch_tol = 1e-3;
	flash_params.split_negative_flash_iter = 6;
	flash_params.rr2_tol = 1e-15;
	flash_params.tpd_tol = 1e-7;
	flash_params.min_z = 1e-13;
	flash_params.verbose = verbose;

	CubicEoS ceos(comp_data, CubicEoS::PR);
	ceos.set_preferred_roots(0, 0.6, EoS::RootFlag::MAX);
	flash_params.add_eos("CEOS", &ceos);
	flash_params.eos_params["CEOS"]->trial_comps = {0, 1, 2, 3};
	flash_params.eos_params["CEOS"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["CEOS"]->stability_tol = 1e-25;
	flash_params.eos_params["CEOS"]->stability_max_iter = 50;
	flash_params.eos_params["CEOS"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	flash_params.eos_params["CEOS"]->rich_phase_order = {2, 3};

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {0};
	flash_params.eos_params["AQ"]->stability_max_iter = 10;
	flash_params.eos_params["AQ"]->use_gmix = true;

	flash_params.split_negative_flash_iter = 5;
	flash_params.set_eos_order({"AQ", "CEOS"});

	std::vector<double> mix_1 = {0.5, 0.15, 0.30, 0.05};
	std::vector<double> mix_2 = {0.5, 0.05, 0.05, 0.40};

	double tol = 3e-2;
	std::vector<Reference> references = {
		// Mixture 1
		Reference(10.000000, 273.150000, mix_1, {0.504587, 0.495413, 0, 0}, tol),
		Reference(20.000000, 273.150000, mix_1, {0.508901, 0.491099, 0, 0}, tol),
		Reference(30.000000, 273.150000, mix_1, {0.512345, 0.487655, 0, 0}, tol),
		Reference(40.000000, 273.150000, mix_1, {0.515024, 0.484976, 0, 0}, tol),
		Reference(50.000000, 273.150000, mix_1, {0.516953, 0.483047, 0, 0}, tol),
		Reference(60.000000, 273.150000, mix_1, {0.517349, 0.41723, 0.0654209, 0}, tol),
		Reference(70.000000, 273.150000, mix_1, {0.516794, 0.307902, 0.175304, 0}, tol),
		Reference(80.000000, 273.150000, mix_1, {0.516316, 0.191352, 0.292332, 0}, tol),
		Reference(90.000000, 273.150000, mix_1, {0.515963, 0, 0.484037, 0}, tol),
		Reference(100.000000, 273.150000, mix_1, {0.515891, 0, 0.484109, 0}, tol),
		Reference(110.000000, 273.150000, mix_1, {0.51589, 0, 0.48411, 0}, tol),
		Reference(120.000000, 273.150000, mix_1, {0.515923, 0, 0.484077, 0}, tol),
		Reference(130.000000, 273.150000, mix_1, {0.515975, 0, 0.484025, 0}, tol),
		Reference(140.000000, 273.150000, mix_1, {0.51604, 0, 0.48396, 0}, tol),

		// Mixture 2
		Reference(10.000000, 273.150000, mix_2, {0.509713, 0.490287, 0, 0}, tol),
		Reference(20.000000, 273.150000, mix_2, {0.510441, 0.162971, 0, 0.326588}, tol),
		Reference(30.000000, 273.150000, mix_2, {0.510885, 0.0842698, 0, 0.404845}, tol),
		Reference(40.000000, 273.150000, mix_2, {0.511322, 0.0484279, 0, 0.44025}, 3.1e-2),
		Reference(50.000000, 273.150000, mix_2, {0.511686, 0.0214761, 0, 0.466838}, tol),
		Reference(60.000000, 273.150000, mix_2, {0.511979, 0, 0, 0.488021}, tol),
		Reference(70.000000, 273.150000, mix_2, {0.512145, 0, 0, 0.487855}, tol),
		Reference(80.000000, 273.150000, mix_2, {0.512314, 0, 0, 0.487686}, tol),
		Reference(90.000000, 273.150000, mix_2, {0.512485, 0, 0, 0.487515}, tol),
		Reference(100.000000, 273.150000, mix_2, {0.512659, 0, 0, 0.487341}, tol),
		Reference(110.000000, 273.150000, mix_2, {0.512834, 0, 0, 0.487166}, tol),
		Reference(120.000000, 273.150000, mix_2, {0.513012, 0, 0, 0.486988}, tol),
		Reference(130.000000, 273.150000, mix_2, {0.513193, 0, 0, 0.486807}, tol),
		Reference(140.000000, 273.150000, mix_2, {0.513375, 0, 0, 0.486625}, tol),
	};

	// references = {
	// 	// Reference(60.000000, 273.150000, mix_1, {0.516794, 0.1753040, 0.307902}),
	// 	// Reference(80.000000, 273.150000, mix_1, {0.5163157513, 0.1913521695, 0.2923320792}),
	// 	Reference(50.000000, 273.150000, mix_2, {0.511686, 0.0214761, 0, 0.466838}, tol),
	// };

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::lnY};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::lnK};
	std::vector<bool> modcholesky = {true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (Reference condition: references)
				{
					// condition.print_conditions(true);
					error_output += condition.test(flash_ptr, verbose, test_derivs);
					if (write) { condition.write_ref(ref_string); }
				}
				write = false;
			}
		}
	}
	ref_string += "\t};\n";

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_stabilityflashN_brine_vapour_h2s(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_brine_vapour_h2s(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stabilityflashN_brine_vapour_oil()
{
	//
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = true;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH BRINE-VAPOUR-OIL\n" : "");
	int error_output = 0;

	std::vector<std::string> comp = {"H2O", "CO2", "nC10"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 73.75, comp_data::Pc["nC10"]};
	comp_data.Tc = {647.14, 304.10, comp_data::Tc["nC10"]};
	comp_data.ac = {0.328, 0.239, comp_data::ac["nC10"]};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.48});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.1});
	comp_data.Mw = {18.015, 44.01, comp_data::Mw["nC10"]};

	FlashParams flash_params(comp_data);
	// flash_params.split_switch_tol = 1e-3;
	flash_params.split_tol = 1e-24;
	// flash_params.tpd_tol = 1e-11;
	flash_params.tpd_close_to_boundary = 1e-2;
	// flash_params.rr2_tol = 1e-15;
	// flash_params.min_z = 1e-13;
	flash_params.verbose = verbose;

	CubicEoS pr(comp_data, CubicEoS::PR);
	pr.set_preferred_roots(0, 0.75, EoS::RootFlag::MAX);
	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->trial_comps = {//InitialGuess::Yi::Wilson,
										  		  0, 1, 2};
	flash_params.eos_params["PR"]->stability_switch_tol = 1e-2;
	flash_params.eos_params["PR"]->stability_max_iter = 50;
	flash_params.eos_params["PR"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};
	// flash_params.eos_params["PR"].rich_phase_order = {2, 1};
	// flash_params.eos_params["PR"].use_gmix = true;

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"] = {//InitialGuess::Yi::Henry,
									 	  0};
	flash_params.eos_params["AQ"]->stability_max_iter = 10;
	flash_params.eos_params["AQ"]->use_gmix = true;

	flash_params.set_eos_order({"AQ", "PR"});

	std::vector<std::vector<double>> zero = {{1.-1e-11, 1e-11, 0.}, {1e-11, 1.-1e-11, 0.}};
	std::vector<double> min_z = {1e-10, 1e-5};
	std::vector<Reference> references = {
		// Freezing point
		Reference(1.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 0.489108, 0.510892}),
		Reference(1.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0.489118, 0.510882}),
		Reference(1.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {0.999995, 0, 5.00109e-06}),
		Reference(1.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 0, 1}),
		Reference(1.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 1}),
		Reference(1.000000, 273.150000, {0.1, 0.1, 0.8}, {0.0994749, 0.0828584, 0.817667}),
		Reference(1.000000, 273.150000, {0.5, 0.25, 0.25}, {0.499011, 0.245544, 0.255445}),
		Reference(1.000000, 273.150000, {0.8, 0.1, 0.1}, {0.8004, 0.0974219, 0.102178}),
		Reference(1.000000, 273.150000, {0.9, 0.05, 0.05}, {0.900863, 0.0480479, 0.0510893}),
		Reference(1.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(1.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 0.366324, 0.633676}),
		Reference(10.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0.366336, 0.633664}),
		Reference(10.000000, 273.150000, {1-1e-11, 5e-11, 5e-11}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(10.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 0, 1}),
		Reference(10.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 1}),
		Reference(10.000000, 273.150000, {0.1, 0.1, 0.8}, {0.100535, 0, 0.899465}),
		Reference(10.000000, 273.150000, {0.5, 0.25, 0.25}, {0.506011, 0.177143, 0.316846}),
		Reference(10.000000, 273.150000, {0.8, 0.1, 0.1}, {0.80984, 0.0634216, 0.126739}),
		Reference(10.000000, 273.150000, {0.9, 0.05, 0.05}, {0.911116, 0.0255144, 0.0633698}),
		Reference(10.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(10.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(100.000000, 273.150000, {1e-10, 0.5, 0.5}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {1e-05, 0.5, 0.49999}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 5.00063e-11}),
		Reference(100.000000, 273.150000, {0.99999, 5e-06, 5e-06}, {0.999995, 0, 5.00101e-06}),
		Reference(100.000000, 273.150000, {5e-11, 1-1e-10, 5e-11}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-06, 0.99999, 5e-06}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-11, 5e-11, 1-1e-10}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 1}),
		Reference(100.000000, 273.150000, {0.5, 0.25, 0.25}, {0.513397, 0, 0.486603}),
		Reference(100.000000, 273.150000, {0.8, 0.1, 0.1}, {0.820111, 0, 0.179889}),
		Reference(100.000000, 273.150000, {0.9, 0.05, 0.05}, {0.919791, 0, 0.0802089}),
		Reference(100.000000, 273.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(100.000000, 273.150000, {1e-11, 1-1e-11, 0}, {0, 0, 1}),

		// Boiling point
		Reference(1.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 0.553318, 0.446682}),
		Reference(1.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 0.553329, 0.446671}),
		Reference(1.000000, 373.150000, {1, 5e-11, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-11, 1, 5e-11}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {5e-11, 5e-11, 1}, {0, 0, 1}),
		Reference(1.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 1}),
		Reference(1.000000, 373.150000, {0.5, 0.25, 0.25}, {0, 0.833195, 0.166805}),
		Reference(1.000000, 373.150000, {0.8, 0.1, 0.1}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {0.9, 0.05, 0.05}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1, 1e-11, 0}, {0, 1, 0}),
		Reference(1.000000, 373.150000, {1e-11, 1, 0}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 0.468561, 0.531439}),
		Reference(10.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 0.468572, 0.531428}),
		Reference(10.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 5.04149e-11}),
		Reference(10.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0.999995, 0, 5.04247e-06}),
		Reference(10.000000, 373.150000, {5e-11, 1, 5e-11}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(10.000000, 373.150000, {5e-11, 5e-11, 1}, {0, 0, 1}),
		Reference(10.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 1}),
		Reference(10.000000, 373.150000, {0.5, 0.25, 0.25}, {0.469364, 0.265078, 0.265558}),
		Reference(10.000000, 373.150000, {0.8, 0.1, 0.1}, {0.788955, 0.104806, 0.10624}),
		Reference(10.000000, 373.150000, {0.9, 0.05, 0.05}, {0.895485, 0.0513816, 0.0531335}),
		Reference(10.000000, 373.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(10.000000, 373.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {1e-10, 0.5, 0.5}, {0, 0, 1}),
		Reference(100.000000, 373.150000, {1e-05, 0.5, 0.49999}, {0, 0, 1}),
		Reference(100.000000, 373.150000, {1-1e-10, 5e-11, 5e-11}, {1, 0, 5.03719e-11}),
		Reference(100.000000, 373.150000, {0.99999, 5e-06, 5e-06}, {0.999995, 0, 5.038e-06}),
		Reference(100.000000, 373.150000, {5e-11, 1-1e-10, 5e-11}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-06, 0.99999, 5e-06}, {0, 1, 0}),
		Reference(100.000000, 373.150000, {5e-11, 5e-11, 1-1e-10}, {0, 0, 1}),
		Reference(100.000000, 373.150000, {5e-06, 5e-06, 0.99999}, {0, 0, 1}),
		Reference(100.000000, 373.150000, {0.1, 0.1, 0.8}, {0.0932796, 0, 0.90672}),
		Reference(100.000000, 373.150000, {0.5, 0.25, 0.25}, {0.501412, 0, 0.498588}),
		Reference(100.000000, 373.150000, {0.8, 0.1, 0.1}, {0.80829, 0, 0.19171}),
		Reference(100.000000, 373.150000, {0.9, 0.05, 0.05}, {0.910002, 0, 0.0899982}),
		Reference(100.000000, 373.150000, {1-1e-11, 1e-11, 0}, {1, 0, 0}),
		Reference(100.000000, 373.150000, {1e-11, 1-1e-11, 0}, {0, 1, 0}),

		// // Difficult conditions
		Reference(30.000000, 330.000000, {0.6, 0.39, 0.01}, {0.602088, 0.3847, 0.0132115}),
	};

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::StabilityVars stab_var: stab_vars)
		{
			flash_params.stability_variables = stab_var;
			for (FlashParams::SplitVars split_var: split_vars)
			{
				flash_params.split_variables = split_var;
				Flash flash(flash_params);
				std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
				for (Reference condition: references)
				{
					// condition.print_conditions(true);
					error_output += condition.test(flash_ptr, verbose, test_derivs);
					if (write) { condition.write_ref(ref_string); }
				}
				write = false;
			}
		}
	}
	ref_string += "\t};\n";

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << ref_string;
		std::cout << "Errors occurred in test_stabilityflashN_brine_vapour_oil(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_brine_vapour_oil(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_negativeflash2_brine_vapour_ions()
{
	//
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = false;
	std::cout << (verbose ? "TESTING 2P NEGATIVEFLASH VAPOUR-BRINE WITH IONS\n" : "");
	int error_output = 0;

	std::vector<std::string> comp = {"H2O", "CO2", "C1"};
	std::vector<std::string> ions1 = {"Na+", "Cl-"};
	std::vector<std::string> ions2 = {"Ca2+", "Cl-"};
	std::vector<int> comp_idxs = {0, 1, 2};

	CompData comp_data(comp, ions1);
	comp_data.Pc = {220.50, 73.75, 46.04};
	comp_data.Tc = {647.14, 304.10, 190.58};
	comp_data.ac = {0.328, 0.239, 0.012};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.47893});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.0936});
	comp_data.Mw = {18.015, 44.01, 16.043};
	comp_data.charge = {1, -1};

	// Constant salinity test
	CubicEoS pr(comp_data, CubicEoS::PR);

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);

	FlashParams flash_params(comp_data);
	flash_params.rr_max_iter = 10;
	flash_params.split_max_iter = 50;
	// flash_params.split_tol = 1e-18;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	flash_params.eos_params["PR"]->set_active_components(comp_idxs);
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->set_active_components(comp_idxs);

	std::vector<std::vector<double>> Z = {{0.5, 0.25, 0.25}, {0.5, 5e-06, 0.499995}};
	std::vector<std::pair<Reference, double>> references = {
		// {Reference(1.01325, 298.15, Z[0], {0.472338242, 0.527661758}), 0.},
		// {Reference(1.01325, 298.15, Z[1], {0.4822510941, 0.5177489059}), 0.},
		// {Reference(100., 335., Z[0], {0.5020456125, 0.4979543875}), 0.},
		// {Reference(100., 335., Z[1], {0.4991218386, 0.5008781614}), 0.},
		// {Reference(54.5, 277.6, Z[0], {0.5100274182, 0.4899725818 }), 0.},
		// {Reference(54.5, 277.6, Z[1], {0.5007545521, 0.4992454479}), 0.},
		// {Reference(1.01325, 298.15, Z[0], {0.4733306958, 0.5266693042}), 1.},
		// {Reference(1.01325, 298.15, Z[1], {0.4828771866, 0.5171228134}), 1.},
		// {Reference(100., 335., Z[0], {0.5010578615, 0.4989421385}), 1.},
		// {Reference(100., 335., Z[1], {0.4990164151, 0.5009835849}), 1.},
		// {Reference(54.5, 277.6, Z[0], {0.5076528839, 0.4923471161}), 1.},
		// {Reference(54.5, 277.6, Z[1], {0.5004981572, 0.4995018428}), 1.},
	};

	std::vector<FlashParams::SplitVars> vars = {FlashParams::nik};//, FlashParams::lnK, FlashParams::lnK_chol};
	std::vector<bool> modcholesky = {false};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			NegativeFlash flash(flash_params, {"AQ", "PR"}, {InitialGuess::Ki::Henry_AV});
			std::unique_ptr<Flash> flash_ptr = std::make_unique<NegativeFlash>(flash);
			for (std::pair<Reference, double> condition: references)
			{
				// Set ion concentration
				std::vector<double> concentrations = {condition.second, condition.second};
				flash_params.eos_params["AQ"]->eos->get_comp_data().set_ion_concentration(concentrations);
				error_output += condition.first.test(flash_ptr, verbose, test_derivs);
			}
		}
	}

	// Include ions into flash
	comp_data = CompData(comp, ions2);
	comp_data.Pc = {220.50, 73.75, 46.04};
	comp_data.Tc = {647.14, 304.10, 190.58};
	comp_data.ac = {0.328, 0.239, 0.012};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.47893});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.0936});
	comp_data.Mw = {18.015, 44.01, 16.043};
	comp_data.charge = {2, -1};

	pr = CubicEoS(comp_data, CubicEoS::PR);
	aq = AQEoS(comp_data, evaluator_map);

	flash_params = FlashParams(comp_data);
	flash_params.rr_max_iter = 10;
	flash_params.split_max_iter = 50;
	flash_params.verbose = verbose;

	flash_params.add_eos("PR", &pr);
	flash_params.add_eos("AQ", &aq);
	flash_params.set_eos_order({"AQ", "PR"});

	references = {
		{Reference(1.01325, 298.15, Z[0], {0.4840357437, 0.5159642563}), 1.},
		{Reference(1.01325, 298.15, Z[1], {0.4839956611, 0.5160043389}), 1.},
		{Reference(100., 335., Z[0], {0.5012634142, 0.4987365858}), 1.},
		{Reference(100., 335., Z[1], {0.4989457184, 0.5010542816}), 1.},
		{Reference(54.5, 277.6, Z[0], {0.5059205224, 0.4940794776}), 1.},
		{Reference(54.5, 277.6, Z[1], {0.5003215263, 0.4996784737}), 1.},
	};

	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (FlashParams::SplitVars var: vars)
		{
			flash_params.split_variables = var;
			NegativeFlash flash(flash_params, {"AQ", "PR"}, {InitialGuess::Ki::Henry_AV});
			std::unique_ptr<Flash> flash_ptr = std::make_unique<NegativeFlash>(flash);

			for (std::pair<Reference, double> condition: references)
			{
				// Set ion and H2O
				std::vector<double> z = condition.first.composition;
				std::vector<double> z_mol = z;
				double z_ion = condition.second * z_mol[0] / 55.509;
				double zH2O = z_mol[0] + 3*z_ion;
				z_mol[0] = z[0]/zH2O * z[0];
				z_mol.push_back(z_ion/zH2O * z[0]);
				z_mol.push_back(2*z_ion/zH2O * z[0]);
				condition.first.composition = z_mol;

				error_output += condition.first.test(flash_ptr, verbose, test_derivs);
				condition.first.composition = z;
			}
		}
	}
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_negativeflash2_brine_vapour_ions(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_negativeflash2_brine_vapour_ions(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stabilityflashN_brine_vapour_si()
{
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	bool test_derivs = true;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH VAPOUR-BRINE-sI\n" : "");
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
	flash_params.eos_params["CEOS"]->trial_comps = {InitialGuess::Yi::Wilson,
										 			0, 1};
	flash_params.eos_params["CEOS"]->stability_switch_tol = 1e-1;
	flash_params.eos_params["CEOS"]->stability_tol = 1e-20;
	flash_params.eos_params["CEOS"]->stability_max_iter = 50;
	flash_params.eos_params["CEOS"]->root_order = {EoS::RootFlag::MAX, EoS::RootFlag::MIN};

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});
	flash_params.add_eos("AQ", &aq);
	flash_params.eos_params["AQ"]->trial_comps = {0};
	flash_params.eos_params["AQ"]->stability_max_iter = 10;
	flash_params.eos_params["AQ"]->use_gmix = true;

	Ballard si(comp_data, "sI");
	flash_params.add_eos("sI", &si);
	flash_params.eos_params["sI"]->trial_comps = {0};
	flash_params.eos_params["sI"]->stability_switch_tol = 1e2;
	flash_params.eos_params["sI"]->stability_max_iter = 20;
	flash_params.eos_params["sI"]->use_gmix = true;

	flash_params.tpd_tol = 1e-11;
	flash_params.rr2_tol = 1e-20;
	flash_params.split_tol = 1e-25;
	flash_params.split_switch_tol = 1e2;
	flash_params.split_negative_flash_iter = 10;
	flash_params.verbose = verbose;
	flash_params.set_eos_order({"AQ", "sI", "CEOS"});

	std::vector<Reference> references = {
		Reference(1.000000, 273.150000, {1e-15, 1}, {0, 0, 1, 0}),
		Reference(1.000000, 273.150000, {1e-05, 0.99999}, {0, 0, 1, 0}),
		Reference(1.000000, 273.150000, {1, 1e-15}, {1, 0, 0, 0}),
		Reference(1.000000, 273.150000, {1, 1e-10}, {1, 0, 0, 0}),
		Reference(1.000000, 273.150000, {0.99999, 1e-05}, {1, 0, 0, 0}),
		Reference(1.000000, 273.150000, {0.2, 0.8}, {0.194981, 0, 0.805019, 0}),
		Reference(1.000000, 273.150000, {0.58, 0.42}, {0.577995, 0, 0.422005, 0}),
		Reference(1.000000, 273.150000, {0.6, 0.4}, {0.598154, 0, 0.401846, 0}),
		Reference(10.000000, 273.150000, {1e-15, 1}, {0, 0, 1, 0}),
		Reference(10.000000, 273.150000, {1e-05, 0.99999}, {0, 0, 1, 0}),
		Reference(10.000000, 273.150000, {1, 1e-15}, {1, 0, 0, 0}),
		Reference(10.000000, 273.150000, {0.99999, 1e-05}, {1, 0, 0, 0}),
		Reference(10.000000, 273.150000, {0.2, 0.8}, {0.201904, 0, 0.798096, 0}),
		Reference(10.000000, 273.150000, {0.58, 0.42}, {0.586921, 0, 0.413079, 0}),
		Reference(10.000000, 273.150000, {0.6, 0.4}, {0.607185, 0, 0.392815, 0}),
		Reference(100.000000, 273.150000, {1e-15, 1}, {0, 0, 0, 1}),
		Reference(100.000000, 273.150000, {1e-05, 0.99999}, {0, 0, 0, 1}),
		Reference(100.000000, 273.150000, {1, 1e-15}, {1, 0, 0, 0}),
		Reference(100.000000, 273.150000, {0.99999, 1e-05}, {1, 0, 0, 0}),
		Reference(100.000000, 273.150000, {0.08, 0.92}, {0, 0.0912801, 0, 0.90872}),
		Reference(100.000000, 273.150000, {0.2, 0.8}, {0, 0.231074, 0, 0.768926}),
		Reference(100.000000, 273.150000, {0.58, 0.42}, {0, 0.673754, 0, 0.326246}),
		Reference(100.000000, 273.150000, {0.6, 0.4}, {0, 0.697053, 0, 0.302947}),
		Reference(100.000000, 273.150000, {0.92, 0.08}, {0.440733, 0.559267, 0, 0}),

		// Difficult conditions
		Reference(30.000000, 273.150000, {0.87755, 0.12245}, {0.0702384, 0.929762, 0, 0}),
		Reference(40.000000, 273.150000, {0.87755, 0.12245}, {0.0714563, 0.928544, 0, 0}),
		Reference(50.000000, 273.150000, {0.87755, 0.12245}, {0.0726628, 0.927337, 0, 0}),
		Reference(60.000000, 273.150000, {0.87755, 0.12245}, {0.0738581, 0.926142, 0, 0}),
		Reference(70.000000, 273.150000, {0.87755, 0.12245}, {0.0750424, 0.924958, 0, 0}),
		Reference(80.000000, 273.150000, {0.87755, 0.12245}, {0.0762156, 0.923784, 0, 0}),
		Reference(40.000000, 273.150000, {0.444445, 0.555555}, {0, 0.51532, 0, 0.48468}),
		Reference(40.000000, 273.150000, {0.250005, 0.749995}, {0, 0.289094, 0, 0.710906}),
		Reference(40.000000, 273.150000, {0.5, 0.5}, {0, 0.579956, 0, 0.420044}),
		Reference(40.000000, 273.150000, {0.749995, 0.250005}, {0, 0.870818, 0, 0.129182}),
		Reference(10.000000, 273.150000, {1e-06, 0.999999}, {0, 0, 1, 0}),
		// Reference(102.564100, 273.150000, {0.00502612, 0.994974}, {0, 0.00393581, 0, 0.996064}),
		Reference(19.743600, 273.150000, {0.522613, 0.477387}, {0, 0.60401, 0.39599, 0}),
		Reference(34.358974, 273.150000, {0.552764, 0.447236}, {0, 0.641748, 0.358252, 0}),
		// Reference(34.358974, 273.150000, {0.869346, 0.130654}, {9.47627e-05, 0.999905, 0, 0}),
		Reference(73.333333, 273.150000, {0.216081, 0.783919}, {0, 0.249722, 0, 0.750278}),
	};

	// references = {
	// 	// Reference(73.3333333, 273.150000, {0.2160809698, 0.7839190302}, {0, 0.2497220275, 0.7502779725}),
	// 	// Reference(100.000000, 273.150000, {0.92, 0.08}, {0, 0.440174346, 0.559825654}),
	// 	// Reference(87.94871795, 273.150000, {0.964823191, 0.03517680905}, {0, 0, 0}),
	// 	// Reference(100.000000, 273.150000, {0.6, 0.4}, {0, 0.697053, 0.302947}),
	// 	// Reference(34.358974 *(1.+1e-6), 273.150000, {0.869346, 0.130654}, {9.47628e-05, 0.999905, 0, 0}),
	// 	// Reference(100.000000, 273.150000, {0.08, 0.92}, {0, 0.09128005931, 0., 0.9087199407}),
	// 	// Reference(10.000010, 273.150000, {0.58, 0.42}, {0.586921, 0, 0.413079, 0}),
	// 	// Reference(100.000000, 273.150000, {0.08, 0.92}, {0, 0.0912801, 0, 0.90872}),
	// 	// Reference(40.000000, 273.150000, {0.250005, 0.749995}, {0, 0.289094, 0, 0.710906}),
	// 	Reference(1.000000, 273.150000, {0.2, 0.8}, {0.194981, 0, 0.805019, 0}),
	// };

	std::string ref_string = "\tstd::vector<Reference> references = {\n";
	bool write = true;

	std::vector<FlashParams::StabilityVars> stab_vars = {FlashParams::Y, FlashParams::lnY, FlashParams::alpha};
	std::vector<FlashParams::SplitVars> split_vars = {FlashParams::nik, FlashParams::lnK, FlashParams::lnK_chol};
	for (FlashParams::StabilityVars stab_var: stab_vars)
	{
		flash_params.stability_variables = stab_var;
		for (FlashParams::SplitVars split_var: split_vars)
		{
			flash_params.split_variables = split_var;
			Flash flash(flash_params);
			std::unique_ptr<Flash> flash_ptr = std::make_unique<Flash>(flash);
			for (Reference condition: references)
			{
				error_output += condition.test(flash_ptr, verbose, test_derivs);
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
		std::cout << "Errors occurred in test_stabilityflashN_brine_vapour_si(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_brine_vapour_si(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_stabilityflashN_brine_vapour_ice()
{
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	bool verbose = false;
	std::cout << (verbose ? "TESTING NP STABILITYFLASH VAPOUR-BRINE WITH ICE\n" : "");
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

	PureSolid ice(comp_data, "Ice");

	FlashParams flash_params(comp_data);
	flash_params.add_eos("PR", &pr);
	flash_params.add_eos("AQ", &aq);
	flash_params.add_eos("Ih", &ice);

	flash_params.rr_max_iter = 10;
	flash_params.split_max_iter = 50;
	flash_params.split_variables = FlashParams::lnK;
	flash_params.verbose = verbose;

	Flash flash(flash_params);
	std::vector<double> pres, temp, nu, x, nu_ref, x_ref;

	pres = {5., 10., 15.};
	temp = {253.15, 263.15, 273.15, 283.15, 298.15};
	std::vector<double> z = {0.8, 0.2-1e-3, 1e-3};
	std::vector<bool> modcholesky = {false, true};
	for (bool modchol: modcholesky)
	{
		flash_params.modChol_split = modchol;
		flash_params.modChol_stability = modchol;
		for (double T: temp)
		{
			for (double p: pres)
			{
				// std::cout << "Calculating T = " << T << "; P = " << p << std::endl;
				error_output += flash.evaluate(p, T, z);
				std::shared_ptr<FlashResults> flash_results = flash.get_flash_results();
				// print("nu", flash_results->nu);
				// print("x", flash_results->X, static_cast<int>(flash_results->nu.size()));
				// print("Error", error_output);
			}
		}
	}
	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_stabilityflashN_brine_vapour_ice(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_stabilityflashN_brine_vapour_ice(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}
