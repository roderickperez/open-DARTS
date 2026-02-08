#include <algorithm>
#include <numeric>
#include <chrono>
#include "dartsflash/eos/eos.hpp"
#include "dartsflash/eos/ideal.hpp"
#include "dartsflash/eos/aq/aq.hpp"
#include "dartsflash/eos/aq/jager.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"
// #include "dartsflash/eos/helmholtz/gerg.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"
#include "dartsflash/eos/iapws/iapws_ice.hpp"
#include "dartsflash/eos/solid/solid.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"
#include "dartsflash/eos/vdwp/munck.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/stability/stability.hpp"

int test_ideal();
int test_cubic();
int test_aq();
int test_iapws();
// int test_gerg();
int test_solid();
int test_vdwp();
int test_gmix();

int main() 
{
    int error_output = 0;

	error_output += test_ideal();
	error_output += test_cubic();
	error_output += test_iapws();
	error_output += test_aq();
	error_output += test_solid();
	error_output += test_vdwp();
	error_output += test_gmix();

    return error_output;
}

int test_ideal()
{
	// Test implementation of ideal gas enthalpy and derivative w.r.t. T
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	int error_output = 0;
	double d;

	std::vector<std::string> comp = {"C1"};
	CompData comp_data(comp);
	comp_data.cpi = {comp_data::cpi["C1"]};

	IdealGas ig(comp_data);
	std::vector<double> x{ 1. };
	double T = 300.;
	double dT = 1e-5;
	double tol = 1e-5;

	// Enthalpy and heat capacity test
	double H0 = ig.hi(T, 0);
	double H1 = ig.hi(T + dT, 0);
	double CP = ig.cpi(T, 0);
	double CP_num = (H1-H0)/dT;
	d = std::log(std::fabs(CP + 1e-15)) - std::log(std::fabs(CP_num + 1e-15));
	if (!(std::fabs(d) < tol)) { print("dH/dT != dH/dT", {CP, CP_num, d}); error_output++; }

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_ideal(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_ideal(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_cubic()
{
	// Test Cubics (Peng-Robinson and Soave-Redlich-Kwong)
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	int error, error_output = 0;
	bool verbose = false;

	// Pure CO2
	if (1)
	{
		double p{ 30. }, v, T{ 300. };
		std::vector<std::string> comp = {"CO2"};
		CompData comp_data(comp);
		comp_data.Pc = {73.75};
		comp_data.Tc = {304.1};
		comp_data.ac = {0.239};
		comp_data.kij = {0.};
		std::vector<double> n = {1.};

		std::vector<bool> volume_shifts = {false, true};
		for (bool volume_shift: volume_shifts)
		{
			CubicEoS pr(comp_data, CubicEoS::PR, volume_shift);
			CubicEoS srk(comp_data, CubicEoS::SRK, volume_shift);

			// Consistency tests for cubics
			v = pr.V(p, T, n);
			error = 0;
			error += pr.dlnphi_test(p, T, n, 1e-3, verbose);
			error += pr.derivatives_test(v, T, n, 2e-4, verbose);
			error += pr.lnphi_test(p, T, n, 2e-4, verbose);
			error += pr.pressure_test(p, T, n, 1e-5, verbose);
			// error += pr.temperature_test(p, T, n, 1e-5);
			error += pr.composition_test(p, T, n, 1e-5, verbose);
			error += pr.pvt_test(p, T, n, 1e-5, verbose);
			error += pr.critical_point_test(n, 1e2, verbose);
			error += pr.properties_test(p, T, n, 1e-5, verbose);
			error += pr.mix_dT_test(T, n, 1e-4);
			if (error || verbose) { print("PR.test() CO2", error); error_output += error; }

			v = srk.V(p, T, n);
			error = 0;
			error += srk.dlnphi_test(p, T, n, 1e-3, verbose);
			error += srk.derivatives_test(v, T, n, 2e-4, verbose);
			error += srk.lnphi_test(p, T, n, 3e-4, verbose);
			error += srk.pressure_test(p, T, n, 1e-5, verbose);
			// error += srk.temperature_test(p, T, n, 1e-5);
			error += srk.composition_test(p, T, n, 1e-5, verbose);
			error += srk.pvt_test(p, T, n, 1e-5, verbose);
			error += srk.critical_point_test(n, 1e2, verbose);
			error += srk.properties_test(p, T, n, 1e-5, verbose);
			error += srk.mix_dT_test(T, n, 1e-4);
			if (error || verbose) { print("SRK.test() CO2", error); error_output += error; }
		}
	}

	// MY10 mixture
	//// Sour gas mixture, data from (Li, 2012)
	if (1)
	{
		double p{ 30. }, v, T{ 300. };
		std::vector<std::string> comp = {"CO2", "N2", "H2S", "C1", "C2", "C3"};
		CompData comp_data(comp);
		comp_data.Pc = {73.819, 33.9, 89.4, 45.992, 48.718, 42.462};
		comp_data.Tc = {304.211, 126.2, 373.2, 190.564, 305.322, 369.825};
		comp_data.ac = {0.225, 0.039, 0.081, 0.01141, 0.10574, 0.15813};
		comp_data.kij = std::vector<double>(6*6, 0.);
		comp_data.set_binary_coefficients(0, {0., -0.02, 0.12, 0.125, 0.135, 0.150});
		comp_data.set_binary_coefficients(1, {-0.02, 0., 0.2, 0.031, 0.042, 0.091});
		comp_data.set_binary_coefficients(2, {0.12, 0.2, 0., 0.1, 0.08, 0.08});
		std::vector<double> n = {0.9, 0.03, 0.04, 0.06, 0.04, 0.03};

		std::vector<bool> volume_shifts = {false, true};
		for (bool volume_shift: volume_shifts)
		{
			CubicEoS pr(comp_data, CubicEoS::PR, volume_shift);
			CubicEoS srk(comp_data, CubicEoS::SRK, volume_shift);

			// Consistency tests for cubics
			v = pr.V(p, T, n);
			error = 0;
			error += pr.dlnphi_test(p, T, n, 1e-3);
			error += pr.derivatives_test(v, T, n, 2e-4);
			error += pr.lnphi_test(p, T, n, 2e-4);
			error += pr.pressure_test(p, T, n, 1e-5);
			// error += pr.temperature_test(p, T, n, 1e-5);
			error += pr.composition_test(p, T, n, 1e-5);
			error += pr.pvt_test(p, T, n, 1e-5);
			error += pr.critical_point_test(n, 1e0);
			error += pr.properties_test(p, T, n, 3e-3);
			error += pr.mix_dT_test(T, n, 1e-4);
			if (error || verbose) { print("PR.test() SourGas", error); error_output += error; }

			v = srk.V(p, T, n);
			error = 0;
			error += srk.dlnphi_test(p, T, n, 1e-3);
			error += srk.derivatives_test(v, T, n, 2e-4);
			error += srk.lnphi_test(p, T, n, 1e-3);
			error += srk.pressure_test(p, T, n, 1e-5);
			// error += srk.temperature_test(p, T, n, 1e-5);
			error += srk.composition_test(p, T, n, 1e-5);
			error += srk.pvt_test(p, T, n, 1e-5);
			error += srk.critical_point_test(n, 1e0);
			error += srk.properties_test(p, T, n, 4e-3);
			error += srk.mix_dT_test(T, n, 1e-4);
			if (error || verbose) { print("SRK.test() SourGas", error); error_output += error; }
		}
	}

	// H2O-CO2 mixture
	if (1)
	{
		std::vector<std::string> comp = {"H2O", "CO2"};
		CompData comp_data(comp);
		comp_data.Pc = {220.50, 73.75};
		comp_data.Tc = {647.14, 304.10};
		comp_data.ac = {0.328, 0.239};
		comp_data.kij = std::vector<double>(2*2, 0.);
		comp_data.set_binary_coefficients(0, {0., 0.19014});
		std::vector<double> n = {1., 0.};

		CubicEoS pr(comp_data, CubicEoS::PR);
		CubicEoS srk(comp_data, CubicEoS::SRK);
		error = pr.pvt_test(1., 360., n, 1e-4);
		if (error || verbose) { print("PR.pvt_test() H2O-CO2", error); error_output += error; }
		// error = pr.critical_point_test(n, 1e0);
		if (error || verbose) { print("PR.critical_point_test() H2O-CO2", error); error_output += error; }
		error = srk.pvt_test(1., 360., n, 1e-4);
		if (error || verbose) { print("SRK.pvt_test() H2O-CO2", error); error_output += error; }
		// error = srk.critical_point_test(n, 1e0);
		if (error || verbose) { print("SRK.critical_point_test() H2O-CO2", error); error_output += error; }
	}

	// 11 component mixture
	if (1)
	{
		std::vector<std::string> comp = {"CO2", "C2", "C3", "C6", "N2+C1", "iC4+nC4", "iC5+nC5", "C7-C15", "C16-C27", "C28-C44", "C45-C80"};
		CompData comp_data(comp);

		comp_data.Mw = {44.0098, 30.0704, 44.0968, 86.1759, 16.1696, 58.1232, 72.1517, 138.9024, 287.0269, 481.4092, 798.4030};
		comp_data.Pc = {73.7646, 48.8387, 42.4552, 29.6882, 45.7788, 37.5365, 33.7809, 24.2755, 16.0835, 14.7207, 14.7919};
		comp_data.Tc = {304.200, 305.400, 369.800, 507.400, 189.410, 420.020, 465.993, 607.025, 751.327, 894.601, 1094.780};
		comp_data.ac = {0.22500, 0.09800, 0.15200, 0.29600, 0.00859, 0.18785, 0.24159, 0.60567, 0.94451, 1.21589, 1.08541};

		comp_data.set_binary_coefficients(0, {0.00, 0.12, 0.12, 0.1184, 0.12, 0.12, 0.10, 0.10, 0.10, 0.10, 0.1});
		comp_data.set_binary_coefficients(1, {0.12, 0.00, 0.00, 0.0004, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(2, {0.12, 0.00, 0.00, 0.0008, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(3, {0.1184, 0.0004, 0.0008, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(4, {0.12, 0.00, 0.00, 0.00, 0.0008, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009});
		comp_data.set_binary_coefficients(5, {0.12, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(6, {0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(7, {0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(8, {0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		comp_data.set_binary_coefficients(9, {0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});
		// comp_data.set_binary_coefficients(10, {0.10, 0.00, 0.00, 0.00, 0.0009, 0.00, 0.00, 0.00, 0.00, 0.00, 0.0});

		CubicEoS pr(comp_data, CubicEoS::PR);
		CubicEoS srk(comp_data, CubicEoS::SRK);
		
		std::vector<std::vector<double>> nn = {{0.06641613561, 0.04806311519, 0.0331903831, 0.00998422608, 0.2891535984, 0.02140980071, 0.01306798678, 0.1141307836, 0.07294298073, 0.1004718332, 0.2311691567},
											   {0.1128686759, 0.0733462345, 0.0464768473, 0.0112070728, 0.5021318403, 0.02763601165, 0.01571531181, 0.1114846953, 0.04674849427, 0.02999235525, 0.02239246092},
											   {0.1434689708, 0.08424731332, 0.04290498632, 0.003556230043, 0.6941066604, 0.01942354845, 0.007698390366, 0.004571267023, 2.26046857e-05, 2.84682784e-08, 1.035786817e-10},
											   {0.06050743237, 0.05455723669, 0.05229796912, 0.02409915385, 0.1748563786, 0.0413980744, 0.02920302198, 0.2919463224, 0.1260073957, 0.08170111346, 0.06342590152},
											   {0.08747947066, 0.06260118661, 0.04391053469, 0.01335348325, 0.3602799875, 0.02844163953, 0.01746332637, 0.1638591177, 0.09204568264, 0.07208690825, 0.05847866284},
    										   {0.1293136564, 0.08025179906, 0.04805633722, 0.009767204589, 0.5944599143, 0.02704002478, 0.014526791, 0.07666346092, 0.01686283727, 0.002664093183, 0.000393881278}
											   };

		for (std::vector<double> n: nn)
		{
			error = pr.critical_point_test(n, 1e0);
			if (error || verbose) { print("PR.critical_point_test() 11C", error); error_output += error; }
			error = srk.critical_point_test(n, 1e0);
			if (error || verbose) { print("SRK.critical_point_test() 11C", error); error_output += error; }
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_cubic(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_cubic(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_aq()
{
	// Test AQ EoS
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error, error_output = 0;
	bool verbose = false;

	// Without ions
    std::vector<std::string> comp{"H2O", "CO2", "C1"};
	CompData comp_data(comp);

	std::vector<double> n{0.95, 0.045, 0.005};
	double p{20.}, T{300.};

	Ziabakhsh2012 zia(comp_data);
	error = zia.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("Ziabakhsh.derivatives_test()", error); error_output += error; }

	Jager2003 jag(comp_data);
	error = jag.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("Jager.derivatives_test()", error); error_output += error; }

	// Test AQEoS
	// Jager (2003) for water and solutes
	AQEoS aq(comp_data, AQEoS::Model::Jager2003);
	error = aq.dlnphi_test(p, T, n, 5e-3);
	if (error || verbose) { print("Jager.dlnphi_test()", error); error_output += error; }
	error = aq.properties_test(p, T, n, 1e-2);
	if (error || verbose) { print("Jager.properties_test()", error); error_output += error; }

	// Ziabakhsh (2012) for water and solutes
	aq = AQEoS(comp_data, AQEoS::Model::Ziabakhsh2012);
	error = aq.dlnphi_test(p, T, n, 3e-3);
	if (error || verbose) { print("Ziabakhsh.dlnphi_test()", error); error_output += error; }
	error = aq.properties_test(p, T, n, 4e-2);
	if (error || verbose) { print("Ziabakhsh.properties_test()", error); error_output += error; }

	// Test mixed AQEoS models: Jager (2003) for water and Ziabakhsh (2012) for solutes
	// Test passing evaluator_map with [CompType, Model]
	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	aq = AQEoS(comp_data, evaluator_map);
	error = aq.dlnphi_test(p, T, n, 5e-3);
	if (error || verbose) { print("AQEoS.dlnphi_test()", error); error_output += error; }
	error = aq.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("AQEoS.properties_test()", error); error_output += error; }

	/*
	// Test passing map of evaluator pointers to AQEoS constructor
	std::map<AQEoS::Model, AQBase*> evaluators = {
		{AQEoS::Model::Jager2003, &jag},
		{AQEoS::Model::Ziabakhsh2012, &zia}
	};
	aq = AQEoS(comp_data, evaluator_map, evaluators);
	error = aq.dlnphi_test(p, T, n, 5e-3);
	if (error || verbose) { print("AQEoS.dlnphi_test()", error); error_output += error; }
	error = aq.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("AQEoS.properties_test()", error); error_output += error; }
	*/

	// With ions
	// Na+ and Cl-
	std::vector<std::string> ions = {"Na+", "Cl-"};
	comp_data = CompData(comp, ions);
	comp_data.charge = {1, -1};

	n = {0.9, 0.045, 0.005, 0.025, 0.025};

	zia = Ziabakhsh2012(comp_data);
	error = zia.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("Ziabakhsh.derivatives_test() NaCl", error); error_output += error; }

	jag = Jager2003(comp_data);
	error = jag.derivatives_test(p, T, n, 2e-2);
	if (error || verbose) { print("Jager.derivatives_test() NaCl", error); error_output += error; }

	// Test Jager2003 with ions
	aq = AQEoS(comp_data, AQEoS::Model::Jager2003);
	error = aq.dlnphi_test(p, T, n, 2.e-2);
	if (error || verbose) { print("Jager.dlnphi_test() NaCl", error); error_output += error; }
	error = aq.properties_test(p, T, n, 1e-2);
	if (error || verbose) { print("Jager.properties_test() NaCl", error); error_output += error; }

	// Test Ziabakhsh2012 with ions
	aq = AQEoS(comp_data, AQEoS::Model::Ziabakhsh2012);
	error = aq.dlnphi_test(p, T, n, 3.e-3);
	if (error || verbose) { print("Ziabakhsh.dlnphi_test() NaCl", error); error_output += error; }
	error = aq.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ziabakhsh.properties_test() NaCl", error); error_output += error; }

	/*
	// Test mixed evaluators
	evaluators[AQEoS::Model::Jager2003] = &jag;
	evaluators[AQEoS::Model::Ziabakhsh2012] = &zia;

	aq = AQEoS(comp_data, evaluator_map, evaluators);
	error = aq.dlnphi_test(p, T, n, 3.e-3);
	if (error || verbose) { print("AQEoS.dlnphi_test() NaCl", error); error_output += error; }
	error = aq.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("AQEoS.properties_test() NaCl", error); error_output += error; }
	*/

	// Ca2+ and Cl-
	ions = {"Ca2+", "Cl-"};
	comp_data = CompData(comp, ions);
	comp_data.charge = {2, -1};

	n = {0.9, 0.045, 0.005, 0.01667, 0.03333};

	zia = Ziabakhsh2012(comp_data);
	error = zia.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("Ziabakhsh.derivatives_test() CaCl2", error); error_output += error; }

	jag = Jager2003(comp_data);
	// error = jag.derivatives_test(p, T, n, 2.1e-2);
	// if (error || verbose) { print("Jager.derivatives_test() CaCl2", error); error_output += error; }

	// Test Jager2003 with ions
	aq = AQEoS(comp_data, AQEoS::Model::Jager2003);
	error = aq.dlnphi_test(p, T, n, 1.e-1);
	if (error || verbose) { print("Jager.dlnphi_test() CaCl2", error); error_output += error; }
	error = aq.properties_test(p, T, n, 1e-2);
	if (error || verbose) { print("Jager.properties_test() CaCl2", error); error_output += error; }

	// Test Ziabakhsh2012 with ions
	aq = AQEoS(comp_data, AQEoS::Model::Ziabakhsh2012);
	error = aq.dlnphi_test(p, T, n, 6.e-3);
	if (error || verbose) { print("Ziabakhsh.dlnphi_test() CaCl2", error); error_output += error; }
	error = aq.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ziabakhsh.properties_test() CaCl2", error); error_output += error; }

	/*
	// Test mixed evaluators
	evaluators[AQEoS::Model::Jager2003] = &jag;
	evaluators[AQEoS::Model::Ziabakhsh2012] = &zia;

	aq = AQEoS(comp_data, evaluator_map, evaluators);
	error = aq.dlnphi_test(p, T, n, 6.e-3);
	if (error || verbose) { print("AQEoS.dlnphi_test() CaCl2", error); error_output += error; }
	error = aq.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("AQEoS.properties_test() CaCl2", error); error_output += error; }
	*/

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_aq(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_aq(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_iapws()
{
	// Test IAPWS-95 EoS
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	int error, error_output = 0;
	bool verbose = false;
	double p{ 30. }, V{ 1.807725808e-05 }, T{ 300. };

	// Pure H2O
	std::vector<std::string> comp = {"H2O"};
	CompData comp_data(comp);
	std::vector<double> n = {2.};

	// use IAPWS95 ideal gas properties
	bool iapws_ideal = true;
	IAPWS95 iapws95(comp_data, iapws_ideal);

	// IAPWS-95 references test
	error = iapws95.references_test(1e-5);
	if (error || verbose) { print("IAPWS95.references_test()", error); error_output += error; }

	// Switch to EoS ideal gas properties
	iapws_ideal = false;
	iapws95 = IAPWS95(comp_data, iapws_ideal);

	// Test of analytical derivatives
	error = iapws95.dlnphi_test(p, T, n, 2.2e-2);
	if (error || verbose) { print("IAPWS95.dlnphi_test()", error); error_output += error; }

    // Consistency tests for IAPWS-95
	error = iapws95.derivatives_test(2*V, T, n, 5e-4);
	if (error || verbose) { print("IAPWS95.derivatives_test()", {static_cast<double>(error), 2*V, T}); error_output += error; }
	double V2 = 2. * iapws95::Mw * 1e-3 / (0.6585693359 * iapws95::rhoc);
	error = iapws95.derivatives_test(V2, 640., n, 5e-2);  // problematic conditions
	if (error || verbose) { print("IAPWS95.derivatives_test()", {static_cast<double>(error), V2, 640.}); error_output += error; }
	error = iapws95.lnphi_test(p, T, n, 2e-4);
	if (error || verbose) { print("IAPWS95.lnphi_test()", error); error_output += error; }
	error = iapws95.pressure_test(p, T, n, 1e-5);
	if (error || verbose) { print("IAPWS95.pressure_test()", error); error_output += error; }
	error = iapws95.temperature_test(p, T, n, 1e-5);
	if (error || verbose) { print("IAPWS95.temperature_test()", error); error_output += error; }
	error = iapws95.composition_test(p, T, n, 1e-5);
	if (error || verbose) { print("IAPWS95.composition_test()", error); error_output += error; }
	error = iapws95.pvt_test(p, T, n, 6e-3);
	if (error || verbose) { print("IAPWS95.pvt_test()", error); error_output += error; }
	error = iapws95.properties_test(p, T, n, 3e-3);
	if (error || verbose) { print("IAPWS95.properties_test()", error); error_output += error; }

	// Test PT solver
	std::vector<double> pressure = logspace(1e-5, 1e4, 10);
	std::vector<double> temperature = {270., 300., 330., 473., 630., 640., 643., 645., 647., 648., 700., 1000.};
	for (double temp: temperature)
	{
		for (double pres: pressure)
		{
			double v = iapws95.V(pres, temp, n);
			if (v < 1.e-2)
			{
				error = iapws95.properties_test(pres, temp, n, 6e-1);
				if (error || verbose) { print("IAPWS95.properties_test() PT", {static_cast<double>(error), pres, temp}); error_output += error; }
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_iapws(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_iapws(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_solid()
{
	// Test Solid EOS
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error, error_output = 0;
	bool verbose = false;
	double p{ 30.}, T{ 273.15 };
	std::vector<double> n{ 2. };

	// Test IAPWSIce EoS
	CompData comp_data({"H2O"});
	IAPWSIce ice(comp_data, true);

	error = ice.references_test(1e-3);
	if (error || verbose) { print("IAPWSIce.references_test()", error); error_output += error; }

	ice = IAPWSIce(comp_data, false);
	error = ice.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("IAPWSIce.derivatives_test()", error); error_output += error; }
	error = ice.dlnphi_test(p, T, n, 1e-4);
	if (error || verbose) { print("IAPWSIce.dlnphi_test()", error); error_output += error; }
	error = ice.properties_test(p, T, n, 1e-3);
	if (error || verbose) { print("IAPWSIce.properties_test()", error); error_output += error; }
	error = ice.pvt_test(p, T, n, 1e-5);
	if (error || verbose) { print("IAPWSIce.pvt_test()", error); error_output += error; }

	// Test for PureSolid EoS for pure component only
	comp_data = CompData({"NaCl"});
	PureSolid s(comp_data, "NaCl");
	error = s.dlnphi_test(p, T, n, 2e-4);
	if (error || verbose) { print("PureSolid.dlnphi_test() NaCl", error); error_output += error; }
	error = s.pvt_test(p, T, n, 1e-5);
	if (error || verbose) { print("PureSolid.pvt_test() NaCl", error); error_output += error; }
	error = s.properties_test(p, T, n, 3e-2);
	if (error || verbose) { print("PureSolid.properties_test() NaCl", error); error_output += error; }
	error = s.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("PureSolid.derivatives_test() NaCl", error); error_output += error; }
	
	// Test PureSolid EoS for Ice for vector of components
	comp_data = CompData({"H2O", "CO2", "C1"});
	n = {1., 0., 0.};
	s = PureSolid(comp_data, "Ice");
	error = s.dlnphi_test(p, T, n, 2e-4);
	if (error || verbose) { print("PureSolid.dlnphi_test() Ice", error); error_output += error; }
	error = s.pvt_test(p, T, n, 1e-5);
	if (error || verbose) { print("PureSolid.pvt_test() Ice", error); error_output += error; }
	error = s.properties_test(p, T, n, 1e-5);
	if (error || verbose) { print("PureSolid.properties_test() Ice", error); error_output += error; }
	error = s.derivatives_test(p, T, n, 2e-5);
	if (error || verbose) { print("PureSolid.derivatives_test() Ice", error); error_output += error; }

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_solid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_solid(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_vdwp()
{
	// Test VdWP EoS implementations
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	int error, error_output = 0;
	bool verbose = false;

	double p{ 30. }, T{ 280. };

	// Single-component sI hydrates
	std::vector<std::string> comp{"H2O", "CO2"};
	CompData comp_data(comp);
	std::vector<double> n{0.9, 0.1};

	Ballard ballard(comp_data, "sI");
	error = ballard.derivatives_test(p, T, n, 2e-4);
	if (error || verbose) { print("Ballard.derivatives_test() sI 1C", error); error_output += error; }
	error = ballard.dlnphi_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ballard.dlnphi_test() sI 1C", error); error_output += error; }
	error = ballard.properties_test(p, T, n, 8e-4);
	if (error || verbose) { print("Ballard.properties_test() sI 1C", error); error_output += error; }

	Munck munck(comp_data, "sI");
	error = munck.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("Munck.derivatives_test() sI 1C", error); error_output += error; }
	error = munck.dlnphi_test(p, T, n, 1e-3);
	if (error || verbose) { print("Munck.dlnphi_test() sI 1C", error); error_output += error; }
	error = munck.properties_test(p, T, n, 2e-4);
	if (error || verbose) { print("Munck.properties_test() sI 1C", error); error_output += error; }

	// Multi-component sI hydrates
	comp = {"H2O", "CO2", "C1"};
	comp_data = CompData(comp);
	n = {0.86, 0.1, 0.04};
	
	ballard = Ballard(comp_data, "sI");
	error = ballard.derivatives_test(p, T, n, 2e-4);
	if (error || verbose) { print("Ballard.derivatives_test() sI 2C", error); error_output += error; }
	error = ballard.dlnphi_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ballard.dlnphi_test() sI 2C", error); error_output += error; }
	error = ballard.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ballard.properties_test() sI 2C", error); error_output += error; }

	munck = Munck(comp_data, "sI");
	error = munck.derivatives_test(p, T, n, 2e-5);
	if (error || verbose) { print("Munck.derivatives_test() sI 2C", error); error_output += error; }
	error = munck.dlnphi_test(p, T, n, 1e-3);
	if (error || verbose) { print("Munck.dlnphi_test() sI 2C", error); error_output += error; }
	error = munck.properties_test(p, T, n, 7e-4);
	if (error || verbose) { print("Munck.properties_test() sI 2C", error); error_output += error; }

	// Single-component sII hydrates
	comp = {"H2O", "CO2"};
	comp_data = CompData(comp);
	n = {0.9, 0.1};

	ballard = Ballard(comp_data, "sII");
	error = ballard.derivatives_test(p, T, n, 2e-4);
	if (error || verbose) { print("Ballard.derivatives_test() sII 1C", error); error_output += error; }
	error = ballard.dlnphi_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ballard.dlnphi_test() sII 1C", error); error_output += error; }
	error = ballard.properties_test(p, T, n, 3e-4);
	if (error || verbose) { print("Ballard.properties_test() sII 1C", error); error_output += error; }

	munck = Munck(comp_data, "sII");
	error = munck.derivatives_test(p, T, n, 1e-5);
	if (error || verbose) { print("Munck.derivatives_test() sII 1C", error); error_output += error; }
	error = munck.dlnphi_test(p, T, n, 1e-3);
	if (error || verbose) { print("Munck.dlnphi_test() sII 1C", error); error_output += error; }
	error = munck.properties_test(p, T, n, 2e-4);
	if (error || verbose) { print("Munck.properties_test() sII 1C", error); error_output += error; }

	// Multi-component sII hydrates
	comp = {"H2O", "CO2", "C1"};
	comp_data = CompData(comp);
	n = {0.86, 0.1, 0.04};
	
	ballard = Ballard(comp_data, "sII");
	error = ballard.derivatives_test(p, T, n, 2e-4);
	if (error || verbose) { print("Ballard.derivatives_test() sII 2C", error); error_output += error; }
	error = ballard.dlnphi_test(p, T, n, 3e-3);
	if (error || verbose) { print("Ballard.dlnphi_test() sII 2C", error); error_output += error; }
	error = ballard.properties_test(p, T, n, 2e-3);
	if (error || verbose) { print("Ballard.properties_test() sII 2C", error); error_output += error; }

	munck = Munck(comp_data, "sII");
	error = munck.derivatives_test(p, T, n, 2e-5);
	if (error || verbose) { print("Munck.derivatives_test() sII 2C", error); error_output += error; }
	error = munck.dlnphi_test(p, T, n, 1e-3);
	if (error || verbose) { print("Munck.dlnphi_test() sII 2C", error); error_output += error; }
	error = munck.properties_test(p, T, n, 3e-3);
	if (error || verbose) { print("Munck.properties_test() sII 2C", error); error_output += error; }

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_vdwp(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_vdwp(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_gmix()
{
	// Test implementation of Gibbs energy of mixing
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	const bool verbose = false;
	int error_output = 0;

	// Binary mixture H2O-CO2
	std::vector<std::string> comp = {"H2O", "CO2"};
	CompData comp_data(comp);
	comp_data.Pc = {220.50, 73.75};
    comp_data.Tc = {647.14, 304.10};
    comp_data.ac = {0.328, 0.239};
    comp_data.kij = std::vector<double>(2*2, 0.);
	comp_data.set_binary_coefficients(0, {0., 0.19014});
	comp_data.cpi = {comp_data::cpi["H2O"], comp_data::cpi["CO2"]};

	IdealGas ig(comp_data);
	CubicEoS ceos(comp_data, CubicEoS::PR);
	ceos.set_preferred_roots(0, 0.6, EoS::RootFlag::MAX);

	std::map<AQEoS::CompType, AQEoS::Model> evaluator_map = {
		{AQEoS::CompType::water, AQEoS::Model::Jager2003},
		{AQEoS::CompType::solute, AQEoS::Model::Ziabakhsh2012},
		{AQEoS::CompType::ion, AQEoS::Model::Jager2003}
	};
	AQEoS aq(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});

	Ballard si(comp_data, "sI");
	Ballard sii(comp_data, "sII");

	PureSolid ice(comp_data, "Ice");
	IAPWSIce iapws_ice(comp_data, true);

	FlashParams flashparams(comp_data);
	// flashparams.add_eos("IG", &ig);
	flashparams.add_eos("CEOS", &ceos);
	flashparams.add_eos("AQ", &aq);
	flashparams.add_eos("sI", &si);
	flashparams.eos_params["sI"]->stability_switch_tol = 1e2;
	flashparams.add_eos("sII", &sii);
	flashparams.eos_params["sII"]->stability_switch_tol = 1e2;
	flashparams.add_eos("Ice", &ice);
	// flashparams.add_eos("Ice", &iapws_ice);
	flashparams.eos_order = {"AQ", "CEOS", "sI", "sII", "Ice"};
	flashparams.verbose = verbose;

	std::vector<std::pair<std::vector<double>, std::vector<double>>> references = {
		{{1., 273.15}, {-1380.56319041565, -6.71936811919951}},
		{{10., 273.15}, {-1378.48291070747, 604.329217115214}},
		{{100., 273.15}, {-1359.0340216406, 924.968193405042}},
		{{1., 373.15}, {-38.6588882880286, -40.9908704568098}},
		{{10., 373.15}, {-19.9721426019564, 809.357755459948}},
		{{100., 373.15}, {0.328967440516408, 1580.67644739892}},
		{{1., 473.15}, {-180.284509180849, -205.748584809929}},
		{{10., 473.15}, {890.032491602665, 879.314554021592}},
		{{100., 473.15}, {1112.36641893965, 1928.7571106981}},
	};

	std::vector<std::vector<double>> n = {
		{0.95, 0.05}, {0.86, 0.14}, {0.05, 0.95}
	};

	for (auto ref: references)
	{
		double p = ref.first[0];
		double T = ref.first[1];
		flashparams.init_eos(p, T);
		std::vector<double> lnphi0 = flashparams.G_pure(p, T);
		std::vector<double> gpure = flashparams.prop_pure(EoS::Property::GIBBS, p, T);

		for (size_t i = 0; i < gpure.size(); i++)
		{
			if (std::fabs(gpure[i]-ref.second[i]) > 1e-6)
			{
				print("Different values for Gpure", {p, T, gpure[i]-ref.second[i]});
				print("result", gpure);
				print("ref", ref.second);
				error_output++;
			}
		}

		// Test for finding local minimum of Gibbs energy of mixing
		TrialPhase trial;

		trial = flashparams.find_ref_comp(p, T, n[0]);
		Stability stab(flashparams);
		stab.init_gmix(trial, lnphi0);
		error_output += stab.run(trial, true);
		if (verbose) { print("x", trial.ymin); print("Gmix", trial.gmin); }

		trial = flashparams.find_ref_comp(p, T, n[1]);
		stab.init_gmix(trial, lnphi0);
		error_output += stab.run(trial, true);
		if (verbose) { print("x", trial.ymin); print("Gmix", trial.gmin); }

		trial = flashparams.find_ref_comp(p, T, n[2]);
		stab.init_gmix(trial, lnphi0);
		error_output += stab.run(trial, true);
		if (verbose) { print("x", trial.ymin); print("Gmix", trial.gmin); }
	}

	// Ternary mixture H2O-CO2-C1
	comp = {"H2O", "CO2", "C1"};
	comp_data = CompData(comp);
	comp_data.Pc = {220.50, 73.75, 46.04};
	comp_data.Tc = {647.14, 304.10, 190.58};
	comp_data.ac = {0.328, 0.239, 0.012};
	comp_data.kij = std::vector<double>(3*3, 0.);
    comp_data.set_binary_coefficients(0, {0., 0.19014, 0.47893});
	comp_data.set_binary_coefficients(1, {0.19014, 0., 0.0936});
	comp_data.Mw = {18.015, 44.01, 16.043};
	comp_data.cpi = {comp_data::cpi["H2O"], comp_data::cpi["CO2"], comp_data::cpi["C1"]};

	ig = IdealGas(comp_data);
	ceos = CubicEoS(comp_data, CubicEoS::PR);
	ceos.set_preferred_roots(0, 0.6, EoS::RootFlag::MAX);

	aq = AQEoS(comp_data, evaluator_map);
	aq.set_eos_range(0, std::vector<double>{0.6, 1.});

	si = Ballard(comp_data, "sI");
	sii = Ballard(comp_data, "sII");

	ice = PureSolid(comp_data, "Ice");

	flashparams = FlashParams(comp_data);
	// flashparams.add_eos("IG", &ig);
	flashparams.add_eos("CEOS", &ceos);
	flashparams.add_eos("AQ", &aq);
	flashparams.add_eos("sI", &si);
	flashparams.eos_params["sI"]->stability_switch_tol = 1e2;
	flashparams.add_eos("sII", &sii);
	flashparams.eos_params["sII"]->stability_switch_tol = 1e2;
	// flashparams.add_eos("Ice", &ice);
	flashparams.eos_order = {"AQ", "CEOS", "sI", "sII"};
	flashparams.verbose = verbose;

	references = {
		{{1., 273.15}, {-1380.43243420251 ,-6.71936811919951, -5.37030069671095}},
		{{10., 273.15}, {-1378.48291070747, 604.329217115214, 616.400590524605}},
		{{100., 273.15}, {-1359.0340216406, 924.968193405042, 1179.20246636859}},
		{{1., 373.15}, {-38.6588882880286, -40.9908704568098, -39.2088811135913}},
		{{10., 373.15}, {-19.9721426019564, 809.357755459948, 816.770994464195}},
		{{100., 373.15}, {0.328967440516408, 1580.67644739892, 1648.9426955284}},
		{{1., 473.15}, {-180.284509180849, -205.748584809929, -202.313966189268}},
		{{10., 473.15}, {890.032491602665, 879.314554021592, 885.984816384403}},
		{{100., 473.15}, {1112.36641893965, 1928.7571106981, 1967.22790041651}},
	};

	n = {
		{0.95, 0.025, 0.025}, {0.025, 0.95, 0.025}, {0.025, 0.025, 0.95}
	};

	for (auto ref: references)
	{
		double p = ref.first[0];
		double T = ref.first[1];
		flashparams.init_eos(p, T);
		std::vector<double> lnphi0 = flashparams.G_pure(p, T);
		std::vector<double> gpure = flashparams.prop_pure(EoS::Property::GIBBS, p, T);

		for (size_t i = 0; i < gpure.size(); i++)
		{
			if (std::fabs(gpure[i]-ref.second[i]) > 1e-6)
			{
				print("Different values for Gpure", {p, T, gpure[i]-ref.second[i]});
				print("result", gpure);
				print("ref", ref.second);
				error_output++;
			}
		}

		// Test for finding local minimum of Gibbs energy of mixing
		TrialPhase trial;

		trial = flashparams.find_ref_comp(p, T, n[0]);
		Stability stab(flashparams);
		stab.init_gmix(trial, lnphi0);
		error_output += stab.run(trial, true);
		if (verbose) { print("x", trial.ymin); print("Gmix", trial.gmin); }

		trial = flashparams.find_ref_comp(p, T, n[1]);
		stab.init_gmix(trial, lnphi0);
		error_output += stab.run(trial, true);
		if (verbose) { print("x", trial.ymin); print("Gmix", trial.gmin); }

		trial = flashparams.find_ref_comp(p, T, n[2]);
		stab.init_gmix(trial, lnphi0);
		error_output += stab.run(trial, true);
		if (verbose) { print("x", trial.ymin); print("Gmix", trial.gmin); }
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_gmix(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_gmix(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}