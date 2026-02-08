#include <chrono>
#include <cmath>
#include <iostream>
#include "dartsflash/rr/rr.hpp"
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"

int test_rr_eq();
int test_rr_minimization();

struct Reference
{
	std::vector<double> Z, K, nu_ref;
    int NP, NC;
    double tolerance;

	Reference(const std::vector<double>& z, const std::vector<double>& k, const std::vector<double>& nu, int np, int nc, double tol) 
	: Z(z), K(k), nu_ref(nu), NP(np), NC(nc), tolerance(tol) {}

	int test(std::unique_ptr<RR>& rr, bool verbose)
	{
		if (verbose)
		{
			std::cout << "==================================\n";
			print("K", K);
			print("z", Z);
		}
		
        int error = rr->solve_rr(Z, K);
        std::vector<double> nu = rr->getnu();
        std::vector<double> X = rr->getx();

		if (verbose)
		{
            std::cout << "\nResults:\n";
			print("nu", nu);
			print("x", X, NP);
		}

		if (error > 0)
		{
			print("Error in RR", error);
			return error;
		}
			
		if (nu.size() != nu_ref.size())
		{
			std::cout << "nu and nu_ref are not the same size\n";
			print("nu", nu);
			print("nu_ref", nu_ref);
			return 1;
		}
		for (size_t j = 0; j < nu_ref.size(); j++)
		{
			if (std::sqrt(std::pow(nu_ref[j]-nu[j], 2)) > tolerance)
			{
				std::cout << "Different values for nu\n";
				print("nu", nu);
				print("nu_ref", nu_ref);
				return 1;
			}
		}

        // Check if z vector is equal to nu*X
        double norm = 0.;
        std::vector<double> z_check(NC, 0.);
        for (int i = 0; i < NC; i++)
        {
            for (int j = 0; j < NP; j++)
            {
                z_check[i] += nu[j] * X[j*NC + i];
            }
            norm += std::sqrt(std::pow(z_check[i]-Z[i], 2));
        }
        if (norm >= tolerance)
        {
            print("z != nu*x, norm", norm);
            print("z", Z);
            print("z_check", z_check);
            return 1;
        }
		return 0;
	}
};

int main() 
{
    /*
        test_rr
        Tests implementation of RR classes (Rachford-Rice).
        Equation solving: RR_EqConvex2
        Minimization: RR_Min
    */

    int error_output = 0;

    error_output += test_rr_eq();
    error_output += test_rr_minimization();

    return error_output;
}

int test_rr_eq() 
{
    // Test RR with negative flash with convex transformations (gradient based equation solving)
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	const bool verbose = false;
	std::cout << (verbose ? "TESTING RR NEGATIVE FLASH CONVEX\n" : "");
    int error_output = 0;
    std::vector<double> z, K;
    double tol = 1e-5;
    
    FlashParams flash_params;
    flash_params.rr2_tol = 1e-10;
    flash_params.rrn_tol = 1e-10;
    flash_params.rr_max_iter = 50;

    // Test 2-phase (Nichita and Leibovici, 2013, Examples 1, 2, 4, 5)
    RR_EqConvex2 rr2(flash_params, 6);

    std::vector<Reference> references = {
        Reference({0.770, 0.200, 0.010, 0.010, 0.005, 0.005}, 
                  {1.00003, 1.00002, 1.00001, 0.99999, 0.99998, 0.99997}, 
                  {1.-32967.22, 32967.22}, 2, 6, 1e-2),
        Reference({0.44, 0.55, 3.88E-03, 2.99E-03, 2.36E-03, 1.95E-03}, 
                  {161.59, 6.90, 0.15, 1.28E-03, 5.86E-06, 2.32E-08}, 
                  {1.-0.9923056, 0.9923056}, 2, 6, tol),
        Reference({0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244}, 
                  {1.000065, 0.999922, 0.999828, 0.999650, 0.999490, 0.999282}, 
                  {1.+264.53877, -264.53877}, 2, 6, tol),
        Reference({0.1789202106, 0.0041006011, 0.7815241261, 0.0164691242, 0.0189859122, 0.0000000257},
                  {445.995819899, 441.311360487, 411.625356748, 339.586063803, 29.7661058122, 0.00596602417}, 
                  {1.-1.00600180527780, 1.00600180527780}, 2, 6, tol),
        Reference({0.2, 0.2, 0.2, 0.2, 0.2, 0.},  // Test zero composition in RR_Eq2
                  {20., 200., 0.02, 0.002, 1.5, 10.},
                  {0.4816674639, 0.5183325361 }, 2, 6, tol)
    };

    for (Reference condition: references)
	{
        std::unique_ptr<RR> rr2_ptr = std::make_unique<RR_EqConvex2>(rr2);
		error_output += condition.test(rr2_ptr, verbose);
	}

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_rr_eq(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_rr_eq(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_rr_minimization()
{
    // Test RR with negative flash with minimization
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
	const bool verbose = false;
	std::cout << (verbose ? "TESTING RR NEGATIVE FLASH MINIMIZATION\n" : "");
    int error_output = 0;
    std::vector<double> z, K;
    double tol = 1e-5;

    FlashParams flash_params;
    flash_params.rr2_tol = 1e-14;
    flash_params.rrn_tol = 1e-14;
    flash_params.rr_max_iter = 100;

    // 3-phase example with inadmissible K-values (Iranshahr, 2010, Fig. 3a)
    RR_Min rr(flash_params, 3, 3);
    // z = {0.3, 0.3, 0.4};
    // K = {0.9, 1.2, 1.3, 2., 1.5, 0.3};
    // if (mic.solve_rr(z, K) != 1)
    // {
    //     error_output++;
    // }

    std::vector<Reference> references = {
        Reference({0.25, 0.5, 0.25},
                  {0.000282756, 274.816, 694.19,
                   0.000208581, 36.7979, 68616.3},
                  {0.2507170793, 0.4656819538, 0.2836009669}, 3, 3, tol),
        // Okuno (2010), Example 4
        Reference({0.08860, 0.81514, 0.09626}, 
                  {0.112359551, 13.72549020, 3.389830508,
                   1.011235955, 0.980392157, 0.847457627},
                  {-14.86, 1.20, 14.66}, 3, 3, tol)
    };
    for (Reference condition: references)
	{
        std::unique_ptr<RR> rr_ptr = std::make_unique<RR_Min>(rr);
		error_output += condition.test(rr_ptr, verbose);
        // error_output += condition.test(&mic, verbose);
	}
    
    // (Iranshahr, 2010, Fig. 2)
    flash_params.rr2_tol = 1e-10;
    flash_params.rrn_tol = 1e-10;
    rr = RR_Min(flash_params, 4, 3);

    references = {
        Reference({0.412, 0.155, 0.369, 0.063},
                  {0.938, 1.446, 0.543, 1.380, 
                   0.713, 0.953, 0.549, 3.533},
                  {1.840055556, -0.8277493941, -0.01230616156}, 3, 4, tol)
    };
    for (Reference condition: references)
	{
        std::unique_ptr<RR> rr_ptr = std::make_unique<RR_Min>(rr);
		error_output += condition.test(rr_ptr, verbose);
        // error_output += condition.test(&mic, verbose);
	}
    
    // // Test 3-phase 5-component (Yan, 2012, example 3)
    flash_params.rr2_tol = 1e-14;
    flash_params.rrn_tol = 1e-14;
    rr = RR_Min(flash_params, 5, 3);

    z = {0.66, 0.03, 0.01, 0.05, 0.25};  // C1, C2, C3, CO2, H2S
    K.resize(10);

    std::vector<double> Pc = {46.04, 48.721, 42.481, 73.75, 89.63};
    std::vector<double> Tc = {190.58, 305.32, 369.83, 304.10, 373.53};
    std::vector<double> ac = {0.012, 0.0995, 0.1523, 0.239, 0.0942};

    // std::vector<double> P = {35., 40., 42., 45.};
    std::vector<double> P = {42.};
    std::vector<std::vector<double>> nu_ref = {{-0.2850498062, 1.444498666, -0.1594488602}};
    double T = 201.;
    for (int j = 0; j < static_cast<int>(P.size()); j++)
    {
        std::vector<double> lnk(5);
        for (int i = 0; i < 5; i++)
        {
            lnk[i] = std::log(Pc[i]/P[j]) + 5.373 * (1. + ac[i]) * (1. - Tc[i]/T);
        }
        for (int i = 0; i < 5; i++)
        {
            K[i] = std::exp(-lnk[i]);
            K[5+i] = std::exp(-lnk[i]);
        }
        K[5] = std::exp(-lnk[0] - 1.);
        K[4] = std::exp(-lnk[4] - 1.);

        Reference ref = Reference(z, K, nu_ref[j], 3, 5, tol);
        std::unique_ptr<RR> rr_ptr = std::make_unique<RR_Min>(rr);
        error_output += ref.test(rr_ptr, verbose);
    }
    
    // Test 3-phase 21-component (Leibovici and Nichita, 2008, Table 1)
    flash_params.rr2_tol = 1e-10;
    flash_params.rrn_tol = 1e-10;
    rr = RR_Min(flash_params, 21, 3);
    references = {
        Reference({0.285714, 0.184210, 0.098280, 0.067017, 0.050844, 0.040959, 0.034293, 0.029492, 0.025871, 0.023041, 0.020770, 0.018906, 0.017349, 0.016029, 0.014896, 0.013912, 0.013051, 0.012289, 0.011612, 0.011006, 0.010459},
                  {2.4867, 4.7980, 2.6758, 1.7495, 1.1434, 0.7687, 0.5209, 0.3583, 0.2470, 0.1717, 0.1191, 0.0840, 0.0591, 0.0419, 0.0298, 0.0212, 0.0152, 0.0106, 0.0078, 0.0056, 0.0039,
                   6.8807, 0.0108, 0.0040, 0.0019, 8.08e-4, 2.64e-4, 7.49e-5, 1.92e-5, 3.95e-6, 8.06e-7, 1.34e-7, 1.79e-8, 2.70e-9, 1.98e-10, 1.55e-11, 1.64e-12, 1.51e-13, 2.58e-14, 2.13e-15, 1.92e-16, 2.38e-17},
                  {0.3609487663, 0.6350370418, 0.004014191875}, 3, 21, tol)
    };

    // // Test 3-phase 7-component
    flash_params.rr2_tol = 1e-14;
    flash_params.rrn_tol = 1e-14;
    rr = RR_Min(flash_params, 7, 3);

    references = {
        // Okuno (2010), Example 1
        Reference({0.204322076984, 0.070970999150, 0.267194323384, 0.296291964579, 0.067046080882, 0.062489248292, 0.031685306730},
                  {1.23466988745, 0.89727701141, 2.29525708098, 1.58954899888, 0.23349348597, 0.02038108640, 1.40715641002,
                   1.52713341421, 0.02456487977, 1.46348240453, 1.16090546194, 0.24166289908, 0.14815282572, 14.3128010831},
                  {0.2529728645, 0.6868328917, 0.06019424378}, 3, 7, tol),
        // Okuno (2010), Example 2
        Reference({0.132266176697, 0.205357472415, 0.170087543100, 0.186151796211, 0.111333894738, 0.034955417168, 0.159847699672},
                  {26.3059904941, 1.91580344867, 1.42153325608, 3.21966622946, 0.22093634359, 0.01039336513, 19.4239894458,
                   66.7435876079, 1.26478653025, 0.94711004430, 3.94954222664, 0.35954341233, 0.09327536295, 12.0162990083},
                  {0.06030232018, 0.4694531641, 0.4702445157}, 3, 7, tol),
        // Okuno (2010), Example 3
        Reference({0.896646630194, 0.046757914522, 0.000021572890, 0.000026632729, 0.016499094171, 0.025646758089, 0.014401397406},
                  {1.64571122126, 1.91627717926, 0.71408616431, 0.28582415424, 0.04917567928, 0.00326226927, 0.00000570946,
                   1.61947897153, 2.65352105653, 0.68719907526, 0.18483049029, 0.01228448216, 0.00023212526, 0.00000003964},
                  {0.1298344631, 0.8701633569, 2.180029611e-06}, 3, 7, tol),
        // Example from Maljamar separator oil, p = 70, xCO2 = 0.95
        Reference({0.95, 0.01177, 0.016475, 0.008565, 0.005495, 0.00287, 0.004825},
                  {0.6573594862, 15.7093378, 57.27372057, 327.8134439, 2917.276454, 43865.17768, 8106290.622,
                   0.9792609678, 3.598229074, 7.145423696, 18.01099508, 57.34736211, 241.1469575, 3821.270214},
                  {-0.0004441469017, 0.07812195105, 0.9223221958}, 3, 7, tol)
    };

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_rr_minimization(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_rr_minimization(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}
