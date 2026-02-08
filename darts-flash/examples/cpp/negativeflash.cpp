#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <cstring>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/flash/negative_flash.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/initial_guess.hpp"
#include "dartsflash/eos/aq/jager.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"

enum mixture { M7, M3, Y8, MY10, C1C16, OILB, MALJAMAR_RES, MALJAMAR_SEP, SOURGAS, BOBSLAUGHTERBLOCK, NORTHWARDESTES };
enum diagram { PT, PX, TX };

int main(int argc, char **argv)
{
	// ./build/examples/stabilityflash [mixture]
 	int error_output = 0;

	FlashParams flash_params;
 	CompData comp_data;
    std::vector<std::string> comp, eos_used;
 	std::vector<double> z, z_init, temp, pres, X;
    std::vector<int> initial_guesses;
    double T = 213.15;
    diagram diagram_type;

    if (argc == 1)
    {
        std::cout << "Please provide a mixture in the arguments list\n";
        return 0;
    }
    else if (std::strcmp(argv[1], "M7") == 0)
    {
        //// Seven-component gas mixture (Michelsen, 1982a fig. 2)
        comp = {"C1", "C2", "C3", "C4", "C5", "C6", "N2"};
        comp_data = CompData(comp);
	    comp_data.Pc = {45.99, 48.72, 42.48, 33.70, 27.40, 21.10, 34.00};
        comp_data.Tc = {190.56, 305.32, 369.83, 469.70, 540.20, 617.70, 126.20};
        comp_data.ac = {0.011, 0.099, 0.152, 0.252, 0.350, 0.490, 0.0377};
	    comp_data.kij = std::vector<double>(7*7, 0.);

		flash_params = FlashParams(comp_data);

    	CubicEoS cubic(comp_data, CubicEoS::PR);
 		flash_params.add_eos("CEOS", &cubic);

	 	eos_used = {"CEOS", "CEOS"};
 		initial_guesses = {InitialGuess::Ki::Wilson_VL};

		pres = {50.};
		temp = {150.};
        // pres = arange<double>(20., 90., 5.);
        // temp = arange<double>(150., 300., 5.);
        z = {0.9430, 0.0270, 0.0074, 0.0049, 0.0027, 0.0010, 0.0140};
        diagram_type = PT;
    }
    else if (std::strcmp(argv[1], "M3") == 0)
    {
        //// Ternary mixture (Michelsen, 1982a fig. 4)
        comp = {"C1", "CO2", "H2S"};
        comp_data = CompData(comp);
	    comp_data.Pc = {46.04, 73.75, 89.63};
        comp_data.Tc = {190.58, 304.10, 373.53};
        comp_data.ac = {0.012, 0.239, 0.0942};
	    comp_data.kij = std::vector<double>(3*3, 0.);

        pres = arange<double>(20., 200., 5.);
        temp = arange<double>(150., 300., 5.);
        z = {0.50, 0.10, 0.40};
        diagram_type = PT;
    }
 	else if (std::strcmp(argv[1], "Y8") == 0)
 	{
        //// Y8 mixture
        comp = {"C1", "C2", "C3", "nC5", "nC7", "nC10"};
        comp_data = CompData(comp);
	    comp_data.Pc = {45.99, 48.72, 42.48, 33.70, 27.40, 21.10};
        comp_data.Tc = {190.56, 305.32, 369.83, 469.70, 540.20, 617.70};
        comp_data.ac = {0.011, 0.099, 0.152, 0.252, 0.350, 0.490};
	    comp_data.kij = std::vector<double>(6*6, 0.);

		flash_params = FlashParams(comp_data);

    	CubicEoS cubic(comp_data, CubicEoS::PR);
 		flash_params.add_eos("CEOS", &cubic);

	 	eos_used = {"CEOS", "CEOS"};
 		initial_guesses = {InitialGuess::Ki::Wilson_VL};

        pres = arange<double>(10., 250., 5.);
        temp = arange<double>(190., 450., 10.);
        z = {0.8097, 0.0566, 0.0306, 0.0457, 0.0330, 0.0244};
        diagram_type = PT;
 	}
    else if (std::strcmp(argv[1], "MY10") == 0)
    {
        //// MY10 mixture
        comp = {"C1", "C2", "C3", "nC4", "nC5", "nC6", "nC7", "nC8", "nC10", "nC14"};
        comp_data = CompData(comp);
        comp_data.Pc = {45.99, 48.72, 42.48, 37.96, 33.70, 30.25, 27.40, 24.9, 21.10, 15.7};
	    comp_data.Tc = {190.56, 305.32, 369.83, 425.12, 469.70, 507.6, 540.20, 568.7, 617.70, 693.0};
    	comp_data.ac = {0.011, 0.099, 0.152, 0.2, 0.252, 0.3, 0.350, 0.399, 0.490, 0.644};
	    comp_data.kij = std::vector<double>(10*10, 0.);
        comp_data.set_binary_coefficients(0, {0., 0., 0., 0.02, 0.02, 0.025, 0.025, 0.035, 0.045, 0.045});

		flash_params = FlashParams(comp_data);

    	CubicEoS cubic(comp_data, CubicEoS::PR);
 		flash_params.add_eos("CEOS", &cubic);

        eos_used = {"CEOS", "CEOS"};
 		initial_guesses = {InitialGuess::Ki::Wilson_VL};

        pres = arange<double>(10., 180., 5.);
        temp = arange<double>(100., 600., 5.);
        z = {0.35, 0.03, 0.04, 0.06, 0.04, 0.03, 0.05, 0.05, 0.30, 0.05};
        diagram_type = PT;
    }
    else if (std::strcmp(argv[1], "C1C16") == 0)
    {
        //// C1C16 mixture
        comp = {"C1", "C16"};
        comp_data = CompData(comp);
        comp_data.Pc = {46.0, 14.0};
	    comp_data.Tc = {190.4, 723.0};
    	comp_data.ac = {0.011, 0.718};
	    comp_data.kij = {0.0, 0.05, 0.05, 0.0};

		flash_params = FlashParams(comp_data);

    	CubicEoS cubic(comp_data, CubicEoS::PR);
 		flash_params.add_eos("CEOS", &cubic);

        eos_used = {"CEOS", "CEOS"};
 		initial_guesses = {InitialGuess::Ki::Wilson_VL};

        pres = arange<double>(150., 300., 5.);
        temp = arange<double>(150., 300., 5.);
        z = {0.7, 0.3};
        diagram_type = PT;
    }
    else if (std::strcmp(argv[1], "H2SC1") == 0)
    {
        //// H2S-C1 mixture
        comp = {"H2S", "C1"};
        comp_data = CompData(comp);
        comp_data.Pc = {89.63, 46.0};
	    comp_data.Tc = {373.53, 190.4};
    	comp_data.ac = {0.0942, 0.011};
	    comp_data.kij = {0.0, 0.0912, 0.0912, 0.0};

		flash_params = FlashParams(comp_data);

    	CubicEoS cubic(comp_data, CubicEoS::PR);
 		flash_params.add_eos("CEOS", &cubic);

	 	eos_used = {"CEOS", "CEOS"};
 		initial_guesses = {InitialGuess::Ki::Wilson_VL};

        pres = std::vector<double>{40.6};
        T = 190.;
        X = std::vector<double>{0.025, 0.5, 0.9};
        z_init = {0., 1.};
        diagram_type = PX;
    }
    else if (std::strcmp(argv[1], "OilB") == 0)
    {
        //// Oil B mixture (Shelton and Yarborough, 1977), data from (Li, 2012)
        comp = {"CO2", "N2", "C1", "C2", "C3", "iC4", "nC4", "iC5", "nC5", "C6",
                "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"};
        comp_data = CompData(comp);
        comp_data.Pc = {73.819, 33.5, 45.4, 48.2, 41.9, 36., 37.5, 33.4, 33.3, 33.9,
                        25.3, 19.1, 14.2, 10.5, 7.5, 4.76};
        comp_data.Tc = {304.211, 126.2, 190.6, 305.4, 369.8, 408.1, 425.2, 460.4, 469.6, 506.35,
                        566.55, 647.06, 719.44, 784.93, 846.33, 919.39};
        comp_data.ac = {0.225, 0.04, 0.008, 0.098, 0.152, 0.176, 0.193, 0.227, 0.251, 0.299,
                        0.3884, 0.5289, 0.6911, 0.8782, 1.1009, 1.4478};
        comp_data.kij = std::vector<double>(16*16, 0.);
        comp_data.set_binary_coefficients(0, {0., -0.02, 0.075, 0.08, 0.08, 0.085, 0.085, 0.085, 0.085, 0.095,
                                              0.095, 0.095, 0.095, 0.095, 0.095, 0.095});
        comp_data.set_binary_coefficients(1, {0., 0., 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.06, 0.05,
                                              0.1, 0.12, 0.12, 0.12, 0.12, 0.12});
        comp_data.set_binary_coefficients(2, {0., 0., 0., 0.003, 0.01, 0.018, 0.018, 0.025, 0.026, 0.036,
                                              0.049, 0.073, 0.098, 0.124, 0.149, 0.181});

        pres = arange<double>(70., 100., 1.);
        X = arange<double>(0.65, 1., 0.01);
        T = 307.6;
        z_init = {0.0011, 0.0048, 0.1630, 0.0403, 0.0297, 0.0036, 0.0329, 0.0158, 0.0215, 0.0332,
                  0.181326, 0.161389, 0.125314, 0.095409, 0.057910, 0.022752};
        diagram_type = PX;
    }
    else if (std::strcmp(argv[1], "MaljamarRes") == 0)
    {
        //// Maljamar reservoir mixture (Orr, 1981), data from (Li, 2012)
        comp = {"CO2", "C1", "C2", "C3", "nC4", "C5-7", "C8-10", "C11-14", "C15-20", "C21-28", "C29+"};
        comp_data = CompData(comp);
        comp_data.Pc = {73.819, 45.4, 48.2, 41.9, 37.5, 28.82, 23.743, 18.589, 14.8, 11.954, 8.523};
        comp_data.Tc = {304.211, 190.6, 305.4, 369.8, 425.2, 516.667, 590., 668.611, 745.778, 812.667, 914.889};
        comp_data.ac = {0.225, 0.008, 0.098, 0.152, 0.193, 0.2651, 0.3644, 0.4987, 0.6606, 0.8771, 1.2789};
        comp_data.kij = std::vector<double>(11*11, 0.);
        comp_data.set_binary_coefficients(0, {0., 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115});
        comp_data.set_binary_coefficients(1, {0., 0., 0., 0., 0., 0.045, 0.055, 0.055, 0.06, 0.08, 0.28});

        pres = arange<double>(70., 96., 1.);
        X = arange<double>(0.60, 1., 0.01);
        T = 305.35;
        z_init = {0., 0.2939, 0.1019, 0.0835, 0.0331, 0.1204, 0.1581, 0.0823, 0.0528, 0.0276, 0.0464};
        diagram_type = PX;
    }
    else if (std::strcmp(argv[1], "MaljamarSep") == 0)
    {
        //// Maljamar separator mixture (Orr, 1981), data from (Li, 2012)
        comp = {"CO2", "C5-7", "C8-10", "C11-14", "C15-20", "C21-28", "C29+"};
        comp_data = CompData(comp);
        comp_data.Pc = {73.9, 28.8, 23.7, 18.6, 14.8, 12.0, 8.5};
	    comp_data.Tc = {304.2, 516.7, 590.0, 668.6, 745.8, 812.7, 914.9};
    	comp_data.ac = {0.225, 0.265, 0.364, 0.499, 0.661, 0.877, 1.279};
        comp_data.kij = std::vector<double>(7*7, 0.);
        comp_data.set_binary_coefficients(0, {0.0, 0.115, 0.115, 0.115, 0.115, 0.115, 0.115});

        pres = arange<double>(64., 80., 1.);
        X = arange<double>(0.65, 1., 0.01);
        T = 305.35;
        z_init = {0.0, 0.2354, 0.3295, 0.1713, 0.1099, 0.0574, 0.0965};
        diagram_type = PX;
    }
    else if (std::strcmp(argv[1], "SourGas") == 0)
    {
        //// Sour gas mixture, data from (Li, 2012)
        comp = {"CO2", "N2", "H2S", "C1", "C2", "C3"};
        comp_data = CompData(comp);
        comp_data.Pc = {73.819, 33.9, 89.4, 45.992, 48.718, 42.462};
        comp_data.Tc = {304.211, 126.2, 373.2, 190.564, 305.322, 369.825};
        comp_data.ac = {0.225, 0.039, 0.081, 0.01141, 0.10574, 0.15813};
        comp_data.kij = std::vector<double>(6*6, 0.);
        comp_data.set_binary_coefficients(0, {0., -0.02, 0.12, 0.125, 0.135, 0.150});
        comp_data.set_binary_coefficients(1, {-0.02, 0., 0.2, 0.031, 0.042, 0.091});
        comp_data.set_binary_coefficients(2, {0.12, 0.2, 0., 0.1, 0.08, 0.08});

        pres = arange<double>(1., 50., 1.);
        X = arange<double>(0.2, 1., 0.05);
        T = 178.8;
        z_init = {0.70592, 0.07026, 0.01966, 0.06860, 0.10559, 0.02967};
        diagram_type = PX;
    }
    else if (std::strcmp(argv[1], "BobSlaughterBlock") == 0)
    {
        //// Maljamar separator mixture (Orr, 1981), data from (Li, 2012)
        comp = {"CO2", "C1", "PC1", "PC2"};
        comp_data = CompData(comp);
        comp_data.Pc = {73.77, 46., 27.32, 17.31};
	    comp_data.Tc = {304.2, 160., 529.03, 795.33};
    	comp_data.ac = {0.225, 0.008, 0.481, 1.042};
        comp_data.kij = std::vector<double>(4*4, 0.);
        comp_data.set_binary_coefficients(0, {0., 0.055, 0.081, 0.105});

        pres = arange<double>(10., 280., 5.);
        X = arange<double>(0.05, 0.95, 0.01);
        T = 313.71;
        z_init = {0.0337, 0.0861, 0.6478, 0.2324};
        diagram_type = PX;
    }
    else if (std::strcmp(argv[1], "NorthWardEstes") == 0)
    {
        //// Maljamar separator mixture (Orr, 1981), data from (Li, 2012)
        comp = {"CO2", "C1", "PC1", "PC2", "PC3", "PC4", "PC5"};
        comp_data = CompData(comp);
        comp_data.Pc = {73.77, 46., 45.05, 33.51, 24.24, 18.03, 17.26};
	    comp_data.Tc = {304.2, 190.6, 343.64, 466.41, 603.07, 733.79, 923.2};
    	comp_data.ac = {0.225, 0.008, 0.13, 0.244, 0.6, 0.903, 1.229};
        comp_data.kij = std::vector<double>(7*7, 0.);
        comp_data.set_binary_coefficients(0, {0., 0.12, 0.12, 0.12, 0.09, 0.09, 0.09});

        pres = arange<double>(40., 200., 5.);
        X = arange<double>(0.05, 0.95, 0.01);
        T = 301.48;
        z_init = {0.0077, 0.2025, 0.118, 0.1484, 0.2863, 0.149, 0.0881};
        diagram_type = PX;
    }
	else if (std::strcmp(argv[1], "BrineVapour") == 0)
	{
		// Brine-vapour mixture
		comp = {"H2O", "CO2"};
		comp_data = CompData(comp);
		comp_data.Pc = {220.50, 73.75};
		comp_data.Tc = {647.14, 304.10};
		comp_data.ac = {0.328, 0.239};
		comp_data.kij = std::vector<double>(2*2, 0.);
        comp_data.set_binary_coefficients(0, {0., 0.19014});
		comp_data.Mw = {18.015, 44.01};

		z = {0.999, 0.001};

		flash_params = FlashParams(comp_data);

		CubicEoS cubic(comp_data, CubicEoS::PR);
		AQEoS aq(comp_data, AQEoS::Model::Ziabakhsh2012);

		flash_params.add_eos("PR", &cubic);
		flash_params.add_eos("AQ", &aq);
		
        eos_used = {"AQ", "PR"};
		initial_guesses = {InitialGuess::Ki::Henry_AV};

		pres = {100.};
		temp = {335.};
		// pres = arange<double>(10., 180., 5.);
        // temp = arange<double>(100., 600., 5.);
        diagram_type = PT;
	}
    else
    {
        std::cout << "Mixture " << argv[1] << " unknown\n";
        return 0;
    }

    int nc = comp.size();

    flash_params.split_variables = FlashParams::lnK;
    flash_params.split_switch_tol = 1e-3;
    flash_params.split_max_iter = 10;
    // flash_params.verbose = true;

    NegativeFlash neg_flash(flash_params, eos_used, initial_guesses);

    // P-T diagram
    if (diagram_type == PT)
    {
        // Initiate loop for P-T
        for (double p : pres)
        {
            for (double t : temp)
            {
                std::cout << "Calculating: P = " << p << ", T = " << t << std::endl;
                error_output += neg_flash.evaluate(p, t, z);
                std::shared_ptr<FlashResults> flash_results = neg_flash.get_flash_results();
                flash_results->print_results();
                print("Error", error_output);
            }
        }
 	}
    else  // P-x diagram
 	{
        for (double p : pres)
        {
            for (double x : X)
            {
                // Correct the feed according to the the CO2 concentration
                std::vector<double> Z(nc);
                Z[0] = x;
                for (int i = 1; i < nc; i++)
                {
                    Z[i] = z_init[i]*(1.0-x);
                }
                std::cout << "Calculating: P = " << p << ", X = " << x << std::endl;
                error_output += neg_flash.evaluate(p, T, Z);
                std::shared_ptr<FlashResults> flash_results = neg_flash.get_flash_results();
                flash_results->print_results();
                print("Error", error_output);
            }
        }
 	}

    std::cout << "Calculation finished." << std::endl;

    flash_params.timer.print_timers();

    return 0;
}