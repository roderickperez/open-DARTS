#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/maths/maths.hpp"
#include "dartsflash/eos/aq/jager.hpp"

namespace jager {
    // Reference conditions
    double R = 8.3145; double T_0 = 298.15; double P_0 = 1.0; double ms_0 = 1.;
    
    // gibbs energy of ideal gas at p0, T0
    std::unordered_map<std::string, double> gi0 = {
        {"H2O", -228700}, {"CO2", -394600}, {"N2", 0}, {"H2S", -33100}, 
        {"C1", -50830}, {"C2", -32900}, {"C3", -23500}, {"iC4", -20900}, {"nC4", -17200}, {"iC5", -15229}, {"nC5", -8370}, 
        {"nC6", -290}, {"nC7", 8120}, {"nC8", 16500}, {"nC9", 24811}, {"nC10", 33220},
        {"Ethene", 68170}, {"Propene", 62760}, {"Benzene", 129700}, {"Toluene", 122100}, {"MeOH", -162600}, {"EOH", -168400}, {"MEG", -304464},
        {"NaCl", -384138.00}, {"CaCl2", -748100}, {"KCl", -409140} 
    };
    // molar enthalpy of ideal gas at p0, T0
    std::unordered_map<std::string, double> hi0 = {
        {"H2O", -242000}, {"CO2", -393800}, {"N2", 0}, {"H2S", -20200}, 
        {"C1", -74900}, {"C2", -84720}, {"C3", -103900}, {"iC4", -134600}, {"nC4", -126200}, {"iC5", -165976}, {"nC5", -146500}, 
        {"nC6", -167300}, {"nC7", -187900}, {"nC8", -208600}, {"nC9", -229028}, {"nC10", -249655},
        {"Ethene", 52320}, {"Propene", 20400}, {"Benzene", 82980}, {"Toluene", 50030}, {"MeOH", -201300}, {"EOH", -235000}, {"MEG", -389314},
        {"NaCl", -411153.00}, {"CaCl2", -795800}, {"KCl", -436747}
    };

    // gibbs energy of pure H2O or 1 molal solution at p0, T0
    std::unordered_map<std::string, double> gi_0 =  {
        {"H2O", -237129}, {"CO2", -385974}, {"N2", 18188}, {"H2S", -27920}, 
        {"C1", -34451}, {"C2", -17000}, {"C3", -7550}, {"iC4", -2900}, {"nC4", -940}, {"iC5", 1451}, {"nC5", 9160}, 
        {"nC6", 18200}, {"nC7", 27500}, {"nC8", 36700}, {"nC9", 45200}, {"nC10", 54140},
        {"Ethene", 81379}, {"Propene", 74935}, {"Benzene", 133888}, {"Toluene", 126608}, {"MeOH", -175937}, {"EOH", -181293}, {"MEG", -325000},
        {"Na+", -261881}, {"Ca2+", -552790}, {"K+", -282462}, {"Cl-", -131039}
    };
    // molar enthalpy of pure H2O or 1 molal solution at p0, T0
    std::unordered_map<std::string, double> hi_0 =  {
        {"H2O", -285830.}, {"CO2", -413798.}, {"N2", -10439.}, {"H2S", -37660.}, 
        {"C1", -87906.}, {"C2", -103136.}, {"C3", -131000.}, {"iC4", -156000.}, {"nC4", -152000.}, {"iC5", -193476.}, {"nC5", -173887.}, 
        {"nC6", -199200.}, {"nC7", -225000.}, {"nC8", -250500.}, {"nC9", -273150.}, {"nC10", -297330.},
        {"Ethene", 35857.}, {"Propene", -1213.}, {"Benzene", 51170.}, {"Toluene", 13724.}, {"MeOH", -246312.}, {"EOH", -287232.}, {"MEG", -320000.},
        {"Na+", -240300.}, {"Ca2+", -543083.}, {"K+", -252170.}, {"Cl-", -167080.} 
    };
	std::vector<double> hi_a = {8.712 * R, 0.125E-2 * R, -0.018E-5 * R};
	// molar volume of pure H2O
	std::vector<std::vector<double>> vi_a = {{31.1251, -2.46176E-2, 8.69425E-6, -6.03348E-10}, // K^0
											 {-1.14154E-1, 2.15663E-4, -7.96939E-8, 5.57791E-12}, // K^-1
											 {3.10034E-4, -6.48160E-7, 2.45391E-10, -1.72577E-14}, // K^-2
											 {-2.48318E-7, 6.47521E-10, -2.51773E-13, 1.77978E-17}}; // K^-3
    
    // Born constants of solutes [eq. 3.6-3.7]
    std::unordered_map<std::string, double> omega = {
        {"CO2", -8368.}, {"N2", -145101.}, {"H2S", -41840.}, 
        {"C1", -133009.}, {"C2", -169870.}, {"C3", -211418.}, {"iC4", -253592.}, {"nC4", -253592.}, {"iC5", -301000.}, {"nC5", -300955.}, 
        {"nC6", -335180.}, {"nC7", -380158.}, {"nC8", -404258.}, {"nC9", -453600.}, {"nC10", -494000.},
        {"Ethene", -167360.}, {"Propene", -232547.}, {"Benzene", -82676.}, {"Toluene", -135896.}, {"MeOH", -61756.}, {"EOH", -85228.}, {"MEG", 406762.},
        {"Na+", 138323.}, {"Ca2+", 517142}, {"K+", 80626}, {"Cl-", 609190.} 
    };
    // Partial molar heat capacity terms [eq. 3.6]
    std::unordered_map<std::string, double> cp1 = {
        {"CO2", 167.50}, {"N2", 149.75}, {"H2S", 135.14}, 
        {"C1", 176.12}, {"C2", 226.67}, {"C3", 277.52}, {"iC4", 330.77}, {"nC4", 330.77}, {"iC5", 373.00}, {"nC5", 373.24}, 
        {"nC6", 424.53}, {"nC7", 472.37}, {"nC8", 522.13}, {"nC9", 571.90}, {"nC10", 621.10},
        {"Ethene", 163.59}, {"Propene", 209.62}, {"Benzene", 338.33}, {"Toluene", 392.98}, {"MeOH", 165.21}, {"EOH", 251.11}, {"MEG", -2.55},
        {"Na+", 76.065}, {"Ca2+", 37.656}, {"K+", 30.962}, {"Cl-", -18.410} 
    };
    std::unordered_map<std::string, double> cp2 = {
        {"CO2", 5304066.}, {"N2", 5046230.}, {"H2S", 2850801.}, 
        {"C1", 6310762.}, {"C2", 9011737.}, {"C3", 11749531.}, {"iC4", 14610096.}, {"nC4", 14610096.}, {"iC5", 16997948}, {"nC5", 16955051.}, 
        {"nC6", 19680558.}, {"nC7", 22283347.}, {"nC8", 24886075.}, {"nC9", 27607264.}, {"nC10", 30256344.},
        {"Ethene", 5846257.}, {"Propene", 8447000.}, {"Benzene", 1072758.}, {"Toluene", 1745012}, {"MeOH", -903211.}, {"EOH", 90828.}, {"MEG", 5711758.},
        {"Na+", -593488.}, {"Ca2+", -1520020}, {"K+", -1079442}, {"Cl-", -1176604.}
    };
    // Partial molar volume terms [eq. 3.7]
    std::unordered_map<std::string, std::vector<double>> vp = {
        {"CO2", {2.614, 3125.9, 11.7721, -129198.}},
        {"N2", {2.596, 3083.0, 11.9407, -129018.}},
        {"H2S", {2.724, 2833.6, 24.9559, -127989.}},
        {"C1", {2.829, 3651.8, 9.7119, -131365.}},
        {"C2", {3.612, 5565.2, 2.1778, -139277.}},
        {"C3", {4.503, 7738.2, -6.3316, -148260.}},
        {"iC4", {5.500, 11014.4, -14.9298, -157256.}}, 
        {"nC4", {5.500, 11014.4, -14.9298, -157256.}},
        {"iC5", {6.300, 12100.0, -23.4000, -166000.}}, 
        {"nC5", {6.282, 12082.2, -23.4091, -166218.}},
        {"nC6", {7.175, 14264.4, -32.0202, -175238.}},
        {"nC7", {8.064, 16435.2, -40.5342, -184213.}},
        {"nC8", {8.961, 18624.1, -49.1356, -193263.}}, 
        {"nC9", {9.830, 20730.0, -57.4000, -201950.}}, 
        {"nC10", {10.710, 22890.0, -65.9000, -210850.}},
        {"Ethene", {3.287, 5288.2, -7.8396, -138131.}}, 
        {"Propene", {4.170, 6925.3, -3.1648, -144900.}},
        {"Benzene", {5.491, 7579.0, 49.4653, -147599.}}, 
        {"Toluene", {6.287, 9169.4, 50.7724, -154176.}},
        {"MeOH", {2.903, 2307.3, 47.7051, -125.809}},
        {"EOH", {3.863, 4166.5, 50.8126, -133495.}},
        {"MEG", {4.000, 0., 0., 0.}},
        {"Na+", {0.7694, -956.04, 13.6231, -114056.}},
        {"Ca2+", {-0.0815, -3034.24, 22.1610, -103721.}}, 
        {"K+", {1.4891, -616.30, 22.7400, -113470.}}, 
        {"Cl-", {1.6870, 2008.74, 23.2756, -119118.}}
    };
    // Dielectric constant of water coefficients [eq. 3.9]
    std::vector<double> eps = {243.9576, 0.039037, -1.01261E-5, // n=0
                            -0.7520846, -2.12309E-4, 6.04961E-8,  // n=1
                            6.60648E-4, 3.18021E-7, -9.33341E-11}; // n=2

    // Solute interaction parameters eq. 3.18-3.20
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Bca = {
        {"Na+", {{"Cl-", {-0.554860699, 4.2795E-3, -6.529E-6}}}},
        {"K+", {{"Cl-", {0.178544751, -9.55043E-4, 1.8208E-6}}}},
        {"Ca2+", {{"Cl-", {0.549244833, -1.870735E-3, 3.3604E-6}}}},
		{"Cl-", {{"Na+", {-0.554860699, 4.2795E-3, -6.529E-6}},
				{"K+", {0.178544751, -9.55043E-4, 1.8208E-6}},
				{"Ca2+", {0.549244833, -1.870735E-3, 3.3604E-6}}}}
    };
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Cca = {
        {"Na+", {{"Cl-", {-0.016131327, -1.25089E-5, 5.89E-8}}}},
        {"K+", {{"Cl-", {-5.546927E-3, 4.22294E-5, -9.038E-8}}}},
        {"Ca2+", {{"Cl-", {-0.011031685, 7.49491E-5, -1.639E-7}}}},
		{"Cl-", {{"Na+", {-0.016131327, -1.25089E-5, 5.89E-8}},
				{"K+", {-5.546927E-3, 4.22294E-5, -9.038E-8}},
				{"Ca2+", {-0.011031685, 7.49491E-5, -1.639E-7}}}}
    };
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> Dca = {
        {"Na+", {{"Cl-", {-1.12161E-3, 2.49474E-5, -4.603E-8}}}},
        {"K+", {{"Cl-", {7.12650E-5, -6.04659E-7, 1.327E-9}}}},
        {"Ca2+", {{"Cl-", {1.08383E-4, -1.03524E-6, 2.3878E-9}}}},
		{"Cl-", {{"Na+", {-1.12161E-3, 2.49474E-5, -4.603E-8}},
				{"K+", {7.12650E-5, -6.04659E-7, 1.327E-9}},
				{"Ca2+", {1.08383E-4, -1.03524E-6, 2.3878E-9}}}}
    };
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> B = {
        {"H2S", {{"H2S", {0.1, -4.7e-4, 0., 0.}}}},
        {"CO2", {{"CO2", {0.107, -4.5e-4, 0., 0.}}}},
        {"C1", {{"Na+", {0.025, 0., 0., -5e-3}}, {"Cl-", {0.025, 0., 0., -5e-3}}}},
        {"C3", {{"Na+", {-0.09809, 4.19e-4, -6.2e-6, 0.}}, {"Cl-", {-0.09809, 4.19e-4, -6.2e-6, 0.}}}},
    };

    double e = 1.60218E-19; double eps0 = 8.85419E-12;
	double Theta = 228.; double Psi = 2600.;

    double Integral::simpson(double x0, double x1, int steps) 
    {
        // integrals solved numerically with simpson's rule
        double s = 0.;
        double h = (x1-x0)/steps;  // interval
    
	    // hi
        for (int i = 0; i < steps; i++) 
        {
            double hix = this->f(x0 + i*h);
            double hixh2 = this->f(x0 + (i+0.5)*h);
            double hixh = this->f(x0 + (i+1)*h);
            s += h*((hix + 4*hixh2 + hixh) / 6);
        }
	    return s;
    }

    IG::IG(std::string component_) : Integral(component_)
    {
        this->gi0 = jager::gi0[component_];
        this->hi0 = jager::hi0[component_];
        this->cpi = comp_data::cpi[component_];
    }
    double IG::H(double T)
    {
        // Integral of H(T)/RT^2 dT from T_0 to T
        return (-(this->hi0 / M_R
                - this->cpi[0] * jager::T_0 
                - 1. / 2 * this->cpi[1] * std::pow(jager::T_0, 2) 
                - 1. / 3 * this->cpi[2] * std::pow(jager::T_0, 3)
                - 1. / 4 * this->cpi[3] * std::pow(jager::T_0, 4)) * (1./T - 1./jager::T_0)
                + (this->cpi[0] * (std::log(T) - std::log(jager::T_0))
                + 1. / 2 * this->cpi[1] * (T - jager::T_0)
                + 1. / 6 * this->cpi[2] * (std::pow(T, 2) - std::pow(jager::T_0, 2))
                + 1. / 12 * this->cpi[3] * (std::pow(T, 3) - std::pow(jager::T_0, 3))));
    }
    double IG::dHdT(double T)
    {
        // Derivative of integral w.r.t. temperature
        return (this->hi0 / M_R + 
                this->cpi[0] * (T-jager::T_0) 
                + 1. / 2 * this->cpi[1] * (std::pow(T, 2)-std::pow(jager::T_0, 2)) 
                + 1. / 3 * this->cpi[2] * (std::pow(T, 3)-std::pow(jager::T_0, 3))
                + 1. / 4 * this->cpi[3] * (std::pow(T, 4)-std::pow(jager::T_0, 4))) / std::pow(T, 2);
    }
    double IG::d2HdT2(double T)
    {
        // Derivative of integral w.r.t. temperature
        return (this->cpi[0] + this->cpi[1] * T + this->cpi[2] * std::pow(T, 2) + this->cpi[3] * std::pow(T, 3)) / std::pow(T, 2) - 2. * this->dHdT(T) / T;
    }
    int IG::test_derivatives(double T, double tol, bool verbose)
    {
        int error_output = 0;
        double dH = this->dHdT(T);
        double d2H = this->d2HdT2(T);

        double d, dT{ 1e-5 };
        double H_ = this->H(T-dT);
        double H1 = this->H(T+dT);
        double dH_num = (H1-H_)/(2*dT);
        double dH_ = this->dHdT(T-dT);
        double dH1 = this->dHdT(T+dT);
        double d2H_num = (dH1-dH_)/(2*dT);

        d = std::log(std::fabs(dH + 1e-15)) - std::log(std::fabs(dH_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager IG dH/dT", {dH, dH_num, d}); error_output++; }
        d = std::log(std::fabs(d2H + 1e-15)) - std::log(std::fabs(d2H_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager IG d2H/dT2", {d2H, d2H_num, d}); error_output++; }
        
        return error_output;
    }

	double H::f(double T) 
    {
		// f(x) = h(T)/RT^2
		if (component == "H2O")  // pure water enthalpy
    	{
        	return (hi_0["H2O"] + hi_a[0] * (T - T_0) 
            		+ 1. / 2 * hi_a[1] * (std::pow(T, 2) - std::pow(T_0, 2)) 
					+ 1. / 3 * hi_a[2] * (std::pow(T, 3) - std::pow(T_0, 3))) / (R*pow(T, 2));
    	}
	    else  // solute molar enthalpy
	    {
			double c1 = cp1[component];
			double c2 = cp2[component];
			double om = omega[component];
			double a = 1.1048e-4;
			double b = -6.881e-7;

			return (hi_0[component] + c1*(T - T_0) - c2*(1./T - 1./T_0) 
					+ om * (a * (T - T_0) + 0.5 * b * (pow(T, 2) - pow(T_0, 2))))
					/ (R * pow(T, 2));
	    }
	}
	double H::F(double T) 
    {
		// Integral of h(T)/RT^2 dT from T_0 to T
		if (component == "H2O")
		{
			return (-(hi_0["H2O"] - hi_a[0]*T_0 - 1. / 2 * hi_a[1] * pow(T_0, 2) - 1. / 3 * hi_a[2] * pow(T_0, 3)) * (1./T - 1./T_0) 
					+ hi_a[0] * (std::log(T) - std::log(T_0)) 
					+ 1. / 2 * hi_a[1] * (T - T_0) 
					+ 1. / 6 * hi_a[2] * (pow(T, 2) - pow(T_0, 2))) / R;
		}
		else
		{
			double c1 = cp1[component];
			double c2 = cp2[component];
			double om = omega[component];
			double a = 1.1048e-4;
			double b = -6.881e-7;

			return (-(hi_0[component] - c1*T_0 + c2/T_0 - om*(a*T_0 + 0.5 * b * pow(T_0, 2))) * (1./T - 1./T_0)
					+ c1 * (std::log(T) - std::log(T_0))
					+ 0.5 * c2 * (std::pow(T, -2) - std::pow(T_0, -2))
					+ om * (a * (std::log(T) - std::log(T_0)) + 0.5 * b * (T - T_0))) / R;
		}
	}
    double H::dFdT(double T) 
    {
        return this->f(T);
    }
    double H::d2FdT2(double T) 
    {
        // f(x) = h(T)/RT^2
        // df(x)/dT = dh(T)/dT/RT^2 - 2 h(T)/RT^3
        double dhdT;
		if (component == "H2O")  // pure water enthalpy
    	{
            dhdT = hi_a[0] + hi_a[1] * T + hi_a[2] * std::pow(T, 2);
    	}
	    else  // solute molar enthalpy
	    {
			double c1 = cp1[component];
			double c2 = cp2[component];
			double om = omega[component];
			double a = 1.1048e-4;
			double b = -6.881e-7;

            dhdT = (c1 + c2*1./std::pow(T, 2) + om * (a + 0.5 * b * 2. * T));
	    }
        return dhdT / (R*pow(T, 2)) - 2 * this->dFdT(T) / T;
    }
    int H::test_derivatives(double T, double tol, bool verbose)
    {
        int error_output = 0;
        double dF = this->dFdT(T);
        double d2F = this->d2FdT2(T);

        double d, dT{ 1e-5 };
        double F_ = this->F(T-dT);
        double F1 = this->F(T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(T-dT);
        double dF1 = this->dFdT(T+dT);
        double d2F_num = (dF1-dF_)/(2*dT);

        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager H dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager H d2F/dT2", {d2F, d2F_num, d}); error_output++; }

        return error_output;
    }

	double V::f(double p) 
    {
		if (component == "H2O") 
    	{
			double vw = (vi_a[0][0] + vi_a[0][1] * p + vi_a[0][2] * std::pow(p, 2) + vi_a[0][3] * std::pow(p, 3))
						+ (vi_a[1][0] + vi_a[1][1] * p + vi_a[1][2] * std::pow(p, 2) + vi_a[1][3] * std::pow(p, 3)) * TT
						+ (vi_a[2][0] + vi_a[2][1] * p + vi_a[2][2] * std::pow(p, 2) + vi_a[2][3] * std::pow(p, 3)) * std::pow(TT, 2)
						+ (vi_a[3][0] + vi_a[3][1] * p + vi_a[3][2] * std::pow(p, 2) + vi_a[3][3] * std::pow(p, 3)) * std::pow(TT, 3);
			return vw * 1e-6 / (R*1e-5*TT);  // m3/mol / (m3.bar/K.mol). K
	    }
	    else 
    	{
			double tau = (5. / 6 * TT - Theta) / (1 + std::exp((TT-273.15) / 5));

			PX px;
    	    return (vp[component][0] 
					+ vp[component][1] / (Psi + p) 
					+ (vp[component][2] + vp[component][3] / (Psi + p)) / (TT - Theta - tau) 
					+ px.f(p, TT)) / (R*TT);  // J/mol.bar / (J/mol.K).K
    	}
	}
    double V::f(double p, double T)
    {
        this->TT = T;
        return this->f(p);
    }
    double V::dfdT(double p, double T)
    {
        if (component == "H2O") 
    	{
			double vw = (vi_a[0][0] + vi_a[0][1] * p + vi_a[0][2] * std::pow(p, 2) + vi_a[0][3] * std::pow(p, 3))
						+ (vi_a[1][0] + vi_a[1][1] * p + vi_a[1][2] * std::pow(p, 2) + vi_a[1][3] * std::pow(p, 3)) * T
						+ (vi_a[2][0] + vi_a[2][1] * p + vi_a[2][2] * std::pow(p, 2) + vi_a[2][3] * std::pow(p, 3)) * std::pow(T, 2)
						+ (vi_a[3][0] + vi_a[3][1] * p + vi_a[3][2] * std::pow(p, 2) + vi_a[3][3] * std::pow(p, 3)) * std::pow(T, 3);
            double dvwdT = vi_a[1][0] + vi_a[1][1] * p + vi_a[1][2] * std::pow(p, 2) + vi_a[1][3] * std::pow(p, 3)
						+ 2. * (vi_a[2][0] + vi_a[2][1] * p + vi_a[2][2] * std::pow(p, 2) + vi_a[2][3] * std::pow(p, 3)) * T
						+ 3. * (vi_a[3][0] + vi_a[3][1] * p + vi_a[3][2] * std::pow(p, 2) + vi_a[3][3] * std::pow(p, 3)) * std::pow(T, 2);
			return 1e-6 / (R*1e-5) * (dvwdT / T - vw / std::pow(T, 2));  // m3/mol / (m3.bar/K.mol). K
	    }
	    else 
    	{
			double tau = (5. / 6 * T - Theta) / (1 + std::exp((T-273.15) / 5));
            double dtaudT = 5. / 6 / (1 + std::exp((T-273.15) / 5)) - (5. / 6 * T - Theta) / std::pow(1 + std::exp((T-273.15) / 5), 2) * std::exp((T-273.15) / 5) / 5;

			PX px;
    	    return (-(vp[component][2] + vp[component][3] / (Psi + p)) / std::pow(T - Theta - tau, 2) * (1. - dtaudT)
					+ px.dfdT(p, T)) / (R*T)
                    - (vp[component][0] 
					+ vp[component][1] / (Psi + p) 
					+ (vp[component][2] + vp[component][3] / (Psi + p)) / (T - Theta - tau) 
					+ px.f(p, T)) / (R*std::pow(T, 2));  // J/mol.bar / (J/mol.K).K
    	}
    }
	double V::F(double p, double T) 
    {
		// Integral of v(p)/RT dp from P_0 to P
		if (component == "H2O")
		{
			return ((vi_a[0][0]/T + vi_a[1][0] + vi_a[2][0]*T + vi_a[3][0]*std::pow(T, 2)) * (p - P_0) 
					+ 1. / 2 * (vi_a[0][1]/T + vi_a[1][1] + vi_a[2][1]*T + vi_a[3][1]*std::pow(T, 2)) * (std::pow(p, 2) - std::pow(P_0, 2)) 
					+ 1. / 3 * (vi_a[0][2]/T + vi_a[1][2] + vi_a[2][2]*T + vi_a[3][2]*std::pow(T, 2)) * (std::pow(p, 3) - std::pow(P_0, 3)) 
					+ 1. / 4 * (vi_a[0][3]/T + vi_a[1][3] + vi_a[2][3]*T + vi_a[3][3]*std::pow(T, 2)) * (std::pow(p, 4) - std::pow(P_0, 4))) / (R*1e-5) * 1e-6;
		}
		else
		{
			this->TT = T;
			return this->simpson(P_0, p, 100);
		}
	}
	double V::dFdP(double p, double T) 
    {
        return this->f(p, T);
    }
    double V::dFdT(double p, double T) 
    {
        if (component == "H2O")
		{
			return ((-vi_a[0][0]/std::pow(T, 2) + vi_a[2][0] + 2*vi_a[3][0]*T) * (p - P_0) 
					+ 1. / 2 * (-vi_a[0][1]/std::pow(T, 2) + vi_a[2][1] + 2*vi_a[3][1]*T) * (std::pow(p, 2) - std::pow(P_0, 2)) 
					+ 1. / 3 * (-vi_a[0][2]/std::pow(T, 2) + vi_a[2][2] + 2*vi_a[3][2]*T) * (std::pow(p, 3) - std::pow(P_0, 3)) 
					+ 1. / 4 * (-vi_a[0][3]/std::pow(T, 2) + vi_a[2][3] + 2*vi_a[3][3]*T) * (std::pow(p, 4) - std::pow(P_0, 4))) / (R*1e-5) * 1e-6;
		}
		else
		{
			double dT = 1e-5;
			double F1 = this->F(p, T+dT);
			double F0 = this->F(p, T);
			return (F1-F0)/dT;

			// double tau = (5. / 6 * T - Theta) / (1 + std::exp((T-273.15) / 5));
			// double dtau_dT = ((1 + std::exp((T-273.15)/5.)) * 5. / 6 - (5. / 6 * T - Theta) / 5 * std::exp((T-273.15)/5.)) / pow(1 + std::exp((T-273.15)/5.), 2);

			// // PX *px{};
			// return (- vp[component][0] / pow(T, 2) * (p - P_0) 
			// 		- vp[component][1] / pow(T, 2) * (std::log(Psi + p) - std::log(Psi + P_0))
			// 		- (vp[component][2] * (p - P_0) + vp[component][3] * (std::log(Psi + p) - std::log(Psi + P_0))) 
			// 		/ pow(T*(T - Theta - tau), 2) * (T - Theta - tau + T * (1-dtau_dT))) * R_inv
			// 		// + px->F(p, T) * R_inv / T
			// 		;
		}
    }
    double V::d2FdPdT(double p, double T) 
    {
        return this->dfdT(p, T);
    }
    double V::d2FdT2(double p, double T) 
    {
		if (component == "H2O")
		{
			return ((2.*vi_a[0][0]/std::pow(T, 3) + 2*vi_a[3][0]) * (p - P_0) 
					+ 1. / 2 * (2.*vi_a[0][1]/std::pow(T, 3) + 2*vi_a[3][1]) * (std::pow(p, 2) - std::pow(P_0, 2)) 
					+ 1. / 3 * (2.*vi_a[0][2]/std::pow(T, 3) + 2*vi_a[3][2]) * (std::pow(p, 3) - std::pow(P_0, 3)) 
					+ 1. / 4 * (2.*vi_a[0][3]/std::pow(T, 3) + 2*vi_a[3][3]) * (std::pow(p, 4) - std::pow(P_0, 4))) / (R*1e-5) * 1e-6;
		}
		else
		{
			double dT = 1e-5;
			double F1 = this->dFdT(p, T+dT);
			double F0 = this->dFdT(p, T);
			return (F1-F0)/dT;

			// double tau = (5. / 6 * T - Theta) / (1 + std::exp((T-273.15) / 5));
			// double dtau_dT = ((1 + std::exp((T-273.15)/5.)) * 5. / 6 - (5. / 6 * T - Theta) / 5 * std::exp((T-273.15)/5.)) / pow(1 + std::exp((T-273.15)/5.), 2);

			// // PX *px{};
			// return (- vp[component][0] / pow(T, 2) * (p - P_0) 
			// 		- vp[component][1] / pow(T, 2) * (std::log(Psi + p) - std::log(Psi + P_0))
			// 		- (vp[component][2] * (p - P_0) + vp[component][3] * (std::log(Psi + p) - std::log(Psi + P_0))) 
			// 		/ pow(T*(T - Theta - tau), 2) * (T - Theta - tau + T * (1-dtau_dT))) * R_inv
			// 		// + px->F(p, T) * R_inv / T
			// 		;
		}
    }
    int V::test_derivatives(double p, double T, double tol, bool verbose)
    {
        int error_output = 0;
        double df = this->dfdT(p, T);
        double dF = this->dFdT(p, T);
        double d2F = this->d2FdT2(p, T);

        double d, dT{ 1e-5 };
        double f_ = this->f(p, T-dT);
        double f1 = this->f(p, T+dT);
        double df_num = (f1-f_)/(2*dT);
        double F_ = this->F(p, T-dT);
        double F1 = this->F(p, T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(p, T-dT);
        double dF1 = this->dFdT(p, T+dT);
        double d2F_num = (dF1-dF_)/(2*dT);

        d = std::log(std::fabs(df + 1e-15)) - std::log(std::fabs(df_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager V df/dT", {df, df_num, d}); error_output++; }
        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager V dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Jager V d2F/dT2", {d2F, d2F_num, d}); error_output++; }
        
        return error_output;
    }

    double PX::f(double p)
    {
        // Term in vi(p)
    	std::vector<double> b(3);
    	for (int i = 0; i < 3; i++)
        {   
    	    b[i] = eps[0 + i] + eps[3 + i] * TT + eps[6 + i] * std::pow(TT, 2);
    	}
		double epsilon = b[0] + b[1] * p + b[2] * std::pow(p, 2);
	    double dedp = b[1] + 2*p*b[2];
		return dedp / std::pow(epsilon, 2);
    }
	double PX::f(double p, double T) 
    {
		this->TT = T;
        return this->f(p);
	}
    double PX::dfdT(double p, double T)
    {
    	std::vector<double> b(3), db(3);
    	for (int i = 0; i < 3; i++)
        {   
    	    b[i] = eps[0 + i] + eps[3 + i] * T + eps[6 + i] * std::pow(T, 2);
            db[i] = eps[3 + i] + eps[6 + i] * T;
    	}
		double epsilon = b[0] + b[1] * p + b[2] * std::pow(p, 2);
        double depsilon = db[0] + db[1] * p + db[2] * std::pow(p, 2);
	    double dedp = b[1] + 2*p*b[2];
        double d2edpdT = db[1] + 2*p*db[2];
		return d2edpdT / std::pow(epsilon, 2) - 2. * dedp * depsilon / std::pow(epsilon, 3);
    }
	double PX::F(double p, double T) {
		// Integral of term in vi(p)
		this->TT = T;
		return simpson(P_0, p, 20);
	}
}

Jager2003::Jager2003(CompData& comp_data) : AQBase(comp_data) {
    gi0.resize(ns);
    gi.resize(ns);
	hi.resize(ns);
	vi.resize(ns);
    lna.resize(ns);
    dlnadxj.resize(ns*ns);
}

void Jager2003::init_PT(double p_, double T_, AQEoS::CompType comp_type) {
    this->p = p_;
    this->T = T_;
    (void) comp_type;

    // H2O and molecular solutes
    for (int i = 0; i < nc; i++)
    {
        // ideal gas Gibbs energy
        jager::IG ig = jager::IG(species[i]);
        double gio = ig.G();
        double hio = ig.H(T);  // eq. 3.3
        gi0[i] = gio - hio;  // eq. 3.2

        // Gibbs energy of aqueous phase
        jager::H h = jager::H(species[i]);
        jager::V v = jager::V(species[i]);
        gi[i] = jager::gi_0[species[i]]/(jager::R*jager::T_0);
        hi[i] = h.F(T);
        vi[i] = v.F(p, T); // - jager::omega[comp] * PX;
    }
    // Ionic solutes
    for (int i = 0; i < ni; i++)
    {
        // ideal gas Gibbs energy of salt
        jager::IG ig = jager::IG(this->compdata.salt);
        double gio = ig.G();
        double hio = ig.H(T);  // eq. 3.3
        gi0[nc + i] = gio - hio;  // eq. 3.2

        // Gibbs energy of ions in aqueous phase
        jager::H h = jager::H(species[i+nc]);
        jager::V v = jager::V(species[i+nc]);
        gi[nc + i] = jager::gi_0[species[i+nc]]/(jager::R*jager::T_0);
        hi[nc + i] = h.F(T);
        vi[nc + i] = v.F(p, T); // - jager::omega[comp] * PX;
    }

    // Initialize vectors
    B0 = std::vector<double>(ns*ns, 0.);
    dB0dP = std::vector<double>(ns*ns, 0.);
    dB0dT = std::vector<double>(ns*ns, 0.);
    B1 = std::vector<double>(ns*ns, 0.);
    for (int i = 0; i < ns; i++)
    {
        if (jager::B.find(species[i]) != jager::B.end())
        {
            std::unordered_map<std::string, std::vector<double>> Bi = jager::B[species[i]];
            for (int j = 0; j < ns; j++)
            {
                if (Bi.find(species[j]) != Bi.end())
                {
                    std::vector<double> b = Bi[species[j]];
                    B0[i*ns + j] = b[0] + b[1] * T + b[2] * p;  // B0
                    B0[j*ns + i] = b[0] + b[1] * T + b[2] * p;  // B0
                    dB0dT[i*ns + j] = b[1];
                    dB0dT[j*ns + i] = b[1];
                    dB0dP[i*ns + j] = b[2];
                    dB0dP[j*ns + i] = b[2];
                    B1[i*ns + j] = b[3];
                    B1[j*ns + i] = b[3];
                }
            }
        }
    }

    // salt related numbers
    B_ca = std::vector<double>(ni*ni, 0.);
    dBcadT = std::vector<double>(ni*ni, 0.);
    d2BcadT2 = std::vector<double>(ni*ni, 0.);
    C_ca = std::vector<double>(ni*ni, 0.);
    dCcadT = std::vector<double>(ni*ni, 0.);
    d2CcadT2 = std::vector<double>(ni*ni, 0.);
    D_ca = std::vector<double>(ni*ni, 0.);
    dDcadT = std::vector<double>(ni*ni, 0.);
    d2DcadT2 = std::vector<double>(ni*ni, 0.);
    for (int j = 0; j < ni; j++)
    {
        int cj = this->charge[j];
        if (cj > 0)  // only cations (+ charge)
        {
            for (int k = 0; k < ni; k++)
            {
                int ck = this->charge[k];
                if (ck < 0)  // only anions (- charge)
                {
                    std::vector<double> B_jk = jager::Bca[species[j+nc]][species[k+nc]];
                    B_ca[j*ni + k] = B_jk[0] + B_jk[1] * T + B_jk[2] * std::pow(T, 2);
                    dBcadT[j*ni + k] = B_jk[1] + 2 * B_jk[2] * T;
                    d2BcadT2[j*ni + k] = 2 * B_jk[2];
                    std::vector<double> C_jk = jager::Cca[species[j+nc]][species[k+nc]];
                    C_ca[j*ni + k] = C_jk[0] + C_jk[1] * T + C_jk[2] * std::pow(T, 2);
                    dCcadT[j*ni + k] = C_jk[1] + 2 * C_jk[2] * T;
                    d2CcadT2[j*ni + k] = 2 * C_jk[2];
                    std::vector<double> D_jk = jager::Dca[species[j+nc]][species[k+nc]];
                    D_ca[j*ni + k] = D_jk[0] + D_jk[1] * T + D_jk[2] * std::pow(T, 2);
                    dDcadT[j*ni + k] = D_jk[1] + 2 * D_jk[2] * T;
                    d2DcadT2[j*ni + k] = 2 * D_jk[2];
                }
            }
        }
    }

    // Debye Huckel parameters
    double eps = (jager::eps[0] + jager::eps[1] * p + jager::eps[2] * std::pow(p, 2))
                + (jager::eps[3] + jager::eps[4] * p + jager::eps[5] * std::pow(p, 2)) * T
                + (jager::eps[6] + jager::eps[7] * p + jager::eps[8] * std::pow(p, 2)) * std::pow(T, 2);
    double depsdP = (jager::eps[1] + 2 * jager::eps[2] * p)
                + (jager::eps[4] + 2 * jager::eps[5] * p) * T
                + (jager::eps[7] + 2 * jager::eps[8] * p) * pow(T, 2);
    double depsdT = jager::eps[3] + jager::eps[4] * p + jager::eps[5] * std::pow(p, 2)
                + 2 * (jager::eps[6] + jager::eps[7] * p + jager::eps[8] * std::pow(p, 2)) * T;
    double d2epsdPdT = (jager::eps[4] + 2 * jager::eps[5] * p) + 2. * (jager::eps[7] + 2 * jager::eps[8] * p) * T;
    double d2epsdT2 = 2 * (jager::eps[6] + jager::eps[7] * p + jager::eps[8] * std::pow(p, 2));
    
    double rho_s = 1000.;
    
    A_DH = std::pow(jager::e, 3)/std::pow(jager::eps0*eps*jager::R*T, 1.5) * std::pow(M_NA, 2)/(8*M_PI) * std::sqrt(2*rho_s); // kg^0.5/mol^0.5
    dA_DHdP = -1.5 * std::pow(jager::e, 3) / std::pow(jager::eps0*eps*jager::R*T, 2.5) * jager::eps0*depsdP*jager::R*T * std::pow(M_NA, 2)/(8*M_PI) * std::sqrt(2*rho_s);
    dA_DHdT = -1.5 * std::pow(jager::e, 3)/std::pow(jager::eps0*eps*jager::R*T, 2.5) * (jager::eps0*depsdT*jager::R*T + jager::eps0*eps*jager::R) * std::pow(M_NA, 2)/(8*M_PI) * std::sqrt(2*rho_s);
    d2A_DHdPdT = (2.5*1.5 * std::pow(jager::e, 3)/std::pow(jager::eps0*eps*jager::R*T, 3.5) * (jager::eps0*depsdP*jager::R*T) * (jager::eps0*depsdT*jager::R*T + jager::eps0*eps*jager::R)
                  -1.5 * std::pow(jager::e, 3)/std::pow(jager::eps0*eps*jager::R*T, 2.5) * (jager::eps0*d2epsdPdT*jager::R*T + jager::eps0*depsdP*jager::R)) * std::pow(M_NA, 2)/(8*M_PI) * std::sqrt(2*rho_s);
    d2A_DHdT2 = 2.5*1.5 * std::pow(jager::e, 3)/std::pow(jager::eps0*eps*jager::R*T, 3.5) * std::pow(jager::eps0*depsdT*jager::R*T + jager::eps0*eps*jager::R, 2) * std::pow(M_NA, 2)/(8*M_PI) * std::sqrt(2*rho_s)
                - 1.5 * std::pow(jager::e, 3)/std::pow(jager::eps0*eps*jager::R*T, 2.5) * (jager::eps0*d2epsdT2*jager::R*T + 2*jager::eps0*depsdT*jager::R) * std::pow(M_NA, 2)/(8*M_PI) * std::sqrt(2*rho_s);

	return;
}

void Jager2003::solve_PT(std::vector<double>& x_, bool second_order, AQEoS::CompType comp_type) 
{
    // Calculate molality of molecular and ionic species
    this->x = x_;  // Water mole fraction to approximate activity
    if (!set_molality) // if not set/calculated yet, calculate
    {
        this->set_species_molality();
    }

    // Ionic strength: I and dI/dxj (eq. 17)
    I = 0.;
    if (ni > 0)
    {
        dIdxj = std::vector<double>(ns, 0.);
        for (int i = 0; i < ni; i++)
        {
            I += 0.5 * std::pow(this->charge[i], 2) * m_s[nc + i];
            dIdxj[nc + i] += 0.5 * std::pow(this->charge[i], 2) * this->dmi_dxi();
            dIdxj[water_index] += 0.5 * std::pow(this->charge[i], 2) * this->dmi_dxw(nc + i);
        }
    }

    // Calculate activities and derivatives
	if (comp_type == AQEoS::CompType::water)
	{
        this->lna[water_index] = this->lnaw();
        if (second_order)
        {
            std::vector<double> dlnaw = this->dlnaw_dxj();
            std::copy(dlnaw.begin(), dlnaw.end(), dlnadxj.begin() + water_index * ns);
        }
	}
	else if (comp_type == AQEoS::CompType::solute)
	{
        for (int i = 0; i < nc; i++)
        {
            if (i != water_index)
            {
                this->lna[i] = this->lnam(i);
                if (second_order)
                {
                    std::vector<double> dlnam = this->dlnam_dxj(i);
                    std::copy(dlnam.begin(), dlnam.end(), dlnadxj.begin() + i * ns);
                }
            }
        }
	}
	else if (comp_type == AQEoS::CompType::ion)
	{
        for (int i = 0; i < ni; i++)
        {
            this->lna[nc + i] = this->lnai(i);
            if (second_order)
            {
                std::vector<double> dlnai = this->dlnai_dxj(i);
                std::copy(dlnai.begin(), dlnai.end(), dlnadxj.begin() + (nc + i) * ns);
            }
        }
	}
	else
	{
		print("Invalid CompType for Jager2003 correlation specified", comp_type);
		exit(1);
	}
    
    // set to false again
    this->set_molality = false;

    return;
}

double Jager2003::lnaw()
{
    // For lna_w (eq. 24)
    double lna_w = 0.;
    double sqrtI = std::sqrt(I);

    // 1st term: ionic contribution j_Ica
    if (ni > 0 && I > 0.)
    {
        // eq. 26
        double j_DH = -2.*A_DH * (1. + sqrtI - 2. * std::log(1. + sqrtI) - 1./(1.+sqrtI));

        double sum_mc{ 0. }, sum_ma{ 0. };
        for (int i = 0; i < ni; i++)
        {
            // sum_mc & sum_ma
            if (this->charge[i] > 0)
            {
                sum_mc += std::pow(this->charge[i], 2) * m_s[i+nc];
            }
            else
            {
                sum_ma += std::pow(this->charge[i], 2) * m_s[i+nc];
            }
        }

        for (int i = 0; i < ni; i++)
        {
            // sum_mcma_jIca
            int ci = this->charge[i];
            if (ci > 0)  // i are only cations (+ charge)
            {
                for (int j = 0; j < ni; j++)
                {
                    int cj = this->charge[j];
                    if (cj < 0)  // j only anions (- charge)
                    {
                        // eq. 27
                        int idx = i*ni + j;
                        double cc_ca = -ci*cj;  // |charge_j * charge_k|
                        double cc_ca_inv = 1./cc_ca;
                        
                        double a = (0.13816 + 0.6*B_ca[idx])*cc_ca/1.5;
                        double jBca1 = (1.+3.*I*cc_ca_inv)/std::pow(1.+1.5*I*cc_ca_inv, 2);
                        double jBca2 = std::log(1.+1.5*I*cc_ca_inv)/(1.5*I*cc_ca_inv);
                        double jBca3 = 0.5*B_ca[idx]*std::pow(I, 2) + 2./3.*C_ca[idx]*std::pow(I, 3) + 0.75*D_ca[idx]*std::pow(I, 4);
                        double j_Bca = 2 * a * I * (jBca1 - jBca2) + 2.*cc_ca_inv * jBca3;

                        // eq. 25
                        double j_Ica = -comp_data::Mw["H2O"]*1e-3 * (2.*I*cc_ca_inv + j_DH + j_Bca);
                        lna_w += m_s[nc + i] * m_s[nc + j] * std::pow(cc_ca, 2) * j_Ica / (sum_mc * sum_ma);
                    }
                }
            }
        }
    }

    // 2nd term: ionic-molecular contribution
    double e = std::exp(-2. * sqrtI);
    for (int i = 0; i < ns; i++)
    {
        if (i != water_index)
        {
            // Pitzer
            for (int j = 0; j < ns; j++)
            {
                if (j != water_index)
                {
                    double B0B1 = B0[i*ns + j] + B1[i*ns + j] * e;  // B0 + B1 exp(-2 sqrt(I))
                    lna_w -= m_s[i] * m_s[j] * B0B1 * comp_data::Mw["H2O"]*1e-3;
                }
            }
        }
    }

    // 3rd term: molecular contribution
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            lna_w -= m_s[i] * comp_data::Mw["H2O"]*1e-3;
        }
    }
    
    return lna_w;
}
double Jager2003::dlnaw_dP()
{
    // Calculate derivative of lnaw with respect to P
    // dlna_w/dP: eq 24
    double dlnawdP = 0.;
    if (ni > 0)  // 1st term
    {
        // Eq 26
        double sqrtI = std::sqrt(I);
        double djDHdP = 2*dA_DHdP * ((1.-std::pow(1.+sqrtI, 2))/(1.+sqrtI) + 2*std::log(1.+sqrtI));

        double sum_mc{ 0. }, sum_ma{ 0. }, d_num{ 0. };
        for (int ii = 0; ii < ni; ii++)
        {
            // sum_mc & sum_ma
            if (charge[ii] > 0)
            {
                sum_mc += std::pow(charge[ii], 2) * m_s[ii+nc];
            }
            else
            {
                sum_ma += std::pow(charge[ii], 2) * m_s[ii+nc];
            }

            if (charge[ii] > 0)
            {
                for (int jj = 0; jj < ni; jj++)
                {
                    if (charge[jj] < 0)
                    {
                        double mcma = m_s[ii+nc] * std::pow(charge[ii], 2) * m_s[jj+nc] * std::pow(charge[jj], 2);

                        // Eq 25
                        double djIdP = -djDHdP;
                        d_num += mcma * djIdP;
                    }
                }
            }
        }
        dlnawdP += comp_data::Mw["H2O"]*1e-3 * d_num / (sum_mc * sum_ma);
    }
    for (int ii = 0; ii < ns; ii++)  // 2nd term
    {
        for (int jj = 0; jj < ns; jj++)
        {
            // eq 21
            dlnawdP -= comp_data::Mw["H2O"]*1e-3 * m_s[ii] * m_s[jj] * dB0dP[ii*ns+jj];
        }
    }
    return dlnawdP;
}
double Jager2003::dlnaw_dT()
{
    // Calculate derivative of lnai with respect to T
    // dlna_w/dT: eq 24
    double dlnawdT = 0.;

    // 1st term
    if (ni > 0)
    {
        // Eq 26
        double sqrtI = std::sqrt(I);
        double djDHdT = 2*dA_DHdT * ((1.-std::pow(1.+sqrtI, 2))/(1.+sqrtI) + 2*std::log(1.+sqrtI));

        double sum_mc{ 0. }, sum_ma{ 0. }, d_num{ 0. };
        for (int ii = 0; ii < ni; ii++)
        {
            // sum_mc & sum_ma
            if (charge[ii] > 0)
            {
                sum_mc += std::pow(charge[ii], 2) * m_s[ii+nc];
            }
            else
            {
                sum_ma += std::pow(charge[ii], 2) * m_s[ii+nc];
            }

            if (charge[ii] > 0)
            {
                for (int jj = 0; jj < ni; jj++)
                {
                    if (charge[jj] < 0)
                    {
                        double mcma = m_s[ii+nc] * std::pow(charge[ii], 2) * m_s[jj+nc] * std::pow(charge[jj], 2);

                        // Eq 27
                        int idx = ii*ni + jj;
                        double cc_ca = -charge[ii]*charge[jj];  // |charge_j * charge_k|
                        double djBcadT = 0.6*dBcadT[idx] * I * cc_ca/1.5;
                        djBcadT *= (1.+3*I/cc_ca)/std::pow(1.+3*I/(2*cc_ca), 2) - std::log(1.+3*I/(2*cc_ca))/(3*I/(2*cc_ca));
                        djBcadT += 2./cc_ca * (0.5*dBcadT[idx]*std::pow(I, 2) + 2./3*dCcadT[idx]*std::pow(I, 3) + 3./4*dDcadT[idx]*std::pow(I, 4));

                        // Eq 25
                        double djIdT = -(djDHdT + djBcadT);
                        d_num += mcma * djIdT;
                    }
                }
            }
        }
        dlnawdT += comp_data::Mw["H2O"]*1e-3 * d_num / (sum_mc * sum_ma);
    }

    // 2nd term
    for (int ii = 0; ii < ns; ii++)
    {
        for (int jj = 0; jj < ns; jj++)
        {
            // eq 21
            dlnawdT -= comp_data::Mw["H2O"]*1e-3 * m_s[ii] * m_s[jj] * dB0dT[ii*ns+jj];
        }
    }
    return dlnawdT;
}
double Jager2003::d2lnaw_dPdT()
{
    // Calculate derivative of lnaw with respect to P
    // dlna_w/dP: eq 24
    double d2lnawdPdT = 0.;
    if (ni > 0)  // 1st term
    {
        // Eq 26
        double sqrtI = std::sqrt(I);
        double d2jDHdPdT = 2*d2A_DHdPdT * ((1.-std::pow(1.+sqrtI, 2))/(1.+sqrtI) + 2*std::log(1.+sqrtI));

        double sum_mc{ 0. }, sum_ma{ 0. }, d_num{ 0. };
        for (int ii = 0; ii < ni; ii++)
        {
            // sum_mc & sum_ma
            if (charge[ii] > 0)
            {
                sum_mc += std::pow(charge[ii], 2) * m_s[ii+nc];
            }
            else
            {
                sum_ma += std::pow(charge[ii], 2) * m_s[ii+nc];
            }

            if (charge[ii] > 0)
            {
                for (int jj = 0; jj < ni; jj++)
                {
                    if (charge[jj] < 0)
                    {
                        double mcma = m_s[ii+nc] * std::pow(charge[ii], 2) * m_s[jj+nc] * std::pow(charge[jj], 2);

                        // Eq 25
                        double d2jIdPdT = -d2jDHdPdT;
                        d_num += mcma * d2jIdPdT;
                    }
                }
            }
        }
        d2lnawdPdT += comp_data::Mw["H2O"]*1e-3 * d_num / (sum_mc * sum_ma);
    }
    
    // 2nd term = 0
    
    return d2lnawdPdT;
}
double Jager2003::d2lnaw_dT2()
{
    // Calculate second derivative of lnai with respect to T
    // d2lna_w/dT2: eq 24
    double d2lnawdT2 = 0.;
    if (ni > 0)  // 1st term
    {
        // Eq 26
        double sqrtI = std::sqrt(I);
        double d2jDHdT2 = 2*d2A_DHdT2 * ((1.-std::pow(1.+sqrtI, 2))/(1.+sqrtI) + 2*std::log(1.+sqrtI));

        double sum_mc{ 0. }, sum_ma{ 0. }, d2_num{ 0. };
        for (int ii = 0; ii < ni; ii++)
        {
            // sum_mc & sum_ma
            if (charge[ii] > 0)
            {
                sum_mc += std::pow(charge[ii], 2) * m_s[ii+nc];
            }
            else
            {
                sum_ma += std::pow(charge[ii], 2) * m_s[ii+nc];
            }

            if (charge[ii] > 0)
            {
                for (int jj = 0; jj < ni; jj++)
                {
                    if (charge[jj] < 0)
                    {
                        double mcma = m_s[ii+nc] * std::pow(charge[ii], 2) * m_s[jj+nc] * std::pow(charge[jj], 2);

                        // Eq 27
                        int idx = ii*ni + jj;
                        double cc_ca = -charge[ii]*charge[jj];  // |charge_j * charge_k|
                        double d2jBcadT2 = 0.6*d2BcadT2[idx] * I * cc_ca/1.5;
                        d2jBcadT2 *= (1.+3*I/cc_ca)/std::pow(1.+3*I/(2*cc_ca), 2) - std::log(1.+3*I/(2*cc_ca))/(3*I/(2*cc_ca));
                        d2jBcadT2 += 2./cc_ca * (0.5*d2BcadT2[idx]*std::pow(I, 2) + 2./3*d2CcadT2[idx]*std::pow(I, 3) + 3./4*d2DcadT2[idx]*std::pow(I, 4));

                        // Eq 25
                        double d2jIdT2 = -(d2jDHdT2 + d2jBcadT2);
                        d2_num += mcma * d2jIdT2;
                    }
                }
            }
        }
        d2lnawdT2 += comp_data::Mw["H2O"]*1e-3 * d2_num / (sum_mc * sum_ma);
    }
    // 2nd term: d2B0/dT2 = 0
    
    return d2lnawdT2;
}
std::vector<double> Jager2003::dlnaw_dxj()
{
    // Calculate composition derivatives of lna_w (eq. 24)
    std::vector<double> dlnawdxj(ns, 0.);
    double sqrtI = std::sqrt(I);

    // 1st term: ionic contribution j_Ica
    if (ni > 0 && I > 0.)
    {
        // eq. 26
        double j_DH = -2.*A_DH * (1. + sqrtI - 2. * std::log(1. + sqrtI) - 1./(1.+sqrtI));
        double djDHdI = -A_DH * sqrtI / std::pow(sqrtI + 1., 2);

        double sum_mc{ 0. }, sum_ma{ 0. };
        for (int i = 0; i < ni; i++)
        {
            // sum_mc & sum_ma
            if (this->charge[i] > 0)
            {
                sum_mc += std::pow(this->charge[i], 2) * m_s[i+nc];
            }
            else
            {
                sum_ma += std::pow(this->charge[i], 2) * m_s[i+nc];
            }
        }

        for (int i = 0; i < ni; i++)
        {
            // sum_mcma_jIca
            int ci = this->charge[i];
            if (ci > 0)  // i are only cations (+ charge)
            {
                for (int j = 0; j < ni; j++)
                {
                    int cj = this->charge[j];
                    if (cj < 0)  // j only anions (- charge)
                    {
                        // eq. 27
                        int idx = i*ni + j;
                        double cc_ca = -ci*cj;  // |charge_j * charge_k|
                        double cc_ca_inv = 1./cc_ca;
                        
                        double a = (0.13816 + 0.6*B_ca[idx])*cc_ca/1.5;
                        double jBca1 = (1.+3.*I*cc_ca_inv)/std::pow(1.+1.5*I*cc_ca_inv, 2);
                        double jBca2 = std::log(1.+1.5*I*cc_ca_inv)/(1.5*I*cc_ca_inv);
                        double jBca3 = 0.5*B_ca[idx]*std::pow(I, 2) + 2./3.*C_ca[idx]*std::pow(I, 3) + 0.75*D_ca[idx]*std::pow(I, 4);
                        double j_Bca = 2 * a * I * (jBca1 - jBca2) + 2.*cc_ca_inv * jBca3;

                        // eq. 25
                        double j_Ica = -comp_data::Mw["H2O"]*1e-3 * (2.*I*cc_ca_inv + j_DH + j_Bca);

                        // eq. 27
                        double djBca1 = 3.*cc_ca_inv / std::pow(1.+1.5*I*cc_ca_inv, 2) - 2.*(1.+3.*I*cc_ca_inv)/std::pow(1.+1.5*I*cc_ca_inv, 3) * 1.5*cc_ca_inv;
                        double djBca2 = 1. / (1. + 1.5*I*cc_ca_inv) / I - std::log(1.+1.5*I*cc_ca_inv)/std::pow(1.5*I*cc_ca_inv, 2) * 1.5*cc_ca_inv;
                        double djBca3 = B_ca[idx]*I + 2.*C_ca[idx]*std::pow(I, 2) + 3.*D_ca[idx]*std::pow(I, 3);                            
                        double djBcadI = 2 * a * (jBca1 - jBca2) + 2*a*I*(djBca1 - djBca2) + 2.*cc_ca_inv * djBca3;
                        double djIcadI = -comp_data::Mw["H2O"]*1e-3 * (2.*cc_ca_inv + djDHdI + djBcadI);

                        // for i == + and j == -
                        double d_denom_dxi = this->dmi_dxi() * std::pow(this->charge[i], 2) * sum_ma;
                        dlnawdxj[nc + i] += (this->dmi_dxi() * m_s[nc + j] * j_Ica / (sum_ma*sum_mc)
                                            + m_s[nc + i] * m_s[nc + j] * djIcadI * dIdxj[i+nc] / (sum_ma*sum_mc)
                                            - m_s[nc + i] * m_s[nc + j] * j_Ica / std::pow(sum_ma*sum_mc, 2) * d_denom_dxi) * std::pow(cc_ca, 2);
                        double d_denom_dxj = this->dmi_dxi() * std::pow(this->charge[j], 2) * sum_mc;
                        dlnawdxj[nc + j] += (m_s[nc + i] * this->dmi_dxi() * j_Ica / (sum_ma*sum_mc)
                                            + m_s[nc + i] * m_s[nc + j] * djIcadI * dIdxj[j+nc] / (sum_ma*sum_mc)
                                            - m_s[nc + i] * m_s[nc + j] * j_Ica / std::pow(sum_ma*sum_mc, 2) * d_denom_dxj) * std::pow(cc_ca, 2);

                        // for j == water
                        // THIS MIGHT NOT BE CORRECT FOR MULTIPLE IONS
                        double d_denom_dxw = std::pow(this->charge[i], 2) * dmi_dxw(i+nc) * sum_ma + std::pow(this->charge[j], 2) * dmi_dxw(j+nc) * sum_mc;
                        dlnawdxj[water_index] += (this->dmi_dxw(nc+i) * m_s[nc + j] * j_Ica / (sum_ma*sum_mc)
                                                + m_s[nc + i] * this->dmi_dxw(nc+j) * j_Ica / (sum_ma*sum_mc)
                                                + m_s[nc + i] * m_s[nc + j] * djIcadI * dIdxj[water_index] / (sum_ma*sum_mc)
                                                - m_s[nc + i] * m_s[nc + j] * j_Ica / std::pow(sum_ma*sum_mc, 2) * d_denom_dxw) * std::pow(cc_ca, 2);
                    }
                }
            }
        }
    }

    // 2nd term: ionic-molecular contribution
    double e = std::exp(-2. * sqrtI);
    double dedI = -e / sqrtI;
    for (int i = 0; i < ns; i++)
    {
        if (i != water_index)
        {
            // Pitzer
            for (int j = 0; j < ns; j++)
            {
                if (j != water_index)
                {
                    double B0B1 = B0[i*ns + j] + B1[i*ns + j] * e;  // B0 + B1 exp(-2 sqrt(I))

                    dlnawdxj[i] -= this->dmi_dxi() * m_s[j] * B0B1 * comp_data::Mw["H2O"]*1e-3;
                    dlnawdxj[j] -= this->dmi_dxi() * m_s[i] * B0B1 * comp_data::Mw["H2O"]*1e-3;
                    dlnawdxj[water_index] -= -2 * m_s[i] * m_s[j] / x[water_index] * B0B1 * comp_data::Mw["H2O"]*1e-3;
                    if (ni > 0 && I > 0.)
                    {
                        double mmB1dedI = m_s[i] * m_s[j] * B1[i*ns + j] * dedI * comp_data::Mw["H2O"]*1e-3;
                        dlnawdxj[i] -= mmB1dedI * dIdxj[i];
                        dlnawdxj[j] -= mmB1dedI * dIdxj[j];
                        dlnawdxj[water_index] -= mmB1dedI * dIdxj[water_index];
                    }
                }
            }
        }
    }

    // 3rd term: molecular contribution
    for (int i = 0; i < nc; i++)
    {
        if (i != water_index)
        {
            dlnawdxj[i] -= this->dmi_dxi() * comp_data::Mw["H2O"]*1e-3;
            dlnawdxj[water_index] -= this->dmi_dxw(i) * comp_data::Mw["H2O"]*1e-3;
        }
    }

    return dlnawdxj;
}
std::vector<double> Jager2003::d2lnaw_dTdxj()
{
    // Calculate derivative of lna_w w.r.t. dT dxj (eq. 24)
    std::vector<double> d2lnawdTdxj(ns, 0.);
    double sqrtI = std::sqrt(I);

    // 1st term: ionic contribution j_Ica
    if (ni > 0 && I > 0.)
    {
        // eq. 26
        double djDHdT = -2.*dA_DHdT * (1. + sqrtI - 2. * std::log(1. + sqrtI) - 1./(1.+sqrtI));
        double d2jDHdTdI = -dA_DHdT * sqrtI / std::pow(sqrtI + 1., 2);

        double sum_mc{ 0. }, sum_ma{ 0. };
        for (int i = 0; i < ni; i++)
        {
            // sum_mc & sum_ma
            if (this->charge[i] > 0)
            {
                sum_mc += std::pow(this->charge[i], 2) * m_s[i+nc];
            }
            else
            {
                sum_ma += std::pow(this->charge[i], 2) * m_s[i+nc];
            }
        }

        for (int i = 0; i < ni; i++)
        {
            // sum_mcma_jIca
            int ci = this->charge[i];
            if (ci > 0)  // i are only cations (+ charge)
            {
                for (int j = 0; j < ni; j++)
                {
                    int cj = this->charge[j];
                    if (cj < 0)  // j only anions (- charge)
                    {
                        // eq. 27
                        double cc_ca = -ci*cj;  // |charge_j * charge_k|
                        double cc_ca_inv = 1./cc_ca;
                        int idx = i*ni + j;
                        
                        double dadT = 0.6*dBcadT[idx] * cc_ca/1.5;

                        double jBca1 = (1.+3.*I*cc_ca_inv)/std::pow(1.+1.5*I*cc_ca_inv, 2);
                        double jBca2 = std::log(1.+1.5*I*cc_ca_inv)/(1.5*I*cc_ca_inv);
                        double djBcadT = 0.6*dBcadT[idx] * I * cc_ca/1.5;
                        djBcadT *= (1.+3*I/cc_ca)/std::pow(1.+3*I/(2*cc_ca), 2) - std::log(1.+3*I/(2*cc_ca))/(3*I/(2*cc_ca));
                        djBcadT += 2./cc_ca * (0.5*dBcadT[idx]*std::pow(I, 2) + 2./3*dCcadT[idx]*std::pow(I, 3) + 3./4*dDcadT[idx]*std::pow(I, 4));

                        double djBca1 = 3.*cc_ca_inv / std::pow(1.+1.5*I*cc_ca_inv, 2) - 2.*(1.+3.*I*cc_ca_inv)/std::pow(1.+1.5*I*cc_ca_inv, 3) * 1.5*cc_ca_inv;
                        double djBca2 = 1. / (1. + 1.5*I*cc_ca_inv) / I - std::log(1.+1.5*I*cc_ca_inv)/std::pow(1.5*I*cc_ca_inv, 2) * 1.5*cc_ca_inv;
                        double d2jBca3dTdI = dBcadT[idx]*I + 2.*dCcadT[idx]*std::pow(I, 2) + 3.*dDcadT[idx]*std::pow(I, 3);
                        double d2jBcadTdI = 2 * dadT * (jBca1 - jBca2) + 2*dadT*I*(djBca1 - djBca2) + 2.*cc_ca_inv * d2jBca3dTdI;
                        double d2jIcadTdI = -comp_data::Mw["H2O"]*1e-3 * (d2jDHdTdI + d2jBcadTdI);

                        // eq. 25
                        double djIdT = -(djDHdT + djBcadT);

                        // for i == + and j == -
                        double d_denom_dxi = this->dmi_dxi() * std::pow(this->charge[i], 2) * sum_ma;
                        d2lnawdTdxj[nc + i] += (this->dmi_dxi() * m_s[nc + j] * djIdT / (sum_ma*sum_mc)
                                                + m_s[nc + i] * m_s[nc + j] * d2jIcadTdI * dIdxj[i+nc] / (sum_ma*sum_mc)
                                                - m_s[nc + i] * m_s[nc + j] * djIdT / std::pow(sum_ma*sum_mc, 2) * d_denom_dxi) * std::pow(cc_ca, 2);
                        double d_denom_dxj = this->dmi_dxi() * std::pow(this->charge[j], 2) * sum_mc;
                        d2lnawdTdxj[nc + j] += (m_s[nc + i] * this->dmi_dxi() * djIdT / (sum_ma*sum_mc)
                                                + m_s[nc + i] * m_s[nc + j] * d2jIcadTdI * dIdxj[j+nc] / (sum_ma*sum_mc)
                                                - m_s[nc + i] * m_s[nc + j] * djIdT / std::pow(sum_ma*sum_mc, 2) * d_denom_dxj) * std::pow(cc_ca, 2);

                        // for j == water
                        // THIS MIGHT NOT BE CORRECT FOR MULTIPLE IONS
                        double d_denom_dxw = std::pow(this->charge[i], 2) * dmi_dxw(i+nc) * sum_ma + std::pow(this->charge[j], 2) * dmi_dxw(j+nc) * sum_mc;
                        d2lnawdTdxj[water_index] += (this->dmi_dxw(nc+i) * m_s[nc + j] * djIdT / (sum_ma*sum_mc)
                                                + m_s[nc + i] * this->dmi_dxw(nc+j) * djIdT / (sum_ma*sum_mc)
                                                + m_s[nc + i] * m_s[nc + j] * d2jIcadTdI * dIdxj[water_index] / (sum_ma*sum_mc)
                                                - m_s[nc + i] * m_s[nc + j] * djIdT / std::pow(sum_ma*sum_mc, 2) * d_denom_dxw) * std::pow(cc_ca, 2);
                    }
                }
            }
        }
    }

    // 2nd term: ionic-molecular contribution
    for (int i = 0; i < ns; i++)
    {
        if (i != water_index)
        {
            // Pitzer
            for (int j = 0; j < ns; j++)
            {
                if (j != water_index)
                {
                    double dB0B1dT = dB0dT[i*ns + j];  // B0 + B1 exp(-2 sqrt(I))

                    d2lnawdTdxj[i] -= this->dmi_dxi() * m_s[j] * dB0B1dT * comp_data::Mw["H2O"]*1e-3;
                    d2lnawdTdxj[j] -= this->dmi_dxi() * m_s[i] * dB0B1dT * comp_data::Mw["H2O"]*1e-3;
                    d2lnawdTdxj[water_index] -= -2 * m_s[i] * m_s[j] / x[water_index] * dB0B1dT * comp_data::Mw["H2O"]*1e-3;
                }
            }
        }
    }
    return d2lnawdTdxj;
}

double Jager2003::lnam(int i)
{
    // For lna_i molecular (eq. 15)
    double lna_m = 0.;
    
    // For molecular/ionic interactions: Pitzer j_P1
    double I_term{ 0. };
    if (I > 0.)
    {
        double sqrtI = std::sqrt(I);
        double I_inv = 1./I;
        double e = std::exp(-2.*sqrtI);
        I_term = I_inv * (1. - (1. + 2.*sqrtI) * e);
    }

    // 1st term
    for (int j = 0; j < ns; j++)
    {
        if (j != water_index)
        {
            double j_P1ij = B0[i*ns + j];
            if (ni > 0 && I > 0.)
            {
                // Pitzer contribution of ions - eq 21 and 22
                j_P1ij += 0.5 * B1[i*ns + j] * I_term;
            }
            // eq 23
            lna_m += 2. * m_s[j] * j_P1ij;
        }
    }

    // 2nd term
    lna_m += std::log(m_s[i]);

    return lna_m;
}
double Jager2003::dlnam_dP(int i)
{
    // dlnai/dP for molecular: eq 23
    double dlnamdP = 0.;
    for (int jj = 0; jj < ns; jj++)
    {
        // eq 21
        double djP1dP = dB0dP[i*ns + jj];
        dlnamdP += 2 * m_s[jj] * djP1dP;
    }
    return dlnamdP;
}
double Jager2003::dlnam_dT(int i)
{
    // dlnai/dT for molecular: eq 23
    double dlnamdT = 0.;
    for (int jj = 0; jj < ns; jj++)
    {
        // eq 21
        double djP1dT = dB0dT[i*ns + jj];
        dlnamdT += 2 * m_s[jj] * djP1dT;
    }
    return dlnamdT;
}
double Jager2003::d2lnam_dPdT(int i)
{
    // d2lnai/dPdT for molecular = 0: eq 23
    (void) i;
    return 0.;
}
double Jager2003::d2lnam_dT2(int i)
{
    // d2lnai/dT2 for molecular: eq 23
    (void) i;

    // d2B0/dT2 = 0
    
    return 0.;
}
std::vector<double> Jager2003::dlnam_dxj(int i)
{
    // For lna_i molecular (eq. 15)
    std::vector<double> dlnamdxj(ns, 0.);
    
    // For molecular/ionic interactions: Pitzer j_P1
    double I_term{ 0. }, dI_term{ 0. };
    if (I > 0.)
    {
        double sqrtI = std::sqrt(I);
        double I_inv = 1./I;
        double e = std::exp(-2.*sqrtI);
        I_term = I_inv * (1. - (1. + 2.*sqrtI) * e);
        dI_term = -std::pow(I_inv, 2) * (1. - (1. + 2.*sqrtI) * e) + I_inv * 2 * e;
    }

    // 1st term
    for (int j = 0; j < ns; j++)
    {
        if (j != water_index)
        {
            double j_P1ij = B0[i*ns + j];
            if (ni > 0 && I > 0.)
            {
                // Pitzer contribution of ions - eq 21 and 22
                j_P1ij += 0.5 * B1[i*ns + j] * I_term;
                
                // eq 23
                double dj_P1ij = 0.5 * B1[i*ns + j] * dI_term;
                dlnamdxj[j] += 2. * m_s[j] * dj_P1ij * dIdxj[j];
                dlnamdxj[water_index] += 2. * m_s[j] * dj_P1ij * dIdxj[water_index];
            }
                        
            dlnamdxj[j] += 2. * this->dmi_dxi() * j_P1ij;
            dlnamdxj[water_index] += 2. * this->dmi_dxw(j) * j_P1ij;
        }
    }

    // 2nd term
    dlnamdxj[i] += 1./m_s[i] * this->dmi_dxi();
    dlnamdxj[water_index] += 1./m_s[i] * this->dmi_dxw(i);
    
    return dlnamdxj;
}
std::vector<double> Jager2003::d2lnam_dTdxj(int i)
{
    // d2lnai/dTdxj for molecular: eq 23
    std::vector<double> d2lnamdTdxj(ns, 0.);
    for (int j = 0; j < ns; j++)
    {
        // eq 21
        if (j != water_index)
        {
            double djP1dT = dB0dT[i*ns + j];
            d2lnamdTdxj[j] = 2 * this->dmi_dxi() * djP1dT;
            d2lnamdTdxj[water_index] += 2 * this->dmi_dxw(j) * djP1dT;
        }
    }
    return d2lnamdTdxj;
}

double Jager2003::lnai(int i)
{
    // For lna_i ions (eq. 16)
    double lna_i = 0.;
    double sqrtI = std::sqrt(I);
    double I_inv = 1./I;
    double e = std::exp(-2.*sqrtI);

    // 1st term: j_LR - eq 18
    double j_LR = -A_DH * sqrtI / (1.+sqrtI);
    lna_i += std::pow(this->charge[i], 2) * j_LR;

    // 2nd term: j_SR - eq 20
    for (int j = 0; j < ni; j++)
    {
        int ci = this->charge[i];
        int cj = this->charge[j];
        if (ci * cj < 0)
        {
            int idx = ci > 0 ? i*ni+j : j*ni + i;
            int zizj = -ci*cj;
            double cicj2 = std::pow(0.5*(std::abs(ci) + std::abs(cj)), 2);
            double j_SR = (0.13816 + 0.6 * B_ca[idx]) * zizj / std::pow(1.+1.5*I/zizj, 2)
                            + B_ca[idx] + C_ca[idx] * I + D_ca[idx] * std::pow(I, 2);
            lna_i += m_s[nc+j] * j_SR * cicj2;
        }
    }

    // 3rd term: j_P1 - eq 21
    for (int j = 0; j < ns; j++)
    {
        // Pitzer contribution of ions - eq 21 and 22
        double j_P1ij = B0[(i+nc)*ns + j] + 0.5 * B1[(i+nc)*ns + j] * I_inv * (1. - (1. + 2.*sqrtI) * e);
        lna_i += 2. * m_s[j] * j_P1ij;
    }

    // 4th term: j_P2 - eq 22
    double sum_mimj_jP2 = 0.;
    for (int j = 0; j < ns; j++)
    {
        if (j != water_index)
        {
            for (int k = 0; k < ns; k++)
            {
                if (k != water_index)
                {
                    // Pitzer contribution of ions - eq 21 and 22
                    double j_P2jk = B1[j*ns + k] * (1. - (1. + 2.*sqrtI + 2.*I) * e);
                    sum_mimj_jP2 += m_s[j] * m_s[k] * j_P2jk;
                }
            }
        }
    }
    lna_i += 0.25 * std::pow(this->charge[i], 2) * std::pow(I_inv, 2) * sum_mimj_jP2;

    // Eq 15: ai = mi/mi0 * j_i -> lnai = ln(m_i) + ln(j_i)
    lna_i += std::log(m_s[i+nc]);

    return lna_i;
}
double Jager2003::dlnai_dP(int i)
{
    // dlnai/dP for ions: eq 16
    double dlnaidP = 0.;

    // 1st term: dj_LR/dP: eq 18
    double sqrtI = std::sqrt(I);
    double dj_LRdP = -dA_DHdP * sqrtI / (1.+sqrtI);
    dlnaidP += std::pow(charge[i], 2) * dj_LRdP;

    // 2nd term: dj_SR/dP = 0 - eq 20

    // 3rd term: dj_P1/dP - eq 21
    for (int jj = 0; jj < nc; jj++)
    {
        double dj_P1dP = dB0dP[(i+nc)*ns + jj];
        dlnaidP += 2 * m_s[jj] * dj_P1dP;
    }

    // 4th term: dj_P2/dP = 0 - eq 22

    return dlnaidP;
}
double Jager2003::dlnai_dT(int i)
{
    // dlnai/dT for ions: eq 16
    double dlnaidT = 0.;

    // 1st term: dj_LR/dT - eq 18
    double sqrtI = std::sqrt(I);
    double dj_LRdT = -dA_DHdT * sqrtI / (1.+sqrtI);
    dlnaidT += std::pow(charge[i], 2) * dj_LRdT;

    // 2nd term: dj_SR/dT - eq 20
    for (int jj = 0; jj < ni; jj++)
    {
        if (charge[i] * charge[jj] < 0)
        {
            int idx = charge[i] > 0 ? i*ni+jj : jj*ni + i;
            int zizj = std::abs(charge[i]*charge[jj]);
            double dj_SRdT = 0.6 * dBcadT[idx] * zizj / std::pow(1.+1.5*I/zizj, 2)
                            + dBcadT[idx] + dCcadT[idx] * I + dDcadT[idx] * std::pow(I, 2);
            dlnaidT += m_s[nc+jj] * dj_SRdT * std::pow(0.5 * (std::abs(charge[i]) + std::abs(charge[jj])), 2);
        }
    }

    // 3rd term: dj_P1/dT - eq 21
    for (int jj = 0; jj < nc; jj++)
    {
        double dj_P1dT = dB0dT[(i+nc)*ns + jj];
        dlnaidT += 2 * m_s[jj] * dj_P1dT;
    }

    // 4th term: dj_P2/dT = 0 - eq 22

    return dlnaidT;
}
double Jager2003::d2lnai_dPdT(int i)
{
    // d2lnai/dPdT for ions: eq 16
    double d2lnaidPdT = 0.;

    // 1st term: d2j_LR/dPdT - eq 18
    double sqrtI = std::sqrt(I);
    double d2j_LRdPdT = -d2A_DHdPdT * sqrtI / (1.+sqrtI);
    d2lnaidPdT += std::pow(charge[i], 2) * d2j_LRdPdT;

    // 2nd term: d2j_SR/dPdT = 0 - eq 20

    // 3rd term: d2j_P1/dPdT = 0 - eq 21
    
    // 4th term: dj_P2/dT = 0 - eq 22

    return d2lnaidPdT;
}
double Jager2003::d2lnai_dT2(int i)
{
      // d2lnai/dT2 for ions: eq 16
    double d2lnaidT2 = 0.;

    // 1st term: d2j_LR/dT2 - eq 18
    double sqrtI = std::sqrt(I);
    double d2j_LRdT2 = -d2A_DHdT2 * sqrtI / (1.+sqrtI);
    d2lnaidT2 += std::pow(charge[i], 2) * d2j_LRdT2;

    // 2nd term: d2j_SR/dT2 - eq 20
    for (int jj = 0; jj < ni; jj++)
    {
        if (charge[i] * charge[jj] < 0)
        {
            int idx = charge[i] > 0 ? i*ni+jj : jj*ni + i;
            int zizj = std::abs(charge[i]*charge[jj]);
            double d2j_SRdT2 = 0.6 * d2BcadT2[idx] * zizj / std::pow(1.+1.5*I/zizj, 2)
                            + d2BcadT2[idx] + d2CcadT2[idx] * I + d2DcadT2[idx] * pow(I, 2);
            d2lnaidT2 += m_s[nc+jj] * d2j_SRdT2 * std::pow(0.5 * (std::abs(charge[i]) + std::abs(charge[jj])), 2);
        }
    }

    // 3rd term: d2j_P1/dT2 - eq 21

    // 3rd term: d2B0/dT2 = 0
    
    // 4th term: d2j_P2/dT2 = 0 - eq 22

    return d2lnaidT2;
}
std::vector<double> Jager2003::dlnai_dxj(int i)
{
    // For lna_i ions (eq. 16)
    std::vector<double> dlnaidxj(ns, 0.);
    double sqrtI = std::sqrt(I);
    double I_inv = 1./I;
    double e = std::exp(-2.*sqrtI);
    double dedI = -e / sqrtI;

    // 1st term: j_LR - eq 18
    double djLRdI = -0.5 * A_DH / (sqrtI * std::pow(1.+sqrtI, 2));
    for (int j = 0; j < ni; j++)
    {
        dlnaidxj[j + nc] += std::pow(this->charge[i], 2) * djLRdI * dIdxj[j+nc];
    }
    dlnaidxj[water_index] += std::pow(this->charge[i], 2) * djLRdI * dIdxj[water_index];

    // 2nd term: j_SR - eq 20
    for (int j = 0; j < ni; j++)
    {
        int ci = this->charge[i];
        int cj = this->charge[j];
        if (ci * cj < 0)
        {
            int idx = ci > 0 ? i*ni+j : j*ni + i;
            int zizj = -ci*cj;
            double cicj2 = std::pow(0.5*(std::abs(ci) + std::abs(cj)), 2);
            double j_SR = (0.13816 + 0.6 * B_ca[idx]) * zizj / std::pow(1.+1.5*I/zizj, 2)
                            + B_ca[idx] + C_ca[idx] * I + D_ca[idx] * std::pow(I, 2);
                
            double djSRdI = -2 * (0.13816 + 0.6 * B_ca[idx]) * zizj / std::pow(1.+1.5*I/zizj, 3) * 1.5/zizj
                            + C_ca[idx] + 2. * D_ca[idx] * I;
            dlnaidxj[i + nc] += m_s[nc+j] * djSRdI * dIdxj[nc+i] * cicj2;
            dlnaidxj[j + nc] += this->dmi_dxi() * j_SR * cicj2 + m_s[nc+j] * djSRdI * dIdxj[nc+j] * cicj2;
            dlnaidxj[water_index] += this->dmi_dxw(nc+j) * j_SR * cicj2 + m_s[nc+j] * djSRdI * dIdxj[water_index] * cicj2;
        }
    }

    // 3rd term: j_P1 - eq 21
    for (int j = 0; j < ns; j++)
    {
        // Pitzer contribution of ions - eq 21 and 22
        double j_P1ij = B0[(i+nc)*ns + j] + 0.5 * B1[(i+nc)*ns + j] * I_inv * (1. - (1. + 2.*sqrtI) * e);
            
        if (j != water_index)
        {
            double dj_P1ij = -0.5 * B1[i*ns + j] * (std::pow(I_inv, 2) * (1. - (1. + 2.*sqrtI) * e) + 
                                                    I_inv * (std::sqrt(I_inv) * e + (1. + 2.*sqrtI) * dedI));
            dlnaidxj[j] += 2. * this->dmi_dxi() * j_P1ij + 2. * m_s[j] * dj_P1ij * dIdxj[j];
            dlnaidxj[water_index] += 2. * this->dmi_dxw(j) * j_P1ij + 2. * m_s[j] * dj_P1ij * dIdxj[water_index];
        }
    }

    // 4th term: j_P2 - eq 22
    double sum_mimj_jP2 = 0.;
    std::vector<double> dsum_mimj_jP2(ns, 0.);
    for (int j = 0; j < ns; j++)
    {
        if (j != water_index)
        {
            for (int k = 0; k < ns; k++)
            {
                if (k != water_index)
                {
                    // Pitzer contribution of ions - eq 21 and 22
                    double j_P2jk = B1[j*ns + k] * (1. - (1. + 2.*sqrtI + 2.*I) * e);
                    sum_mimj_jP2 += m_s[j] * m_s[k] * j_P2jk;

                    double djP2jkdI = B1[j*ns + k] * ((1. - (1. + 2.*sqrtI + 2.*I)) * dedI - (std::sqrt(I_inv) + 2.) * e);
                
                    for (int ii = 0; ii < ns; ii++)
                    {
                        dsum_mimj_jP2[ii] += m_s[j] * m_s[k] * djP2jkdI * dIdxj[ii];
                    }
                    dsum_mimj_jP2[j] += this->dmi_dxi() * m_s[k] * j_P2jk;
                    dsum_mimj_jP2[k] += m_s[j] * this->dmi_dxi() * j_P2jk;
                    dsum_mimj_jP2[water_index] += (this->dmi_dxw(j) * m_s[k] + m_s[j] * this->dmi_dxw(k)) * j_P2jk;
                }
            }
        }
    }
    for (int j = 0; j < ns; j++)
    {
        dlnaidxj[j] += -0.5 * std::pow(this->charge[i], 2) * std::pow(I_inv, 3) * dIdxj[j] * sum_mimj_jP2
                    + 0.25 * std::pow(this->charge[i], 2) * std::pow(I_inv, 2) * dsum_mimj_jP2[j];
    }

    // Eq 15: ai = mi/mi0 * j_i -> lnai = ln(m_i) + ln(j_i)
    dlnaidxj[i+nc] += 1./m_s[i+nc] * this->dmi_dxi();
    dlnaidxj[water_index] += 1./m_s[i+nc] * this->dmi_dxw(i+nc);

    return dlnaidxj;
}
std::vector<double> Jager2003::d2lnai_dTdxj(int i)
{
    // dlnai/dT for ions: eq 16
    std::vector<double> d2lnaidTdxj(ns, 0.);
    double sqrtI = std::sqrt(I);

    // 1st term: dj_LR/dT - eq 18
    double d2jLRdIdT = -0.5 * dA_DHdT / (sqrtI * std::pow(1.+sqrtI, 2));
    for (int j = 0; j < ni; j++)
    {
        d2lnaidTdxj[j + nc] += std::pow(this->charge[i], 2) * d2jLRdIdT * dIdxj[j+nc];
    }
    d2lnaidTdxj[water_index] += std::pow(this->charge[i], 2) * d2jLRdIdT * dIdxj[water_index];

    // 2nd term: j_SR - eq 20
    for (int j = 0; j < ni; j++)
    {
        int ci = this->charge[i];
        int cj = this->charge[j];
        if (ci * cj < 0)
        {
            int idx = ci > 0 ? i*ni+j : j*ni + i;
            int zizj = -ci*cj;
            double cicj2 = std::pow(0.5*(std::abs(ci) + std::abs(cj)), 2);
            double dj_SRdT = 0.6 * dBcadT[idx] * zizj / std::pow(1.+1.5*I/zizj, 2)
                            + dBcadT[idx] + dCcadT[idx] * I + dDcadT[idx] * std::pow(I, 2);
            double d2jSRdTdI = -2 * 0.6 * dBcadT[idx] * zizj / std::pow(1.+1.5*I/zizj, 3) * 1.5/zizj
                            + dCcadT[idx] + 2. * dDcadT[idx] * I;
            d2lnaidTdxj[i + nc] += m_s[nc+j] * d2jSRdTdI * dIdxj[nc+i] * cicj2;
            d2lnaidTdxj[j + nc] += this->dmi_dxi() * dj_SRdT * cicj2 + m_s[nc+j] * d2jSRdTdI * dIdxj[nc+j] * cicj2;
            d2lnaidTdxj[water_index] += this->dmi_dxw(nc+j) * dj_SRdT * cicj2 + m_s[nc+j] * d2jSRdTdI * dIdxj[water_index] * cicj2;
        }
    }

    // 3rd term: j_P1 - eq 21
    for (int j = 0; j < ns; j++)
    {
        // Pitzer contribution of ions - eq 21 and 22
        if (j != water_index)
        {
            double dj_P1ijdT = dB0dT[(i+nc)*ns + j];
            d2lnaidTdxj[j] += 2. * this->dmi_dxi() * dj_P1ijdT;
            d2lnaidTdxj[water_index] += 2. * this->dmi_dxw(j) * dj_P1ijdT;
        }
    }

    // 4th term: dj_P2/dT = 0 - eq 22

    return d2lnaidTdxj;
}

double Jager2003::lnphii(int i) 
{
    // Calculate fugacity coefficient
    double mu = gi[i] - hi[i] + vi[i] + lna[i];  // eq. 3.5/3.10
    return mu - gi0[i] - std::log(x[i]) - std::log(p);
}
double Jager2003::dlnphii_dP(int i) 
{
    jager::V v = jager::V(species[i]);
    double dlnaidP;
    if (i == water_index)
    {
        dlnaidP = this->dlnaw_dP();
    }
    else if (i < nc)
    {
        dlnaidP = this->dlnam_dP(i);
    }
    else
    {
        dlnaidP = this->dlnai_dP(i-nc);
    }
    double dmuidP = v.dFdP(p, T) + dlnaidP;
    return dmuidP - 1./p;
}
double Jager2003::dlnphii_dT(int i) 
{
    double dgi0dT, dlnaidT;
    if (i == water_index)
    {
        jager::IG ig = jager::IG(species[i]);
        dgi0dT = -ig.dHdT(T);
        dlnaidT = this->dlnaw_dT();
    }
    else if (i < nc)
    {
        jager::IG ig = jager::IG(species[i]);
        dgi0dT = -ig.dHdT(T);
        dlnaidT = this->dlnam_dT(i);
    }
    else
    {
        // ideal_gas::IdealGas ideal = ideal_gas::IdealGas("NaCl");
        jager::IG ig = jager::IG(this->compdata.salt);
        dgi0dT = -ig.dHdT(T);
        dlnaidT = this->dlnai_dT(i-nc);
    }
    jager::H h = jager::H(species[i]);
    jager::V v = jager::V(species[i]);
    double dmuidT = -h.dFdT(T) + v.dFdT(p, T) + dlnaidT;
    
    return dmuidT - dgi0dT;
}
double Jager2003::d2lnphii_dPdT(int i) 
{
    double d2lnaidPdT;
    if (i == water_index)
    {
        d2lnaidPdT = this->d2lnaw_dPdT();
    }
    else if (i < nc)
    {
        d2lnaidPdT = this->d2lnam_dPdT(i);
    }
    else
    {
        d2lnaidPdT = this->d2lnai_dPdT(i-nc);
    }
    jager::V v = jager::V(species[i]);
    double d2muidPdT = v.d2FdPdT(p, T) + d2lnaidPdT;
    
    return d2muidPdT;
}
double Jager2003::d2lnphii_dT2(int i) 
{
    double d2gi0dT2, d2lnaidT2;
    if (i == water_index)
    {
        jager::IG ig = jager::IG(species[i]);
        d2gi0dT2 = -ig.d2HdT2(T);
        d2lnaidT2 = this->d2lnaw_dT2();
    }
    else if (i < nc)
    {
        jager::IG ig = jager::IG(species[i]);
        d2gi0dT2 = -ig.d2HdT2(T);
        d2lnaidT2 = this->d2lnam_dT2(i);
    }
    else
    {
        // ideal_gas::IdealGas ideal = ideal_gas::IdealGas("NaCl");
        jager::IG ig = jager::IG(this->compdata.salt);
        d2gi0dT2 = -ig.d2HdT2(T);
        d2lnaidT2 = this->d2lnai_dT2(i-nc);
    }
    jager::H h = jager::H(species[i]);
    jager::V v = jager::V(species[i]);
    double d2muidT2 = -h.d2FdT2(T) + v.d2FdT2(p, T) + d2lnaidT2;
    
    return d2muidT2 - d2gi0dT2;
}
double Jager2003::dlnphii_dxj(int i, int j) 
{
    // dlnphii/dxj = dlnai/dxj - 1/xi dxi/dxj
    if (i == j)
    {
        // dxi/dxj = 1-xi
        return dlnadxj[i*ns + j] - (1.-x[i])/x[i];
    }
    else
    {
        // dxi/dxj = -xi
        return dlnadxj[i*ns + j] + 1.;
    }
}
std::vector<double> Jager2003::d2lnphii_dTdxj(int i)
{
    // dlnphii/dxj = d2lnai/dTdxj
    if (i == water_index)
    {
        return this->d2lnaw_dTdxj();
    }
    else if (i < nc)
    {
        return this->d2lnam_dTdxj(i);
    }
    else
    {
        return this->d2lnai_dTdxj(i-nc);
    }
}

double Jager2003::lnphi0(double X, double T_, bool pt)
{
    // Calculate pure water Gibbs energy
    (void) pt;
    (void) T_;
    this->p = X;
    double mu = gi[water_index] - hi[water_index] + vi[water_index];  // eq. 3.5/3.10
    return mu - gi0[water_index] - std::log(p);
}

int Jager2003::derivatives_test(double p_, double T_, std::vector<double>& x_, double tol, bool verbose)
{
    int error_output = 0;

    double p0 = p_;
    double T0 = T_;
    std::vector<double> x0 = x_;
    double d;

    this->init_PT(p0, T0, AQEoS::CompType::water);
    this->solve_PT(x0, true, AQEoS::CompType::water);
    this->solve_PT(x0, true, AQEoS::CompType::solute);
    if (ni > 0) { this->solve_PT(x0, true, AQEoS::CompType::ion); }

    // Test derivatives of ig, h and v terms
    jager::IG ig = jager::IG("H2O");
	jager::H h = jager::H("H2O");
    jager::V v = jager::V("H2O");
	error_output += ig.test_derivatives(T0, tol, verbose);
	error_output += h.test_derivatives(T0, tol, verbose);
	error_output += v.test_derivatives(p0, T0, tol, verbose);
    // v = jager::V("CO2");
    // error_output += v.test_derivatives(p0, T0, tol, verbose);

    // Test derivatives of lna terms
    std::vector<double> lnai0(ns), dlnaidP(ns), dlnaidT(ns), d2lnaidPdT(ns), d2lnaidT2(ns);
    std::vector<std::vector<double>> dlnaidxj(ns), d2lnaidTdxj(ns);
    for (int i = 0; i < ns; i++)
    {
        lnai0[i] = this->lna[i];
        if (i == water_index)
        {
            dlnaidP[i] = this->dlnaw_dP();
            dlnaidT[i] = this->dlnaw_dT();
            dlnaidxj[i] = this->dlnaw_dxj();
            d2lnaidPdT[i] = this->d2lnaw_dPdT();
            d2lnaidT2[i] = this->d2lnaw_dT2();
            d2lnaidTdxj[i] = this->d2lnaw_dTdxj();
        }
        else if (i < nc)
        {
            dlnaidP[i] = this->dlnam_dP(i);
            dlnaidT[i] = this->dlnam_dT(i);
            dlnaidxj[i] = this->dlnam_dxj(i);
            d2lnaidPdT[i] = this->d2lnam_dPdT(i);
            d2lnaidT2[i] = this->d2lnam_dT2(i);
            d2lnaidTdxj[i] = this->d2lnam_dTdxj(i);
        }
        else
        {
            dlnaidP[i] = this->dlnai_dP(i-nc);
            dlnaidT[i] = this->dlnai_dT(i-nc);
            dlnaidxj[i] = this->dlnai_dxj(i-nc);
            d2lnaidPdT[i] = this->d2lnai_dPdT(i-nc);
            d2lnaidT2[i] = this->d2lnai_dT2(i-nc);
            d2lnaidTdxj[i] = this->d2lnai_dTdxj(i-nc);
        }
    }

    // Test analytical derivatives of lna with respect to P
    double dp = 1e-5;
    this->init_PT(p0+dp, T0, AQEoS::CompType::water);
    this->solve_PT(x0, false, AQEoS::CompType::water);
    this->solve_PT(x0, false, AQEoS::CompType::solute);
    if (ni > 0) { this->solve_PT(x0, false, AQEoS::CompType::ion); }

    for (int i = 0; i < ns; i++)
    {
        double dlnanum = (this->lna[i]-lnai0[i])/dp;
        d = std::log(std::fabs(dlnaidP[i] + 1e-15)) - std::log(std::fabs(dlnanum + 1e-15));
        if (verbose || d > tol)
        {
            print("comp", i);
            print("dlnai/dP", {dlnaidP[i], dlnanum, d});
            error_output++;
        }
    }

    // Test analytical derivatives of lna with respect to T
    double dT = 1e-5;
    this->init_PT(p0, T0+dT, AQEoS::CompType::water);
    this->solve_PT(x0, true, AQEoS::CompType::water);
    this->solve_PT(x0, true, AQEoS::CompType::solute);
    if (ni > 0) { this->solve_PT(x0, true, AQEoS::CompType::ion); }

    for (int i = 0; i < ns; i++)
    {
        double dlnanum = (this->lna[i]-lnai0[i])/dT;
        d = std::log(std::fabs(dlnaidT[i] + 1e-15)) - std::log(std::fabs(dlnanum + 1e-15));
        if (verbose || d > tol)
        {
            print("comp", i);
            print("dlnai/dT", {dlnaidT[i], dlnanum, d});
            error_output++;
        }

        // d2lnai/dPdT
        double dlnai1;
        if (i == water_index)
        {
            dlnai1 = this->dlnaw_dP();
        }
        else if (i < nc)
        {
            dlnai1 = this->dlnam_dP(i);
        }
        else
        {
            dlnai1 = this->dlnai_dP(i-nc);
        }
        
        double d2lnanum = (dlnai1-dlnaidP[i])/dT;
        d = std::log(std::fabs(d2lnaidPdT[i] + 1e-15)) - std::log(std::fabs(d2lnanum + 1e-15));
        if (verbose || d > tol)
        {
            print("comp", i);
            print("d2lnai/dPdT", {d2lnaidPdT[i], d2lnanum, d});
            error_output++;
        }

        // d2lnai/dT2
        if (i == water_index)
        {
            dlnai1 = this->dlnaw_dT();
        }
        else if (i < nc)
        {
            dlnai1 = this->dlnam_dT(i);
        }
        else
        {
            dlnai1 = this->dlnai_dT(i-nc);
        }
        
        d2lnanum = (dlnai1-dlnaidT[i])/dT;
        d = std::log(std::fabs(d2lnaidT2[i] + 1e-15)) - std::log(std::fabs(d2lnanum + 1e-15));
        if (verbose || d > tol)
        {
            print("comp", i);
            print("d2lnai/dT2", {d2lnaidT2[i], d2lnanum, d});
            error_output++;
        }
    }

    // Test analytical derivatives of lna with respect to composition (mole fractions)
    double dx = 1e-6;
    this->init_PT(p0, T0, AQEoS::CompType::water);
	for (int k = 0; k < ns; k++)
	{
		// Transform to +dz
		double dxk = dx * x0[k];
		x0[k] += dxk;
        for (int ii = 0; ii < ns; ii++)
        {
            x0[ii] /= (1. + dxk);
        }
		
		// Numerical derivative of lnphi w.r.t. zk
        this->solve_PT(x0, true, AQEoS::CompType::water);
        this->solve_PT(x0, true, AQEoS::CompType::solute);
        if (ni > 0) { this->solve_PT(x0, true, AQEoS::CompType::ion); }
        std::vector<double> lnai1 = this->lna;
    
		// Compare analytical and numerical
		for (int j = 0; j < ns; j++)
		{
			double dlnaidxk_an = dlnaidxj[j][k];
            double dlnaidxk = (lnai1[j] - lnai0[j]);
			double dlnaidxk_num = dlnaidxk/dxk;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(dlnaidxk_num + 1e-15)) - std::log(std::fabs(dlnaidxk_an + 1e-15));
			if (verbose || std::isnan(dlnaidxk_an) || (!(std::fabs(d) < tol) && std::fabs(dlnaidxk) > 1e-14))
			{
				print("comp j, xk", {j, k});
                print("lna0, lna1", {dlnaidxk, dxk});
				print("dlnaj/dxk", {dlnaidxk_an, dlnaidxk_num, d});
				error_output++;
			}

            double dlnaidT1;
            if (j == water_index)
            {
                dlnaidT1 = this->dlnaw_dT();
            }
            else if (j < nc)
            {
                dlnaidT1 = this->dlnam_dT(j);
            }
            else
            {
                dlnaidT1 = this->dlnai_dT(j-nc);
            }
            double d2lnaidTdxk_an = d2lnaidTdxj[j][k];
            double d2lnaidTdxk = (dlnaidT1 - dlnaidT[j]);
			double d2lnaidTdxk_num = d2lnaidTdxk/dxk;
			// Use logarithmic scale to compare
			d = std::log(std::fabs(d2lnaidTdxk_num + 1e-15)) - std::log(std::fabs(d2lnaidTdxk_an + 1e-15));
			if (verbose || std::isnan(d2lnaidTdxk_an) || (!(std::fabs(d) < tol) && std::fabs(d2lnaidTdxk) > 1e-14))
			{
				print("comp j, xk", {j, k});
				print("d2lnaj/dTdxk", {d2lnaidTdxk_an, d2lnaidTdxk_num, d});
				error_output++;
			}
		}

		// Return to original z
        for (int ii = 0; ii < ns; ii++)
        {
            x0[ii] *= (1. + dxk);
        }
        x0[k] -= dxk;
	}
    return error_output;
}
