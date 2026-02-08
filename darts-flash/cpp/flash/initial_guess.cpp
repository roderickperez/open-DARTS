#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>

#include "dartsflash/global/components.hpp"
#include "dartsflash/global/units.hpp"
#include "dartsflash/flash/initial_guess.hpp"
#include "dartsflash/flash/trial_phase.hpp"

using namespace std;

namespace henry {
	// Henry's constant
	// https://acp.copernicus.org/articles/15/4399/2015/acp-15-4399-2015.pdf
	double T_0 = 298.15;
	std::unordered_map<std::string, double> H0 = {
		{"CO2", 33.}, {"N2", 0.64}, {"H2S", 100.}, {"H2", 0.78},
		{"C1", 1.4}, {"C2", 1.9}, {"C3", 1.5}, {"iC4", 0.91}, {"nC4", 1.2}, {"iC5", 0.7}, {"nC5", 0.8},
		{"nC6", 0.61}, {"nC7", 0.44}, {"nC8", 0.31}, {"nC9", 0.2}, {"nC10", 0.14},
	}; // from Sander (2015)
	std::unordered_map<std::string, double> dlnH = {
		{"CO2", 2400.}, {"N2", 1600.}, {"H2S", 2100.}, {"H2", 530},
		{"C1", 1900.}, {"C2", 2400.}, {"C3", 2700.}, {"iC4", 2700.}, {"nC4", 3100.}, {"iC5", 3400.}, {"nC5", 3400.},
		{"nC6", 3800.}, {"nC7", 4100.}, {"nC8", 4300.}, {"nC9", 5000.}, {"nC10", 5000.},
	}; // from Sander (2015)
}

namespace sI {
	// sI Hydrate parameters
	// Ballard (2002) - A.1.5
	std::unordered_map<std::string, std::vector<double>> a = {
		{"CO2", {15.8336435, 3.119, 0, 3760.6324, 1090.27777, 0, 0}},
		{"H2S", {31.209396, -4.20751374, 0.761087, 8340.62535, -751.895, 182.905, 0}},
		{"C1", {27.474169, -0.8587468, 0, 6604.6088, 50.8806, 1.57577, -1.4011858}},
		{"C2", {14.81962, 6.813994, 0, 3463.9937, 2215.3, 0, 0}}
	};
	std::unordered_map<bool, std::vector<double>> a_N2 = {
		{false, {173.2164, -0.5996, 0, 24751.6667, 0, 0, 0, 1.441, -37.0696, -0.287334, -2.07405E-5, 0, 0}},  // without H2S
		{true, {71.67484, -1.75377, -0.32788, 25180.56, 0, 0, 0, 56.219655, -140.5394, 0, 8.0641E-4, 366006.5, 978852}}  // with H2S
	};
}

namespace sII {
	// sII Hydrate parameters
	// Ballard (2002) - A.1.5
	std::unordered_map<std::string, std::vector<double>> a = {
		{"CO2", {9.0242, 0, 0, -207.033, 0, 6.7588E-4, -6.992E-3, -6.0794E-4, -9.026E-2, 0, 0, 0, 0.0186833, 0, 0, 8.82E-5, 7.78015E-3, 0, 0}},
		{"N2", {1.78857, 0, -0.019667, -6.187, 0, 0, 0, 5.259E-5, 0, 0, 0, 0, 0, 0, 192.39, 0, 3.051E-5, 1.1E-7, 0}},
		{"H2S", {-6.42956, 0.06192, 0, 82.627, 0, -1.0718E-4, 0, 0, 3.493522, -0.64405, 0, 0, 0, -184.257, 0, -1.30E-6, 0, 0, 0}},
		{"C1", {-0.45872, 0, 0, 31.6621, -3.4028, -7.702E-5, 0, 0, 1.8641, -0.78338, 0, 0, 0, -77.6955, 0, -2.3E-7, -6.102E-5, 0, 0}},
		{"C2", {3.21799, 0, 0, -290.283, 181.2694, 0, 0, -1.893E-5, 1.882, -1.19703, -402.166, -4.897688, 0.0411205, -68.8018, 25.6306, 0, 0, 0, 0}},
		{"C3", {-7.51966, 0, 0, 47.056, 0, -1.697E-5, 7.145E-4, 0, 0, 0.12348, 79.34, 0, 0.0160778, 0, -14.684, 5.50E-6, 0, 0, 0}},
		{"nC4", {-37.211, 0.86564, 0, 732.2, 0, 0, 0, 1.9711E-3, -15.6144, 0, 0, -4.56576, 0, 0, 300.55350, 0, 0.0151942, -1.26E-6, 0}},
		{"iC4", {-9.55128, 0, 0, 0, 0, 0, 0.001251, 2.1036E-6, 2.40904, -2.75945, 0, 0, 0, 0, -0.28974, 0, -1.6476E-3, -1.0E-8, 0}},
	};
}

namespace salt {
	std::unordered_map<std::string, std::string> pure_comp = {{"NaCl", "NaCl"}, {"CaCl2", "CaCl2"}, {"KCl", "KCl"}};

    std::unordered_map<std::string, std::vector<double>> bi = {
		{"NaCl", {7.1811e-4, 1.1215e-3, -4.3613e-6, 5.7463e-9}},
		{"CaCl2", {3.9347e-1, -5.3151e-3, 2.2103e-5, 2.6414e-8}},
		{"KCl", {10.9, -9.6298e-2, 2.9116e-4, -2.9591e-7}}
	};
	std::unordered_map<std::string, std::vector<double>> ci = {
		{"NaCl", {-1.70e-5, 1.0e-9, 1.0e-10, 1.0e-14}},
		{"CaCl2", {-2.10e-5, 1.0e-9, 1.0e-10, 1.0e-14}},
		{"KCl", {-2.53e-5, 1.0e-9, 1.0e-10, 1.0e-14}}
	};
}

InitialGuess::InitialGuess(CompData& comp_data_) 
{
	this->comp_data = comp_data_;
	this->units = comp_data.units;

	this->components = comp_data.components;
	this->ions = comp_data.ions;
	this->nc = comp_data.nc;
	this->ni = comp_data.ni;
	this->ns = nc + ni;

	this->water_index = comp_data.water_index;
	this->ypure = std::vector<double>(ns, 0.99);
}

void InitialGuess::init(double p_, double T_)
{
	// Translate input units
    p = (this->units.pressure == Units::PRESSURE::BAR) ? p_ : units.input_to_bar(p_);
    T = (this->units.temperature == Units::TEMPERATURE::KELVIN) ? T_ : units.input_to_kelvin(T_);
    return;
}

std::vector<double> InitialGuess::evaluate(std::vector<int>& initial_guesses)
{
	// Generate lnK-values for direct use in PhaseSplit algorithm
	// Wilson, Henry and Hydrate-Vapour correlations
	int nk = static_cast<int>(initial_guesses.size());
	std::vector<double> lnk(nk * ns);

	int j = 0;
	for (int type : initial_guesses)
	{
		switch (type)
		{
			case Ki::Wilson_VL:
			case Ki::Wilson_LV:
			{
				bool inverse = (type == Ki::Wilson_VL) ? true : false;
				std::vector<double> kwilson = this->k_wilson(inverse);
				for (int i = 0; i < ns; i++)
				{
					lnk[j*ns + i] = kwilson[i];
				}
				j++;
				break;
			}
			case Ki::Henry_VA:
			case Ki::Henry_AV:
			{
				bool inverse = (type == Ki::Henry_VA) ? true : false;
				std::vector<double> khenry = this->k_henry(inverse);
				for (int i = 0; i < ns; i++)
				{
					lnk[j*ns + i] = khenry[i];
				}
				j++;
				break;
			}
			// case K::Ice_IA:
			// case K::Ice_AI:
			// {
			// 	bool inverse = (type == K::Ice_IA) ? false : true;
			// 	std::vector<double> k_ice = this->k_aq_ice(inverse);
			// 	for (int i = 0; i < ns; i++)
			// 	{
			// 		lnk[j*ns + i] = k_ice[i];
			// 	}
			// 	j++;
			// 	break;
			// }
			default:
			{
				std::cout << "Invalid initial guess for phase split\n";
				exit(1);
			}
		}
		j++;
	}
	return lnk;
}

std::vector<TrialPhase> InitialGuess::evaluate(int eos_idx, std::string eos_name, std::vector<int>& initial_guesses, std::vector<TrialPhase>& ref_comps)
{
	// Generate Y-values for use in Stability algorithm
	// According to [Li and Firoozabadi 2012] a few sets of initial guesses are used
    // It is important to point out that this procedure can be changed depending on the components present
    // Trial 1 (Vapor) // Trial 2 (Liquid) // Trial 3 (Vapor)  // Trial 4 (Liquid) // Trial 5 (Pure components)
	std::vector<TrialPhase> trial_comps;

	int j = 0;
	for (int type: initial_guesses)
	{
		switch (type)
		{
			case Yi::Wilson:
			{
				// Generate Wilson and Henry K-values for generating set of compositions Y
				std::vector<double> kwilson = this->k_wilson(false);
				std::transform(kwilson.begin(), kwilson.end(), kwilson.begin(), [](double lnk) { return std::exp(lnk); });

				for (TrialPhase ref_comp: ref_comps)
				{
					trial_comps.resize(j + 2, TrialPhase(eos_idx, eos_name, ref_comp.Y));
					for (int i = 0; i < ns; i++)
					{
						trial_comps[j].Y[i] = ref_comp.Y[i] / kwilson[i];
						trial_comps[j+1].Y[i] = ref_comp.Y[i] * kwilson[i];
					}
					j += 2;
				}
				break;
			}
			case Yi::Wilson13:
			{
				// Generate Wilson and Henry K-values for generating set of compositions Y
				std::vector<double> kwilson = this->k_wilson(false);
				std::transform(kwilson.begin(), kwilson.end(), kwilson.begin(), [](double lnk) { return std::exp(lnk); });

				for (TrialPhase ref_comp: ref_comps)
				{
					trial_comps.resize(j + 2, TrialPhase(eos_idx, eos_name, ref_comp.Y));
					for (int i = 0; i < ns; i++)
					{
						trial_comps[j].Y[i] = ref_comp.Y[i] / std::cbrt(kwilson[i]);
						trial_comps[j+1].Y[i] = ref_comp.Y[i] * std::cbrt(kwilson[i]);
					}
					j += 2;
				}
				break;
			}
			case Yi::Henry:
			{
				if (water_index >= nc)
				{
					std::cout << "Henry's law initial guess specified, but no H2O is present\n";
					exit(1);
				}

				std::vector<double> y = this->y_henry();
				trial_comps.resize(j + 1, TrialPhase(eos_idx, eos_name, y));

				j++;
				break;
			}
			case Yi::Free:
			{
				// Generate Wilson and Henry K-values for generating set of compositions Y
				std::vector<double> kwilson = this->k_wilson(false);
				std::transform(kwilson.begin(), kwilson.end(), kwilson.begin(), [](double lnk) { return std::exp(lnk); });

				for (TrialPhase ref_comp: ref_comps)
				{
					trial_comps.resize(j + 2, TrialPhase(eos_idx, eos_name, ref_comp.Y));
					for (int i = 0; i < ns; i++)
					{
						trial_comps[j].Y[i] = ref_comp.Y[i] / kwilson[i];
						trial_comps[j+1].Y[i] = ref_comp.Y[i] * kwilson[i];
					}
					j += 2;
				}
				break;
			}
			case Yi::Min:
			{
				// Generate initial guess from ymin rich component composition and overall composition of other components
				std::vector<double> y = this->y_min(rich_idx, rich_comp, ref_comps[0].Y);
				trial_comps.resize(j + 1, TrialPhase(eos_idx, eos_name, y));

				j++;
				break;
			}
			default:
			{
				if (type >= Yi::Pure && type < nc)
				{
					for (TrialPhase ref_comp: ref_comps)
					{
						if (ref_comp.Y[type] > this->zero)
						{
							std::vector<double> y = this->y_pure(type);
							trial_comps.resize(j + 1, TrialPhase(eos_idx, eos_name, y));
							j++;
							break;
						}
					}
					break;
				}
				else
				{
					std::cout << "Invalid initial guess for stability test\n";
					exit(1);
				}
			}
		}
	}
	return trial_comps;
}

std::vector<double> InitialGuess::k_wilson(bool inverse)
{
	// Wilson's ideal K-value
	std::vector<double> lnK(ns);

	for (int i = 0; i < nc; i++)
	{
		if (i == water_index)
		{
			lnK[i] = std::log((-133.67 + 0.63288 * T) / p + 3.19211E-3 * p);
		}
		else
		{
			lnK[i] = std::log(comp_data.Pc[i] / p) + 5.373 * (1. + comp_data.ac[i]) * (1. - comp_data.Tc[i] / T);
		}
	}

	// Ions
	for (int i = 0; i < ni; i++)
	{
		lnK[i + nc] = 0.;
	}

	// Inverse
	if (inverse)
	{
		for (int i = 0; i < nc; i++)
		{
			lnK[i] = -lnK[i];
		}
	}
	return lnK;
}

std::vector<double> InitialGuess::y_henry()
{
	// Approximate aqueous phase composition with Henry's constants
	std::vector<double> yhenry(nc);

	for (int i = 0; i < nc; i++)
	{
		if (i != water_index)
		{
			double H = henry::H0[components[i]] * std::exp(henry::dlnH[components[i]] * (1./T - 1./henry::T_0));
			double ca = H*p;
			double rho_Aq = 1100.;
			double Vm = comp_data.Mw[water_index]*1E-3/rho_Aq;
			yhenry[i] = ca*Vm;
		}
		else
		{
			yhenry[i] = 1.;
		}
	}

	return yhenry;
}

std::vector<double> InitialGuess::k_henry(bool inverse)
{
	// Raoult's/Henry's law initial K-value for phase Aq
	std::vector<double> lnK(ns);
	for (int i = 0; i < nc; i++)
	{
		if (i == water_index)
		{
			// Raoult's law for H2O - Ballard (2002) - Appendix A.1.2
			double lnpsat = 12.048399 - 4030.18245 / (T - 38.15);
			lnK[i] = lnpsat - std::log(p);
		}
		else
		{
			// Henry's law for solutes in dilute solution - Sander (2015)
			double x_iV = 1.;
			double H = henry::H0[components[i]] * std::exp(henry::dlnH[components[i]] * (1./T - 1./henry::T_0));
			double ca = H*p;
			double rho_Aq = 1100.;
			double Vm = comp_data.Mw[water_index]*1E-3/rho_Aq;
			double x_iAq = ca*Vm;
			lnK[i] = std::log(x_iV) - std::log(x_iAq);
		}
	}

	// Ions
	for (int i = 0; i < ni; i++)
	{
		lnK[nc + i] = inverse ? 14. : -14.;
	}

	// Inverse
	if (inverse)
	{
		for (int i = 0; i < nc; i++)
		{
			lnK[i] = -lnK[i];
		}
	}

	return lnK;
}

std::vector<double> InitialGuess::k_vapour_sI(bool inverse) 
{
	// Initial K-value for sI
	// Ballard (2002) - Appendix A.1.5
	std::vector<double> lnK(nc);
	double x_wH = 0.88;

	for (int i = 0; i < nc; i++)
	{
		if (i == water_index)
		{
			// Kw_VAq
			double lnpsat = 12.048399 - 4030.18245 / (T - 38.15);
			double lnKw_VAq = lnpsat - std::log(p);
			// Kw_IAq
			double p0 = 6.11657E-3;
			double T0 = 273.1576;
			double Ti = T0 - 7.404E-3*(p-p0) - 1.461E-6*std::pow(p-p0, 2.);
			double x_wAq = 1. + 8.33076E-3*(T-Ti) + 3.91416E-5*std::pow(T-Ti, 2.);
			double Kw_IAq = 1./x_wAq;
			lnK[i] = lnKw_VAq - std::log(x_wH*Kw_IAq);
		}
		else if (components[i] == "N2")
		{
			std::vector<double> a;

			bool H2S = std::find(components.begin(), components.end(), "H2S") != components.end();
			a = sI::a_N2[H2S]; // depends on presence of H2S

			double lnKi_wf = a[0] + a[1]*std::log(p) + a[2]*std::pow(std::log(p), 2.) - (a[3] + a[4]*std::log(p) + a[5]*std::pow(std::log(p), 2.) + a[6]*std::pow(std::log(p), 3.))/T
							+ a[7]/p + a[8]/(std::pow(p, 2)) + a[9]*T + a[10]*p + a[11]*std::log(p)/std::pow(T, 2) + a[12]/std::pow(T, 2);
			lnK[i] = lnKi_wf - std::log(1.-x_wH);
		}
		else
		{
			std::string comp = components[i];
			double lnKi_wf = sI::a[comp][0] + sI::a[comp][1]*std::log(p) + sI::a[comp][2]*std::pow(std::log(p), 2.) - (sI::a[comp][3]
							+ sI::a[comp][4]*std::log(p) + sI::a[comp][5]*std::pow(std::log(p), 2.) + sI::a[comp][6]*std::pow(std::log(p), 3.))/T;
			lnK[i] = lnKi_wf - std::log(1.-x_wH);
		}
	}

	// Ions
	for (int i = 0; i < ni; i++)
	{
		lnK[i + nc] = 0.;
	}

	// Inverse
	if (inverse)
	{
		for (int i = 0; i < nc; i++)
		{
			lnK[i] = -lnK[i];
		}
	}
	return lnK;
}

std::vector<double> InitialGuess::k_vapour_sII(bool inverse) 
{
	// Initial K-value for sII
	// Ballard (2002) - Appendix A.1.5
	std::vector<double> lnK(nc);
	double x_wH = 0.90;

	for (int i = 0; i < nc; i++)
	{
		if (i == water_index)
		{
			// Kw_VAq
			double lnpsat = 12.048399 - 4030.18245 / (T - 38.15);
			double lnKw_VAq = lnpsat - std::log(p);
			// Kw_IAq
			double p0 = 6.11657E-3;
			double T0 = 273.1576;
			double Ti = T0 - 7.404E-3*(p-p0) - 1.461E-6*std::pow(p-p0, 2.);
			double x_wAq = 1. + 8.33076E-3*(T-Ti) + 3.91416E-5*std::pow(T-Ti, 2.);
			double Kw_IAq = 1./x_wAq;
			lnK[i] = lnKw_VAq - std::log(x_wH*Kw_IAq);
		}
		else
		{
			std::string comp = components[i];
			double lnKi_wf = std::exp(sII::a[comp][0] + sII::a[comp][1]*T + sII::a[comp][2]*p + sII::a[comp][3]/T + sII::a[comp][4]/p + sII::a[comp][5]*p*T + sII::a[comp][6]*std::pow(T, 2)
				+ sII::a[comp][7] * std::pow(p, 2) + sII::a[comp][8]*p/T + sII::a[comp][9]*std::log(p/T) + sII::a[comp][10]/std::pow(p, 2) + sII::a[comp][11]*T/p + sII::a[comp][12]*std::pow(T, 2)/p
				+ sII::a[comp][13]*p/std::pow(T, 2) + sII::a[comp][14]*T/std::pow(p, 3) + sII::a[comp][15]*std::pow(T, 3) + sII::a[comp][16]*std::pow(p, 3)/std::pow(T, 2) + sII::a[comp][17]*std::pow(T, 4)) + sII::a[comp][18]*std::log(p);
			lnK[i] = lnKi_wf - std::log(1.-x_wH);
		}
	}

	// Ions
	for (int i = 0; i < ni; i++)
	{
		lnK[i + nc] = 0.;
	}

	// Inverse
	if (inverse)
	{
		for (int i = 0; i < nc; i++)
		{
			lnK[i] = -lnK[i];
		}
	}
	return lnK;
}

std::vector<double> InitialGuess::k_aq_ice(bool inverse)
{
	// Initial K-value for Ice
	// Ballard (2002) - Appendix A.1.3
	double sign = (inverse) ? -1. : 1.;
	std::vector<double> lnK(nc, sign * 16.);

	double T_0 = 273.1576; // K
	double P_0 = 6.11657e-3;  // bar
	double T_ice = T_0 - 7.404e-3 * (p - P_0) - 1.461e-6 * std::pow(p - P_0, 2);
	double x_wA = 1. + 8.33076e-3 * (T - T_ice) + 3.91416e-5 * std::pow(T - T_ice, 2);
	lnK[water_index] = sign * std::log(x_wA);
	return lnK;
}

// std::vector<double> InitialGuess::k_aq_salt(bool inverse)
// {
// 	// Initial K-value for Salt
// 	// Ballard (2002) - Appendix A.1.4
// 	double sign = (inverse) ? -1. : 1.;
// 	std::vector<double> lnK(nc, sign * 16.);

// 	std::vector<double> bi = salt::bi["NaCl"];
// 	std::vector<double> ci = salt::ci["NaCl"];
// 	double x_iA = 0.;
// 	for (int i = 0; i < 4; i++)
// 	{
// 		double ai = bi[i] * (1. - std::exp(ci[i] * (p - 1.)));
// 		x_iA += ai * std::pow(T, i);
// 	}

// 	lnK[salt_index] = sign * std::log(x_iA);
// 	return;
// }

std::vector<double> InitialGuess::y_pure(int j, double pure)
{
	std::vector<double> Y(nc, 0.);
	double Ypure = (!std::isnan(pure)) ? pure : this->ypure[j];

	for (int i = 0; i < nc; i++)
    {
    	if (i == j)
        {
            Y[i] = Ypure;
        }
        else
        {
            Y[i] = (1.-Ypure)/(nc-1);
        }
    }
	return Y;
}

std::vector<double> InitialGuess::y_min(int rich_idx_, double rich_comp_, std::vector<double>& composition)
{
    std::vector<double> Y(ns);
    Y[rich_idx_] = rich_comp_;

    double Yi_inv = 1./(std::accumulate(composition.begin(), composition.end(), 0.) - composition[rich_idx_]);

    for (int i = 0; i < ns; i++)
    {
        if (i != rich_idx_)
        {
            Y[i] = (1.-Y[rich_idx_]) * composition[i] * Yi_inv;
        }
    }
    return Y;
}
