#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include <numeric>

#include "dartsflash/maths/maths.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"

namespace ziabakhsh { // parameters for Ziabaksh correlation for Aq phase
	std::vector<double> Psw = {-7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502};
	std::unordered_map<std::string, std::vector<double>> labda = {
		{"CO2", {-0.0652869, 1.6790636E-4, 40.838951, 0, 0, -3.9266518E-2, 0, 2.1157167E-2, 6.5486487E-6, 0, 0, 0}},
		{"N2", {-2.0939363, 3.1445269E-3, 3.913916E2, -2.9973977E-7, 0, -1.5918098E-5, 0, 0, 0, 0, 0, 0}},
		{"H2S", {1.03658689, -1.1784797E-3, -1.7754826E2, -4.5313285E-4, 0, 0, 0, 0, 0, 0.4775165E2, 0, 0}},
		{"C1", {-5.7066455E-1, 7.2997588E-4, 1.5176903E2, 3.1927112E-5, 0, -1.642651E-5, 0, 0, 0, 0, 0, 0}},
		{"C2", {-2.143686, 2.598765E-3, 4.6942351E2, -4.6849541E-5, 0, 0, 0, 0, 0, 0, -8.4616602E-10, 1.095219E-6}},
		{"C3", {0.513068, -0.000958, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"iC4", {0.52862384, -1.0298104E-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"nC4", {0.52862384, -1.0298104E-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"H2", {-2.0939363, 3.1445269E-3, 3.913916E2, -2.9973977E-7, 0, -1.5918098E-5, 0, 0, 0, 0, 0, 0}},
		// {"iC5", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		// {"nC5", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		// {"nC6", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		// {"nC7", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		// {"nC8", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		// {"nC9", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		// {"nC10", {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	};
	std::unordered_map<std::string, std::vector<double>> ksi = {
		{"CO2", {-1.144624E-2, 2.8274958E-5, 1.3980876E-2, -1.4349005E-2}},
		{"N2", {-6.3981858E-3, 0, 0, 0}},
		{"H2S", {-0.010274152, 0, 0, 0}},
		{"C1", {-2.9990084E-3, 0, 0, 0}},
		{"C2", {-1.0165947E-2, 0, 0, 0}},
		{"C3", {-0.007485, 0, 0, 0}},
		{"iC4", {0.0206946, 0, 0, 0}},
		{"nC4", {0.0206946, 0, 0, 0}},
		{"H2", {-6.3981858E-3, 0, 0, 0}},
	};
	std::unordered_map<std::string, double> eta = {
		{"CO2", -0.114535}, {"N2", -0.008194}, {"H2S", 0.77357854}, 
		{"C1", -0.092248}, {"C2", -0.6091}, {"C3", -1.1471}, {"iC4", -1.6849}, {"nC4", -1.6849},
		{"H2", 0.1}
	};
	std::unordered_map<std::string, double> tau = {
		{"CO2", -5.279063}, {"N2", -5.175337}, {"H2S", 0.27049433}, 
		{"C1", -5.779280}, {"C2", -16.8037}, {"C3", -25.3879}, {"iC4", -33.8492}, {"nC4", -33.8492},
		{"H2", -6.6}
	};
	std::unordered_map<std::string, double> beta = {
		{"CO2", 6.187967}, {"N2", 6.906469}, {"H2S", 0.27543436}, 
		{"C1", 7.262730}, {"C2", 20.0628}, {"C3", 28.2616}, {"iC4", 36.1457}, {"nC4", 36.1457},
		{"H2", 7.3}
	};
	std::unordered_map<std::string, double> Gamma = {
		{"CO2", 0}, {"N2", 0}, {"H2S", 0}, 
		{"C1", 0}, {"C2", 0}, {"C3", 0}, {"iC4", 0}, {"nC4", 0},
		{"H2", 0}
	};
	
	double R{ 83.145 };
}

Ziabakhsh2012::Ziabakhsh2012(CompData& comp_data) : AQBase(comp_data) 
{
	lnk_H.resize(nc);
	labda.resize(nc);
	ksi.resize(nc);
}

void Ziabakhsh2012::init_PT(double p_, double T_, AQEoS::CompType comp_type)
{
	// Calculate composition-independent parameters of Ziabakhsh EoS
	this->p = p_;
	this->T = T_;
	double tc = T - 273.15;  // temperature in Celsius
	double R = ziabakhsh::R;

	if (comp_type == AQEoS::CompType::water)
	{
		// Expression for H2O
		double logK0_H2O = -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3);
		K0_H2O = std::pow(10., logK0_H2O);  // equilibrium constant for H2O at 1 bar
		lnKw = std::log(K0_H2O) + (p - 1.0) * V_H2O / (R * T);
	}
	else if (comp_type == AQEoS::CompType::solute)
	{
		// Expression for non-ionic solutes
		double denom = 1./(0.9998396 + 18.224944E-3 * tc - 7.922210E-6 * std::pow(tc, 2) - 55.44846E-9 * std::pow(tc, 3) + 149.7562E-12 * std::pow(tc, 4) - 393.2952E-15 * std::pow(tc, 5));
		double ddenom = 18.224944E-3 - 2.* 7.922210E-6 * tc - 3.* 55.44846E-9 * std::pow(tc, 2.) + 4.* 149.7562E-12 * std::pow(tc, 3.) - 5.* 393.2952E-15 * std::pow(tc, 4.);
		double d2denom = - 2.* 7.922210E-6 - 6.* 55.44846E-9 * tc + 12.* 149.7562E-12 * std::pow(tc, 2.) - 20.* 393.2952E-15 * std::pow(tc, 3.);
		double V0 = (1. + 18.159725E-3 * tc) * denom;
		double dV0dT = 18.159725E-3 * denom 
						- (1. + 18.159725E-3 * tc) * ddenom * std::pow(denom, 2);
		double d2V0dT2 = -2. * 18.159725E-3 * ddenom * std::pow(denom, 2)
						- (1. + 18.159725E-3 * tc) * (d2denom * std::pow(denom, 2) - 2* std::pow(ddenom, 2) * std::pow(denom, 3));

		tc = 50.; // correlation breaks down at high T
		double B = 19654.320 + 147.037 * tc - 2.21554 * std::pow(tc, 2) + 1.0478E-2 * std::pow(tc, 3) - 2.2789E-5 * std::pow(tc, 4);
		double dBdT = 147.037 - 2.* 2.21554 * tc + 3.* 1.0478E-2 * std::pow(tc, 2.) - 4.* 2.2789E-5 * std::pow(tc, 3.);
		double A1 = 3.2891 - 2.3910E-3 * tc + 2.8446E-4 * std::pow(tc, 2) - 2.8200E-6 * std::pow(tc, 3) + 8.477E-9 * std::pow(tc, 4);
		double dA1dT = - 2.3910E-3 + 2.* 2.8446E-4 * tc - 3.* 2.8200E-6 * std::pow(tc, 2.) + 4.* 8.477E-9 * std::pow(tc, 3.);
		double A2 = 6.245E-5 - 3.913E-6 * tc - 3.499E-8 * std::pow(tc, 2) + 7.942E-10 * std::pow(tc, 3) - 3.299E-12 * std::pow(tc, 4);
		double dA2dT = - 3.913E-6 - 2.* 3.499E-8 * tc + 3.* 7.942E-10 * std::pow(tc, 2.) - 4.* 3.299E-12 * std::pow(tc, 3.);
		dBdT = 0; dA1dT = 0; dA2dT = 0;
		// double d2BdT2{ 0. }, d2A1dT2{ 0. }, d2A2dT2{ 0. };

		denom = 1./(B + A1 * p + A2 * std::pow(p, 2));
		double V = V0 - V0 * p * denom;  // volume of pure water at p[cm3 / g]
		double dVdT = dV0dT - (dV0dT * p) * denom 
							+ V0 * p * (dBdT + dA1dT * p + dA2dT * std::pow(p, 2)) * std::pow(denom, 2);
		double d2VdT2 = d2V0dT2 - d2V0dT2 * p * denom;
							// - dV0dT * p * (dBdT + dA1dT * p + dA2dT * std::pow(p, 2)) * std::pow(denom, 2)
							// - (dV0dT * p * (dBdT + dA1dT * p + dA2dT * std::pow(p, 2)) * std::pow(denom, 2) 
							//	+ V0 * p * (d2BdT2 + d2A1dT2 * p + d2A2dT2 * std::pow(p, 2)) * std::pow(denom, 2)
							//	- 2. * V0 * p * std::pow(dBdT + dA1dT * p + dA2dT * std::pow(p, 2), 2) * std::pow(denom, 3)));
		double dVdP = -V0 * denom + V0*p * (A1 + 2. * A2 * p) * std::pow(denom, 2);
		double d2VdPdT = -dV0dT * denom + (dV0dT*p * (A1 + 2. * A2 * p) + V0*p * (dA1dT + 2. * dA2dT * p)) * std::pow(denom, 2);

		rho0_H2O = 1. / V;  // density of pure water at p[g / cm3]
		drho0_H2OdT = -1. / std::pow(V, 2) * dVdT;
		drho0_H2OdP = -1. / std::pow(V, 2) * dVdP;
		d2rho0_H2OdPdT = 2. / std::pow(V, 3) * dVdT * dVdP - 1. / std::pow(V, 2) * d2VdPdT;
		d2rho0_H2OdT2 = 2. / std::pow(V, 3) * std::pow(dVdT, 2) - 1. / std::pow(V, 2) * d2VdT2;

		double tau = 1. - T / comp_data::Tc["H2O"];  // 1-Tr
		double dtaudT = -1. / comp_data::Tc["H2O"];
		std::vector<double> Psw = ziabakhsh::Psw;

		double term = Psw[0] * tau + Psw[1] * std::pow(tau, 1.5) + Psw[2] * std::pow(tau, 3.) + Psw[3] * std::pow(tau, 3.5) + Psw[4] * std::pow(tau, 4.0) + Psw[5] * std::pow(tau, 7.5);
		double dterm = Psw[0] + Psw[1] * 1.5*std::pow(tau, 0.5) + Psw[2] * 3.*std::pow(tau, 2.) + Psw[3] * 3.5 * std::pow(tau, 2.5) + Psw[4] * 4. * std::pow(tau, 3.) + Psw[5] * 7.5 * std::pow(tau, 6.5);
		double d2term = Psw[1] * 0.75*std::pow(tau, -0.5) + Psw[2] * 6.*tau + Psw[3] * 2.5*3.5 * std::pow(tau, 1.5) + Psw[4] * 12. * std::pow(tau, 2.) + Psw[5] * 6.5*7.5 * std::pow(tau, 5.5);

		double exp_ = comp_data::Tc["H2O"] / T * term;
		double dexpdT = -comp_data::Tc["H2O"] / std::pow(T, 2) * term + comp_data::Tc["H2O"] / T * dtaudT * dterm;
		double d2expdT2 = comp_data::Tc["H2O"] * (2. / std::pow(T, 3) * term - 1. / std::pow(T, 2) * dterm * dtaudT
						- 1. / std::pow(T, 2) * dtaudT * dterm + 1. / T * d2term * std::pow(dtaudT, 2));

		// Ps = Pc * exp(a(T)) 
		double P_s = comp_data::Pc["H2O"] * std::exp(exp_);
		double dPsdT = comp_data::Pc["H2O"] * std::exp(exp_) * dexpdT;
		double d2PsdT2 = comp_data::Pc["H2O"] * std::exp(exp_) * (std::pow(dexpdT, 2) + d2expdT2);

		// df0/dT = dPs/dT * exp(c(T)) + Ps * exp(c(T)) * c'(T)
		f0_H2O = P_s * std::exp((p - P_s) * Mw * V / (R * T));  // fugacity of pure water[bar]
		df0_H2OdT = dPsdT * std::exp((p - P_s) * Mw * V / (R * T))
							+ P_s * std::exp((p - P_s) * Mw * V / (R * T)) * Mw/R * (-dPsdT * V/T + (p-P_s) * dVdT / T - (p-P_s)*V / std::pow(T, 2));
		d2f0_H2OdT2 = d2PsdT2 * std::exp((p - P_s) * Mw * V / (R * T)) 
							+ 2 * dPsdT * std::exp((p - P_s) * Mw * V / (R * T)) * Mw/R * (-dPsdT * V/T + (p-P_s) * dVdT / T - (p-P_s)*V / std::pow(T, 2))
							+ P_s * std::exp((p - P_s) * Mw * V / (R * T)) * 
									(std::pow(Mw/R * (-dPsdT * V/T + (p-P_s) * dVdT / T - (p-P_s)*V / std::pow(T, 2)), 2) + 
									Mw/R * (-d2PsdT2 * V/T + (p-P_s) * d2VdT2 / T + 2. * (p-P_s)*V / std::pow(T, 3)));
		df0_H2OdP = P_s * std::exp((p - P_s) * Mw * V / (R * T)) * Mw/(R*T) * (V + (p-P_s) * dVdP);
		d2f0_H2OdPdT = dPsdT * std::exp((p - P_s) * Mw * V / (R * T)) * Mw/(R*T) * (V + (p-P_s) * dVdP)
					 + P_s * std::exp((p - P_s) * Mw * V / (R * T)) * (p - P_s) * Mw * V / (R * T) * Mw / R * (-dPsdT * V/T + (p - P_s) * (dVdT / T - V / std::pow(T, 2))) * Mw/(R*T) * (V + (p-P_s) * dVdP)
					 + P_s * std::exp((p - P_s) * Mw * V / (R * T)) * (-Mw/(R*std::pow(T, 2)) * (V + (p-P_s) * dVdP) + Mw/(R*T) * (dVdT - dPsdT * dVdP + (p-P_s) * d2VdPdT));

		// Construct constant part of fugacity coefficients
		for (int i = 0; i < nc; i++) 
		{
			if (i != water_index)  // non-H2O components
			{
				std::string comp = species[i];
				if (ziabakhsh::labda.find(comp) != ziabakhsh::labda.end())
				{
					std::vector<double> ai = ziabakhsh::labda[comp];
					std::vector<double> bi = ziabakhsh::ksi[comp];
					double dB = ziabakhsh::tau[comp] + ziabakhsh::Gamma[comp] * p + ziabakhsh::beta[comp] * std::sqrt(1000. / T);
					lnk_H[i] = (1 - ziabakhsh::eta[comp]) * std::log(f0_H2O) + ziabakhsh::eta[comp] * std::log(R * T / Mw * rho0_H2O) + 2. * rho0_H2O * dB;
					labda[i] = ai[0] + ai[1] * T + ai[2] / T + ai[3] * p + ai[4] / p + ai[5] * p / T + ai[6] * T / std::pow(p, 2) + ai[7] * p / (630 - T) + ai[8] * T * std::log(p) + ai[9] * p / std::pow(T, 2) + ai[10] * std::pow(p, 2) * T + ai[11] * p * T;
					ksi[i] = bi[0] + bi[1] * T + bi[2] * p / T + bi[3] * p / (630.0 - T);
				}
				else
				{
					lnk_H[i] = NAN;
				}
			}
		}
	}
	else if (comp_type == AQEoS::CompType::ion)
	{
		// Expression for ions non-existent
	}
	else
	{
		print("Invalid CompType for Ziabakhsh2012 correlation specified", comp_type);
		exit(1);
	}
	return;
}

void Ziabakhsh2012::solve_PT(std::vector<double>& x_, bool second_order, AQEoS::CompType comp_type) 
{
	// Calculate molality of molecular and ionic species
	this->x = x_;  // Water mole fraction to approximate activity

	if (comp_type == AQEoS::CompType::water)
	{

	}
	else if (comp_type == AQEoS::CompType::solute)
	{
    	// Find effective salt molalities
		m_c = 0.;
		m_ac = 0.;
		if (ni > 0)
		{
			for (int i = 0; i < ni; i++)
			{	
				int ci = this->charge[i];
				if (ci > 0)
				{
					double msi = this->mi(nc+i);
					m_c += msi * ci;
					for (int j = 0; j < ni; j++)
					{
						if (this->charge[j] < 0)
						{
							m_ac += msi * this->mi(nc+j);
						}
					}
				}
			}
			if (second_order)
			{
				dmcdxj = std::vector<double>(ns, 0.);
				dmacdxj = std::vector<double>(ns, 0.);
				for (int i = 0; i < ni; i++)
				{
					int ci = this->charge[i];
					if (ci > 0)
					{
						double msi = this->mi(nc+i);
						dmcdxj[nc+i] += this->dmi_dxi() * ci;
						dmcdxj[water_index] += this->dmi_dxw(nc+i) * ci;
						for (int j = 0; j < ni; j++)
						{
							if (this->charge[j] < 0)
							{
								double msj = this->mi(nc + j);
								dmacdxj[nc+i] += this->dmi_dxi() * msj;
								dmacdxj[nc+j] += msi * this->dmi_dxi();
								dmacdxj[water_index] += this->dmi_dxw(nc+i) * msj + msi * this->dmi_dxw(j+nc);
							}
						}
					}	
				}
			}
		}
	}
	else if (comp_type == AQEoS::CompType::ion)
	{
		// Expression for ions non-existent
	}
	else
	{
		print("Invalid CompType for Ziabakhsh2012 correlation specified", comp_type);
		exit(1);
	}
	
    return;
}

double Ziabakhsh2012::lnphii(int i) 
{
	// Calculate fugacity coefficient
	if (i == water_index)
	{
		// return lnKw + std::log(x[water_index]) - std::log(p);
		return lnKw - std::log(p);
	}
	else if (i < nc)
	{
		if (!std::isnan(lnk_H[i]))
		{
			double lnji = 2. * m_c * labda[i] + m_ac * ksi[i];
			return lnji + lnk_H[i] - std::log(p);
		}
		else
		{
			return 20.;
		}
	}
	else
	{
		return 0.;
	}
}
double Ziabakhsh2012::dlnphii_dP(int i) 
{
	// Calculate derivative of fugacity coefficient with respect to P
	if (i == water_index)
	{
		// lnKw = std::log(K0_H2O) + (p - 1.0) * V_H2O / (R * T);
		// dlnKw = 0 + V_H2O/RT
		double dlnKw_dP = V_H2O / (ziabakhsh::R * T);
		return dlnKw_dP - 1./p;
	}
	else if (i < nc)
	{
		if (ziabakhsh::labda.find(species[i]) == ziabakhsh::labda.end())
		{
			return 0.;
		}
		else
		{
			double dlnj_dP = 0.;
			if (ni > 0)
			{
				std::vector<double> ai = ziabakhsh::labda[species[i]];
				std::vector<double> bi = ziabakhsh::ksi[species[i]];
				double dlabda_dP = ai[3] - ai[4] / std::pow(p, 2) + ai[5] / T - 2.* ai[6] * T / std::pow(p, 3.) + ai[7] / (630 - T) + ai[8] * T * 1./p + ai[9] / std::pow(T, 2) + 2.* ai[10] * p * T + ai[11] * T;
				double dksi_dP = bi[2] / T + bi[3] / (630.0 - T);
				dlnj_dP = 2 * m_c * dlabda_dP + m_ac * dksi_dP;
			}

			double dB = ziabakhsh::tau[species[i]] + ziabakhsh::Gamma[species[i]] * p + ziabakhsh::beta[species[i]] * std::sqrt(1000. / T);
			double dBdP = ziabakhsh::Gamma[species[i]];
			double dlnkH_dP = (1. - ziabakhsh::eta[species[i]]) * 1./f0_H2O * df0_H2OdP
								+ ziabakhsh::eta[species[i]] * 1./(ziabakhsh::R * T / Mw * rho0_H2O) * ziabakhsh::R * T / Mw * drho0_H2OdP
								+ 2. * (drho0_H2OdP * dB + rho0_H2O * dBdP);
			return dlnj_dP + dlnkH_dP - 1./p;
		}
	}
	else
	{
		// no expression for ions fugacity -> derivative is zero
		return 0.;
	}
}
double Ziabakhsh2012::dlnphii_dT(int i) 
{
	// Calculate derivative of fugacity coefficient with respect to T
	if (i == water_index)
	{
		// lnKw = std::log(K0_H2O) + (p - 1.0) * V_H2O / (R * T);
		// logK0_H2O = -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3);
		// K0_H2O = std::pow(10., logK0_H2O);  // equilibrium constant for H2O at 1 bar

		// dlnKw/dT = d/dT log(K0_H2O) - (p-1.0)*V_H2O / (R * T^2);
		// d/dT log(K0_H2O) = 1/(K0_H2O) * d/dT K0_H2O
		// d/dT K0_H2O = 10^c * ln (10) * dc/dT
		double tc = T - 273.15;
		double dKdT = std::pow(10, -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3)) 
					* std::log(10.) * (3.097E-2 - 2. * 1.098E-4 * tc + 3. * 2.048E-7 * std::pow(tc, 2));
		double dlogK_dT = 1/(K0_H2O) * dKdT;
		return dlogK_dT - (p-1.0) * V_H2O / (ziabakhsh::R * pow(T, 2));
	}
	else if (i < nc)
	{
		// lnphii = lnj(i) + lnk_H[i] - ln(p)
		// dlnphii/dT = dlnj/dT + dlnkH/dT

		if (ziabakhsh::labda.find(species[i]) == ziabakhsh::labda.end())
		{
			return 0.;
		}
		else
		{
			// lnj = 2 * m_c * labda[i] + m_ac * ksi[i];
			// dlnj/dT = 2 * m_c * d/dT labda + m_ac * d/dT ksi;
			double dlnj_dT = 0.;
			if (ni > 0)
			{
				std::vector<double> ai = ziabakhsh::labda[species[i]];
				std::vector<double> bi = ziabakhsh::ksi[species[i]];
				double dlabda_dT = ai[1] - ai[2] / std::pow(T, 2) - ai[5] * p / std::pow(T, 2) + ai[6] / std::pow(p, 2) + ai[7] * p / std::pow(630 - T, 2) + ai[8] * std::log(p) - 2. * ai[9] * p / std::pow(T, 3) + ai[10] * std::pow(p, 2) + ai[11] * p;
				double dksi_dT = bi[1] - bi[2] * p / std::pow(T, 2) + bi[3] * p / std::pow(630.0 - T, 2);
				dlnj_dT = 2 * m_c * dlabda_dT + m_ac * dksi_dT;
			}

			// lnkH = (1 - eta) * std::log(f0_H2O) + eta * ln(R * T / M * rho0_H2O) + 2. * rho0_H2O * dB;
			double dB = ziabakhsh::tau[species[i]] + ziabakhsh::Gamma[species[i]] * p + ziabakhsh::beta[species[i]] * std::sqrt(1000. / T);
			double dBdT = ziabakhsh::beta[species[i]] * std::sqrt(1000.) * -0.5 * std::pow(T, -1.5);
			double dlnkH_dT = (1 - ziabakhsh::eta[species[i]]) * 1./f0_H2O * df0_H2OdT
							+ ziabakhsh::eta[species[i]] * 1./(ziabakhsh::R * T / Mw * rho0_H2O) * (ziabakhsh::R / Mw * rho0_H2O + ziabakhsh::R * T / Mw * drho0_H2OdT)
							+ 2. * (drho0_H2OdT * dB + rho0_H2O * dBdT);
			return dlnj_dT + dlnkH_dT;
		}
	}
	else
	{
		// no expression for ions fugacity -> derivative is zero
		return 0.;
	}
}
double Ziabakhsh2012::d2lnphii_dPdT(int i) 
{
	// Calculate derivative of fugacity coefficient with respect to P
	if (i == water_index)
	{
		// lnKw = std::log(K0_H2O) + (p - 1.0) * V_H2O / (R * T);
		// dlnKw = 0 + V_H2O/RT
		// d2lnKw/dPdT = dV_H2O/dP / RT
		return -V_H2O / (ziabakhsh::R * std::pow(T, 2));
	}
	else if (i < nc)
	{
		if (ziabakhsh::labda.find(species[i]) == ziabakhsh::labda.end())
		{
			return 0.;
		}
		else
		{
			double d2lnj_dPdT = 0.;
			if (ni > 0)
			{
				std::vector<double> ai = ziabakhsh::labda[species[i]];
				std::vector<double> bi = ziabakhsh::ksi[species[i]];
				double d2labda_dPdT = - ai[5] / std::pow(T, 2) - 2.* ai[6] / std::pow(p, 3.) + ai[7] / std::pow(630 - T, 2) + ai[8]/p - 2. * ai[9] / std::pow(T, 3) + 2.* ai[10] * p + ai[11];
				double d2ksi_dPdT = -bi[2] / std::pow(T, 2) + bi[3] / std::pow(630.0 - T, 2);
				d2lnj_dPdT = 2 * m_c * d2labda_dPdT + m_ac * d2ksi_dPdT;
			}

			double dB = ziabakhsh::tau[species[i]] + ziabakhsh::Gamma[species[i]] * p + ziabakhsh::beta[species[i]] * std::sqrt(1000. / T);
			double dBdP = ziabakhsh::Gamma[species[i]];
			double dBdT = ziabakhsh::beta[species[i]] * std::sqrt(1000.) * -0.5 * std::pow(T, -1.5);
			double d2lnkH_dPdT = (1. - ziabakhsh::eta[species[i]]) * (1./f0_H2O * d2f0_H2OdPdT - 1./std::pow(f0_H2O, 2) * df0_H2OdT * df0_H2OdP)
								+ ziabakhsh::eta[species[i]] * (1./std::pow(ziabakhsh::R * T / Mw * rho0_H2O, 2) * (ziabakhsh::R / Mw * rho0_H2O + ziabakhsh::R * T / Mw * drho0_H2OdT) * ziabakhsh::R * T / Mw * drho0_H2OdP 
															  + 1./(ziabakhsh::R * T / Mw * rho0_H2O) * (ziabakhsh::R / Mw * drho0_H2OdP + ziabakhsh::R * T / Mw * d2rho0_H2OdPdT))
								+ 2. * (d2rho0_H2OdPdT * dB + drho0_H2OdP * dBdT + drho0_H2OdT * dBdP);
			return d2lnj_dPdT + d2lnkH_dPdT;
		}
	}
	else
	{
		// no expression for ions fugacity -> derivative is zero
		return 0.;
	}
}
double Ziabakhsh2012::d2lnphii_dT2(int i) 
{
	// Calculate second derivative of fugacity coefficient with respect to T
	if (i == water_index)
	{
		// lnKw = std::log(K0_H2O) + (p - 1.0) * V_H2O / (R * T);
		// logK0_H2O = -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3);
		// K0_H2O = std::pow(10., logK0_H2O);  // equilibrium constant for H2O at 1 bar

		// dlnKw/dT = d/dT log(K0_H2O) - (p-1.0)*V_H2O / (R * T^2);
		// d/dT log(K0_H2O) = 1/(K0_H2O) * d/dT K0_H2O
		// d/dT K0_H2O = 10^c * ln (10) * dc/dT
		double tc = T - 273.15;
		double dKdT = std::pow(10, -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3)) 
					* std::log(10.) * (3.097E-2 - 2. * 1.098E-4 * tc + 3. * 2.048E-7 * std::pow(tc, 2));
		double d2KdT2 = std::pow(10, -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3)) 
						* std::pow(std::log(10.), 2) * std::pow(3.097E-2 - 2. * 1.098E-4 * tc + 3. * 2.048E-7 * std::pow(tc, 2), 2) 
						+ std::pow(10, -2.209 + 3.097E-2 * tc - 1.098E-4 * std::pow(tc, 2) + 2.048E-7 * std::pow(tc, 3)) 
						* std::log(10.) * (- 2. * 1.098E-4 + 6. * 2.048E-7 * tc);
		double d2logK_dT2 = d2KdT2/K0_H2O - std::pow(dKdT/K0_H2O, 2);
		return d2logK_dT2 + 2. * (p-1.0) * V_H2O / (ziabakhsh::R * pow(T, 3));
	}
	else if (i < nc)
	{
		// lnphii = lnj(i) + lnk_H[i] - ln(p)
		// dlnphii/dT = dlnj/dT + dlnkH/dT

		if (ziabakhsh::labda.find(species[i]) == ziabakhsh::labda.end())
		{
			return 0.;
		}
		else
		{
			// lnj = 2 * m_c * labda[i] + m_ac * ksi[i];
			// dlnj/dT = 2 * m_c * d/dT labda + m_ac * d/dT ksi;
			// d2lnj/dT2 = 2 * m_c * d2/dT2 labda + m_ac * d2/dT2 ksi
			double d2lnj_dT2 = 0.;
			if (ni > 0)
			{
				std::vector<double> ai = ziabakhsh::labda[species[i]];
				std::vector<double> bi = ziabakhsh::ksi[species[i]];

				double d2labda_dT2 = 2. * ai[2] / std::pow(T, 3) + 2. * ai[5] * p / std::pow(T, 3) + 2 * ai[7] * p / std::pow(630 - T, 3) + 6. * ai[9] * p / std::pow(T, 4);
				double d2ksi_dT2 = 2. * bi[2] * p / std::pow(T, 3) + 2 * bi[3] * p / std::pow(630.0 - T, 3);
				d2lnj_dT2 = 2 * m_c * d2labda_dT2 + m_ac * d2ksi_dT2;
			}

			// lnkH = (1 - eta) * ln(f0_H2O) + eta * ln(R * T / M * rho0_H2O) + 2. * rho0_H2O * dB;
			// dlnkH/dT = (1 - eta) / f0_H2O * df0_H2O/dT + eta * 1/(RT/Mw rho0_H2O) * (R/Mw * rho0_H2O + RT/Mw drho0_H2O/dT) + 2 dB drho0/dT + 2 rho0 ddB/dT
			// d2lnkH/dT2 = -(1-eta)/f0_H2O^2 * (df0/dT)^2 + (1-eta)/f0 d2f0/dT2 
			//				- eta (R/Mw * rho0_H2O + RT/Mw drho0_H2O/dT)^2/(RT/Mw rho0)^2 + eta/(RT/Mw rho0) * (2 R/Mw drho0/dT + RT/Mw d2rho0/dT2)
			//				+ 4 ddB/dT drho0/dT + 2 dB d2rho/dT2 + 2 rho0 d2dB/dT2
			double dB = ziabakhsh::tau[species[i]] + ziabakhsh::Gamma[species[i]] * p + ziabakhsh::beta[species[i]] * std::sqrt(1000.) / std::sqrt(T);
			double dBdT = ziabakhsh::beta[species[i]] * std::sqrt(1000.) * -0.5 * std::pow(T, -1.5);
			double d2BdT2 = ziabakhsh::beta[species[i]] * std::sqrt(1000.) * 0.75 * std::pow(T, -2.5);

			double d2lnkH_dT2 = (1 - ziabakhsh::eta[species[i]]) * (-std::pow(df0_H2OdT/f0_H2O, 2) + d2f0_H2OdT2/f0_H2O)
								+ ziabakhsh::eta[species[i]] * (-std::pow(ziabakhsh::R / Mw * (rho0_H2O + T * drho0_H2OdT)/(ziabakhsh::R * T / Mw * rho0_H2O), 2)
																+ ziabakhsh::R / Mw * (2 * drho0_H2OdT + T * d2rho0_H2OdT2)/(ziabakhsh::R * T / Mw * rho0_H2O))
								+ 2. * (d2rho0_H2OdT2 * dB + 2. * drho0_H2OdT * dBdT + rho0_H2O * d2BdT2);
			return d2lnj_dT2 + d2lnkH_dT2;
		}
	}
	else
	{
		// no expression for ions fugacity -> derivative is zero
		return 0.;
	}
}
double Ziabakhsh2012::dlnphii_dxj(int i, int j) 
{
	// Derivative of lnphii with respect to mole fraction xj
	if (i == water_index)  // i = water
	{
		// if (j == water_index)  // j = water
		// {
		// 	return 1./x[water_index];
		// }
		// else  // j = gases and ions
		// {
		// 	return 0.;
		// }
		return 0.;
	}
	else if (i < nc)  // i = gases
	{
		if (ni > 0)
		{
			// Derivative of lnj_i w.r.t. xj
			// Only necessary if ions are present, otherwise lnj reduces to zero
			return 2. * dmcdxj[j] * labda[i] + dmacdxj[j] * ksi[i];

			// if (j == water_index)
			// {
			// 	// j = water
			// 	// ions contribution to derivative - if no ions present, derivative reduces to zero
			// 	// dlnjdxj = -2. * m_c * labda[i] / x[water_index] + 2. * m_ac / std::pow(x[water_index], 2) * ksi[i];
			// 	return 2. * dmcdxj[water_index] * labda[i] + dmacdxj[water_index] * ksi[i];
			// }
			// else if (j < nc)
			// {
			// 	// j = gases
			// 	return 0.;
			// }
			// else
			// {
			// 	// j = ions
			// 	return 2. * dmcdxj[j] * labda[i] + dmacdxj[j] * ksi[i];
			// }
		}
		else
		{
			return 0.;
		}
	}
	else  // i = ions
	{
		// no expression for ions fugacity -> derivative is zero
		return 0.;
	}
}
std::vector<double> Ziabakhsh2012::d2lnphii_dTdxj(int i) 
{
	// Derivative of lnphii with respect to mole fraction xj
	if (i == water_index)  // i = water
	{
		// if (j == water_index)  // j = water
		// {
		// 	return 1./x[water_index];
		// }
		// else  // j = gases and ions
		// {
		// 	return 0.;
		// }
		return std::vector<double>(ns, 0.);
	}
	else if (i < nc)  // i = gases
	{
		if (ni > 0)
		{
			std::vector<double> ai = ziabakhsh::labda[species[i]];
			std::vector<double> bi = ziabakhsh::ksi[species[i]];
			double dlabda_dT = ai[1] - ai[2] / std::pow(T, 2) - ai[5] * p / std::pow(T, 2) + ai[6] / std::pow(p, 2) + ai[7] * p / std::pow(630 - T, 2) + ai[8] * std::log(p) - 2. * ai[9] * p / std::pow(T, 3) + ai[10] * std::pow(p, 2) + ai[11] * p;
			double dksi_dT = bi[1] - bi[2] * p / std::pow(T, 2) + bi[3] * p / std::pow(630.0 - T, 2);

			std::vector<double> d2lnphiidTdxj(ns);
			for (int j = 0; j < ns; j++)
			{
				d2lnphiidTdxj[j] = 2. * dmcdxj[j] * dlabda_dT + dmacdxj[j] * dksi_dT;
			}
			return d2lnphiidTdxj;
		}
		else
		{
			return std::vector<double>(ns, 0.);
		}
	}
	else  // i = ions
	{
		// no expression for ions fugacity -> derivative is zero
		return std::vector<double>(ns, 0.);
	}
}

double Ziabakhsh2012::lnphi0(double X, double T_, bool pt)
{
	// Calculate pure H2O Gibbs energy
	(void) X;
	(void) T_;
	(void) pt;
	return this->lnphii(water_index);
}
