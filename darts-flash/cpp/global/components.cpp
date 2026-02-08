#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/global/units.hpp"

namespace comp_data {
    std::unordered_map<std::string, int> charge = {{"Na+", 1}, {"K+", 1}, {"Ca2+", 2}, {"Mg2+", 2}, {"Cl-", -1}};

    std::unordered_map<std::string, double> Tc = {
        {"H2O", 647.14}, {"CO2", 304.10}, {"N2", 126.20}, {"H2S", 373.53}, {"H2", 33.145},
        {"C1", 190.58}, {"C2", 305.32}, {"C3", 369.83}, {"iC4", 407.85}, {"nC4", 425.12}, {"iC5", 460.45}, {"nC5", 469.70}, 
        {"nC6", 507.60}, {"nC7", 540.20}, {"nC8", 569.32}, {"nC9", 594.6}, {"nC10", 617.7}, 
        {"Ethene", 282.34}, {"Propene", 365.57}, {"Benzene", 562.16}, {"Toluene", 591.80}, {"MeOH", 512.58}, {"EOH", 516.25}, {"MEG", 645.00},
    };
    std::unordered_map<std::string, double> Pc = {
        {"H2O", 220.50}, {"CO2", 73.75}, {"N2", 34.00}, {"H2S", 89.63}, {"H2", 12.93},
        {"C1", 46.04}, {"C2", 48.721}, {"C3", 42.481}, {"iC4", 36.40}, {"nC4", 37.960}, {"iC5", 33.77}, {"nC5", 33.701}, 
        {"nC6", 30.251}, {"nC7", 27.40}, {"nC8", 24.97}, {"nC9", 22.88}, {"nC10", 21.2}, 
        {"Ethene", 50.401}, {"Propene", 46.650}, {"Benzene", 48.981}, {"Toluene", 41.060}, {"MeOH", 80.959}, {"EOH", 63.835}, {"MEG", 75.300},
    };
    std::unordered_map<std::string, double> ac = {
        {"H2O", 0.328}, {"CO2", 0.239}, {"N2", 0.0377}, {"H2S", 0.0942}, {"H2", -0.219},
        {"C1", 0.012}, {"C2", 0.0995}, {"C3", 0.1523}, {"iC4", 0.1844}, {"nC4", 0.2002}, {"iC5", 0.227}, {"nC5", 0.2515}, 
        {"nC6", 0.3013}, {"nC7", 0.3495}, {"nC8", 0.396}, {"nC9", 0.445}, {"nC10", 0.489}, 
        {"Ethene", 0.0865}, {"Propene", 0.1398}, {"Benzene", 0.2100}, {"Toluene", 0.2621}, {"MeOH", 0.5656}, {"EOH", 0.6371}, {"MEG", 0.2300},
    };
    std::unordered_map<std::string, double> Z_ra = {  // https://ars.els-cdn.com/content/image/1-s2.0-S037838121630437X-mmc1.pdf
        {"H2O", 0.2307}, {"CO2", 0.2719}, {"N2", 0.2902}, {"H2S", 0.2818}, {"H2", 0.3192},
        {"C1", 0.2893}, {"C2", 0.2814}, {"C3", 0.2770}, {"iC4", 0.2745}, {"nC4", 0.2731}, {"iC5", 0.2715}, {"nC5", 0.2681}, 
        {"nC6", 0.2638}, {"nC7", 0.2605}, {"nC8", 0.2569}, {"nC9", 0.2544}, {"nC10", 0.2517}, 
        {"Ethene", 0.2820}, {"Propene", 0.2776}, {"Benzene", 0.2697}, {"Toluene", 0.2643}, {"MeOH", 0.2313}, {"EOH", 0.2430}, {"MEG", 0.2126},
    };
    std::unordered_map<std::string, double> Mw = {
        {"H2O", 18.015}, {"CO2", 44.01}, {"N2", 28.013}, {"H2S", 34.10}, {"H2", 2.016},
        {"C1", 16.043}, {"C2", 30.07}, {"C3", 44.097}, {"iC4", 58.124}, {"nC4", 58.124}, {"iC5", 72.151}, {"nC5", 72.151}, 
        {"nC6", 86.178}, {"nC7", 100.205}, {"nC8", 114.231}, {"nC9", 128.257}, {"nC10", 142.2848},
        {"Ethene", 28.054}, {"Propene", 42.081}, {"Benzene", 78.114}, {"Toluene", 92.141}, {"MeOH", 32.040}, {"EOH", 46.070}, {"MEG", 62.068},
        {"Na+", 22.99}, {"Ca2+", 40.078}, {"K+", 39.098}, {"Mg2+", 24.305}, {"Cl-", 35.453}, {"NaCl", 58.443} 
    };

	std::unordered_map<std::string, std::unordered_map<std::string, double>> kij = {
		{"H2O", {{"H2O", 0.}, {"CO2", 0.19014}, {"N2", 0.32547}, {"H2S", 0.105}, {"C1", 0.47893}, {"C2", 0.5975}, {"C3", 0.5612}, {"iC4", 0.508}, {"nC4", 0.5569}, {"iC5", 0.5}, {"nC5", 0.5260}, {"nC6", 0.4969}, {"nC7", 0.4880}, {"nC8", 0.48}, {"nC9", 0.48}, {"nC10", 0.48}, {"H2", 0.}, }},
        {"CO2", {{"H2O", 0.19014}, {"CO2", 0.}, {"N2", -0.02}, {"H2S", 0.120}, {"C1", 0.125}, {"C2", 0.1350}, {"C3", 0.1500}, {"iC4", 0.13}, {"nC4", 0.1336}, {"iC5", 0.13}, {"nC5", 0.1454}, {"nC6", 0.1167}, {"nC7", 0.1209}, {"nC8", 0.1}, {"nC9", 0.1}, {"nC10", 0.1}, {"H2", 0.}, }},
        {"N2",  {{"H2O", 0.32547}, {"CO2", -0.02}, {"N2", 0.}, {"H2S", 0.2}, {"C1", 0.031}, {"C2", 0.042}, {"C3", 0.091}, {"iC4", 0.1}, {"nC4", 0.0596}, {"iC5", 0.1}, {"nC5", 0.0917}, {"nC6", 0.1552}, {"nC7", 0.1206}, {"nC8", 0.1}, {"nC9", 0.1}, {"nC10", 0.1}, {"H2", 0.}, }},
        {"H2S", {{"H2O", 0.105}, {"CO2", 0.120}, {"N2", 0.2}, {"H2S", 0.}, {"C1", 0.1}, {"C2", 0.08}, {"C3", 0.08}, {"iC4", 0.06}, {"nC4", 0.0564}, {"iC5", 0.06}, {"nC5", 0.0655}, {"nC6", 0.0465}, {"nC7", 0.0191}, {"nC8", 0}, {"nC9", 0}, {"nC10", 0.1}, {"H2", 0.}, }},
        {"C1",  {{"H2O", 0.47893}, {"CO2", 0.125}, {"N2", 0.031}, {"H2S", 0.1}, {"C1", 0.}, {"C2", 0.00518}, {"C3", 0.01008}, {"iC4", 0.026717}, {"nC4", 0.0152}, {"iC5", 0.0206}, {"nC5", 0.0193}, {"nC6", 0.0258}, {"nC7", 0.0148}, {"nC8", 0.037}, {"nC9", 0.03966}, {"nC10", 0.048388}, {"H2", -0.1622}, }},
        {"C2",  {{"H2O", 0.5975}, {"CO2", 0.1350}, {"N2", 0.042}, {"H2S", 0.08}, {"C1", 0.00518}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"C3",  {{"H2O", 0.5612}, {"CO2", 0.1500}, {"N2", 0.091}, {"H2S", 0.08}, {"C1", 0.01008}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"iC4", {{"H2O", 0.508}, {"CO2", 0.13}, {"N2", 0.1}, {"H2S", 0.06}, {"C1", 0.026717},{"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC4", {{"H2O", 0.5569}, {"CO2", 0.1336}, {"N2", 0.0596}, {"H2S", 0.0564}, {"C1", 0.0152}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"iC5", {{"H2O", 0.5}, {"CO2", 0.13}, {"N2", 0.1}, {"H2S", 0.06}, {"C1", 0.0206}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC5", {{"H2O", 0.5260}, {"CO2", 0.1454}, {"N2", 0.0917}, {"H2S", 0.0655}, {"C1", 0.0193}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC6", {{"H2O", 0.4969}, {"CO2", 0.1167}, {"N2", 0.1552}, {"H2S", 0.0465}, {"C1", 0.0258}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC7", {{"H2O", 0.4880}, {"CO2", 0.1209}, {"N2", 0.1206}, {"H2S", 0.0191}, {"C1", 0.0148}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC8", {{"H2O", 0.48}, {"CO2", 0.1}, {"N2", 0.1}, {"H2S", 0}, {"C1", 0.037}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC9", {{"H2O", 0.48}, {"CO2", 0.1}, {"N2", 0.1}, {"H2S", 0}, {"C1", 0.03966}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"nC10", {{"H2O", 0.48}, {"CO2", 0.1}, {"N2", 0.1}, {"H2S", 0.1}, {"C1", 0.048388}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"nC5", 0.}, {"iC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
        {"H2", {{"H2O", 0.}, {"CO2", 0.}, {"N2", 0.}, {"H2S", 0.}, {"C1", -0.1622}, {"C2", 0.}, {"C3", 0.}, {"iC4", 0.}, {"nC4", 0.}, {"iC5", 0.5}, {"nC5", 0.}, {"nC6", 0.}, {"nC7", 0.}, {"nC8", 0.}, {"nC9", 0.}, {"nC10", 0.}, {"H2", 0.}, }},
	};

    // ideal gas heat capacity parameters [eq. 3.4]
    std::unordered_map<std::string, std::vector<double>> cpi =  {
        {"H2O", {3.8747, 0.0231E-2, 0.1269E-5, -0.4321E-9}},
        {"CO2", {2.6751, 0.7188E-2, -0.4208E-5, 0.8977E-9}},
        {"N2", {3.4736, -0.0189E-2, 0.0971E-5, -0.3453E-9}},
        {"H2S", {3.5577, 0.1574E-2, 0.0686E-5, -0.3959E-9}},
        {"H2", {0., 0., 0., 0.}},
        {"C1", {2.3902, 0.6039E-2, 0.1525E-5, -1.3234E-9}},
        {"C2", {0.8293, 2.0752E-2, -0.7699E-5, 0.8756E-9}},
        {"C3", {-0.4861, 3.6629E-2, -1.8895E-5, 3.8143E-9}},
        {"C4", {1.1410, 3.9846E-2, -1.3326E-5, -3.3940E-10}},
        {"iC4", {-0.9511, 4.9999E-2, -2.7651E-5, 5.9982E-9}},
        {"nC4", {0.4755, 4.4650E-2, -2.2041E-5, 4.2068E-9}},
        {"iC5", {-1.9942, 6.6725E-2, -3.9738E-5, 9.1735E-9}},
        {"nC5", {0.8142, 5.4598E-2, -2.6997E-5, 5.0824E-9}},
        {"nC6", {0.8338, 6.6373E-2, -3.444E-5, 6.9342E-9}},
        {"nC7", {-0.6184, 8.1268E-2, -4.388E-5, 9.2037E-9}},
        {"nC8", {-0.7327, 9.2691E-2, -5.0421E-5, 10.6429E-9}},
        {"nC9", {0.3779, 8.1419E-2, -2.3178E-5, -3.5833E-9}},
        {"nC10", {-0.9511, 11.5490E-2, -6.3555E-5, 13.5917E-9}},
        {"MeOH", {2.2896, 1.1000E-2, -0.1464e-5, -0.9662e-9}},
        {"EOH", {2.3902, 2.5191e-2, -1.2475e-5, 2.4104e-9}},
        {"MEG", {4.2904, 2.9845e-2, -1.7995e-5, 3.6181e-9}},
        {"Ethene", {0.4750, 1.8795e-2, -1.0029e-5, 2.1235e-9}}, 
        {"Propene", {0.3789, 2.8638e-2, -1.4643e-5, 2.9589e-9}}, 
        {"Benzene", {-4.3528, 5.8261e-2, -3.7942e-5, 9.3295e-9}}, 
        {"Toluene", {-4.1328, 6.7214e-2, -4.1414e-5, 9.6616e-9}},
        {"NaCl", {5.526, 0.1963e-2, 0., 0.}},
        {"CaCl2", {8.646, 0.153, 0., 0.}},
        {"KCl", {6.17, 0., 0., 0.}},
    };

}

CompData::CompData(const std::vector<std::string>& components_, const std::vector<std::string>& ions_)
{
    this->components = components_;
    this->ions = ions_;
    this->salt = (ions.size() > 0) ? ((ions[0] == "Na+") ? "NaCl" : "CaCl2") : "";
    this->nc = static_cast<int>(components.size());
    this->ni = static_cast<int>(ions.size());
    this->ns = nc + ni;
    this->units = Units();
    this->water_index = std::distance(components.begin(), std::find(components.begin(), components.end(), "H2O"));
    this->T_0 = 298.15;
    this->P_0 = 1.;
    this->V_0 = this->units.R * this->T_0 / this->P_0;

    // Reserve memory for vectors
    this->Pc = std::vector<double>(nc);
    this->Tc = std::vector<double>(nc);
    this->ac = std::vector<double>(nc);
    this->Z_ra = std::vector<double>(nc);
    this->Mw = std::vector<double>(ns);
    this->kij = std::vector<double>(nc*nc, 0.);

    this->H0.resize(nc);
    this->dlnH0.resize(nc);

    this->charge = std::vector<int>(ni);
    this->m_i.resize(ni);

    // Heat capacities
    this->cpi.resize(ns, std::vector<double>(4));
    for (int i = 0; i < nc; i++)
    {
        Pc[i] = comp_data::Pc[components[i]];
        Tc[i] = comp_data::Tc[components[i]];
        ac[i] = comp_data::ac[components[i]];
        Z_ra[i] = comp_data::Z_ra[components[i]];
        Mw[i] = comp_data::Mw[components[i]];
        cpi[i] = comp_data::cpi[components[i]];
    }
    for (int i = 0; i < ni; i++)
    {
        Mw[i+nc] = comp_data::Mw[ions[i]];
        cpi[i+nc] = comp_data::cpi["NaCl"];
        charge[i] = comp_data::charge[ions[i]];
    }
}

void CompData::set_binary_coefficients(int i, std::vector<double> kij_)
{
    for (int j = 0; j < nc; j++)
    {
        kij[i*nc + j] = kij_[j];
        kij[j*nc + i] = kij_[j];
    }
    return;
}

void CompData::set_ion_concentration(std::vector<double> concentrations, CompData::ConcentrationUnit unit)
{
	this->constant_salinity = true;
	if (unit == CompData::ConcentrationUnit::molality)
	{
		for (int ii = 0; ii < ni; ii++)
		{
			m_i[ii] = concentrations[ii];
		}
	}
	else  // if (unit == AQEoS::ConcentrationUnit::weight)
	{
		for (int ii = 0; ii < ni; ii++)
		{
			m_i[ii] = concentrations[ii];
		}
	}
	return;
}

double CompData::get_molar_weight(std::vector<double>& n)
{
    return std::inner_product(n.begin(), n.end(), this->Mw.begin(), 0.);
}
