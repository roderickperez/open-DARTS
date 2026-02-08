//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_GLOBAL_COMPONENTS_H
#define OPENDARTS_FLASH_GLOBAL_COMPONENTS_H
//--------------------------------------------------------------------------

#include <unordered_map>
#include <vector>
#include <string>
#include "dartsflash/global/units.hpp"

enum Component : int { H2O = 0, CO2, N2, H2S, H2, C1, C2, C3, nC4, iC4, nC5, iC5, nC6, nC7, nC8, nC9, nC10, nCX, Ethene, Propene, Benzene, Toluene, MeOH, EOH, MEG };
enum Ion : int { Na = 0, Ca, K, Cl };

namespace comp_data {
	extern std::unordered_map<std::string, int> charge;
	extern std::unordered_map<std::string, double> Pc, Tc, ac, Z_ra, Mw;  //, H0, dlnH0;
    extern std::unordered_map<std::string, std::vector<double>> cpi;
	extern std::unordered_map<std::string, std::unordered_map<std::string, double>> kij;
}

struct CompData
{
    enum ConcentrationUnit : int { molality = 0, weight };

    int nc, ni, ns, water_index;
    Units units;
    std::vector<std::string> components, ions;
    std::string salt;

    std::vector<double> Pc, Tc, ac, Z_ra, Mw, kij;  // Critical properties and parameters for cubic EoS

    std::vector<std::vector<double>> cpi;  // ideal gas heat capacity coefficients
    double T_0, P_0, V_0;  // reference conditions for ideal gas properties

    std::vector<double> H0, dlnH0;  // Henry's law parameters
    std::vector<int> charge;  // charge of ions
    std::vector<double> m_i;  // molality of ions
    bool constant_salinity = false;

    CompData() { }
    CompData(const std::vector<std::string>& components_, const std::vector<std::string>& ions_={});
	~CompData() = default;

    void set_units(Units& units_) { this->units = units_; }
    void set_binary_coefficients(int i, std::vector<double> kij_);
    void set_ion_concentration(std::vector<double> concentrations, CompData::ConcentrationUnit unit = CompData::ConcentrationUnit::molality);
    double get_molar_weight(std::vector<double>& n);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_GLOBAL_COMPONENTS_H
//--------------------------------------------------------------------------
