//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_GLOBAL_UNITS_H
#define OPENDARTS_FLASH_GLOBAL_UNITS_H
//--------------------------------------------------------------------------

#include <unordered_map>

struct Units
{
    enum class PRESSURE { BAR, PA, KPA, MPA, ATM, PSIA } pressure;
    enum class TEMPERATURE { KELVIN, CELSIUS, FAHRENHEIT, RANKINE } temperature;
    enum class VOLUME { M3, CM3, L, FT3 } volume;
    enum class ENERGY { J, CAL } energy;

    double pressure_in, pressure_out;
    std::pair<double, double> temperature_in, temperature_out;
    double volume_in, volume_out;
    double energy_in, energy_out;
    double R, R_inv;

    Units();
    Units(PRESSURE p, TEMPERATURE t, VOLUME v, ENERGY e);

    double input_to_bar(double p);
    double bar_to_output(double p);

    double input_to_kelvin(double t);
    double kelvin_to_output(double t);

    double input_to_m3(double v);
    double m3_to_output(double v);

    double input_to_J(double e);
    double J_to_output(double e);
};

namespace unit_conversion {
    extern std::unordered_map<Units::PRESSURE, double> input_to_bar, bar_to_output, R_to_pressure;
    extern std::unordered_map<Units::TEMPERATURE, std::pair<double, double>> input_to_kelvin, kelvin_to_output;
    extern std::unordered_map<Units::TEMPERATURE, double> R_to_temperature;
    extern std::unordered_map<Units::VOLUME, double> input_to_m3, m3_to_output, R_to_volume;
    extern std::unordered_map<Units::ENERGY, double> input_to_J, J_to_output, R_to_energy;
}

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_GLOBAL_UNITS_H
//--------------------------------------------------------------------------
