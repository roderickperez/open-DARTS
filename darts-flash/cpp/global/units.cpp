#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/units.hpp"

namespace unit_conversion {
    std::unordered_map<Units::PRESSURE, double> input_to_bar = {
        {Units::PRESSURE::BAR, 1.},
        {Units::PRESSURE::PA, 1.e-5},
        {Units::PRESSURE::KPA, 1.e-2},
        {Units::PRESSURE::MPA, 1.e1},
        {Units::PRESSURE::ATM, 1.01325},
        {Units::PRESSURE::PSIA, 14.503773800722}
    };
    std::unordered_map<Units::PRESSURE, double> bar_to_output = {
        {Units::PRESSURE::BAR, 1.},
        {Units::PRESSURE::PA, 1.e5},
        {Units::PRESSURE::KPA, 1.e2},
        {Units::PRESSURE::MPA, 1.e-1},
        {Units::PRESSURE::ATM, 1./1.01325},
        {Units::PRESSURE::PSIA, 1./14.503773800722}
    };
    std::unordered_map<Units::PRESSURE, double> R_to_pressure = {
        {Units::PRESSURE::BAR, 1.e-5},
        {Units::PRESSURE::PA, 1.},
        {Units::PRESSURE::KPA, 1.e-3},
        {Units::PRESSURE::MPA, 1.e-6},
        {Units::PRESSURE::ATM, 1.e-5/1.01325},
        {Units::PRESSURE::PSIA, 1.e-5/14.503773800722}
    };

    std::unordered_map<Units::TEMPERATURE, std::pair<double, double>> input_to_kelvin = {
        {Units::TEMPERATURE::KELVIN, {1., 1.}},
        {Units::TEMPERATURE::CELSIUS, {1., -273.15}},
        {Units::TEMPERATURE::FAHRENHEIT, {1./1.8, 459.67}},
        {Units::TEMPERATURE::RANKINE, {1./1.8, 0.}}
    };
    std::unordered_map<Units::TEMPERATURE, std::pair<double, double>> kelvin_to_output = {
        {Units::TEMPERATURE::KELVIN, {1., 1.}},
        {Units::TEMPERATURE::CELSIUS, {1., 273.15}},
        {Units::TEMPERATURE::FAHRENHEIT, {1.8, -459.67}},
        {Units::TEMPERATURE::RANKINE, {1.8, 0.}}
    };
    std::unordered_map<Units::TEMPERATURE, double> R_to_temperature = {
        {Units::TEMPERATURE::KELVIN, 1.},
        {Units::TEMPERATURE::CELSIUS, 1.},
        {Units::TEMPERATURE::FAHRENHEIT, 1./1.8},
        {Units::TEMPERATURE::RANKINE, 1./1.8}
    };

    std::unordered_map<Units::VOLUME, double> input_to_m3 = {
        {Units::VOLUME::M3, 1.},
        {Units::VOLUME::CM3, 1.e-6},
        {Units::VOLUME::L, 1.e-3},
        {Units::VOLUME::FT3, 1./std::pow(0.3048, 3)}
    };
    std::unordered_map<Units::VOLUME, double> m3_to_output = {
        {Units::VOLUME::M3, 1.},
        {Units::VOLUME::CM3, 1.e6},
        {Units::VOLUME::L, 1.e3},
        {Units::VOLUME::FT3, std::pow(0.3048, 3)}
    };
    std::unordered_map<Units::VOLUME, double> R_to_volume = {
        {Units::VOLUME::M3, 1.},
        {Units::VOLUME::CM3, 1.e6},
        {Units::VOLUME::L, 1.e3},
        {Units::VOLUME::FT3, std::pow(0.3048, 3)}
    };

    std::unordered_map<Units::ENERGY, double> input_to_J = {
        {Units::ENERGY::J, 1.},
        {Units::ENERGY::CAL, 1./0.239005736},
    };
    std::unordered_map<Units::ENERGY, double> J_to_output = {
        {Units::ENERGY::J, 1.},
        {Units::ENERGY::CAL, 0.239005736},
    };
    std::unordered_map<Units::ENERGY, double> R_to_energy = {
        {Units::ENERGY::J, 1.},
        {Units::ENERGY::CAL, 0.239005736},
    };
}

Units::Units() : Units(PRESSURE::BAR, TEMPERATURE::KELVIN, VOLUME::M3, ENERGY::J) {}

Units::Units(PRESSURE p, TEMPERATURE t, VOLUME v, ENERGY e)
{
    // Gas constant R
    R = M_R;  // 8.31446261815324 J/K.mol or m3.Pa/K.mol

    // Pressure input and output
    pressure = p;
    pressure_in = unit_conversion::input_to_bar[pressure];
    pressure_out = unit_conversion::bar_to_output[pressure];
    R *= unit_conversion::R_to_pressure[pressure];

    // Temperature input and output
    temperature = t;
    temperature_in = unit_conversion::input_to_kelvin[temperature];
    temperature_out = unit_conversion::kelvin_to_output[temperature];
    R *= unit_conversion::R_to_temperature[temperature];

    // Volume input and output
    volume = v;
    volume_in = unit_conversion::input_to_m3[volume];
    volume_out = unit_conversion::m3_to_output[volume];
    R *= unit_conversion::R_to_volume[volume];

    // Energy input and output
    energy = e;
    energy_in = unit_conversion::input_to_J[energy];
    energy_out = unit_conversion::J_to_output[energy];
    R *= unit_conversion::R_to_energy[energy];

    // Calculate 1/R, used often
    R_inv = 1./R;
}

double Units::input_to_bar(double p)
{
    // Translate from input unit to bar
    return p * pressure_in;
}

double Units::bar_to_output(double p)
{
    // Translate from bar to output unit
    return p * pressure_out;
}

double Units::input_to_kelvin(double t)
{
    // Translate from input unit to Kelvin
    return (t + temperature_in.first) * temperature_in.second;
}

double Units::kelvin_to_output(double t)
{
    // Translate from Kelvin to output unit
    return t * temperature_out.first + temperature_out.second;
}

double Units::input_to_m3(double v)
{
    // Translate from input unit to m3
    return v * volume_in;
}

double Units::m3_to_output(double v)
{
    // Translate from m3 to output unit
    return v * volume_out;
}

double Units::input_to_J(double e)
{
    // Translate from input unit to J
    return e * energy_in;
}

double Units::J_to_output(double e)
{
    // Translate from J to output unit
    return e * energy_out;
}