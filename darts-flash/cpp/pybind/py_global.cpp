#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/global/timer.hpp"
#include "dartsflash/global/units.hpp"
#include "dartsflash/maths/root_finding.hpp"

namespace py = pybind11;

void pybind_global(py::module& m)
{
    using namespace pybind11::literals;  // bring in '_a' literal

    py::enum_<StateSpecification>(m, "StateSpecification", "State specification for P-based flash (PT/PH/PS)")
        .value("TEMPERATURE", StateSpecification::TEMPERATURE)
        .value("ENTHALPY", StateSpecification::ENTHALPY)
        .value("ENTROPY", StateSpecification::ENTROPY)
        .export_values()
        ;

    // Expose Timer class
    py::class_<Timer> timer(m, "Timer", R"pbdoc(
            This class contains timers.
            )pbdoc");

    py::enum_<Timer::timer>(timer, "timer", "Timer keys")
        .value("FLASH", Timer::timer::FLASH)
        .value("STABILITY", Timer::timer::STABILITY)
        .value("SPLIT", Timer::timer::SPLIT)
        .value("EOS", Timer::timer::EOS)
        .value("TOTAL", Timer::timer::TOTAL)
        .export_values()
        ;
    
    // Timers
    timer.def("start", &Timer::start, R"pbdoc(
            :param key: Timer key
            :type key: str
            )pbdoc", "key"_a)
        .def("stop", &Timer::stop, R"pbdoc(
            :param key: Timer key
            :type key: str
            )pbdoc", "key"_a)
        .def("print_timers", &Timer::print_timers, R"pbdoc(
            This method prints all tracked timers.
            )pbdoc")
        ;

    // Expose Units class and enums
    py::class_<Units> units(m, "Units", R"pbdoc(
            This class contains units and performs unit conversions.
            )pbdoc");

    py::enum_<Units::PRESSURE>(units, "PRESSURE", "Pressure units")
        .value("BAR", Units::PRESSURE::BAR)
        .value("PA", Units::PRESSURE::PA)
        .value("KPA", Units::PRESSURE::KPA)
        .value("MPA", Units::PRESSURE::MPA)
        .value("ATM", Units::PRESSURE::ATM)
        .value("PSIA", Units::PRESSURE::PSIA)
        .export_values()
        ;

    py::enum_<Units::TEMPERATURE>(units, "TEMPERATURE", "Temperature units")
        .value("KELVIN", Units::TEMPERATURE::KELVIN)
        .value("CELSIUS", Units::TEMPERATURE::CELSIUS)
        .value("FAHRENHEIT", Units::TEMPERATURE::FAHRENHEIT)
        .value("RANKINE", Units::TEMPERATURE::RANKINE)
        .export_values()
        ;

    py::enum_<Units::VOLUME>(units, "VOLUME", "Volume units")
        .value("M3", Units::VOLUME::M3)
        .value("CM3", Units::VOLUME::CM3)
        .value("L", Units::VOLUME::L)
        .value("FT3", Units::VOLUME::FT3)
        .export_values()
        ;

    py::enum_<Units::ENERGY>(units, "ENERGY", "Energy units")
        .value("J", Units::ENERGY::J)
        .value("CAL", Units::ENERGY::CAL)
        .export_values()
        ;

    units.def(py::init<Units::PRESSURE, Units::TEMPERATURE, Units::VOLUME, Units::ENERGY>(), R"pbdoc(
            This is the constructor of Units.

            :param pressure: Pressure unit, default is BAR
            :param temperature: Temperature unit, default is KELVIN
            :param volume: Volume unit, default is M3
            :param energy: Energy unit, default is J
            )pbdoc", py::arg("pressure")=Units::PRESSURE::BAR, py::arg("temperature")=Units::TEMPERATURE::KELVIN,
                     py::arg("volume")=Units::VOLUME::M3, py::arg("energy")=Units::ENERGY::J)
        .def_readonly("R", &Units::R)
        ;
    
    // Expose CompData class
    py::class_<CompData> compdata(m, "CompData", R"pbdoc(
            This class contains component specific data that is required for correlations in EoS and InitialGuess.

            It contains a :class:`Units` object that specifies the units of measurement for each of the properties.
            )pbdoc");
    
    compdata.def(py::init<const std::vector<std::string>&, const std::vector<std::string>&>(), R"pbdoc(
            This is the constructor of CompData

            :param components: List of components
            :type components: list
            :param ions: List of ions
            :type ions: list
            )pbdoc", "components"_a, "ions"_a=std::vector<std::string>{})

        .def_readonly("nc", &CompData::nc, R"pbdoc(Number of components.)pbdoc")
        .def_readonly("ni", &CompData::ni, R"pbdoc(Number of ions.)pbdoc")
        .def_readonly("ns", &CompData::ns, R"pbdoc(Number of species (components+ions).)pbdoc")

        .def_readwrite("units", &CompData::units, R"pbdoc(
            Input/output units
            )pbdoc")

        .def_readwrite("Mw", &CompData::Mw, R"pbdoc(
            List of component molar weight.
            )pbdoc")
        .def_readwrite("cpi", &CompData::cpi, R"pbdoc(
            List of component ideal gas heat capacity coefficients.
            )pbdoc")
        .def_readwrite("T0", &CompData::T_0, R"pbdoc(
            Ideal heat capacity reference temperature.
            )pbdoc")
        
        .def_readwrite("Pc", &CompData::Pc, R"pbdoc(
            List of component critical pressures Pc.
            )pbdoc")
        .def_readwrite("Tc", &CompData::Tc, R"pbdoc(
            List of component critical temperatures Tc.
            )pbdoc")
        .def_readwrite("ac", &CompData::ac, R"pbdoc(
            List of component acentric factors ac.
            )pbdoc")
        
        .def_readwrite("kij", &CompData::kij, R"pbdoc(
            List of lists of binary interaction coefficients.
            )pbdoc")
        .def("set_binary_coefficients", &CompData::set_binary_coefficients, R"pbdoc(
            This is a function to set binary interaction coefficients for component i.

            :param i: Component index
            :type i: int
            :param kij: List of binary interaction coefficients for component i
            :type kij: list
            )pbdoc", "i"_a, "kij"_a)

        .def_readwrite("charge", &CompData::charge, R"pbdoc(
            List of ion charges.
            )pbdoc")
        .def("set_ion_concentration", &CompData::set_ion_concentration, R"pbdoc(
            :param concentrations: List of ion concentrations
            :type concentrations: list
            :param unit: Concentration unit (molality, weight fraction)
            :type unit: CompData::ConcentrationUnit
            )pbdoc", "concentrations"_a, "unit"_a)
        ;

    py::enum_<CompData::ConcentrationUnit>(compdata, "ConcentrationUnit", "Unit for ion concentration")
        .value("molality", CompData::ConcentrationUnit::molality)
        .value("weight", CompData::ConcentrationUnit::weight)
        .export_values()
        ;

    py::class_<RootFinding>(m, "RootFinding", R"pbdoc(
            This class contains methods to find the root of an equation using bisection or Brent's method.
            )pbdoc")
        .def(py::init<>())

        .def("bisection", &RootFinding::bisection)
        .def("brent", &RootFinding::brent)
        .def("getx", &RootFinding::getx)
        ;
}
