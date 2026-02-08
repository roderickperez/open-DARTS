#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include "dartsflash/stability/stability.hpp"

namespace py = pybind11;

void pybind_stability(py::module& m)
{
    using namespace pybind11::literals;  // bring in '_a' literal
    
    // Expose Stability class
    py::class_<Stability>(m, "Stability", R"pbdoc(
            This is the base class for performing stability tests.
            )pbdoc")
        .def(py::init<FlashParams&>(), R"pbdoc(
            :param flash_params: FlashParams object
            )pbdoc", "flash_params"_a)

        .def("init", &Stability::init, R"pbdoc(
            Initialise stability algorithm at p, T, with reference phase compositions
            
            :param ref_comps: List of reference phases.
            :type ref_comps: list of TrialPhase objects
            )pbdoc", "ref_comps"_a)

        .def("run", &Stability::run, R"pbdoc(
            Run stability test from trial phase composition initial guess Y.
            
            :param trial_comp: Trial phase
            :type trial_comp: TrialPhase object
            :param gmix_min: Switch for finding minimum of Gmix surface
            :type gmix_min: bool
            )pbdoc", "trial_comp"_a, py::arg("gmix_min")=false)
        ;
}