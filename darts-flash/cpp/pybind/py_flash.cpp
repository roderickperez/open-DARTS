// #ifdef PYBIND11_ENABLED
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <memory>

#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/negative_flash.hpp"
#include "dartsflash/flash/px_flash.hpp"
#include "dartsflash/flash/initial_guess.hpp"
#include "dartsflash/flash/trial_phase.hpp"

namespace py = pybind11;

template <class FlashBase = Flash> class PyFlash : public FlashBase {
public:
    /* Inherit the constructors */
    using FlashBase::FlashBase;

    /* Trampoline (need one for each virtual function) */
    int evaluate(double p_, double T_, std::vector<double>& z_, bool start_from_feed = true) override {
        PYBIND11_OVERRIDE(
            int,  /* Return type */
            Flash, /* Parent class */
            evaluate,   /* Name of function in C++ (must match Python name) */
            p_, T_, z_, start_from_feed    /* Argument(s) */
        );
    }
    int evaluate(double p_, double T_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results) override {
        PYBIND11_OVERRIDE(
            int, Flash, evaluate, p_, T_, z_, flash_results
        );
    }
    int evaluate(double p_, double T_) override {
        PYBIND11_OVERRIDE(
            int, Flash, evaluate, p_, T_
        );
    }
};

template <class FlashResultsBase = FlashResults> class PyFlashResults : public FlashResultsBase, public py::trampoline_self_life_support {
public:
    /* Inherit the constructors */
    using FlashResultsBase::FlashResultsBase;

    void get_derivs(std::vector<double>& dnudP_, std::vector<double>& dnudT_, std::vector<double>& dnudzk_,
                    std::vector<double>& dxdP_, std::vector<double>& dxdT_, std::vector<double>& dxdzk_) override {
        PYBIND11_OVERRIDE(
            void, FlashResults, get_derivs, dnudP_, dnudT_, dnudzk_, dxdP_, dxdT_, dxdzk_
        );
    }

    void print_results(bool derivs = false) override {
        PYBIND11_OVERRIDE(
            void, FlashResults, print_results, derivs
        );
    }
    void print_derivs() override {
        PYBIND11_OVERRIDE(
            void, FlashResults, print_derivs
        );
    }
};

void pybind_flash(py::module& m) 
{
    using namespace pybind11::literals;  // bring in '_a' literal

    // Helper aliases and copy routines to write directly into provided numpy arrays
    using ArrayD = py::array_t<double, py::array::c_style | py::array::forcecast>;
    auto copy_vec_to_array = [](const std::vector<double>& src, ArrayD& dst, const char* name)
    {
        auto buf = dst.request();
        if (static_cast<size_t>(buf.itemsize) != sizeof(double))
        {
            throw py::type_error(std::string("Expected dtype float64 for ") + name);
        }

        size_t total = 1;
        for (auto s: buf.shape) { total *= static_cast<size_t>(s); }
        if (total != src.size())
        {
            throw py::value_error(std::string("Size mismatch for ") + name + ": expected "
                                  + std::to_string(src.size()) + ", got " + std::to_string(total));
        }

        double* ptr = static_cast<double*>(buf.ptr);
        std::copy(src.begin(), src.end(), ptr);
    };
    auto copy_scalar_to_array = [](double value, ArrayD& dst, const char* name)
    {
        auto buf = dst.request();
        if (static_cast<size_t>(buf.itemsize) != sizeof(double))
        {
            throw py::type_error(std::string("Expected dtype float64 for ") + name);
        }

        size_t total = 1;
        for (auto s: buf.shape) { total *= static_cast<size_t>(s); }
        if (total != 1)
        {
            throw py::value_error(std::string("Size mismatch for ") + name + ": expected 1, got "
                                  + std::to_string(total));
        }

        double* ptr = static_cast<double*>(buf.ptr);
        ptr[0] = value;
    };

    py::class_<Flash, PyFlash<>> flash(m, "Flash", R"pbdoc(
            This is a base class for Flash.

            Each Flash child class overrides the methods for `evaluate(p, T, z)`.
            )pbdoc");
    
    flash.def(py::init<FlashParams&>(), R"pbdoc(
            This is the constructor of the Flash base class.

            :param flashparams: Flash parameters object
            :type flashparams: FlashParams
            )pbdoc", py::arg("flashparams"))
        
        .def("evaluate", py::overload_cast<double, double>(&Flash::evaluate), R"pbdoc(
            Evaluate single-component equilibrium at (P, T)

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "T"_a)
        .def("evaluate", py::overload_cast<double, double, std::vector<double>&, bool>(&Flash::evaluate), R"pbdoc(
            Evaluate multicomponent flash at (P, T, z)

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param z: Feed composition
            :type z: list
            :param start_from_feed: Bool to indicate whether to start from feed or previous results
            :type start_from_feed: bool
            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "T"_a, "z"_a, py::arg("start_from_feed")=true)
        .def("evaluate", py::overload_cast<double, double, std::vector<double>&, std::shared_ptr<FlashResults>>(&Flash::evaluate), R"pbdoc(
            Evaluate multicomponent flash at (P, T, z) starting from (extrapolated) previous results

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param z: Feed composition
            :type z: list
            :param flash_results: FlashResults object of (extrapolated) previous results
            :type flash_results: FlashResults
            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "T"_a, "z"_a, "flash_results"_a)
        
        .def("get_flash_results", &Flash::get_flash_results, R"pbdoc(
            :returns: Results of flash
            :rtype: FlashResults
            )pbdoc", py::arg("derivs")=false)
        .def("extrapolate_flash_results", &Flash::extrapolate_flash_results, R"pbdoc(
            :param p: New pressure
            :type p: double
            :param T: New temperature
            :type T: double
            :param z: New feed composition
            :type z: list
            :param flash_results: FlashResults object of previous results
            :type flash_results: FlashResults

            :returns: Extrapolated FlashResults
            :rtype: FlashResults
            )pbdoc", "p"_a, "T"_a, "z"_a, "flash_results"_a)
        // using array = py::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>;

        .def("find_stationary_points", &Flash::find_stationary_points, R"pbdoc(
            Determine stationary points of TPD function at (multiphase) composition X

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param X: List of reference compositions
            :type X: list
            
            :returns: Set of stationary points
            :rtype: list
            )pbdoc", "p"_a, "T"_a, "X"_a)
        ;

    py::class_<FlashResults, PyFlashResults<>, py::smart_holder>(m, "FlashResults", R"pbdoc(
            This is a struct containing PT-flash results and derivatives.
            )pbdoc")
        // Flash results
        .def_readonly("pressure", &FlashResults::pressure)
        .def_readonly("temperature", &FlashResults::temperature)
        .def_readonly("np", &FlashResults::np)
        .def_readonly("phase_state_id", &FlashResults::phase_state_id)
        .def_readonly("nu", &FlashResults::nuj)
        .def_readonly("X", &FlashResults::Xij)
        .def_readonly("eos_idx", &FlashResults::eos_idx)
        .def_readonly("root_type", &FlashResults::root_type)
        
        // Get partial derivatives
        .def("get_derivs",
             [copy_vec_to_array](FlashResults& self,
                                 ArrayD dnudP, ArrayD dnudT, ArrayD dnudzk,
                                 ArrayD dxdP, ArrayD dxdT, ArrayD dxdzk)
             {
                 std::vector<double> v_dnudP, v_dnudT, v_dnudzk, v_dxdP, v_dxdT, v_dxdzk;
                 self.get_derivs(v_dnudP, v_dnudT, v_dnudzk, v_dxdP, v_dxdT, v_dxdzk);

                 copy_vec_to_array(v_dnudP, dnudP, "dnudP");
                 copy_vec_to_array(v_dnudT, dnudT, "dnudT");
                 copy_vec_to_array(v_dnudzk, dnudzk, "dnudzk");
                 copy_vec_to_array(v_dxdP, dxdP, "dxdP");
                 copy_vec_to_array(v_dxdT, dxdT, "dxdT");
                 copy_vec_to_array(v_dxdzk, dxdzk, "dxdzk");
             },
             R"pbdoc(
            Get partial derivatives of PT-flash and write them into provided numpy arrays.

            All destination arrays must be writable, C-contiguous, and sized exactly to the
            returned data:
              - dnudP, dnudT: shape (np_tot,)
              - dnudzk:       shape (np_tot * nc,)
              - dxdP, dxdT:   shape (np_tot * nc,)
              - dxdzk:        shape (np_tot * nc * nc)
            )pbdoc",
             "dnudP"_a, "dnudT"_a, "dnudzk"_a, "dxdP"_a, "dxdT"_a, "dxdzk"_a)

        // Calculate phase and total properties
        .def("phase_prop", py::overload_cast<EoS::Property>(&FlashResults::phase_prop), "Method to calculate phase molar property for all phases")
        .def("total_prop", &FlashResults::total_prop, "Method to calculate total thermodynamic property at equilibrium state")

        // Print results and derivatives
        .def("print_results", &FlashResults::print_results, "Print flash results and optional derivatives", py::arg("derivs")=false)
        .def("print_derivs", &FlashResults::print_derivs, "Print flash derivatives")
        ;

    py::class_<PXFlashResults, PyFlashResults<PXFlashResults>, FlashResults, py::smart_holder>(m, "PXFlashResults", R"pbdoc(
            This is a struct containing PX-flash results and derivatives.
            )pbdoc")

        // Get access to PT-flash objects
        .def("get_pt_results", &PXFlashResults::get_pt_results, R"pbdoc(
            Get access to FlashResults object of PT-flash

            :param is_a: Switch to get FlashResults at a or c
            :type is_a: bool
            )pbdoc", "is_a"_a)

        // Get partial derivatives of temperature w.r.t. P, X, z
        .def("get_dT_derivs",
             [copy_vec_to_array, copy_scalar_to_array](PXFlashResults& self,
                                                       ArrayD dTdP, ArrayD dTdX, ArrayD dTdzk)
             {
                 double v_dTdP = 0., v_dTdX = 0.;
                 std::vector<double> v_dTdzk;
                 self.get_dT_derivs(v_dTdP, v_dTdX, v_dTdzk);

                 copy_scalar_to_array(v_dTdP, dTdP, "dTdP");
                 copy_scalar_to_array(v_dTdX, dTdX, "dTdX");
                 copy_vec_to_array(v_dTdzk, dTdzk, "dTdzk");
             },
             R"pbdoc(
            Get partial derivatives of temperature w.r.t. P, X, z and write them into provided numpy arrays.

            Expected destination shapes:
              - dTdP, dTdX: shape (1,)
              - dTdzk:      shape (nc,)
            )pbdoc",
             "dTdP"_a, "dTdX"_a, "dTdzk"_a)
        ;

    py::class_<NegativeFlash, PyFlash<NegativeFlash>, Flash>(m, "NegativeFlash", R"pbdoc(
            This is the two-phase implementation of negative flash.

            It evaluates two-phase negative flash and returns single-phase composition if either phase fraction is negative.
            )pbdoc")
        .def(py::init<FlashParams&, const std::vector<std::string>&, const std::vector<int>&>(), R"pbdoc(
            This is the constructor of the NegativeFlash class.

            :param flashparams: Flash parameters object
            :type flashparams: FlashParams
            :param eos_used: List of EoS names
            :type eos_used: list
            :param initial_guesses: List of initial guesses for K-values
            :type initial_guesses: list
            )pbdoc", "flashparams"_a, "eos_used"_a, "initial_guesses"_a)
        ;

    py::class_<PXFlash, PyFlash<PXFlash>, Flash>(m, "PXFlash", R"pbdoc(
            This is the implementation of P-based flash (PH/PS).

            )pbdoc")

        .def(py::init<FlashParams&, StateSpecification>(), R"pbdoc(
            This is the constructor of the PXFlash class.

            :param flashparams: Flash parameters object
            :type flashparams: FlashParams
            :param state_spec: State specification for P-based flash (ENTHALPY/ENTROPY)
            :type state_spec: StateSpecification
            )pbdoc", "flashparams"_a, "state_spec"_a)
        
        .def("evaluate", py::overload_cast<double, double>(&PXFlash::evaluate), R"pbdoc(
            Evaluate single-component equilibrium at (P, X)

            :param p: Pressure
            :type p: double
            :param X: Enthalpy/entropy
            :type X: double
            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "X"_a)
        .def("evaluate", py::overload_cast<double, double, std::vector<double>&, bool>(&PXFlash::evaluate), R"pbdoc(
            Evaluate multicomponent flash at (P, X, z)

            :param p: Pressure
            :type p: double
            :param X: Enthalpy/entropy
            :type X: double
            :param z: Feed composition
            :type z: list
            :param start_from_feed: Bool to indicate whether to start from feed or previous results and temperature
            :type start_from_feed: bool

            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "X"_a, "z"_a, py::arg("start_from_feed")=true)
        .def("evaluate", py::overload_cast<double, double, std::vector<double>&, std::shared_ptr<FlashResults>>(&PXFlash::evaluate), R"pbdoc(
            Evaluate multicomponent flash at (P, X, z) starting from (extrapolated) previous results

            :param p: Pressure
            :type p: double
            :param X: Enthalpy/entropy
            :type X: double
            :param z: Feed composition
            :type z: list
            :param flash_results: FlashResults object of (extrapolated) previous results
            :type flash_results: FlashResults

            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "X"_a, "z"_a, "flash_results"_a)

        .def("evaluate_PT", py::overload_cast<double, double>(&PXFlash::evaluate_PT), R"pbdoc(
            Evaluate single-component equilibrium at (P, T)

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "T"_a)
        .def("evaluate_PT", py::overload_cast<double, double, std::vector<double>&>(&PXFlash::evaluate_PT), R"pbdoc(
            Evaluate multicomponent flash at (P, T, z)

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param z: Feed composition
            :type z: list
            :returns: Error output of flash procedure
            :rtype: int
            )pbdoc", "p"_a, "T"_a, "z"_a)

        .def("get_pt_flash_results", &PXFlash::get_pt_flash_results, R"pbdoc(
            :returns: Results of PT-flash
            :rtype: FlashResults
            )pbdoc", py::arg("derivs")=false)
        .def("get_flash_results", &PXFlash::get_flash_results, R"pbdoc(
            :returns: Results of PX-flash
            :rtype: PXFlashResults
            )pbdoc", py::arg("derivs")=false)
        .def("extrapolate_flash_results", &PXFlash::extrapolate_flash_results, R"pbdoc(
            :param p: New pressure
            :type p: double
            :param T: New temperature
            :type T: double
            :param z: New feed composition
            :type z: list
            :param flash_results: PXFlashResults object of previous results
            :type flash_results: PXFlashResults

            :returns: Extrapolated PXFlashResults
            :rtype: PXFlashResults
            )pbdoc", "p"_a, "T"_a, "z"_a, "flash_results"_a)
        ;

    py::class_<FlashParams> flash_params(m, "FlashParams", R"pbdoc(
            This is a class that contains all required parameters that can be passed to a flash algorithm.
            )pbdoc");

    flash_params.def(py::init<CompData&>(), R"pbdoc(
            :param comp_data: Component data object
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)

        .def_readwrite("timer", &FlashParams::timer, "Timer object")
        .def_readwrite("units", &FlashParams::units, "Units object")

        // Flash-related parameters
        .def_readwrite("min_z", &FlashParams::min_z, "Minimum value for composition")
        .def_readwrite("y_pure", &FlashParams::y_pure, "Mole fraction for preferred EoS range")

        .def_readwrite("rr2_tol", &FlashParams::rr2_tol, "Tolerance for two-phase Rachford-Rice norm")
        .def_readwrite("rrn_tol", &FlashParams::rrn_tol, "Tolerance for N-phase Rachford-Rice norm")
        .def_readwrite("rr_max_iter", &FlashParams::rr_max_iter, "Maximum number of iterations for Rachford-Rice procedure")

        .def_readwrite("split_tol", &FlashParams::split_tol, "Tolerance for phase split norm")
        .def_readwrite("split_switch_tol", &FlashParams::split_switch_tol, "Tolerance for switch to Newton in phase split")
        .def_readwrite("split_switch_diff", &FlashParams::split_switch_diff, "If decrease in log(norm) between two SSI iterations is below this number (and tol < switch_tol), switch to Newton - make use of effectiveness of SSI")
        .def_readwrite("split_line_tol", &FlashParams::split_line_tol, "Tolerance for line search in phase split")
        .def_readwrite("split_max_iter", &FlashParams::split_max_iter, "Maximum number of iterations for phase split")
        .def_readwrite("split_line_iter", &FlashParams::split_line_iter, "Maximum number of iterations for line search in phase split")
        .def_readwrite("split_negative_flash_iter", &FlashParams::split_negative_flash_iter, "Number of iterations to quit split if negative flash")
        .def_readwrite("split_variables", &FlashParams::split_variables, "Variables for phase split: 0) n_ik, 1) lnK, 2) lnK-chol")
        .def_readwrite("modChol_split", &FlashParams::modChol_split, "Switch for modified Cholesky iterations in split")
        
        .def_readwrite("stability_variables", &FlashParams::stability_variables, "Variables for stability: 0) Y, 1) lnY, 2) alpha")
        .def_readwrite("modChol_stability", &FlashParams::modChol_stability, "Switch for modified Cholesky iterations in split")
        .def_readwrite("tpd_tol", &FlashParams::tpd_tol, "Tolerance for comparing tpd; also used to determine limit for tpd that is considered stable")
        .def_readwrite("tpd_1p_tol", &FlashParams::tpd_1p_tol, "Tolerance to determine limit for tpd that is considered stable")
        .def_readwrite("tpd_close_to_boundary", &FlashParams::tpd_close_to_boundary, "Tolerance to check if too close to phase boundary and switch PhaseSplit variables")
        .def_readwrite("comp_tol", &FlashParams::comp_tol, "Tolerance for comparing compositions")

        .def_readwrite("T_min", &FlashParams::T_min, "Minimum temperature for PXFlash")
        .def_readwrite("T_max", &FlashParams::T_max, "Maximum temperature for PXFlash")
        .def_readwrite("T_init", &FlashParams::T_init, "Initial temperature for PXFlash")
        .def_readwrite("pxflash_type", &FlashParams::pxflash_type, "Root finding algorithm for PXFlash: 0) BRENT, 1) BRENT_NEWTON")
        .def_readwrite("pxflash_Ftol", &FlashParams::pxflash_Ftol, "Function tolerance for specification equation in PXFlash: |X-Xspec| < Ftol")
        .def_readwrite("pxflash_Ttol", &FlashParams::pxflash_Ttol, "Temperature tolerance for switch to locate_phase_boundary in PXFlash: Tmax-Tmin < Ttol")
        .def_readwrite("phase_boundary_Gtol", &FlashParams::phase_boundary_Gtol, "Function tolerance for locating equal Gibbs energies: |Ga-Gb| < Gtol")
        .def_readwrite("phase_boundary_Ttol", &FlashParams::phase_boundary_Ttol, "Temperature tolerance for locating equal Gibbs energies: Tmax-Tmin < Ttol")
        
        .def_readwrite("save_performance_data", &FlashParams::save_performance_data, "Option to save performance data of the flash")
        .def_readwrite("verbose", &FlashParams::verbose, "Verbose level")

        // EoS-related parameters
        .def_readwrite("eos_params", &FlashParams::eos_params, "Map of EoSParams object associated with each EoS object")
        .def_readwrite("eos_order", &FlashParams::eos_order, "Order of EoS for output")
        .def_readwrite("vl_eos_name", &FlashParams::vl_eos_name, "Name of VL EoS object for phase identification")
        .def_readwrite("light_comp_idx", &FlashParams::light_comp_idx, "Index of lightest component to perform initial guess on for vapour phase identification")
        .def_readwrite("np_max", &FlashParams::np_max, "Maximum number of phases according to EoS order, root and rich phase order")
        .def("add_eos", &FlashParams::add_eos, R"pbdoc(
            Add EoSParams object to map. This function creates a copy of the EoS object inside the EoSParams struct.

            :param name: EoS name
            :type name: str
            :param eos: EoS object
            :type eos: EoS
            )pbdoc", "name"_a, "eos"_a)
        .def("set_eos_order", &FlashParams::set_eos_order, R"pbdoc(
            Set order of EoS. This function counts the maximum number of phases and creates a map of unique phase states.

            :param name: List of EoS names
            )pbdoc", "eos_order"_a)
        .def("get_phase_state", py::overload_cast<std::vector<int>&>(&FlashParams::get_phase_state), R"pbdoc(
            Get phase state id. This function calculates binary id of phase idxs and returns phase state id.

            :param phase_idxs: List of phase idxs

            :returns: Phase state id
            :rtype: int
            )pbdoc", "phase_idxs"_a)
        .def("get_phase_state", py::overload_cast<int>(&FlashParams::get_phase_state), R"pbdoc(
            Get phase state string from phase_states_map

            :param phase_state: Phase state id

            :returns: Phase state name
            :rtype: str
            )pbdoc", "phase_state"_a)
        .def_readwrite("phase_states_str", &FlashParams::phase_states_str, "Vector of phase state names")
        .def("init_eos", &FlashParams::init_eos, R"pbdoc(
            Initialize EoS parameters at (P,T)

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            )pbdoc", "p"_a, "T"_a)
        .def("find_ref_comp", &FlashParams::find_ref_comp, R"pbdoc(
            Find stable phase at state (P,T,n)

            :param p: Pressure
            :type p: float
            :param T: Temperature
            :type T: float
            :param n: List of compositions
            :type n: list
            )pbdoc", "p"_a, "T"_a, "n"_a)
        .def("find_pure_phase", &FlashParams::find_pure_phase, R"pbdoc(
            Find pure phases at state (P,T)

            :param p: Pressure
            :type p: float
            :param T: Temperature
            :type T: float
            :param Gpure: List of pure component Gibbs free energies
            :type Gpure: list
            )pbdoc", "p"_a, "T"_a, "Gpure"_a)
        .def("G_pure", &FlashParams::G_pure)
        .def("H_pure", &FlashParams::H_pure)
        .def("prop_pure", &FlashParams::prop_pure)
        .def("prop_1p", &FlashParams::prop_1p)
        .def("prop_np", &FlashParams::prop_np)
        ;

    py::enum_<FlashParams::SplitVars>(flash_params, "SplitVars", "Primary variables for phase split")
        .value("nik", FlashParams::SplitVars::nik)
        .value("lnK", FlashParams::SplitVars::lnK)
        .value("lnK_chol", FlashParams::SplitVars::lnK_chol)
        .export_values()
        ;

    py::enum_<FlashParams::StabilityVars>(flash_params, "StabilityVars", "Primary variables for stability")
        .value("Y", FlashParams::StabilityVars::Y)
        .value("lnY", FlashParams::StabilityVars::lnY)
        .value("alpha", FlashParams::StabilityVars::alpha)
        .export_values()
        ;

    py::enum_<FlashParams::PXFlashType>(flash_params, "PXFlashType", "Root finding algorithm for PXFlash")
        .value("BRENT", FlashParams::PXFlashType::BRENT)
        .value("BRENT_NEWTON", FlashParams::PXFlashType::BRENT_NEWTON)
        .export_values()
        ;

    // Expose TrialPhase and InitialGuess classes
    py::class_<TrialPhase>(m, "TrialPhase", R"pbdoc(
            This is a struct containing all information for a trial phase
            )pbdoc")
        .def(py::init<int, std::string, std::vector<double>&, EoS::RootFlag>(), R"pbdoc(
            :param eos_idx: Index of EoS in FlashParams::eos_order
            :param eos_name: Name of EoS of trial phase
            :param Y: Trial phase composition Y
            :param root: EoS.RootFlag root type
            )pbdoc", "eos_idx"_a, "eos_name"_a, "Y"_a, py::arg("root")=EoS::RootFlag::STABLE)

        .def_readonly("Y", &TrialPhase::Y)
        .def_readonly("y", &TrialPhase::y)
        .def_readonly("tpd", &TrialPhase::tpd)
        .def_readonly("eos_name", &TrialPhase::eos_name)
        .def_readonly("eos_idx", &TrialPhase::eos_idx)
        .def_readonly("root_type", &TrialPhase::root)

        .def("print_point", &TrialPhase::print_point, R"pbdoc(
            Function to print out trial phase information
            )pbdoc")
        ;

    py::class_<InitialGuess> ig(m, "InitialGuess", R"pbdoc(
            This is a base class for providing initial guesses to Flash, Stability or PhaseSplit algorithms.

            It contains methods for evaluating Wilson's (1968) correlation for vapour-liquid, Henry's law coefficients (Sander, 2015)
            vapour-hydrate and `pure phase` ideal K-values.
            )pbdoc");
    
    ig.def(py::init<CompData&>(), R"pbdoc(
            This is the constructor for generating initial guesses for Stability algorithms

            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)
        
        .def("init", &InitialGuess::init, R"pbdoc(
            Initialize P and T inside InitialGuess object.

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            )pbdoc", "p"_a, "T"_a)
        
        .def("wilson", &InitialGuess::k_wilson, R"pbdoc(        
            :returns: List of Wilson (1968) K-values at P-T
            :rtype: list
            )pbdoc")
        .def("henry", &InitialGuess::k_henry, R"pbdoc(        
            :returns: List of Henry's law K-values at P-T, Sander (2015)
            :rtype: list
            )pbdoc")
        .def("y_henry", &InitialGuess::y_henry, R"pbdoc(        
            :returns: List of Henry's law K-values at P-T, Sander (2015)
            :rtype: list
            )pbdoc")
        .def("vapour_sI", &InitialGuess::k_vapour_sI, R"pbdoc(        
            :returns: List of vapour-sI K-values at P-T
            :rtype: list
            )pbdoc")
        .def("vapour_sII", &InitialGuess::k_vapour_sII, R"pbdoc(        
            :returns: List of vapour-sII K-values at P-T
            :rtype: list
            )pbdoc")
        .def("y_pure", &InitialGuess::y_pure, R"pbdoc(        
            :param j: j'th component
            :type j: int
            :param pure: Mole fraction of pure component, default is 0.9
            :type pure: double
            :returns: List of pure component K-values at P-T-x
            :rtype: list
            )pbdoc", "j"_a, py::arg("pure")=NAN)
        .def("set_ypure", &InitialGuess::set_ypure, "Set mole fraction for pure phase guess of component j", "j"_a, "pure"_a)
        .def("set_ymin", &InitialGuess::set_ymin, "Set mole fraction for ymin phase guess", "rich_idx"_a, "rich_comp"_a)
        ;

    py::enum_<InitialGuess::Ki>(ig, "Ki", "Correlation type for phase split initial guess (K)")
        .value("Wilson_VL", InitialGuess::Ki::Wilson_VL)
        .value("Wilson_LV", InitialGuess::Ki::Wilson_LV)
        .value("Henry_VA", InitialGuess::Ki::Henry_VA)
        .value("Henry_AV", InitialGuess::Ki::Henry_AV)
        .export_values()
        ;
    
    py::enum_<InitialGuess::Yi>(ig, "Yi", "Correlation type for stability test initial guess (Y)")
        .value("Wilson", InitialGuess::Yi::Wilson)
        .value("Wilson13", InitialGuess::Yi::Wilson13)
        .value("Henry", InitialGuess::Yi::Henry)
        .value("sI", InitialGuess::Yi::sI)
        .value("sII", InitialGuess::Yi::sII)
        .value("sH", InitialGuess::Yi::sH)
        .value("Min", InitialGuess::Yi::Min)
        .value("Pure", InitialGuess::Yi::Pure)
        .export_values()
        ;
};

// #endif //PYBIND11_ENABLED