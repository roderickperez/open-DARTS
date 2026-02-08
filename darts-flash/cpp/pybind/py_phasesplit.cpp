#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include "dartsflash/phase-split/basesplit.hpp"
#include "dartsflash/phase-split/twophasesplit.hpp"
#include "dartsflash/phase-split/multiphasesplit.hpp"

namespace py = pybind11;

template <class BaseSplitBase = BaseSplit> class PyBaseSplit : public BaseSplitBase {
public:
    /* Inherit the constructors */
    using BaseSplitBase::BaseSplitBase;

    /* Trampoline (need one for each virtual function) */
    Eigen::MatrixXd construct_U() override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd, BaseSplit, construct_U,
        );
    }
    Eigen::MatrixXd construct_Uinv() override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd, BaseSplit, construct_Uinv,
        );
    }
    Eigen::MatrixXd construct_PHI() override {
        PYBIND11_OVERRIDE_PURE(
            Eigen::MatrixXd, BaseSplit, construct_PHI,
        );
    }
    Eigen::MatrixXd construct_H(Eigen::MatrixXd& U, Eigen::MatrixXd& PHI) override {
        PYBIND11_OVERRIDE(
            Eigen::MatrixXd, BaseSplit, construct_H, U, PHI
        );
    }
    Eigen::MatrixXd construct_J(Eigen::MatrixXd& PHI, Eigen::MatrixXd& U_inv) override {
        PYBIND11_OVERRIDE(
            Eigen::MatrixXd, BaseSplit, construct_J, PHI, U_inv
        );
    }
};

void pybind_phasesplit(py::module& m)
{
    using namespace pybind11::literals;  // bring in '_a' literal
    
    // Expose BaseSplit class
    py::class_<BaseSplit, PyBaseSplit<>>(m, "BaseSplit", R"pbdoc(
            This is the base class for phase split calculation.
            )pbdoc")
        .def(py::init<FlashParams&, int>(), R"pbdoc(
            This is the constructor for the BaseSplit class.
            )pbdoc", "flash_params"_a, "np"_a)

        .def("run", &BaseSplit::run, R"pbdoc(
            Run multiphase split algorithm at p, T, z, with initial guess lnK
            )pbdoc", "z"_a, "lnk"_a, "eos"_a)

        .def("nu", &BaseSplit::getnu, R"pbdoc(
            :returns: List of phase fractions
            :rtype: list
            )pbdoc")
        .def("x", &BaseSplit::getx, R"pbdoc(
            :returns: List of phase compositions
            :rtype: list
            )pbdoc")
        ;

    // Expose TwoPhaseSplit class
    py::class_<TwoPhaseSplit, PyBaseSplit<TwoPhaseSplit>, BaseSplit>(m, "TwoPhaseSplit", R"pbdoc(
            This is the BaseClass implementation for split calculations with two phases.
            )pbdoc")
        .def(py::init<FlashParams&>(), R"pbdoc(
            :param flash_params: FlashParams object
            )pbdoc", "flash_params"_a)
        ;

    // Expose MultiPhaseSplit class
    py::class_<MultiPhaseSplit, PyBaseSplit<MultiPhaseSplit>, BaseSplit>(m, "MultiPhaseSplit", R"pbdoc(
            This is the BaseClass implementation for split calculations with three or more phases.
            )pbdoc")
        .def(py::init<FlashParams&, int>(), R"pbdoc(
            :param np: Number of phases
            :type np: int
            :param flash_params: FlashParams object
            )pbdoc", "flash_params"_a, "np"_a)
        ;
}
