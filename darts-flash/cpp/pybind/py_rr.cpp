#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

#include "dartsflash/rr/rr.hpp"

namespace py = pybind11;

template <class RRBase = RR> class PyRR : public RRBase {
public:
    using RRBase::RRBase;

    int solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_={}) override {
        PYBIND11_OVERRIDE_PURE(
            int, RR, solve_rr, z_, K_, nonzero_comp_
        );
    }
    std::vector<double> getx() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, RR, getx,
        );
    }
};

void pybind_rr(py::module& m) 
{
    using namespace pybind11::literals;  // bring in '_a' literal
    
    // Expose RR class
    py::class_<RR, PyRR<>>(m, "RR", R"pbdoc(
            This is the base class for solving the Rachford-Rice equation.
            )pbdoc")
        .def(py::init<FlashParams&, int, int>(), R"pbdoc(
            :param flash_params: FlashParams object that contains all numerical parameters for calculation
            :param np: Number of phases
            :type np: int
            :param nc: Number of components
            :type nc: int
            )pbdoc", "flash_params"_a, "np"_a, "nc"_a)
        .def("solve_rr", &RR::solve_rr, R"pbdoc(
            This function solves the N-phase RR equation

            :param z: List of feed composition
            :type z: list
            :param K: List of equilibrium constants K
            :type K: list
            :param nonzero_comp: List of equilibrium constants K
            :type nonzero_comp: list
            :returns: Error output of RR-procedure
            :rtype: int
            )pbdoc", "z"_a, "K"_a, "nonzero_comp"_a)
        .def("getnu", &RR::getnu, R"pbdoc(
            :returns: List of phase fractions
            :rtype: list
            )pbdoc")
        .def("getx", &RR::getx, R"pbdoc(
            :returns: List of phase compositions
            :rtype: list
            )pbdoc")
        ;

    py::class_<RR_Eq2, PyRR<RR_Eq2>, RR>(m, "RR_Eq2", R"pbdoc(
            This is the two-phase negative flash implementation based on equation solving with bisection and Newton. 
            
            For reference, see Whitson and Michelsen (1989).
            )pbdoc")
        .def(py::init<FlashParams&, int>(), R"pbdoc(
            :param flash_params: FlashParams object that contains all numerical parameters for calculation
            :param nc: Number of components
            :type nc: int
            )pbdoc", "flash_params"_a, "nc"_a)
        ;

    py::class_<RR_EqConvex2, PyRR<RR_EqConvex2>, RR>(m, "RR_EqConvex2", R"pbdoc(
            This is the two-phase negative flash implementation based on equation solving with convex transformations.
            
            For reference, see Nichita and Leibovici (2013).
            )pbdoc")
        .def(py::init<FlashParams&, int>(), R"pbdoc(
            :param flash_params: FlashParams object that contains all numerical parameters for calculation
            :param nc: Number of components
            :type nc: int
            )pbdoc", "flash_params"_a, "nc"_a)
        ;

    py::class_<RR_EqN, PyRR<RR_EqN>, RR>(m, "RR_EqN", R"pbdoc(
            This is the generalized multiphase negative flash implementation based on equation solving with bisection and Newton.
            
            For reference, see Iranshahr (2010).
            )pbdoc")
        .def(py::init<FlashParams&, int, int>(), R"pbdoc(
            :param flash_params: FlashParams object that contains all numerical parameters for calculation
            :param np: Number of phases
            :type np: int
            :param nc: Number of components
            :type nc: int
            )pbdoc", "flash_params"_a, "np"_a, "nc"_a)
        ;

    py::class_<RR_Min, PyRR<RR_Min>, RR>(m, "RR_Min", R"pbdoc(
            This is the normal flash implementation based on minimization.

            For reference, see Michelsen (1994).
            )pbdoc")
        .def(py::init<FlashParams&, int, int>(), R"pbdoc(
            :param flash_params: FlashParams object that contains all numerical parameters for calculation
            :param np: Number of phases
            :type np: int
            :param nc: Number of components
            :type nc: int
            )pbdoc", "flash_params"_a, "np"_a, "nc"_a)
        ;

    py::class_<RR_MinNeg, PyRR<RR_MinNeg>, RR>(m, "RR_MinNeg", R"pbdoc(
            This is the negative flash implementation based on minimization.

            For reference, see Yan and Stenby (2012).
            )pbdoc")
        .def(py::init<FlashParams&, int, int>(), R"pbdoc(
            :param flash_params: FlashParams object that contains all numerical parameters for calculation
            :param np: Number of phases
            :type np: int
            :param nc: Number of components
            :type nc: int
            )pbdoc", "flash_params"_a, "np"_a, "nc"_a)
        ;
}
