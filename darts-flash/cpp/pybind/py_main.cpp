#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void pybind_eos(py::module &);
void pybind_global(py::module &);
void pybind_flash(py::module &);
// void pybind_phasesplit(py::module &);
void pybind_stability(py::module &);
// void pybind_rr(py::module &);


PYBIND11_MODULE(libflash, m) {
    m.doc() = R"pbdoc(
        This is the documentation of `dartsflash.libflash`.

        In the libflash module, the user can perform thermodynamic calculations:
        - Equations of State for a range of vapour, liquid, solid phases
        - Stability test, multiphase split, multiphase flash algorithms
        - Thermodynamic property calculation using Equations of State
        )pbdoc" ;

    // make possible to pass STL-containers by reference in function arguments
    // py::bind_vector<std::vector<int>>(m, "index_vector", py::module_local(true), py::buffer_protocol());
    // py::bind_vector<std::vector<double>>(m, "value_vector", py::module_local(true), py::buffer_protocol());
    // py::bind_vector<std::vector<std::string>>(m, "string_vector");

    pybind_eos(m);
    pybind_global(m);
    pybind_flash(m);
    // pybind_phasesplit(m);
    // pybind_rr(m);
    pybind_stability(m);

}
