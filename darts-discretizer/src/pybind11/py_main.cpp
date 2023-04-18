#include "py_global.h"
#include "utils.h"

namespace py = pybind11;

void pybind_elem(py::module &m);
void pybind_mesh(py::module &m);
void pybind_discretizer(py::module &m);
void pybind_linalg(py::module &m);

PYBIND11_MODULE(discretizer, m)
{
	py::bind_vector<std::vector<index_t>>(m, "index_vector", py::module_local(), py::buffer_protocol());
	py::bind_vector<std::vector<value_t>>(m, "value_vector", py::module_local(), py::buffer_protocol())
		.def(py::pickle(
			[](const std::vector<value_t> &p) { // __getstate__
		py::tuple t(p.size());
		for (int i = 0; i < p.size(); i++)
			t[i] = p[i];

		return t;
	},
			[](py::tuple t) { // __setstate__
		std::vector<value_t> p(t.size());

		for (int i = 0; i < p.size(); i++)
			p[i] = t[i].cast<value_t>();

		return p;
	}));
	
	m.def("load_single_float_keyword", utils::load_single_keyword<value_t>);
	m.def("load_single_int_keyword", utils::load_single_keyword<index_t>);

	pybind_elem(m);
	pybind_linalg(m);
	pybind_mesh(m);
	pybind_discretizer(m);
}
