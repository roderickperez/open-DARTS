#include "commons.h"
#include <pybind11/stl.h>

/*namespace py = pybind11;*/

// commons related bindings
using namespace commons;


PYBIND11_MODULE(commons, m) {
  m.doc() = "Shared functions and objects of DARTS.";

  m.def("counter", &counter);
}


