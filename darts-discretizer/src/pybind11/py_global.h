#ifndef PYGLOBAL_H_
#define PYGLOBAL_H_

#include "pybind11.h"
#include "numpy.h"
#include "stl.h"
#include "stl_bind.h"

#include <vector>
#include <valarray>

typedef int index_t;
typedef double value_t;

PYBIND11_MAKE_OPAQUE(std::vector<index_t>);
PYBIND11_MAKE_OPAQUE(std::vector<value_t>);

#endif /* PYGLOBAL_H_ */