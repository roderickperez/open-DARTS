#ifndef GPU_BECNHMARK_CU_H
#define GPU_BECNHMARK_CU_H

#include "interp_table.h"


/**
 * Host function that copies the data and launches the work on GPU
 */
interp_value_t* gpuInterpolation(interp_value_t *val1, interp_value_t *val2, unsigned size, interp_table *tbl);

#endif
