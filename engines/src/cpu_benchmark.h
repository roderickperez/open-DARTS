#ifndef CPU_BECNHMARK_H
#define CPU_BECNHMARK_H

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#ifdef _MSC_VER 
#include <time.h>
#endif
#include "2d_interpolation.h"


#include "interp_table.h"


interp_value_t* cpuInterpolation (interp_value_t *val1, interp_value_t *val2, unsigned size, interp_table *tbl)
{

  int nIter = 10;
  float msecTotal = 0.0f, msecPerMatrixMul;
  double flopsPerMatrixMul, gigaFlops;

  interp_value_t *cpuRes = new interp_value_t[size];
  interp_value_t *cpuResDer1 = new interp_value_t[size];
  interp_value_t *cpuResDer2 = new interp_value_t[size];

  int ax1_points = tbl->ax1_npoints;
  interp_value_t ax1_min = tbl->ax1_min;
  interp_value_t ax1_max = tbl->ax1_max;
  interp_value_t ax1_step_inv = (ax1_points - 1) / (ax1_max - ax1_min);

  int ax2_points = tbl->ax2_npoints;
  interp_value_t ax2_min = tbl->ax2_min;
  interp_value_t ax2_max = tbl->ax2_max;
  interp_value_t ax2_step_inv = (ax2_points - 1) / (ax2_max - ax2_min);

  interp_value_t *cpuInterp = tbl->data;
  
  // now estimate cpu performance

  // warmup
  bilinear_interpolation<interp_index_t, interp_value_t> (size, ax1_points, ax1_min, ax1_step_inv,
    ax2_points, ax2_min, ax2_step_inv,
    cpuInterp, val1, val2,
    cpuRes, cpuResDer1, cpuResDer2);
  clock_t cstart = clock ();
  clock_t cend = 0;
  for (int j = 0; j < nIter; j++)
  {
    bilinear_interpolation<interp_index_t, interp_value_t> (size, ax1_points, ax1_min, ax1_step_inv,
      ax2_points, ax2_min, ax2_step_inv,
      cpuInterp, val1, val2,
      cpuRes, cpuResDer1, cpuResDer2);
  }
  cend = clock ();
  msecTotal = ((double)cend - (double)cstart) / CLOCKS_PER_SEC * 1000;

  msecPerMatrixMul = msecTotal / nIter;
  flopsPerMatrixMul = 28 * size;
  gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf ("CPU Performance= %.2f GFlop/s, Time= %.3f msec \n", gigaFlops, msecPerMatrixMul);

  //for (int i = 0; i < 5; i++)
  //  printf ("Val1 = %lf\tVal2 = %lf; \tRes = %e\tDer1 = %e\tDer2 = %e \r\n", val1[i], val2[i], cpuRes[i], cpuResDer1[i], cpuResDer2[i]);
  return cpuRes;
}
#endif

