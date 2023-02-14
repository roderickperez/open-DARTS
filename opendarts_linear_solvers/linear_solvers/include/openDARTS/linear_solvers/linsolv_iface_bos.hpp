//*************************************************************************
//    Copyright (c) 2023
//    Delft University of Technology, the Netherlands
//
//    This file is part of the open Delft Advanced Research Terra Simulator (opendarts)
//
//    opendarts is free software: you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as
//    published by the Free Software Foundation, either version 3 of the
//    License, or (at your option) any later version.
//
//    DARTS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public
//    License along with DARTS. If not, see <http://www.gnu.org/licenses/>.
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_BOS_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_BOS_HPP
//--------------------------------------------------------------------------

#include "openDARTS/auxiliary/timer_node.hpp"
#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linear_solver_base.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_iface_bos : public linsolv_iface
    {

      public:
        linsolv_iface_bos () {};

        virtual ~linsolv_iface_bos () {};

        virtual int init(opendarts::linear_solvers::csr_matrix_base *A, 
          int max_iters, 
          opendarts::config::mat_float tolerance)
        { 
          // TODO: This can, and will, go horribly wrong, this must be changed --> comes from previous code
          return this->init(static_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A), max_iters, tolerance);
        };

        virtual int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A, int max_iters, double tolerance) = 0;

        virtual int setup(opendarts::linear_solvers::csr_matrix_base *A)
        {
          // TODO: This can, and will, go horribly wrong, this must be changed --> comes from previous code
          return this->setup(static_cast<opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *>(A));
        };
        
        virtual int setup (opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A) = 0;

        opendarts::linear_solvers::linear_solver_base *get_bos_solver ()
        { 
          return solver;
        };

        opendarts::linear_solvers::linear_solver_base *solver;

    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_IFACE_HPP
//--------------------------------------------------------------------------
