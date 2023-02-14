//*************************************************************************
//    Copyright (c) 2022
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

// *************************************************************************
// BOS GMRES wrapper
// *************************************************************************

//--------------------------------------------------------------------------
#ifndef OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_GMRES_HPP
#define OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_GMRES_HPP
//--------------------------------------------------------------------------

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/csr_matrix_base.hpp"
#include "openDARTS/linear_solvers/linsolv_iface_bos.hpp"
#include "openDARTS/linear_solvers/linear_solver_base.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE>
    class linsolv_bos_gmres : public opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE>
    {
      public:
        linsolv_bos_gmres(int gpu_mode_in = 0);

        ~linsolv_bos_gmres();
        
        // TODO: This implementation is a complete mess.has similar structure to
        //       linsolv_bos_cpr but does not inherit from linear_solver_base. review!
    

        // setup preconditioner
        int setup(opendarts::linear_solvers::csr_matrix_base *matrix) override;

        int set_prec(opendarts::linear_solvers::linsolv_iface *prec_in) override;

        int init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
            opendarts::config::index_t max_iters, 
            opendarts::config::mat_float tolerance) override;
          
        int setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in) override;

        int solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X) override;
      
        opendarts::config::index_t get_n_iters() override;

        opendarts::config::mat_float get_residual() override;
        
        opendarts::linear_solvers::linear_solver_base *solver;
        opendarts::linear_solvers::linsolv_iface_bos<N_BLOCK_SIZE> *prec;
        opendarts::linear_solvers::csr_matrix_base *A;
        int gpu_mode;
        
    };
  } // namespace linear_solvers
} // namespace opendarts

//--------------------------------------------------------------------------
#endif // OPENDARTS_LINEAR_SOLVERS_LINSOLV_BOS_GMRES_HPP
//--------------------------------------------------------------------------
