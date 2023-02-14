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

#include <iostream>

#include "openDARTS/config/data_types.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/linsolv_iface.hpp"
#include "openDARTS/linear_solvers/linsolv_bos_bilu0.hpp"

namespace opendarts
{
  namespace linear_solvers
  {
    template <uint8_t N_BLOCK_SIZE> 
    linsolv_bos_bilu0<N_BLOCK_SIZE>::linsolv_bos_bilu0()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::linsolv_bos_bilu0" << std::endl;
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    linsolv_bos_bilu0<N_BLOCK_SIZE>::~linsolv_bos_bilu0()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::~linsolv_bos_bilu0" << std::endl;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_bilu0<N_BLOCK_SIZE>::set_prec(opendarts::linear_solvers::linsolv_iface *prec_in)
    {
      (void) prec_in;
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::set_prec" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_bilu0<N_BLOCK_SIZE>::init(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in, 
      opendarts::config::index_t max_iters, 
      opendarts::config::mat_float tolerance)
    {
      (void) A_in;
      (void) max_iters;
      (void) tolerance; 
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::init" << std::endl;
      
      return 1;
    }
    
    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_bilu0<N_BLOCK_SIZE>::setup(opendarts::linear_solvers::csr_matrix<N_BLOCK_SIZE> *A_in)
    {
      (void) A_in;
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::setup" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    int linsolv_bos_bilu0<N_BLOCK_SIZE>::solve(opendarts::config::mat_float *B, opendarts::config::mat_float *X)
    {
      (void) B; 
      (void) X;
      
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::solve" << std::endl;
      
      return 1;
    }

    template <uint8_t N_BLOCK_SIZE> 
    opendarts::config::index_t linsolv_bos_bilu0<N_BLOCK_SIZE>::get_n_iters()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::get_n_iters" << std::endl;
      
      return 0;
    }

    template <uint8_t N_BLOCK_SIZE> 
    opendarts::config::mat_float linsolv_bos_bilu0<N_BLOCK_SIZE>::get_residual()
    {
      std::cout << "NOT IMPLEMENTED: linsolv_bos_bilu0::get_residual" << std::endl;
      
      return 1000.0;
    }
    
    template class linsolv_bos_bilu0<1>;
    template class linsolv_bos_bilu0<2>;
    template class linsolv_bos_bilu0<3>;
    template class linsolv_bos_bilu0<4>;
    template class linsolv_bos_bilu0<5>;
    template class linsolv_bos_bilu0<6>;
    template class linsolv_bos_bilu0<7>;
    template class linsolv_bos_bilu0<8>;
    template class linsolv_bos_bilu0<9>;
    template class linsolv_bos_bilu0<10>;
    template class linsolv_bos_bilu0<11>;
    template class linsolv_bos_bilu0<12>;
    template class linsolv_bos_bilu0<13>;
  } // namespace linear_solvers
} // namespace opendarts
