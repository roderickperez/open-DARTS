#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

#include "openDARTS/config/version.hpp"
#include "openDARTS/linear_solvers/csr_matrix.hpp"
#include "openDARTS/linear_solvers/data_types.hpp"

#include "test_common.hpp"

int test_transpose();
int test_build_transpose();
int test_build_transpose_struct();

int main()
{
  /*
    test_04__transpose
    Tests csr_matrix.transpose function. To do that, generates a tridiagonal matrix
    with a block size of 4 and transposes it. The output is compared to a reference
    result.
  */
  int error_output = 0;
  
  // Test the csr_matrix::transpose function A -> A^t
  error_output += test_transpose();
  
  // Test the csr_matrix::build_transpose function A^t <- A
  error_output += test_build_transpose();
  
  // Test the csr_matrix::build_transpose_struct function A^t <- A (only structure, no values)
  error_output += test_build_transpose_struct();
  
  return error_output;
  
}


int test_transpose()
{
  // bool files_are_equal = true;
  int error_output = 0;
  bool files_are_equal = true;
  opendarts::config::index_t n = 12;

  // The output file to save the matrix to
  std::string output_filename("test_04__transpose.txt");

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("test_04__transpose_ref.txt");         // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  // Generate the matrix with block size 4
  opendarts::linear_solvers::csr_matrix<4> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n,
      opendarts::linear_solvers::testing::block_fill_option::
          index_filled_block); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the
                               // -2, 0, and 2 diagonals

  // Compute transpose
  opendarts::linear_solvers::csr_matrix<4> A_t;
  A.transpose(A_t);

  // Save the matrix in human readable format
  error_output = A_t.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::human_readable); // save the matrix to file

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  return error_output;
}


int test_build_transpose()
{
  // bool files_are_equal = true;
  int error_output = 0;
  bool files_are_equal = true;
  opendarts::config::index_t n = 12;

  // The output file to save the matrix to
  std::string output_filename("test_04__build_transpose.txt");

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("test_04__transpose_ref.txt");         // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  // Generate the matrix with block size 4
  opendarts::linear_solvers::csr_matrix<4> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n,
      opendarts::linear_solvers::testing::block_fill_option::
          index_filled_block); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the
                               // -2, 0, and 2 diagonals

  // Compute transpose
  opendarts::linear_solvers::csr_matrix<4> A_t;
  A_t.build_transpose(&A);

  // Save the matrix in human readable format
  error_output = A_t.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::human_readable); // save the matrix to file

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  return error_output;
}

int test_build_transpose_struct()
{
  // bool files_are_equal = true;
  int error_output = 0;
  bool files_are_equal = true;
  opendarts::config::index_t n = 12;

  // The output file to save the matrix to
  std::string output_filename("test_04__build_transpose_struct.txt");

  //  The reference ascii file
  std::string data_path_prefix = opendarts::config::get_cmake_openDARTS_source_dir() +
                                 std::string("/data/tests/linear_solvers/"); // the path to the data folder
  std::string reference_base_filename("test_04__build_transpose_struct_ref.txt");         // the filename of the reference file
  std::string reference_filename = data_path_prefix +
                                   reference_base_filename; // get the full path to the reference file

  // Generate the matrix with block size 4
  opendarts::linear_solvers::csr_matrix<4> A;
  opendarts::linear_solvers::testing::generate_tridiagonal_matrix(A, n,
      opendarts::linear_solvers::testing::block_fill_option::
          index_filled_block); // populate the matrix, in this case a tridiagonal matrix with the values -2, 1, 2 in the
                               // -2, 0, and 2 diagonals

  // Compute transpose
  opendarts::linear_solvers::csr_matrix<4> A_t;
  A_t.build_transpose_struct(&A);

  // Save the matrix in human readable format
  error_output = A_t.export_matrix_to_file(output_filename,
      opendarts::linear_solvers::sparse_matrix_export_format::human_readable); // save the matrix to file

  // Compare the generated output to the reference output
  files_are_equal = opendarts::linear_solvers::testing::compare_files(output_filename,
      reference_filename); // check if equal to reference

  if (files_are_equal)
  {
    error_output = 0;
  }
  else
  {
    error_output = 1;
  }

  return error_output;
}
