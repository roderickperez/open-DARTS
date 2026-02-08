//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_MATHS_MODIFIEDCHOLESKY_H
#define OPENDARTS_FLASH_MATHS_MODIFIEDCHOLESKY_H
//--------------------------------------------------------------------------
////////////////////////////////////////////////////////////////////////////////////////////////////
///
///  From Numerical Recipies
///
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

#include <Eigen/Dense>


class ModifiedCholeskyS99
{
public:
    //Constructor.
    ModifiedCholeskyS99();
    // Destructor
    ~ModifiedCholeskyS99();
    // Given a positive matrix and its size,
    // construct and store the cholesky decomposition of A
    int initialize(Eigen::MatrixXd &A_in, int);
    int solve(Eigen::VectorXd& b_in, Eigen::VectorXd& x_out); // Solve Ax=b, input=b, output=x

private:
    Eigen::MatrixXd m_el, m_piv;
    std::vector<double> m_elinvii;
    std::vector<int> m_pivot;
    int        m_n;
    bool       m_isposdef;
    int        m_indnonpositive;
    double     m_valnpositive;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_MATHS_MODIFIEDCHOLESKY_H
//--------------------------------------------------------------------------
