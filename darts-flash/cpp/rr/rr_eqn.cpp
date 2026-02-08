#include <iostream>
#include <cmath>
#include <algorithm>

#include "dartsflash/rr/rr.hpp"
#include "dartsflash/global/global.hpp"
#include "dartsflash/maths/maths.hpp"
#include <Eigen/Dense>

RR_EqN::RR_EqN(FlashParams& flash_params, int nc_, int np_) : RR(flash_params, nc_, np_) {
	v_mid.resize(np-1);
	v_min.resize(np-1);
	v_max.resize(np-1);
    f.resize(np-1);

	// Output vector, contains all NP values of V
	nu.resize(np);
}

bool RR_EqN::bounded() {
	// Calculate directional derivative of mi(V) along mj(V)
	// gij = Dm_i dotted with u_j
	// Dm_i[i, k] = K[i, k] - 1
	// u_j[j] = [1, 1, ..., sum((K[j, k] - 1)/(1-K[j, np-1]))]

	std::vector<double> Dm_i(np-1, 0.);
	std::vector<double> u_j(np-1, 1.);
	std::vector<double> gij(nc*nc, 0.);
	for (int i = 0; i < nc; i++)
	{
		for (int k = 0; k < np-1; k++)
		{
			Dm_i[k] = K[k*nc + i] - 1;
		}
		for (int j = 0; j < nc; j++)
		{
			if (np > 2)
			{
				u_j[np-2] = 0;
				for (int k = 0; k < np-2; k++)
				{
					u_j[np-2] += (K[k*nc + j] - 1)/(1-K[(np-2)*nc + j]);
				}
			}
			for (int k = 0; k < np-1; k++)
			{
				gij[i*nc + j] += Dm_i[k]*u_j[k];
			}
		}
	}

	// Check if there exist j, k such that gji*gki < 0 for i = 0, ..., NC-1
	std::vector<bool> gi(nc, false);
	for (int i = 0; i < nc; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			for (int k = 0; k < nc; k++)
			{
				if (gij[j*nc+i] * gij[k*nc+i] < 0)
				{
					gi[i] = true;
					k = nc; j = nc; // go to next i in loop
				}
			}
		}
		if (gi[i] == false)
		{
			return false;  // there exist no j, k such that gji*gki < 0 for this i, so region is not bounded -> exit loop
		}
	}

	return true;
}

double RR_EqN::fdf(std::vector<double> V_j, int J) {
	// Calculate objective function f_j of J-th phase fraction, and derivative of f_j w.r.t. v_j
	f[J] = 0.;
    double dF{ 0. };
	for (int i = 0; i < nc; i++)
	{
		double m_i = 1.;
		for (int k = 0; k < np-1; k++)
		{
			m_i += V_j[k] * (K[k*nc + i] - 1.);
		}
		f[J] += z[i] * (1.-K[J*nc+i]) / m_i; // f_j
		dF -= z[i] * std::pow(1.-K[J*nc+i], 2) / std::pow(m_i, 2.); // df_j/dv_j
	}

	return dF;
}

int RR_EqN::solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_) {
    this->init(z_, K_, nonzero_comp_);
	error_output = 0;

	// Check if domain is bounded: admissible K-values (gji*gki<0)
	// If not admissible, gives output 1
	if (!this->bounded())
	{
        std::cout << "unbounded" << std::endl;
		return 1;
	}
	// // Calculate limits of V domain
    int J = np-1;
    error_output += this->bounds(v_mid, J); // Calculate v_min and v_max of j - based on values for v_mid[j+]
    v_mid[J-1] = (v_min[J-1] + v_max[J-1]) * 0.5;

    // // Run recursive RR loop starting from NP-1 (so index=NP-2)
    error_output += this->rr_loop(J-1);

    // Update nu
    nu[0] = 1.;
	for (int j = 1; j < np; j++)
	{
		nu[j] = v_mid[j-1];
		nu[0] -= v_mid[j-1];
	}

	return error_output;
}

int RR_EqN::bounds(std::vector<double> V_j, int J) {
	// Calculate V_min and V_max from corner points of domain
	std::vector<double> Vmin(J);
	std::vector<double> Vmax(J);

	// This function calculates all intersections of planes m_i(v) and then checks which ones are actual corner points and range of the domain for V

	// Find all unique combinations of i of length NP-1
	Combinations c(nc, J);  // contains vector with (NP-1)*(number of intersections) entries, stores combinations of i for unique intersections of m_i(v)
	int ni = c.n_combinations;  // Total number of intersections between mi and mj: NC!/((NP-1)!(NC-(NP-1))!)
	int count_corners = 0;

	// Solve linear system with Eigen library to find all V_j at intersection of m_i's: Ax = b
	// m_i = 1 + sum(K_ik-1)*v_k = 0 -> sum(K_ik-1)*v_k = -1 for each m_i
	// A_ik = [K_ik-1]; b_k = [-1, ..., -1]
	Eigen::MatrixXd A(J, J);
	Eigen::VectorXd b(J);
	Eigen::VectorXd v(J);

	// If all but the three intersecting lines for m_i(v) > 0, then this intersection is a cornerpoint of the domain for V
	// Calculate m_i at intersections for all i = 0, ..., NC-1
	// std::cout << "index ";
	// for (int i = 0; i < ni; i++)
	// {
	// 	for (int j = 0; j < J; j++)
	// 	{
	// 		std::cout << c.getIndex(i, j) << " ";
	// 	}
	// }
	// std::cout << std::endl;
	// exit(1);

	for (int i = 0; i < ni; i++)
	{
		b = Eigen::VectorXd::Constant(J, -1.);

		// Fill matrix A for combination of m_i's
		for (int j = 0; j < J; j++)
		{
			int index = c.getIndex(i, j); // index returns component j for i-th combination
			for (int k = 0; k < J; k++)
			{
				// loop over phase k to fill row j
				A(j, k) = K[k*nc + index] - 1.; // A_jk = K_jk-1
			}
			for (int k = J; k < np-1; k++) {
				b(j) -= V_j[k]*(K[k*nc + index] - 1.);  // if V_j+ are known
			}
		}
		v = A.partialPivLu().solve(b);  // with LU factorization
		// print("v", v);

		// If all but the three intersecting lines for m_i(v) > 0, then this intersection is a cornerpoint of the domain for V
		// Calculate m_i at intersections for all i = 0, ..., NC-1
		std::vector<double> m_i(nc, 1.); // value of m_i's at the ni intersections = 1 + sum((K_kj-1)*v_j)
		for (int k = 0; k < nc; k++)
		{
			for (int j = 0; j < J; j++)
			{
				m_i[k] += v(j) * (K[j*nc + k]-1.); // += v_j*(K_kj-1)
				// std::cout << K[j*nc+k] << " " << v(j) * (K[j*nc + k]-1.) << std::endl;
			}
			for (int j = J; j < np-1; j++)
			{
				m_i[k] += V_j[j] * (K[j*nc + k]-1.);
			}
		}
		// Count the number of m_i's > 0
		int count = 0;
		for (int k = 0; k < nc; k++)
		{
			// std::cout << "mi[k] " << m_i[k] << std::endl;
			if (m_i[k] > 1e-15)
			{
				// std::cout << "count\n";
				count++;
			}
		}
		// print("count", count);
		if (count >= nc-J)
		{
			count_corners++;
			for (int j = 0; j < J; j++)
			{
				// std::cout << "intersect " << J << " " << j << " " << v(j) << std::endl;
				// v_j, min
				if (( v(j) < Vmin[j] ) || (count_corners == 1))
				{
					Vmin[j] = v(j);
				}
				// v_j, max
				if (( v(j) > Vmax[j] ) || (count_corners == 1))
				{
					Vmax[j] = v(j);
				}
			}
		}
	}

	if (count_corners - J < 1)
	{
		std::cout << "Not enough corner points found: unable to calculate limits in "  << J << " dimensions" << std::endl;
		return 1;
	}
	// std::cout << J << " corners " << count_corners << std::endl;
	// for (int j = 0; j < J; j++)
	// {
	// 	std::cout << j << " " << Vmin[j] << " " << Vmax[j] << std::endl;
	// }

	v_min[J-1] = Vmin[J-1];
    v_max[J-1] = Vmax[J-1];
	// if (J == 1)
	// {
	// 	exit(1);
	// }
	return 0;
}

int RR_EqN::rr_loop(int J) {
	// This function recursively calculates the value of v[J] for which f_j[v] = 0
	// It is repeated until |f_j[v]| < eps
	// If f_j for j-1 has not yet converged, the function moves down one level to find [v_j-]
	for (int iter = 0; iter < max_iter; iter++)
	{
		// If j > 0, then move down one level (j-1)
		if (J > 0)
		{
			// Guess V_mid for all j- phases
			error_output += this->bounds(v_mid, J); // Calculate Vmin and Vmax of j - based on values for V_mid[j+]

			// if previous v[j] is out of bounds, recalculate v_mid[j]. Could save iterations if not necessary
			if (!(v_min[J-1] < v_mid[J-1]) || !(v_mid[J-1] < v_max[J-1]))
			{
				v_mid[J-1] = (v_min[J-1] + v_max[J-1]) * 0.5;
			}

			error_output += this->rr_loop(J-1);
		}

		double df = this->fdf(v_mid, J);
		double f_df = f[J] / df;

		// if f[J] < 0 -> f[J] = 0 is above v[J], update v_min[J] to v_mid[J]
		if (f[J] < 0)
		{
			v_min[J] = v_mid[J];
			if (v_mid[J] - f_df > v_max[J])
			{
				// std::cout << "Correction applied, upper limit" << std::endl;
				v_mid[J] = (v_min[J] + v_max[J]) * 0.5;
			}
			else
			{
				v_mid[J] += f_df;
			}
		}
		else
		{ // else, f[J] > 0 -> f[J] = 0 is below v[J], update v_max[J] to v_mid[J]
			v_max[J] = v_mid[J];
			if (v_mid[J] - f_df < v_min[J])
			{
				// std::cout << "Correction applied, lower limit" << std::endl;
				v_mid[J] = (v_mid[J] + v_min[J]) * 0.5;
			}
			else
			{
				v_mid[J] += f_df;
			}
		}
		if (std::abs(f[J]) < rrn_tol)
		{
			// std::cout << J << " " << v_mid[J] << std::endl;
			return 0;
		}
	}
	// std::cout << "Max iter\n";
	return 1;
}
