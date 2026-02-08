#include <limits>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <cassert>

#include "dartsflash/maths/modifiedcholeskys99.hpp"
// -------------------------------------------
// ----- Constructor -------------------------
// -------------------------------------------
ModifiedCholeskyS99::ModifiedCholeskyS99()
{
    m_indnonpositive = -1;
    m_valnpositive = 0.0;
}

// ------------------------------------------
// ----- Destructor -------------------------
// ------------------------------------------
ModifiedCholeskyS99::~ModifiedCholeskyS99()
{
}

// ------------------------------------------
// ----- Initialize -------------------------
// ------------------------------------------
int ModifiedCholeskyS99::initialize(Eigen::MatrixXd &A, int meth)
{
    // Structure conversion
	m_n = A.rows();
    m_el = Eigen::MatrixXd::Zero(m_n, m_n);

	m_pivot.resize(m_n);
	std::iota(m_pivot.begin(), m_pivot.end(), 0);

	Eigen::MatrixXd A2 = A;

	double delta = 0.0;
	double deltaprev = 0.0;

	double macheps = std::numeric_limits<double>::epsilon();

	double tau    = std::pow(macheps, 1./3.);
	double taubar = std::pow(macheps, 2./3.);
	double mu = 0.1;

	bool phaseone = true;

	double maxx = 0.0;
	double minn = 0.0;

	std::vector<double> ee(m_n, 0.0);

	std::vector<double> g(m_n, 0.0);

	double gamma = std::fabs(A2(0, 0));
	for (int i = 1; i < m_n; i++)
	{
		if (gamma < std::fabs(A2(i, i))) 
		{
			gamma = std::fabs(A2(i, i));
		}
	}

	int j = 0;

	while (j < m_n && phaseone)
	{
		minn = 1e+80;
		maxx = -1e+80;
		int indmax = j;

		for (int i = j; i < m_n; i++)
		{
			if (maxx < A2(i, i))
			{
				maxx = A2(i, i);
				indmax = i;
			}

			if (minn > A2(i, i)) 
			{
				minn = A2(i, i);
			}
		}
		if (maxx < taubar*gamma || minn < -mu*maxx)
		{
			phaseone = false;
		}
		else
		{
			// Pivot
			if (indmax != j)
			{
				int l = m_pivot[indmax];
				m_pivot[indmax] = m_pivot[j];
				m_pivot[j] = l;

				for (int i = 0; i < m_n; i++)
				{
					double temp = A2(i, j);
					A2(i, j) = A2(i, indmax);
					A2(i, indmax) = temp;
				}

				for (int i = 0; i < m_n; i++)
				{
					double temp = m_el(j, i);
					m_el(j, i) = m_el(indmax, i);
					m_el(indmax, i) = temp;
				}

				for (int i = 0; i < m_n; i++)
				{
					double temp = A2(j, i);
					A2(j, i) = A2(indmax, i);
					A2(indmax, i) = temp;
				}
			}

			minn = 1e+80;
			for (int i = j+1; i < m_n; i++)
			{
				if (minn > A2(i, i) - A2(i, j) / A2(j, j) * A2(i, j)) 
				{
					minn = A2(i, i) - A2(i, j) / A2(j, j) * A2(i, j);
				}
			}

			if (minn < -mu*tau)
			{
				phaseone = false;
			}
			else
			{
				m_el(j, j) = std::sqrt(A2(j, j));
				for (int i = j + 1; i < m_n; i++)
				{
					m_el(i, j) = A2(i, j) / m_el(j, j);
				}

				for (int i = j + 1; i < m_n; i++)
				{
					for (int k = j + 1; k < m_n; k++)
					{
						A2(i, k) = A2(i, k) - m_el(i, j)*m_el(k, j);
					}
				}
				j++;
			}
		}
	}

	// Phasetwo
	if (!phaseone && j == m_n)
	{
		delta = -A2(m_n - 1, m_n - 1) + std::max(-tau*A2(m_n - 1, m_n - 1) / (1.0 - tau), taubar*gamma);
		A2(m_n - 1, m_n - 1) += delta;
		m_el(m_n - 1, m_n - 1) = std::sqrt(A2(m_n - 1, m_n - 1));
		ee[m_n - 1] = delta;
	}
	else if (!phaseone && j < m_n)
	{
		int k = j - 1; // Number of iterations perfoemed in phase one

		// Calculate lower Gershgorin bounds of A(k+1)
		for (int i = j; i < m_n; i++)
		{
			double sum1 = 0.0;
			for (int l = j; l <= i - 1; l++) 
			{
				sum1 += std::fabs(A2(i, l));
			}
			double sum2 = 0.0;
			for (int l = i + 1; l < m_n; l++)
			{
				sum2 += std::fabs(A2(l, i));
			}
			g[i] = A2(i, i) - sum1 - sum2;
		}

		// Modified CholeskyB Decomposition
		for (j = k + 1; j < m_n - 2; j++)
		{
			maxx = -1e+20;
			int indmax = -1;
			for (int l = j; l < m_n; l++)
			{
				if (maxx < g[l])
				{
					indmax = l;
					maxx = g[l];
				}
			}

			// Pivot on maximum lower Gerschgorin bound estimate
			if (indmax != j)
			{
				int l = m_pivot[indmax];
				m_pivot[indmax] = m_pivot[j];
				m_pivot[j] = l;

				for (int i = 0; i < m_n; i++)
				{
					double temp = A2(i, j);
					A2(i, j) = A2(i, indmax);
					A2(i, indmax) = temp;
				}

				for (int i = 0; i < m_n; i++)
				{
					double temp = m_el(j, i);
					m_el(j, i) = m_el(indmax, i);
					m_el(indmax, i) = temp;
				}

				for (int i = 0; i < m_n; i++)
				{
					double temp = A2(j, i);
					A2(j, i) = A2(indmax, i);
					A2(indmax, i) = temp;
				}

				double temp = g[indmax];
				g[indmax] = g[j];
				g[j] = temp;
			}
			// Calculate Ejj and add to diagonal
			double sum1 = 0.0;
			for (int i = j + 1; i < m_n; i++) 
			{
				sum1 += std::fabs(A2(i, j));
			}
			delta = std::max(std::max(0.0, -A2(j, j) + std::max(sum1, taubar*gamma)), deltaprev);

			if (delta > 0.0)
			{
				A2(j, j) += delta;
				deltaprev = delta;
				ee[j] = delta;
			}

			// Update Gerschgorin bound estimates
			if (std::fabs(A2(j, j) - sum1) > 1e-16)
			{
				double sum2 = -1.0 + sum1 / A2(j, j);
				for (int i = j + 1; i < m_n; i++) 
				{
					g[i] += std::fabs(A2(i, j)) * sum2;
				}
			}
			// Perform the jth iteration of factorization
			m_el(j, j) = std::sqrt(A2(j, j));
			for (int i = j + 1; i < m_n; i++)
			{
				m_el(i, j) = A2(i, j) / m_el(j, j);
			}

			for (int i = j + 1; i < m_n; i++)
			{
				for (k = j + 1; k < m_n; k++)
				{
					A2(i, k) = A2(i, k) - m_el(i, j) * m_el(k, j);
				}
			}
		}

		// Final 2*2 submatrix
		double b = A2(m_n - 2, m_n - 2) - A2(m_n - 1, m_n - 1); // (In fact -b)
		double c = A2(m_n - 1, m_n - 2) * A2(m_n - 1, m_n - 2); // In fact -c
		double dd;

		if(meth == 1)
		{
			// Stability
			dd = A2(m_n - 2, m_n - 2) - A2(m_n - 1, m_n - 1);
		}
		else
		{
			// Flash
			dd = A2(m_n - 2, m_n - 2) + A2(m_n - 1, m_n - 1);
		}

		delta = b*b + 4*c;

		double lambdal, lambdah;
		if (delta <= 0.0)
		{
			lambdal = dd / 2.0;
			lambdah = dd / 2.0;
		}
		else
		{
			lambdal = (dd - std::sqrt(delta))/2.0;
			lambdah = (dd + std::sqrt(delta))/2.0;
		}

		delta = std::max(std::max(0.0, -lambdal + tau*std::max((lambdah - lambdal) / (1.0 - tau), taubar*gamma)), deltaprev);
		// delta = std::max(std::max(0.0, -lambdal + std::max(tau*(lambdah - lambdal) / (1.0 - tau), taubar*gamma)), deltaprev);
		if (delta > 0.0)
		{
			A2(m_n - 2, m_n - 2) += delta;
			A2(m_n - 1, m_n - 1) += delta;
			deltaprev = delta;
			ee[m_n - 2] = delta;
			ee[m_n - 1] = delta;
		}
		m_el(m_n - 2, m_n - 2) = std::sqrt(A2(m_n - 2, m_n - 2));
		m_el(m_n - 1, m_n - 2) = A2(m_n - 1, m_n - 2)/m_el(m_n - 2, m_n - 2);
		m_el(m_n - 1, m_n - 1) = std::sqrt(A2(m_n - 1, m_n - 1) - m_el(m_n - 1, m_n - 2) * m_el(m_n - 1, m_n - 2));
	}

	m_piv = Eigen::MatrixXd::Zero(m_n, m_n);
	for (int i = 0; i < m_n; i++)
	{
		m_piv(m_pivot[i], i) = 1.0;
	}

    m_elinvii.resize(m_n);
    for (int i = 0; i < m_n; i++)
    {
        m_elinvii[i] = 1.0/m_el(i, i);
		for (j = 0; j < i; j++)
        {
            m_el(j, i) = 0.0;
        }
    }
    m_isposdef = true;
    return 0;
}


// -------------------------------------
// ----- Solve -------------------------
// -------------------------------------
int ModifiedCholeskyS99::solve(Eigen::VectorXd& b_in, Eigen::VectorXd& x_out)
{
	// int error = 0;
	assert((b_in.size() == this->m_n) && "Length of b vector not equal to n\n");
	assert((x_out.size() == this->m_n) && "Length of x vector not equal to n\n");

	std::vector<double> b(m_n, 0.0);
	std::vector<double> z(m_n, 0.0);

	for (int i = 0; i < m_n; i++)
	{
		b[i] = 0.0;
		for (int k = 0; k < m_n; k++)
		{
			b[i] += m_piv(k, i) * b_in(k);
		}
	}

    //solve Ly=b
    for (int i = 0; i < m_n; i++)
    {
        double sum = b[i];
        for (int k = i-1; k >= 0; k--)
        {
            sum -= m_el(i, k)*z[k];
        }
        z[i] = sum*m_elinvii[i];
    }

    //solve L'x=y
    for (int i = m_n - 1; i >= 0; i--)
    {
        double sum = z[i];
        for (int k = i+1; k < m_n; k++)
        {
            sum -= m_el(k, i)*z[k];
        }
        z[i] = sum * m_elinvii[i];
    }

	for (int i = 0; i < m_n; i++)
	{
		x_out(i) = 0.0;
		for (int k = 0; k < m_n; k++)
		{
			x_out(i) += m_piv(i, k) * z[k];
			if(std::isnan(x_out(i))){return 1;}
		}
	}

	return 0;
}
