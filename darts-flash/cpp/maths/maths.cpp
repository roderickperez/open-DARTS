#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>

#include "dartsflash/global/global.hpp"
#include "dartsflash/maths/maths.hpp"

namespace maths
{
    // Precompute factorials of integers
    std::vector<int> factorial = {Factorial<0>::value, Factorial<1>::value, Factorial<2>::value, Factorial<3>::value, Factorial<4>::value, Factorial<5>::value,
                                  Factorial<6>::value, Factorial<7>::value, Factorial<8>::value, Factorial<9>::value, Factorial<10>::value, Factorial<11>::value,
                                  };

    // Precompute sets of combinations of integers
    // std::vector<std::vector<std::vector<std::vector<int>>>> combinations = get_combinations(NC_MAX, NP_MAX);
}

std::vector<std::complex<double>> cubic_roots_analytical(double a2, double a1, double a0) {
    // Analytical solution of roots for normalized cubic polynomial
    // Following McNamee and Pan (2013) - Numerical Methods for Roots of Polynomials, Chapter 12
    // f(Z) = Z^3 + a2 Z^2 + a1 Z + a0 = 0
    double P = a1 - std::pow(a2, 2) / 3;
    double Q = a0 - a2 * a1 / 3 + 2 * std::pow(a2, 3) / 27;
    double D_ = std::pow(P, 3) / 27 + std::pow(Q, 2) / 4;

    std::vector<std::complex<double>> ZZ(3);

    if (D_ > 0.) 
    {
        // Cardano's method
        double uu = std::cbrt(-Q/2 + std::sqrt(D_));
        double vv = std::cbrt(-Q/2 - std::sqrt(D_));
        std::complex<double> w(-0.5, 0.5*std::sqrt(3.));
        std::complex<double> w2 = std::conj(w);

        ZZ[0] = uu + vv - a2 / 3;
        ZZ[1] = w*uu + w2*vv - a2 / 3;
        ZZ[2] = w2*uu + w*vv - a2 / 3;
    } 
    else 
    {
        // Trigonomic method
        for (int k = 0; k < 3; k++) 
        {
            double theta = std::acos(-Q/2 * std::sqrt(-27. / std::pow(P, 3)));
            ZZ[k] = -a2 / 3 + 2 * std::sqrt(-P / 3) * std::cos((theta + 2.*M_PI*k) / 3);
        }
    }

    return ZZ;
}

std::vector<std::complex<double>> cubic_roots_iterative(double a2, double a1, double a0, double tol, int max_iter)
{
    // Iterative solution of roots for normalized cubic polynomial, Deiters (2014)
    // f(Z) = Z^3 + a2 Z^2 + a1 Z + a0 = 0
    std::vector<std::complex<double>> ZZ(3);

    // Find inflection point Z_infl and discriminant d
    double Z_infl = -a2 / 3;
    double d = std::pow(a2, 2) - 3.*a1;

    // Based on discriminant value, determine the initial guess for first root
    double Z1;
    if (d > 0.)
    {
        // Cubic polynomial has a maximum and a minimum (doesn't mean that three real roots exist, however)
        // Based on the sign of the function f(Z_infl), we choose the initial guess for finding one root to be Z_low or Z_high
        // This ensures that there is no extremum between the initial guess and the nearest root
        double f_infl = std::pow(Z_infl, 3) + a2 * std::pow(Z_infl, 2) + a1 * Z_infl + a0;
        if (f_infl > 0.)
        {
            // Inflection point above f = 0 -> choose Z_low
            Z1 = Z_infl - 2. / 3 * std::sqrt(d);
        }
        else if (f_infl < 0.)
        {
            // Inflection point below f = 0 -> choose Z_high
            Z1 = Z_infl + 2. / 3 * std::sqrt(d);
        }
        else
        {
            // f(Z_infl) = 0 and the inflection point is itself one of the roots
            Z1 = Z_infl;
        }
    }
    else if (d < 0.)
    {
        // Cubic polynomial does not have extrema and inflection point can be used as initial guess
        Z1 = Z_infl;
    }
    else
    {
        // If D = 0, the inflection point has a slope equal to zero and the inflection point is readily obtained
        double f_infl = std::pow(Z_infl, 3) + a2 * std::pow(Z_infl, 2) + a1 * Z_infl + a0;
        Z1 = Z_infl - std::cbrt(f_infl);
    }
    
    // If we need to iterate to find the solution, use Halley's method
    int it = 0;
    while (it < max_iter)
    {
        it++;
        double fz = std::pow(Z1, 3) + a2 * std::pow(Z1, 2) + a1 * Z1 + a0;
        double df = 3. * std::pow(Z1, 2) + 2. * a2 * Z1 + a1;
        double d2f = 6. * Z1 + 2. * a2;
        double dZ = fz * df / (std::pow(df, 2) - 0.5 * fz * d2f);
        Z1 -= dZ;

        if (std::fabs(dZ) < tol)
        {
            break;
        }
        else if (it == max_iter)
        {
            print("Max iter for iterative cubic root finding", dZ);
        }
    }
    ZZ[0] = Z1;

    // Having found the first root, the other two can be obtained analytically
    // Determine discriminant d of 'deflated' polynomial
    double c1 = Z1 + a2;
    double c2 = c1 * Z1 + a1;
    d = std::pow(c1, 2) * 0.25 - c2;
    if (d >= 0.)
    {
        // Second and third roots are real
        ZZ[1] = -c1 * 0.5 - std::sqrt(d);
        ZZ[2] = -c1 * 0.5 + std::sqrt(d);
    }
    else
    {
        // Second and third roots are complex conjugates
        ZZ[1] = std::complex<double>(-c1 * 0.5, std::sqrt(-d));
        ZZ[2] = std::conj(ZZ[1]);
    }

    return ZZ;
}

Combinations::Combinations(int n_elem, int comb_length)
{
	// https://www.geeksforgeeks.org/print-all-possible-combinations-of-r-elements-in-a-given-array-of-size-n/
    if (n_elem < comb_length) { std::cout << "Invalid combination requested! " << n_elem << " < " << comb_length << "\n"; exit(1); }
	n_elements = n_elem; combination_length = comb_length;
    n_combinations = maths::factorial[n_elements]/(maths::factorial[combination_length]*maths::factorial[n_elements-combination_length]);

    combination = std::vector<int>(n_elements);
	for (int i = 0; i < n_elements; i++)
	{
		combination[i] = i;   // array with component numbers for each m_i(v)
	}

    if (n_combinations > 1)
    {
        combinations.resize(n_combinations);
	    std::vector<int> temp(combination_length, 0);
	    this->unique_combinations(temp, 0, n_elements-1, 0);
    }
    else
    {
        combinations = std::vector<std::vector<int>>{combination};
    }
}

void Combinations::unique_combinations(std::vector<int> temp, int start, int end, int index)
{
	/* std::vector<int> arr ---> Input Array
	std::vector<int> data ---> Temporary array to store current combination
	start & end ---> Starting and ending indexes in arr
	index ---> Current index in data
	r ---> Size of a combination to be printed */
	// This code is contributed by rathbhupendra
	if (index == combination_length)
	{
        combinations[j] = temp;
        j++;
		return;
	}

	// replace index with all possible elements. The condition "end-i+1 >= r-index" makes sure
	// that including one element at index will make a combination with remaining elements at remaining positions
	for (int i = start; i <= end && end - i + 1 >= combination_length - index; i++)
	{
		temp[index] = combination[i];
		this->unique_combinations(temp, i+1, end, index+1);
	}
}

std::vector<std::vector<std::vector<std::vector<int>>>> get_combinations(int max_elem, int max_len)
{
    std::vector<std::vector<std::vector<std::vector<int>>>> combinations(max_elem + 1);
    for (int n_elem = 1; n_elem <= max_elem; n_elem++)
    {
        combinations[n_elem] = std::vector<std::vector<std::vector<int>>>(max_len + 1);
        for (int comb_len = 1; comb_len <= max_len; comb_len++)
        {
            if (comb_len <= n_elem)
            {
                Combinations c(n_elem, comb_len);
                combinations[n_elem][comb_len] = c.combinations;
            }
        }
    }
    return combinations;
}
