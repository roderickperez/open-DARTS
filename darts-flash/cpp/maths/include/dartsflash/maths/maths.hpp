//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_MATHS_MATHS_H
#define OPENDARTS_FLASH_MATHS_MATHS_H
//--------------------------------------------------------------------------

#include <vector>
#include <complex>
#include <map>

namespace maths {
    extern std::vector<int> factorial;
    // extern std::vector<std::vector<std::vector<std::vector<int>>>> combinations;
}

template <int n>
struct Factorial
{
    enum { value = n * Factorial<n-1>::value };
};

template <>
struct Factorial<0>
{
    enum { value = 1 };
};

std::vector<std::complex<double>> cubic_roots_analytical(double a2, double a1, double a0);
std::vector<std::complex<double>> cubic_roots_iterative(double a2, double a1, double a0, double tol, int max_iter=50);

// Class to find set of combinations of integer indices from set of n_elem idxs and length comb_length
struct Combinations
{
    std::vector<int> combination;
    std::vector<std::vector<int>> combinations;
    int n_combinations, n_elements, combination_length; // number of combinations, number of unique elements, length of combinations
    int j = 0;

    Combinations(int n_elem, int comb_length);
    int getIndex(int i, int k) { return combinations[i][k]; }

    void unique_combinations(std::vector<int> temp, int start, int end, int index);
};

std::vector<std::vector<std::vector<std::vector<int>>>> get_combinations(int max_elem, int max_len);

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_MATHS_MATHS_H
//--------------------------------------------------------------------------
