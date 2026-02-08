//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_GLOBAL_GLOBAL_H
#define OPENDARTS_FLASH_GLOBAL_GLOBAL_H
//--------------------------------------------------------------------------

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>

#define M_NA 6.02214076e23 // Avogadro's number [mol-1]
#define M_kB 1.380649e-23 // Boltzmann constant [J/K]
#define M_R (M_kB * M_NA) // Gas constant [J/mol.K]

#define NC_MAX 10
#define NP_MAX 5

enum class StateSpecification : int { TEMPERATURE = 0, ENTHALPY, ENTROPY };

template <typename... Ts>
void not_implemented_void(Ts...)
{
    return;
}

template <typename... Ts>
double not_implemented_double(Ts...)
{
    return NAN;
}

template <typename... Ts>
std::vector<double> not_implemented_vector(size_t len, Ts...)
{
    return std::vector<double>(len, NAN);
}

template <typename T>
void print(std::string text, T var, int precision=15) {
    std::cout << text << ": " << std::setprecision(precision) << var << std::endl;
    return;
}

template <typename T>
void print(std::string text, const std::initializer_list<T>& list, int precision=15)
{
    std::cout << text << ": " << std::setprecision(precision);
    for( T elem : list )
    {
        std::cout << elem << " ";
    }
    std::cout << "\n";
    return;
}

template <typename T>
void print(std::string text, const std::vector<T>& vec, int n_rows=1, int precision=15) {
    std::cout << text << ": " << std::setprecision(precision);
    int n_cols = static_cast<int>(vec.size()) / n_rows;
    for (int i = 0; i < n_rows; i++)
    {
        for (int j = 0; j < n_cols; j++)
        {
            std::cout << vec[i*n_cols + j] << " ";
        }
        std::cout << "\n";
    }
    return;
}

template <typename T>
void print(std::string text, const std::vector<std::vector<T>>& vecvec, int n_rows=1, int precision=15) {
    std::cout << text << ":\n" << std::setprecision(precision);
    for (std::vector<T> vec : vecvec)
    {
        int n_cols = static_cast<int>(vec.size()) / n_rows;
        for (int i = 0; i < n_rows; i++)
        {
            for (int j = 0; j < n_cols; j++)
            {
                std::cout << vec[i*n_cols + j] << " ";
            }
            std::cout << "\n";
        }
    }
    return;
}

template<typename T>
std::vector<T> arange(T start, T stop, T step = 1) 
{
    // Templated function to generate arange from start to stop with specified stepsize
    std::vector<T> values;
    for (T value = start; value < stop; value += step)
    {
        values.push_back(value);
    }
    return values;
}

template<typename T>
std::vector<T> linspace(T start, T stop, int array_len) 
{
    // Templated function to generate linearly spaced array from start to stop with specified array length
    std::vector<T> values;
    T dx = (stop-start)/(array_len-1);
    for (int i = 0; i < array_len; i++)
    {
        values.push_back(start + i*dx);
    }
    return values;
}

template<typename T>
std::vector<T> logspace(T start, T stop, int array_len, T base=10.)
{
    // Function to generate array with numbers linearly scaled in logspace from start to stop with specified logbase and array length
    std::vector<T> values;    
    T log_start = std::log(start) / std::log(base);
    T log_diff = (std::log(stop) / std::log(base) - log_start) / (array_len-1);

    for (int i = 0; i < array_len; i++)
    {
        T logvalue = log_start + i * log_diff;
        values.push_back(std::pow(base, logvalue));
    }
    return values;
}

bool compare_compositions(std::vector<double>& X0, std::vector<double>& X1, double tol);

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_GLOBAL_GLOBAL_H
//--------------------------------------------------------------------------
