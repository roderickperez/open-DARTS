#include <complex>
#include <math.h>
#include <chrono>
#include "dartsflash/global/global.hpp"
#include "dartsflash/maths/geometry.hpp"
#include "dartsflash/maths/maths.hpp"
#include "dartsflash/maths/modifiedcholeskys99.hpp"
#include "dartsflash/maths/linesearch.hpp"
#include "dartsflash/maths/root_finding.hpp"

int test_combinations();
int test_simplex();
int test_function_pass();
int test_rootfinding();
int test_line_search();
int test_cubic_roots();

int main() 
{
    int error_output = 0;

    error_output += test_combinations();
    error_output += test_simplex();
    error_output += test_function_pass();
    error_output += test_rootfinding();
    error_output += test_line_search();
    error_output += test_cubic_roots();

    return error_output;
}

int test_combinations()
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error_output = 0;

    std::vector<std::pair<Combinations, std::vector<std::vector<int>>>> combinations = {
        {Combinations(2, 2), {{0, 1}}},
        {Combinations(3, 2), {{0, 1}, {0, 2}, {1, 2}}},
        {Combinations(4, 2), {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}}},
        {Combinations(3, 3), {{0, 1, 2}}},
        {Combinations(4, 3), {{0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}}},
        {Combinations(5, 3), {{0, 1, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 3}, {0, 2, 4}, {0, 3, 4}, {1, 2, 3}, {1, 2, 4}, {1, 3, 4}, {2, 3, 4}}},
        {Combinations(4, 4), {{0, 1, 2, 3}}},
        {Combinations(5, 4), {{0, 1, 2, 3}, {0, 1, 2, 4}, {0, 1, 3, 4}, {0, 2, 3, 4}, {1, 2, 3, 4}}},
    };

    for (auto it: combinations)
    {
        for (size_t i = 0; i < it.first.combinations.size(); i++)
        {
            std::vector<int> combination = it.first.combinations[i];
            for (size_t ii = 0; ii < combination.size(); ii++)
            {
                if (combination[ii] != it.second[i][ii])
                {
                    error_output++;
                    std::cout << "Different combinations\n";
        			print("n_elements, combination_length", std::vector<int>{it.first.n_elements, it.first.combination_length});
        			print("combinations", it.first.combinations);
		        	print("ref", it.second);
                    break;
                }
            }
        }
    }
    
    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_combinations(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_combinations(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	return error_output;
}

int test_simplex()
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error_output = 0;

    // Is between two points? (2D simplex/line)
    std::vector<std::vector<double>> coords = {{0.}, {1.}};
    std::vector<std::vector<double>> points = {{0.2}, {1.}, {1.2}};
    std::vector<bool> in_simplex = {true, true, false};
    for (size_t i = 0; i < points.size(); i++)
    {
        if (is_in_simplex(points[i], coords) != in_simplex[i])
        {
            error_output++;
            std::cout << (!in_simplex[i] ? "Point is in 2D simplex (line), but shouldn't be" : "Point isn't in 2D simplex (line), but should be") << "\n";
            print("Point", points[i]);
        }
    }

    coords = {{0.5}, {2.}};
    points = {{0.2}, {0.5}, {1.}, {2.}, {3.}};
    in_simplex = {false, true, true, true, false};
    for (size_t i = 0; i < points.size(); i++)
    {
        if (is_in_simplex(points[i], coords) != in_simplex[i])
        {
            error_output++;
            std::cout << (!in_simplex[i] ? "Point is in 2D simplex (line), but shouldn't be" : "Point isn't in 2D simplex (line), but should be") << "\n";
            print("Point", points[i]);
        }
    }

    // Is in triangle? (3D simplex/triangle)
    coords = {{0., 0.}, {0., 1.}, {1., 0.}};
    points = {{0.2, 0.2}, {0., 1.}, {0.5, 0.5}, {1., 1.}};
    in_simplex = {true, true, true, false};
    for (size_t i = 0; i < points.size(); i++)
    {
        if (is_in_simplex(points[i], coords) != in_simplex[i])
        {
            error_output++;
            std::cout << (!in_simplex[i] ? "Point is in 3D simplex (triangle), but shouldn't be" : "Point isn't in 3D simplex (triangle), but should be") << "\n";
            print("Point", points[i]);
        }
    }

    coords = {{1., 4.}, {6., 3.}, {7.1, 6.5}};
    points = {{1., 4.}, {2., 4.2}, {3.5, 3.5}, {0.2, 0.2}, {7.1-1e-12, 6.5}};
    in_simplex = {true, true, true, false, false};
    for (size_t i = 0; i < points.size(); i++)
    {
        if (is_in_simplex(points[i], coords) != in_simplex[i])
        {
            error_output++;
            std::cout << (!in_simplex[i] ? "Point is in 3D simplex (triangle), but shouldn't be" : "Point isn't in 3D simplex (triangle), but should be") << "\n";
            print("Point", points[i]);
        }
    }

    // Is in tetrahedron? (4D simplex/tetrahedron)
    coords = {{0., 0., 0.}, {0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}};
    points = {{0.2, 0.2, 0.2}, {0.25, 0.25, 0.25}, {0., 1., 0.}, {1., 1., 0.}, {0., 1., 1.}, {0.5, 0.5, 0.5}, {1., 1., 1.}};
    in_simplex = {true, true, true, false, false, false, false};
    for (size_t i = 0; i < points.size(); i++)
    {
        if (is_in_simplex(points[i], coords) != in_simplex[i])
        {
            error_output++;
            std::cout << (!in_simplex[i] ? "Point is in 4D simplex (tetrahedron), but shouldn't be" : "Point isn't in 4D simplex (tetrahedron), but should be") << "\n";
            print("Point", points[i]);
        }
    }

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_simplex(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_simplex(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_cubic_roots()
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error_output = 0;

    double a2{ -0.92455459054338163 }, a1{ 0.27900309322530026 }, a0{ -0.027600384437325549 };
    a2 = -1;
    a1 = 0.1356257109;
    a0 = -0.006523118931;
    a2 = -0.9222039261;
    a1 = 0.2834866938;
    a0 = -0.02904806022;
    std::vector<std::complex<double>> Z_analytical = cubic_roots_analytical(a2, a1, a0);
    std::vector<std::complex<double>> Z_iterative = cubic_roots_iterative(a2, a1, a0, 1e-14);
    (void) Z_analytical;
    (void) Z_iterative;

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_cubic_roots(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_cubic_roots(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

double fn_pointer(std::function<double(double, double)> fun, double a, double b) { return fun(a, b); }
double add(double a, double b) { return a+b; }
int test_function_pass()
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error_output = 0;
    auto f = std::bind(&add, std::placeholders::_1, std::placeholders::_2);
    error_output += (fn_pointer(f, 1., 1.5) == 2.5) ? 0 : 1;
    error_output += (fn_pointer(f, -1., 3.2) == 2.2) ? 0 : 1;

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_function_pass(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_function_pass(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

struct RootFindingTest
{
    int test_case;
    double n;

    RootFindingTest(int test_case_, double n_=0.) : test_case(test_case_), n(n_) { }

    double evaluate(double x)
    {
        double val = 0.;
        switch (this->test_case)
		{
			case 0:
			{
                val = std::sin(x) - 0.5 * x;
				break;
			}
            case 1:
            {
                for (double i = 0; i < 20; i++)
                {
                    val -= 2. * std::pow(2. * (i+1.) - 5., 2) / std::pow(x - std::pow(i+1., 2), 3);
                }
                break;
            }
            case 2:
            {
                double a{ -40. }, b{ -1. };
                val = a * x * std::exp(b * x);
                break;
            }
            case 3:
            {
                double a{ -100. }, b{ -2. };
                val = a * x * std::exp(b * x);
                break;
            }
            case 4:
            {
                double a{ -200. }, b{ -3. };
                val = a * x * std::exp(b * x);
                break;
            }
            case 5:
            {
                double a{ 0.2 };
                val = std::pow(x, n) - a;
                break;
            }
            case 6:
            {
                double a{ 1. };
                val = std::pow(x, n) - a;
                break;
            }
            case 7:
            {
                val = std::sin(x) - 0.5;
                break;
            }
            case 8:
            {
                val = 2. * x * std::exp(-n) - 2. * std::exp(-n * x) + 1.;
                break;
            }
            case 9:
            {
                val = (1.+std::pow(1.-n, 2)) * x - std::pow(1.-n*x, 2);
                break;
            }
            case 10:
            {
                val = std::pow(x, 2) - std::pow(1.-x, n);
                break;
            }
            case 11:
            {
                val = (1.+std::pow(1.-n, 4)) * x - std::pow(1.-n*x, 4);
                break;
            }
            case 12:
            {
                val = std::exp(-n*x) * (x-1.) + std::pow(x, n);
                break;
            }
            case 13:
            {
                val = (n*x - 1.)/((n-1.)*x);
                break;
            }
            case 14:
            {
                val = (x == 0.) ? 0. : x * std::exp(-std::pow(x, 2));
                break;
            }
            case 15:
            {
                val = (x > 0.) ? n/20. * (x/1.5 + std::sin(x) - 1.) : -n/20.;
                break;
            }
            case 16:
            {
                if (x < 0.)
                {
                    val = -0.859;
                }
                else if (x < 2.e-3/(1.+n))
                {
                    val = std::exp((n+1.) * x/(2.e3));
                }
                else
                {
                    val = std::exp(1.) - 1.859;
                }
                break;
            }
			default:
        	{
            	std::cout << "Invalid RootFindingTest::test_case specified\n";
                exit(1);
    	    }
		}
        return val;
    }
    double gradient(double x)
    {
        double grad = 0.;
        switch (this->test_case)
		{
			case 0:
			{
                grad = std::cos(x) - 0.5;
				break;
			}
            case 1:
            {
                for (double i = 0; i < 20; i++)
                {
                    grad += 6. * std::pow(2 * (i+1.) - 5., 2) / std::pow(x - std::pow(i+1., 2), 4);
                }
                break;
            }
            case 2:
            {
                double a{ -40. }, b{ -1. };
                grad = a * std::exp(b * x) + a * b * x * std::exp(b * x);
                break;
            }
            case 3:
            {
                double a{ -100. }, b{ -2. };
                grad = a * std::exp(b * x) + a * b * x * std::exp(b * x);
                break;
            }
            case 4:
            {
                double a{ -200. }, b{ -3. };
                grad = a * std::exp(b * x) + a * b * x * std::exp(b * x);
                break;
            }
            case 5:
            {
                grad = n * std::pow(x, n-1.);
                break;
            }
            case 6:
            {
                grad = n * std::pow(x, n-1.);
                break;
            }
            case 7:
            {
                grad = std::cos(x);
                break;
            }
            case 8:
            {
                grad = 2. * std::exp(-n) + 2. * n * std::exp(-n * x);
                break;
            }
            case 9:
            {
                grad = 1.+std::pow(1.-n, 2) + 2. * n * (1.-n*x);
                break;
            }
            case 10:
            {
                grad = 2. * x + n * std::pow(1.-x, n-1);
                break;
            }
            case 11:
            {
                grad = 1.+std::pow(1.-n, 4) + 4. * n * std::pow(1.-n*x, 3);
                break;
            }
            case 12:
            {
                grad = -n * std::exp(-n*x) * (x-1.) + std::exp(-n*x) + n * std::pow(x, n-1.);
                break;
            }
            case 13:
            {
                grad = n/((n-1.)*x) - (n-1.) * (n*x - 1.)/std::pow((n-1.)*x, 2);
                break;
            }
            case 14:
            {
                grad = (x == 0.) ? 0. : std::exp(-std::pow(x, 2)) + 2 * std::pow(x, 2) * std::exp(-std::pow(x, 2));
                break;
            }
            case 15:
            {
                grad = (x > 0.) ? n/20. * (1./1.5 + std::cos(x)) : 0.;
                break;
            }
            case 16:
            {
                if (x < 0.)
                {
                    grad = 0.;
                }
                else if (x < 2.e-3/(1.+n))
                {
                    grad = (n+1.)/(2.e3) * std::exp((n+1.) * x/(2.e3));
                }
                else
                {
                    grad = 0.;
                }
                break;
            }
			default:
        	{
            	std::cout << "Invalid RootFindingTest::test_case specified\n";
                exit(1);
    	    }
		}
        return grad;
    }
};
struct Polynomial
{
    std::vector<double> a, b;
    double c;

    Polynomial(std::vector<double> a_) : a(a_), b(a_), c(0.) { }
    Polynomial(std::vector<double> a_, std::vector<double> b_, double c_) : a(a_), b(b_), c(c_) { }

    double evaluate(double x)
    {
        std::vector<double> d = (x < c) ? a : b;

        double y = 0.;
        int order = static_cast<int>(d.size()-1);
        for (int i = 0; i <= order; i++)
        {
            y += d[i] * std::pow(x, order-i);
        }
        return y;
    }
    double gradient(double x)
    {
        std::vector<double> d = (x < c) ? a : b;

        double dy = 0.;
        int order = static_cast<int>(d.size()-1);
        for (int i = 0; i <= order-1; i++)
        {
            dy += (order-i) * d[i] * std::pow(x, order-i-1);
        }
        return dy;
    }
};
int test_rootfinding()
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error_output = 0;

    // Test root finding methods by means of examples presented in Alefeld, Potra, Shi (M.Com 1993), Table 5.2
    std::vector<std::tuple<RootFindingTest, double, double, double>> rootfindingtests = {
        // First 
        {RootFindingTest(0), 1.895494267, M_PI/2., M_PI},  // RootFindingTest, solution, lower bound, upper bound
        
        {RootFindingTest(1), 3.022915347, std::pow(1., 2) + 1e-9, std::pow(2., 2) - 1e-9},  // n = 1
        {RootFindingTest(1), 6.683753561, std::pow(2., 2) + 1e-9, std::pow(3., 2) - 1e-9},  // n = 2
        {RootFindingTest(1), 11.23870166, std::pow(3., 2) + 1e-9, std::pow(4., 2) - 1e-9},  // n = 3
        {RootFindingTest(1), 19.67600008, std::pow(4., 2) + 1e-9, std::pow(5., 2) - 1e-9},  // n = 4
        {RootFindingTest(1), 29.82822733, std::pow(5., 2) + 1e-9, std::pow(6., 2) - 1e-9},  // n = 5
        {RootFindingTest(1), 41.9061162, std::pow(6., 2) + 1e-9, std::pow(7., 2) - 1e-9},  // n = 6
        {RootFindingTest(1), 55.9535958, std::pow(7., 2) + 1e-9, std::pow(8., 2) - 1e-9},  // n = 7
        {RootFindingTest(1), 71.98566559, std::pow(8., 2) + 1e-9, std::pow(9., 2) - 1e-9},  // n = 8
        {RootFindingTest(1), 90.00886854, std::pow(9., 2) + 1e-9, std::pow(10., 2) - 1e-9},  // n = 9
        {RootFindingTest(1), 110.0265327, std::pow(10., 2) + 1e-9, std::pow(11., 2) - 1e-9},  // n = 10
        {RootFindingTest(1), 132.0405517, std::pow(11., 2) + 1e-9, std::pow(12., 2) - 1e-9},  // n = 11
        {RootFindingTest(1), 156.0521144, std::pow(12., 2) + 1e-9, std::pow(13., 2) - 1e-9},  // n = 12
        {RootFindingTest(1), 182.0620604, std::pow(13., 2) + 1e-9, std::pow(14., 2) - 1e-9},  // n = 13
        {RootFindingTest(1), 210.071101, std::pow(14., 2) + 1e-9, std::pow(15., 2) - 1e-9},  // n = 14
        {RootFindingTest(1), 240.0800483, std::pow(15., 2) + 1e-9, std::pow(16., 2) - 1e-9},  // n = 15
        {RootFindingTest(1), 272.0902669, std::pow(16., 2) + 1e-9, std::pow(17., 2) - 1e-9},  // n = 16
        {RootFindingTest(1), 306.1051233, std::pow(17., 2) + 1e-9, std::pow(18., 2) - 1e-9},  // n = 17
        {RootFindingTest(1), 342.1369454, std::pow(18., 2) + 1e-9, std::pow(19., 2) - 1e-9},  // n = 18
        {RootFindingTest(1), 380.2687097, std::pow(19., 2) + 1e-9, std::pow(20., 2) - 1e-9},  // n = 19
        
        {RootFindingTest(2), 31., -9., 31.},
        {RootFindingTest(3), 21.26190621, -9., 31.},
        {RootFindingTest(4), 21.17187495, -9., 31.},
        
        {RootFindingTest(5, 4.), 0.668740305, 0., 5.},  // n = 4
        {RootFindingTest(5, 6.), 0.7647244913, 0., 5.},  // n = 6
        {RootFindingTest(5, 8.), 0.817765434, 0., 5.},  // n = 8
        {RootFindingTest(5, 10.), 0.8513399225, 0., 5.},  // n = 10
        {RootFindingTest(5, 12.), 0.8744852722, 0., 5.},  // n = 12

        {RootFindingTest(6, 8.), 1., 0.95, 4.05},  // n = 8
        {RootFindingTest(6, 10.), 1., 0.95, 4.05},  // n = 10
        {RootFindingTest(6, 12.), 1., 0.95, 4.05},  // n = 12
        {RootFindingTest(6, 14.), 1., 0.95, 4.05},  // n = 14

        {RootFindingTest(7), 0.5235987756, 0., 1.5},
        
        {RootFindingTest(8, 1.), 0.4224777096, 0., 1.},  // n = 1
        {RootFindingTest(8, 2.), 0.3066994105, 0., 1.},  // n = 2
        {RootFindingTest(8, 3.), 0.2237054577, 0., 1.},  // n = 3
        {RootFindingTest(8, 4.), 0.1717191475, 0., 1.},  // n = 4
        {RootFindingTest(8, 5.), 0.1382571551, 0., 1.},  // n = 5
        {RootFindingTest(8, 15.), 0.04620981015, 0., 1.},  // n = 15
        {RootFindingTest(8, 20.), 0.03465735902, 0., 1.},  // n = 20

        {RootFindingTest(9, 1.), 0.3819660113, 0., 1.},  // n = 1
        {RootFindingTest(9, 2.), 0.1909830056, 0., 1.},  // n = 2
        {RootFindingTest(9, 5.), 0.03840255184, 0., 1.},  // n = 5
        {RootFindingTest(9, 10.), 0.009900009998, 0., 1.},  // n = 10
        {RootFindingTest(9, 15.), 0.004424691748, 0., 1.},  // n = 15
        {RootFindingTest(9, 20.), 0.002493750039, 0., 1.},  // n = 20

        {RootFindingTest(10, 1.), 0.6180339887, 0., 1.},  // n = 1
        {RootFindingTest(10, 2.), 0.5, 0., 1.},  // n = 2
        {RootFindingTest(10, 5.), 0.3459548158, 0., 1.},  // n = 5
        {RootFindingTest(10, 10.), 0.2451223338, 0., 1.},  // n = 10
        {RootFindingTest(10, 15.), 0.1955476235, 0., 1.},  // n = 15
        {RootFindingTest(10, 20.), 0.1649209573, 0., 1.},  // n = 20

        {RootFindingTest(11, 1.), 0.275508041, 0., 1.},  // n = 1
        {RootFindingTest(11, 2.), 0.1377540205, 0., 1.},  // n = 2
        {RootFindingTest(11, 4.), 0.01030528378, 0., 1.},  // n = 4
        {RootFindingTest(11, 5.), 0.003617108179, 0., 1.},  // n = 5
        {RootFindingTest(11, 8.), 0.0004108729185, 0., 1.},  // n = 8
        {RootFindingTest(11, 15.), 2.59895759e-05, 0., 1.},  // n = 15
        {RootFindingTest(11, 20.), 7.668595119e-06, 0., 1.},  // n = 20

        {RootFindingTest(12, 1.), 0.4010581375, 0., 1.},  // n = 1
        {RootFindingTest(12, 5.), 0.5161535188, 0., 1.},  // n = 5
        {RootFindingTest(12, 10.), 0.5395222269, 0., 1.},  // n = 10
        {RootFindingTest(12, 15.), 0.5481822943, 0., 1.},  // n = 15
        {RootFindingTest(12, 20.), 0.5527046667, 0., 1.},  // n = 20

        {RootFindingTest(13, 2.), 0.5, 0.01, 1.},  // n = 2
        {RootFindingTest(13, 5.), 0.2, 0.01, 1.},  // n = 5
        {RootFindingTest(13, 15.), 0.06666666667, 0.01, 1.},  // n = 15
        {RootFindingTest(13, 20.), 0.05, 0.01, 1.},  // n = 20

        {RootFindingTest(14), 0., -1., 4.},

        {RootFindingTest(15, 1.), 0.623806519, -1.e4, M_PI/2.},  // n = 1
        {RootFindingTest(15, 2.), 0.623806519, -1.e4, M_PI/2.},  // n = 2
        {RootFindingTest(15, 3.), 0.623806519, -1.e4, M_PI/2.},  // n = 3
        {RootFindingTest(15, 4.), 0.623806519, -1.e4, M_PI/2.},  // n = 4
        {RootFindingTest(15, 5.), 0.623806519, -1.e4, M_PI/2.},  // n = 5
        {RootFindingTest(15, 6.), 0.623806519, -1.e4, M_PI/2.},  // n = 6
        {RootFindingTest(15, 7.), 0.623806519, -1.e4, M_PI/2.},  // n = 7
        {RootFindingTest(15, 8.), 0.623806519, -1.e4, M_PI/2.},  // n = 8
        {RootFindingTest(15, 9.), 0.623806519, -1.e4, M_PI/2.},  // n = 9
        {RootFindingTest(15, 10.), 0.623806519, -1.e4, M_PI/2.},  // n = 10
        {RootFindingTest(15, 11.), 0.623806519, -1.e4, M_PI/2.},  // n = 11
        {RootFindingTest(15, 12.), 0.623806519, -1.e4, M_PI/2.},  // n = 12
        {RootFindingTest(15, 13.), 0.623806519, -1.e4, M_PI/2.},  // n = 13
        {RootFindingTest(15, 14.), 0.623806519, -1.e4, M_PI/2.},  // n = 14
        {RootFindingTest(15, 15.), 0.623806519, -1.e4, M_PI/2.},  // n = 15
        {RootFindingTest(15, 16.), 0.623806519, -1.e4, M_PI/2.},  // n = 16
        {RootFindingTest(15, 17.), 0.623806519, -1.e4, M_PI/2.},  // n = 17
        {RootFindingTest(15, 18.), 0.623806519, -1.e4, M_PI/2.},  // n = 18
        {RootFindingTest(15, 19.), 0.623806519, -1.e4, M_PI/2.},  // n = 19
        {RootFindingTest(15, 20.), 0.623806519, -1.e4, M_PI/2.},  // n = 20
        {RootFindingTest(15, 21.), 0.623806519, -1.e4, M_PI/2.},  // n = 21
        {RootFindingTest(15, 22.), 0.623806519, -1.e4, M_PI/2.},  // n = 22
        {RootFindingTest(15, 23.), 0.623806519, -1.e4, M_PI/2.},  // n = 23
        {RootFindingTest(15, 24.), 0.623806519, -1.e4, M_PI/2.},  // n = 24
        {RootFindingTest(15, 25.), 0.623806519, -1.e4, M_PI/2.},  // n = 25
        {RootFindingTest(15, 26.), 0.623806519, -1.e4, M_PI/2.},  // n = 26
        {RootFindingTest(15, 27.), 0.623806519, -1.e4, M_PI/2.},  // n = 27
        {RootFindingTest(15, 28.), 0.623806519, -1.e4, M_PI/2.},  // n = 28
        {RootFindingTest(15, 29.), 0.623806519, -1.e4, M_PI/2.},  // n = 29
        {RootFindingTest(15, 30.), 0.623806519, -1.e4, M_PI/2.},  // n = 30
        {RootFindingTest(15, 31.), 0.623806519, -1.e4, M_PI/2.},  // n = 31
        {RootFindingTest(15, 32.), 0.623806519, -1.e4, M_PI/2.},  // n = 32
        {RootFindingTest(15, 33.), 0.623806519, -1.e4, M_PI/2.},  // n = 33
        {RootFindingTest(15, 34.), 0.623806519, -1.e4, M_PI/2.},  // n = 34
        {RootFindingTest(15, 35.), 0.623806519, -1.e4, M_PI/2.},  // n = 35
        {RootFindingTest(15, 36.), 0.623806519, -1.e4, M_PI/2.},  // n = 36
        {RootFindingTest(15, 37.), 0.623806519, -1.e4, M_PI/2.},  // n = 37
        {RootFindingTest(15, 38.), 0.623806519, -1.e4, M_PI/2.},  // n = 38
        {RootFindingTest(15, 39.), 0.623806519, -1.e4, M_PI/2.},  // n = 39
        {RootFindingTest(15, 40.), 0.623806519, -1.e4, M_PI/2.},  // n = 40

        // {RootFindingTest(16, 20.), 0.5235987756, -1.e4, 1.e-4},  // n = 20
        // {RootFindingTest(16, 21.), 0.5235987756, -1.e4, 1.e-4},  // n = 21
        // {RootFindingTest(16, 22.), 0.5235987756, -1.e4, 1.e-4},  // n = 22
        // {RootFindingTest(16, 23.), 0.5235987756, -1.e4, 1.e-4},  // n = 23
        // {RootFindingTest(16, 24.), 0.5235987756, -1.e4, 1.e-4},  // n = 24
        // {RootFindingTest(16, 25.), 0.5235987756, -1.e4, 1.e-4},  // n = 25
        // {RootFindingTest(16, 26.), 0.5235987756, -1.e4, 1.e-4},  // n = 26
        // {RootFindingTest(16, 27.), 0.5235987756, -1.e4, 1.e-4},  // n = 27
        // {RootFindingTest(16, 28.), 0.5235987756, -1.e4, 1.e-4},  // n = 28
        // {RootFindingTest(16, 29.), 0.5235987756, -1.e4, 1.e-4},  // n = 29
        // {RootFindingTest(16, 30.), 0.5235987756, -1.e4, 1.e-4},  // n = 30
        // {RootFindingTest(16, 31.), 0.5235987756, -1.e4, 1.e-4},  // n = 31
        // {RootFindingTest(16, 32.), 0.5235987756, -1.e4, 1.e-4},  // n = 32
        // {RootFindingTest(16, 33.), 0.5235987756, -1.e4, 1.e-4},  // n = 33
        // {RootFindingTest(16, 34.), 0.5235987756, -1.e4, 1.e-4},  // n = 34
        // {RootFindingTest(16, 35.), 0.5235987756, -1.e4, 1.e-4},  // n = 35
        // {RootFindingTest(16, 36.), 0.5235987756, -1.e4, 1.e-4},  // n = 36
        // {RootFindingTest(16, 37.), 0.5235987756, -1.e4, 1.e-4},  // n = 37
        // {RootFindingTest(16, 38.), 0.5235987756, -1.e4, 1.e-4},  // n = 38
        // {RootFindingTest(16, 39.), 0.5235987756, -1.e4, 1.e-4},  // n = 39
        // {RootFindingTest(16, 40.), 0.5235987756, -1.e4, 1.e-4},  // n = 40

        // {RootFindingTest(16, 100.), 0.5235987756, -1.e4, 1.e-4},  // n = 100
        // {RootFindingTest(16, 200.), 0.5235987756, -1.e4, 1.e-4},  // n = 200
        // {RootFindingTest(16, 300.), 0.5235987756, -1.e4, 1.e-4},  // n = 300
        // {RootFindingTest(16, 400.), 0.5235987756, -1.e4, 1.e-4},  // n = 400
        // {RootFindingTest(16, 500.), 0.5235987756, -1.e4, 1.e-4},  // n = 500
        // {RootFindingTest(16, 600.), 0.5235987756, -1.e4, 1.e-4},  // n = 600
        // {RootFindingTest(16, 700.), 0.5235987756, -1.e4, 1.e-4},  // n = 700
        // {RootFindingTest(16, 800.), 0.5235987756, -1.e4, 1.e-4},  // n = 800
        // {RootFindingTest(16, 900.), 0.5235987756, -1.e4, 1.e-4},  // n = 900
        // {RootFindingTest(16, 1000.), 0.5235987756, -1.e4, 1.e-4},  // n = 1000

    };

    for (auto ref: rootfindingtests)
    {
        auto f = std::bind(&RootFindingTest::evaluate, std::get<0>(ref), std::placeholders::_1);
        double a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        double root = 0.5 * (a+b);
        RootFinding rootfinding;
        int output = rootfinding.bisection(f, root, a, b, 1e-12, 1e-12);
        root = rootfinding.getx();

        if ((std::fabs(root - std::get<1>(ref)) > 1e-9 && std::fabs(std::get<0>(ref).evaluate(root)) > 1e-9) || output > 0)
        {
            print("Bisection root finding not successful", {static_cast<double>(std::get<0>(ref).test_case), 
                                                            root, std::get<1>(ref),
                                                            std::get<0>(ref).evaluate(root)});
            error_output++;
        }
    }
    for (auto ref: rootfindingtests)
    {
        auto f = std::bind(&RootFindingTest::evaluate, std::get<0>(ref), std::placeholders::_1);
        auto df = std::bind(&RootFindingTest::gradient, std::get<0>(ref), std::placeholders::_1);
        double a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        double root = 0.5 * (a+b);
        RootFinding rootfinding;
        int output = rootfinding.bisection_newton(f, df, root, a, b, 1e-12, 1e-12);
        root = rootfinding.getx();

        if ((std::fabs(root - std::get<1>(ref)) > 1e-9 && std::fabs(std::get<0>(ref).evaluate(root)) > 1e-9) || output > 0)
        {
            print("Bisection-Newton root finding not successful", {static_cast<double>(std::get<0>(ref).test_case), 
                                                                   root, std::get<1>(ref),
                                                                   std::get<0>(ref).evaluate(root)});
            error_output++;
        }
    }
    for (auto ref: rootfindingtests)
    {
        auto f = std::bind(&RootFindingTest::evaluate, std::get<0>(ref), std::placeholders::_1);
        double a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        double root = 0.5 * (a+b);
        RootFinding rootfinding;
        int output = rootfinding.brent(f, root, a, b, 1e-12, 1e-12);
        root = rootfinding.getx();

        if ((std::fabs(root - std::get<1>(ref)) > 1e-9 && std::fabs(std::get<0>(ref).evaluate(root)) > 1e-9) || output > 0)
        {
            print("Brent root finding not successful", {static_cast<double>(std::get<0>(ref).test_case), 
                                                        root, std::get<1>(ref),
                                                        std::get<0>(ref).evaluate(root)});
            error_output++;
        }
    }
    for (auto ref: rootfindingtests)
    {
        auto f = std::bind(&RootFindingTest::evaluate, std::get<0>(ref), std::placeholders::_1);
        auto df = std::bind(&RootFindingTest::gradient, std::get<0>(ref), std::placeholders::_1);
        double a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        double root = 0.5 * (a+b);
        RootFinding rootfinding;
        int output = rootfinding.brent_newton(f, df, root, a, b, 1e-12, 1e-12);
        root = rootfinding.getx();

        if ((std::fabs(root - std::get<1>(ref)) > 1e-9 && std::fabs(std::get<0>(ref).evaluate(root)) > 1e-9) || output > 0)
        {
            print("Brent-Newton root finding not successful", {static_cast<double>(std::get<0>(ref).test_case), 
                                                               root, std::get<1>(ref),
                                                               std::get<0>(ref).evaluate(root)});
            error_output++;
        }
    }

    std::vector<std::tuple<Polynomial, double, double, double>> polynomials = {
        // Linear polynomial
        // {Polynomial({3., 1.5}), -0.5, -5., 5.},  // Polynomial, solution, lower bound, upper bound

        // // Quadratic polynomial
        // {Polynomial({2., -3., -1.5}), 1.895643924, -5., 5.},

        // // Cubic polynomial
        // {Polynomial({1., 3., 2., 6.}), -3., -5., 5.},
        // {Polynomial({1., 1., -5., 3.}), -3., -5., 5.},

        // // Quartic polynomial
        // {Polynomial({1., 0., 3., 0., -4}), 1., -5., 5.},

        // // Quintic polynomial
        // {Polynomial({1., 0., 2., 0., -0.5, 3.}), -1.044499064, -5., 5.},

        // // Discontinuous polynomials
        // {Polynomial({3., 1.5}, {3., 1.5}, 0.), -0.5, -5., 5.},
        // {Polynomial({3., -0.5}, {3., 1.5}, 0.), 0., -5., 5.},
        // {Polynomial({1., -2.5}, {1., -3., 3.}, 0.), 0., -5., 5.},
        // {Polynomial({1., -2.5}, {1., -3., 3.}, 2.), 2., -5., 5.},
    };

    for (auto ref: polynomials)
    {
        auto f = std::bind(&Polynomial::evaluate, std::get<0>(ref), std::placeholders::_1);
        double root{ 0. }, a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        RootFinding rootfinding;
        int output = rootfinding.bisection(f, root, a, b, 1e-12, 1e-14);
        root = rootfinding.getx();

        if (std::fabs(root - std::get<1>(ref)) > 1e-8 || output > 0)
        {
            print("Bisection Polynomial root finding not successful", {root, std::get<1>(ref)});
            error_output++;
        }
    }
    for (auto ref: polynomials)
    {
        auto f = std::bind(&Polynomial::evaluate, std::get<0>(ref), std::placeholders::_1);
        auto df = std::bind(&Polynomial::gradient, std::get<0>(ref), std::placeholders::_1);
        double root{ 0. }, a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        RootFinding rootfinding;
        int output = rootfinding.bisection_newton(f, df, root, a, b, 1e-12, 1e-14);
        root = rootfinding.getx();

        if (std::fabs(root - std::get<1>(ref)) > 1e-8 || output > 0)
        {
            print("Bisection-Newton Polynomial root finding not successful", {root, std::get<1>(ref)});
            error_output++;
        }
    }
    for (auto ref: polynomials)
    {
        auto f = std::bind(&Polynomial::evaluate, std::get<0>(ref), std::placeholders::_1);
        double root{ 0. }, a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        RootFinding rootfinding;
        int output = rootfinding.brent(f, root, a, b, 1e-12, 1e-14);
        root = rootfinding.getx();

        if (std::fabs(root - std::get<1>(ref)) > 1e-8 || output > 0)
        {
            print("Brent Polynomial root finding not successful", {root, std::get<1>(ref)});
            error_output++;
        }
    }
    for (auto ref: polynomials)
    {
        auto f = std::bind(&Polynomial::evaluate, std::get<0>(ref), std::placeholders::_1);
        auto df = std::bind(&Polynomial::gradient, std::get<0>(ref), std::placeholders::_1);
        double root{ 0. }, a{ std::get<2>(ref) }, b{ std::get<3>(ref) };
        RootFinding rootfinding;
        int output = rootfinding.brent_newton(f, df, root, a, b, 1e-12, 1e-14);
        root = rootfinding.getx();

        if (std::fabs(root - std::get<1>(ref)) > 1e-8 || output > 0)
        {
            print("Brent-Newton Polynomial root finding not successful", {root, std::get<1>(ref)});
            error_output++;
        }
    }

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_rootfinding(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_rootfinding(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}

int test_line_search()
{
    std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();
    int error_output = 0;

    // Test the line search method considering a minimization of f(x) problem
    // Ha = -b | H = grad(b) ; b = grad(f) ; a = delta(x)

    double lamb = 1.;
    bool check;

    // Application to polynomial functions
    // f(x,y) = x1^4 + x1^2 + x2^2 | x = [ x1 x2 ]

    LineSearch  linesearch{2};
    // Add gradient b
    Eigen::VectorXd b = Eigen::VectorXd::Zero(2);
    b << -6., -2.;
    // Add step a
    Eigen::VectorXd a = Eigen::VectorXd::Zero(2);
    a << -3., -1.;

    // New values of independent variable vector x_old before calculation
    Eigen::VectorXd x(2);
    x << -2., 0.;
    // Old values of independent variable vector x_old before calculation
    Eigen::VectorXd x_old(2); 
    x_old << 1., 1.;
    // Objective function f(x), new value
    double f = 20.;
    // Objective function f(x), old value
    double f_old = 3.;

    check = linesearch.init(lamb, x_old, f_old, b, a, 1.0);

    while(f > f_old && check)
    {
        if(f > f_old)
        {
            check = linesearch.process(x, f);
            lamb = linesearch.get_alam();
            
            x = x_old + lamb * a;
            f = std::pow(x[0],4) + std::pow(x[0],2) + std::pow(x[1],2);
        }
        else
        {
            break;
        }
    }

    // Reduced the objective function ?
    check = (f_old - f > 0.);
    error_output = !check;

    std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	double dt = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() * 1e-6;
	if (error_output > 0)
	{
		std::cout << "Errors occurred in test_line_search(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
	else
	{
		std::cout << "No errors occurred in test_line_search(): " << error_output;
		std::cout << " - Time: " << dt << " seconds\n";
	}
    return error_output;
}