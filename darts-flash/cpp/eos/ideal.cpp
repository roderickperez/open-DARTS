#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include "dartsflash/eos/ideal.hpp"
#include "dartsflash/maths/maths.hpp"

IdealGas::IdealGas(CompData& comp_data) : EoS(comp_data) { }

void IdealGas::init_PT(double p_, double T_)
{
    this->p = p_; this->T = T_;
    return;
}
void IdealGas::solve_PT(std::vector<double>::iterator n_it, bool second_order)
{
    this->N = std::accumulate(n_it, n_it + ns, 0.);
    (void) second_order;
    return;
}
std::vector<double> IdealGas::lnphi0(double X, double T_, bool pt)
{
    (void) X;
    (void) T_;
    (void) pt;
    return std::vector<double>(ns, 0.);
}

void IdealGas::init_VT(double V_, double T_)
{
    this->v = V_;
    this->T = T_;
    return;
}
void IdealGas::solve_VT(std::vector<double>::iterator n_it, bool second_order)
{
    this->N = std::accumulate(n_it, n_it + ns, 0.);
    (void) second_order;
    return;
}

double IdealGas::P(double V_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    (void) pt;
    this->N = std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);
    return N * this->units.R * T_ / V_;
}
double IdealGas::V(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    (void) pt;
    this->N = std::accumulate(n_.begin() + start_idx, n_.begin() + start_idx + ns, 0.);
    return N * this->units.R * T_ / p_;
}
double IdealGas::rho(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    return this->compdata.get_molar_weight(n_) * 1e-3 / this->V(p_, T_, n_, start_idx, pt);
}
