#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <numeric>
#include <unordered_map>

#include "dartsflash/maths/maths.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"

namespace ballard {
    // Following hydrate fugacity model described in Ballard (2002)
	double R{ 8.314472 }; // gas constant
    double R_inv = 1./R;
    double T_0{ 298.15 }; // reference temperature
    double P_0{ 1 }; // reference pressure [bar]

    // gibbs energy of ideal gas at p0, T0
    std::unordered_map<std::string, double> gi0 = {
        {"H2O", -228700}, {"CO2", -394600}, {"N2", 0}, {"H2S", -33100}, 
        {"C1", -50830}, {"C2", -32900}, {"C3", -23500}, {"iC4", -20900}, {"nC4", -17200}, {"iC5", -15229}, {"nC5", -8370}, {"nC6", -290}, {"nC7", 8120},
        {"NaCl", -384138.00} 
    };
    // molar enthalpy of ideal gas at p0, T0
    std::unordered_map<std::string, double> hi0 = {
        {"H2O", -242000}, {"CO2", -393800}, {"N2", 0}, {"H2S", -20200}, 
        {"C1", -74900}, {"C2", -84720}, {"C3", -103900}, {"iC4", -134600}, {"nC4", -126200}, {"iC5", -165976}, {"nC5", -146500}, {"nC6", -167300}, {"nC7", -187900} ,
        {"NaCl", -411153.00}
    };

    // water and empty hydrate constants
    double gw_00{ -228700 }; // formation Gibbs energy of H2O in ideal gas
    double hw_00{ -242000 }; // enthalpy of formation of H2O in ideal gas
    std::vector<double> hw_a = { 3.8747*R, 0.0231E-2*R, 0.1269E-5*R, -0.4321E-9*R }; // ideal gas heat capacity parameters of H2O in ideal gas
    
    std::unordered_map<std::string, double> g_B0 = {{"sI", -235537.85}, {"sII", -235627.53}, {"sH", -235491.02}}; // formation Gibbs energy of pure hydrate phases (sI, sII, sH)
    std::unordered_map<std::string, double> h_B0 = {{"sI", -291758.77}, {"sII", -292044.10}, {"sH", -291979.26}}; // enthalpy of formation of pure hydrate phases (sI, sII, sH)
    std::vector<double> h_B_a = { 0.735409713 * R, 1.4180551E-2 * R, -1.72746E-5 * R, 63.5104E-9 * R }; // heat capacity parameters of pure hydrate phases
    
    std::unordered_map<std::string, double> v_0{{"sI", 22.712}, {"sII", 22.9456}, {"sH", 24.2126}}; // molar volume of pure hydrate phases at reference p, T (sI, sII, sH)
    std::unordered_map<std::string, std::vector<double>> v_a = {
        {"sI", {3.384960E-4, 5.400990E-7, -4.769460E-11, 3E-5}},   // molar volume parameters of pure sI hydrate
        {"sII", {2.029776E-4, 1.851168E-7, -1.879455E-10, 3E-6}},
        {"sH", {3.575490E-4, 6.294390E-7, 0, 3E-7}}};

    // cage parameters
    double l0 = 1E-13;
    std::unordered_map<std::string, double> a = {{"sI", 25.74}, {"sII", 260.}, {"sH", 0.}}; // parameter a in eq. 4.39 (sI, sII, sH) [J/cm3]
    std::unordered_map<std::string, double> b = {{"sI", -481.32}, {"sII", -68.64}, {"sH", 0.}}; // parameter b in eq. 4.39 (sI, sII, sH) [J/cm3]
    std::unordered_map<std::string, double> a_0 = {{"sI", 11.99245}, {"sII", 17.1}, {"sH", 11.09826}}; // standard lattice parameters for sI, sII, sH [eq. 4.42, p. ]

    std::unordered_map<std::string, std::vector<int>> zn = {{"sI", {8, 12, 8, 4, 8, 4}}, {"sII", {2, 6, 12, 12, 12, 4}}, {"sH", {20, 20, 36}}}; // #water in layers
    std::unordered_map<std::string, std::vector<int>> n_shells = {{"sI", {2, 4}}, {"sII", {3, 3}}, {"sH", {1, 1, 1}}};
    std::unordered_map<std::string, std::vector<double>> Rn {
        {"sI", {3.83E-10, 3.96E-10, 4.06E-10, 4.25E-10, 4.47E-10, 4.645E-10}},  // radius of layers S0, S1, L0, L1, L2, L3 (sI)
        {"sII", {3.748E-10, 3.845E-10, 3.956E-10, 4.635E-10, 4.715E-10, 4.729E-10}},  // radius of layers S0, S1, S2, L0, L1, L2 (sII)
        {"sH", {3.91E-10, 4.06E-10, 5.71E-10}}};  // radius of layers S, M, L (sH)

    // guest parameters
    std::unordered_map<std::string, double> ai = {{"CO2", 0.6805E-10}, {"N2", 0.3526E-10},  {"H2S", 0.3600E-10}, {"C1", 0.3834E-10}, {"C2", 0.5651E-10}, {"C3", 0.6502E-10}, {"iC4", 0.8706E-10}, {"nC4", 0.9379E-10}, {"iC5", 0.9868E-10}, {"nC5", 0}, {"nC6", 0}, {"nC7", 0}, }; // hard core radius
    std::unordered_map<std::string, double> sigma = {{"CO2", 2.97638E-10}, {"N2", 3.13512E-10}, {"H2S", 3.10000E-10}, {"C1", 3.14393E-10}, {"C2", 3.24693E-10}, {"C3", 3.41670E-10}, {"iC4", 3.41691E-10}, {"nC4", 3.51726E-10}, {"iC5", 3.5455E-10}, {"nC5", 0}, {"nC6", 0}, {"nC7", 0}, }; // soft core radius
    std::unordered_map<std::string, double> eik = {{"CO2", 175.405}, {"N2", 127.426}, {"H2S", 212.047}, {"C1", 155.593}, {"C2", 188.181}, {"C3", 192.855}, {"iC4", 198.333}, {"nC4", 197.254}, {"iC5", 199.56}, {"nC5", 0}, {"nC6", 0}, {"nC7", 0}, }; // potential well depth/k
    std::unordered_map<std::string, double> Di = {{"CO2", 4.603}, {"N2", 4.177}, {"H2S", 4.308}, {"C1", 4.247}, {"C2", 5.076}, {"C3", 5.745}, {"iC4", 6.306}, {"nC4", 6.336}, {"iC5", 6.777}, {"nC5", 0}, {"nC6", 0}, {"nC7", 0}, }; // molecular diameter of component i
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<double>>> dr_im = {
        {"sI", {{"CO2", {0, 5.8282E-3}}, {"N2", {1.7377E-2, 0}}, {"H2S", {1.7921E-2, 0}}, {"C1", {1.7668E-2, 0}}, {"C2", {0, 1.5773E-2}}, {"C3", {0, 2.9839E-2}}, {"iC4", {0, 0}}, {"nC4", {0, 0}}, {"iC5", {0, 0}}, {"nC5", {0, 0}}, {"nC6", {0, 0}}, {"nC7", {0, 0}},}}, // repulsive const in cage S, L, sI
        {"sII", {{"CO2", {2.2758E-3, 1.2242E-2}}, {"N2", {2.0652E-3, 1.1295E-2}}, {"H2S", {2.1299E-3, 1.1350E-2}}, {"C1", {2.0998E-3, 1.1383E-2}}, {"C2", {2.5097E-3, 1.4973E-2}}, {"C3", {0,  2.5576E-2}}, {"iC4", {0, 3.6E-2}}, {"nC4", {0, 3.6593E-2}}, {"iC5", {0, 4.7632E-2}}, {"nC5", {0, 0}}, {"nC6", {0, 0}}, {"nC7", {0, 0}},}}
    }; // repulsive const in cage S, L, sII
    
    // std::vector<double> dr_im3 = { 0 }; // volume of sH is assumed constant [p. 66]
    std::unordered_map<std::string, std::unordered_map<std::string, double>> kappa_iH = {  // compressibility parameters p. 118
        {"sI", {{"CO2", 1E-6}, {"N2", 1.1E-5}, {"H2S", 5E-6}, {"C1", 1E-5}, {"C2", 1E-8}, {"C3", 1E-7}, {"iC4", 0}, {"nC4", 0}, {"iC5", 0}, {"nC5", 0}, {"nC6", 0}, {"nC7", 0},}},
        {"sII", {{"CO2", 1E-5}, {"N2", 1.1E-5}, {"H2S", 1E-5}, {"C1", 5E-5}, {"C2", 1E-7}, {"C3", 1E-6}, {"iC4", 1E-8}, {"nC4", 1E-8}, {"iC5", 1E-8}, {"nC5", 0}, {"nC6", 0}, {"nC7", 0},}}};

    IG::IG(std::string component_) : Integral(component_)
    {
        this->gi0 = ballard::gi0[component_];
        this->hi0 = ballard::hi0[component_];
        this->cpi = comp_data::cpi[component_];
    }
    double IG::H(double T)
    {
        // Integral of H(T)/RT^2 dT from T_0 to T
        return (-(this->hi0 / M_R
                - this->cpi[0] * ballard::T_0 
                - 1. / 2 * this->cpi[1] * std::pow(ballard::T_0, 2) 
                - 1. / 3 * this->cpi[2] * std::pow(ballard::T_0, 3)
                - 1. / 4 * this->cpi[3] * std::pow(ballard::T_0, 4)) * (1./T - 1./ballard::T_0)
                + (this->cpi[0] * (std::log(T) - std::log(ballard::T_0))
                + 1. / 2 * this->cpi[1] * (T - ballard::T_0)
                + 1. / 6 * this->cpi[2] * (std::pow(T, 2) - std::pow(ballard::T_0, 2))
                + 1. / 12 * this->cpi[3] * (std::pow(T, 3) - std::pow(ballard::T_0, 3))));
    }
    double IG::dHdT(double T)
    {
        // Derivative of integral w.r.t. temperature
        return (this->hi0 / M_R + 
                this->cpi[0] * (T-ballard::T_0) 
                + 1. / 2 * this->cpi[1] * (std::pow(T, 2)-std::pow(ballard::T_0, 2)) 
                + 1. / 3 * this->cpi[2] * (std::pow(T, 3)-std::pow(ballard::T_0, 3))
                + 1. / 4 * this->cpi[3] * (std::pow(T, 4)-std::pow(ballard::T_0, 4))) / std::pow(T, 2);
    }
    double IG::d2HdT2(double T)
    {
        // Second derivative of integral w.r.t. temperature
        return (this->cpi[0] + this->cpi[1] * T + this->cpi[2] * std::pow(T, 2) + this->cpi[3] * std::pow(T, 3)) / std::pow(T, 2) - 2. * this->dHdT(T) / T;
    }
    int IG::test_derivatives(double T, double tol, bool verbose)
    {
        int error_output = 0;
        double dH = this->dHdT(T);
        double d2H = this->d2HdT2(T);

        double d, dT{ 1e-5 };
        double H_ = this->H(T-dT);
        double H1 = this->H(T+dT);
        double dH_num = (H1-H_)/(2*dT);
        double dH_ = this->dHdT(T-dT);
        double dH1 = this->dHdT(T+dT);
        double d2H_num = (dH1-dH_)/(2*dT);

        d = std::log(std::fabs(dH + 1e-15)) - std::log(std::fabs(dH_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard IG dH/dT", {dH, dH_num, d}); error_output++; }
        d = std::log(std::fabs(d2H + 1e-15)) - std::log(std::fabs(d2H_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard IG d2H/dT2", {d2H, d2H_num, d}); error_output++; }
        
        return error_output;
    }

    double HB::f(double T) 
    {
        // molar enthalpy of empty hydrate lattice [eq. 4.40];
        return (h_B0[phase] 
                + h_B_a[0] * (T-T_0)
                + 1. / 2 * h_B_a[1] * (std::pow(T, 2)-std::pow(T_0, 2)) 
                + 1. / 3 * h_B_a[2] * (std::pow(T, 3)-std::pow(T_0, 3)) 
                + 1. / 4 * h_B_a[3] * (std::pow(T, 4)-std::pow(T_0, 4))) * R_inv / std::pow(T, 2);
    }
    double HB::F(double T) 
    {
        return (-(h_B0[phase] 
                - h_B_a[0] * T_0 
                - 1. / 2 * h_B_a[1] * std::pow(T_0, 2) 
                - 1. / 3 * h_B_a[2] * std::pow(T_0, 3) 
                - 1. / 4 * h_B_a[3] * std::pow(T_0, 4)) * (1./T - 1./T_0)
                + h_B_a[0] * (std::log(T) - std::log(T_0))
                + 1. / 2 * h_B_a[1] * (T - T_0)
                + 1. / 6 * h_B_a[2] * (std::pow(T, 2)-std::pow(T_0, 2)) 
                + 1. / 12 * h_B_a[3] * (std::pow(T, 3)-std::pow(T_0, 3))) * R_inv;
    }
    double HB::dFdT(double T) 
    {
        return this->f(T);
    }
    double HB::d2FdT2(double T) 
    {
        // f(x) = h(T)/RT^2
        // df(x)/dT = dh(T)/dT/RT^2 - 2 h(T)/RT^3
        double dhdT = (h_B_a[0] + h_B_a[1] * T + h_B_a[2] * std::pow(T, 2) + h_B_a[3] * std::pow(T, 3)) * R_inv;
        
        return dhdT / std::pow(T, 2) - 2 * this->dFdT(T) / T;
    }
    int HB::test_derivatives(double T, double tol, bool verbose)
    {
        int error_output = 0;
        double dF = this->dFdT(T);
        double d2F = this->d2FdT2(T);

        double d, dT{ 1e-5 };
        double F_ = this->F(T-dT);
        double F1 = this->F(T+dT);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(T-dT);
        double dF1 = this->dFdT(T+dT);
        double d2F_num = (dF1-dF_)/(2*dT);

        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard HB dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard HB d2F/dT2", {d2F, d2F_num, d}); error_output++; }

        return error_output;
    }

    double VB::f(double p, double T) 
    {
        // molar volume of empty hydrate lattice [eq. 4.41]
        double aa = v_a[phase][0] * (T-T_0) 
                    + v_a[phase][1] * std::pow(T-T_0, 2) 
                    + v_a[phase][2] * std::pow(T-T_0, 3);
        double bb = - v_a[phase][3];
        return v_0[phase]*1e-6 * std::exp(aa+bb*(p-P_0)) * R_inv / (1E-5 * T);
    }
    double VB::dfdT(double p, double T)
    {
        // molar volume of empty hydrate lattice [eq. 4.41]
        double aa = v_a[phase][0] * (T-T_0) 
                    + v_a[phase][1] * std::pow(T-T_0, 2) 
                    + v_a[phase][2] * std::pow(T-T_0, 3);
        double daa = v_a[phase][0]
                    + 2. * v_a[phase][1] * (T-T_0) 
                    + 3. * v_a[phase][2] * std::pow(T-T_0, 2);
        double bb = - v_a[phase][3];
        return v_0[phase]*1e-6 * R_inv / 1E-5 * (std::exp(aa+bb*(p-P_0)) * daa / T - std::exp(aa+bb*(p-P_0)) / std::pow(T, 2));
    }
    double VB::F(double p, double T) 
    {
        // int e^cx dx = 1/c e^cx
        // int v0 exp(a + b*p)/RT dp = v0 exp(a)/RT * int exp(b*p) dp = v0 exp(a)/bRT * exp(b*p)
        double aa = v_a[phase][0] * (T-T_0) 
                    + v_a[phase][1] * std::pow(T-T_0, 2) 
                    + v_a[phase][2] * std::pow(T-T_0, 3);
        double bb = -v_a[phase][3];
        return v_0[phase]*1e-6 * std::exp(aa-bb*P_0) * R_inv / (1e-5*T) * (std::exp(bb*p) / bb - std::exp(bb*P_0) / bb);
    }
    double VB::dFdP(double p, double T) 
    {
        return this->f(p, T);
    }
    double VB::dFdT(double p, double T) 
    {
        double aa = v_a[phase][0] * (T-T_0) 
                    + v_a[phase][1] * std::pow(T-T_0, 2) 
                    + v_a[phase][2] * std::pow(T-T_0, 3);
        double bb = -v_a[phase][3];

        double da_dT = v_a[phase][0] 
                    + 2*v_a[phase][1] * (T-T_0) 
                    + 3*v_a[phase][2] * std::pow(T-T_0, 2);
        
        double d_dT = (da_dT * std::exp(aa-bb*P_0) - std::exp(aa-bb*P_0)/T) * R_inv / (1e-5 * T);
        
        return v_0[phase]*1e-6 * d_dT * (std::exp(bb*p) / bb - std::exp(bb*P_0) / bb);
    }
    double VB::d2FdPdT(double p, double T) 
    {
        return this->dfdT(p, T);
    }
    double VB::d2FdT2(double p, double T) 
    {
        // Second derivative of integral w.r.t. temperature
        double aa = v_a[phase][0] * (T-T_0) 
                + v_a[phase][1] * std::pow(T-T_0, 2) 
                + v_a[phase][2] * std::pow(T-T_0, 3);
        double bb = -v_a[phase][3];

        double da_dT = v_a[phase][0] 
                    + 2*v_a[phase][1] * (T-T_0) 
                    + 3*v_a[phase][2] * std::pow(T-T_0, 2);
        double d2a_dT2 = 2*v_a[phase][1] + 6*v_a[phase][2]*(T-T_0);
        
        double d_dT = (T * da_dT * std::exp(aa-bb*P_0) - std::exp(aa-bb*P_0)) * R_inv / (1e-5 * std::pow(T, 2));
        double d2_dT2 = (T * d2a_dT2 + T * std::pow(da_dT, 2)) * std::exp(aa-bb*P_0) * R_inv / (1e-5 * std::pow(T, 2)) - 2. * d_dT / T;
        
        return v_0[phase]*1e-6 * d2_dT2 * (std::exp(bb*p) / bb - std::exp(bb*P_0) / bb);
    }
    int VB::test_derivatives(double p, double T, double tol, bool verbose)
    {
        int error_output = 0;
        double df = this->dfdT(p, T);
        double dF = this->dFdT(p, T);
        double d2F = this->d2FdT2(p, T);

        double d, dT{ 1e-5 };
        double f_ = this->f(p, T-dT);
        double f1 = this->f(p, T+dT);
        double F_ = this->F(p, T-dT);
        double F1 = this->F(p, T+dT);
        double df_num = (f1-f_)/(2*dT);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(p, T-dT);
        double dF1 = this->dFdT(p, T+dT);
        double d2F_num = (dF1-dF_)/(2*dT);

        d = std::log(std::fabs(df + 1e-15)) - std::log(std::fabs(df_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard VB df/dT", {df, df_num, d}); error_output++; }
        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard VB dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard VB d2F/dT2", {d2F, d2F_num, d}); error_output++; }
        
        return error_output;
    }

    VH::VH(std::string phase_, std::vector<std::string> components_) : Integral(phase_)
    {
        this->components = components_;
        this->nc = static_cast<int>(components.size());
        this->water_index = std::distance(components.begin(), std::find(components.begin(), components.end(), "H2O"));
    }
    double VH::f(double p, double T, std::vector<double> theta) 
    {
        // cage occupancy volume change

        // Kappa & v0
        double v0, kappa;

        if (phase == "sH") 
        { 
            kappa = 1E-7; 
            v0 = v_0["sH"]; 
        }
        else 
        {
            // V_0(x) [eq. 4.42]
            // Function f(theta_im) [eq. 6.18/6.19]
            // fractional occupancy average molecular diameter for 5^12 cage
            double D_{ 0 };
            for (int i = 0; i < nc; i++) 
            {
                D_ += theta[i] * Di[components[i]];
            }

            std::vector<double> f_im(vdwp::n_cages[phase] * nc);
            for (int i = 0; i < nc; i++) 
            {
                if (i != water_index) 
                {
                    // For first cage (type 5^12), use
                    f_im[0 + i] = (1. + vdwp::zm[phase][0] / vdwp::nH2O[phase]) * theta[i] / (1. + vdwp::zm[phase][0] / vdwp::nH2O[phase] * theta[i]) * std::exp(Di[components[i]] - D_);
                    // For other cages, use
                    for (int m = 1; m < vdwp::n_cages[phase]; m++) 
                    {
                        f_im[nc*m + i] = (1. + vdwp::zm[phase][m] / vdwp::nH2O[phase]) * theta[nc*m + i] / (1. + vdwp::zm[phase][m] / vdwp::nH2O[phase] * theta[nc*m + i]);
                    }
                }
            }
            
            double sum_m{ 0 };
            for (int m = 0; m < vdwp::n_cages[phase]; m++) 
            {
                double sum_i{ 0 };
                for (int i = 0; i < nc; i++) 
                {
                    if (i != water_index) 
                    {
                        sum_i += f_im[nc*m + i] * dr_im[phase][components[i]][m];
                    }
                }
                sum_m += vdwp::Nm[phase][m] * sum_i;
            }

            kappa = 0.;
            for (int i = 0; i < nc; i++) 
            {
                if (i != water_index) 
                {
                    kappa += kappa_iH[phase][components[i]] * theta[nc + i];
                }
            }
            v0 = std::pow(a_0[phase] + sum_m, 3) * M_NA / vdwp::nH2O[phase] * 1E-24; // compositional dependence of molar volume [4.42]
        }

        // Molar volume of hydrate [eq. 4.41]
        double exponent = v_a[phase][0]*(T-T_0) + v_a[phase][1]*std::pow((T-T_0), 2) + v_a[phase][2]*std::pow((T-T_0), 3) - kappa*(p-P_0);
        return v0 * std::exp(exponent) * 1E-6; // m3/mol [eq. 4.41]
    }

    /*
    double VH::dfdT(double p, double T, std::vector<double> theta, std::vector<double> dthetadT) {
        // cage occupancy volume change

        // Kappa & v0
        double v0, kappa, dv0{ 0. }, dkappa{ 0. };

        if (phase == "sH") 
        { 
            kappa = 1E-7; 
            v0 = v_0["sH"]; 
        }
        else 
        {
            // V_0(x) [eq. 4.42]
            // Function f(theta_im) [eq. 6.18/6.19]
            // fractional occupancy average molecular diameter for 5^12 cage
            double D_{ 0 };
            for (int i = 0; i < nc; i++) 
            {
                D_ += theta[i] * Di[components[i]];
            }

            std::vector<double> f_im(vdwp::n_cages[phase] * nc);
            for (int i = 0; i < nc; i++) 
            {
                if (i != water_index) 
                {
                    // For first cage (type 5^12), use
                    f_im[0 + i] = (1. + vdwp::zm[phase][0] / vdwp::nH2O[phase]) * theta[i] / (1. + vdwp::zm[phase][0] / vdwp::nH2O[phase] * theta[i]) * std::exp(Di[components[i]] - D_);
                    // For other cages, use
                    for (int m = 1; m < vdwp::n_cages[phase]; m++) 
                    {
                        f_im[nc*m + i] = (1. + vdwp::zm[phase][m] / vdwp::nH2O[phase]) * theta[nc*m + i] / (1. + vdwp::zm[phase][m] / vdwp::nH2O[phase] * theta[nc*m + i]);
                    }
                }
            }
            
            double sum_m{ 0 };
            for (int m = 0; m < vdwp::n_cages[phase]; m++) 
            {
                double sum_i{ 0 };
                for (int i = 0; i < nc; i++) 
                {
                    if (i != water_index) 
                    {
                        sum_i += f_im[nc*m + i] * dr_im[phase][components[i]][m];
                    }
                }
                sum_m += vdwp::Nm[phase][m] * sum_i;
            }

            kappa = 0.;
            for (int i = 0; i < nc; i++) 
            {
                if (i != water_index) 
                {
                    kappa += kappa_iH[phase][components[i]] * theta[nc + i];
                    dkappa += kappa_iH[phase][components[i]] * dthetadT[nc + i];
                }
            }
            v0 = std::pow(a_0[phase] + sum_m, 3) * M_NA / vdwp::nH2O[phase] * 1E-24; // compositional dependence of molar volume [4.42]
            dv0 = 0.;
        }

        // Molar volume of hydrate [eq. 4.41]
        double exponent = v_a[phase][0]*(T-T_0) + v_a[phase][1]*std::pow((T-T_0), 2) + v_a[phase][2]*std::pow((T-T_0), 3) - kappa*(p-P_0);
        double dexp = -(dkappa * (p-P_0) + kappa);
        double vH = v0 * std::exp(exponent) * 1E-6; // m3/mol [eq. 4.41]
        return dv0 * std::exp(exponent) * 1e-6 + v0 * std::exp(exponent) * 1e-6 * dexp;
    }

    std::vector<double> VH::dfdxj(double p, double T, std::vector<double> theta, std::vector<double> dthetadxj) {
        (void) p; (void) T; (void) dthetadxj;
        return {0.};
    }

    double VH::F(double p, double T, std::vector<double> theta) {
        // integrals solved numerically with simpson's rule
        double s = 0.;
        int steps = 20;
        double h = (p-P_0)/steps;  // interval
    
    	// vh
        for (int i = 0; i < steps; i++) 
        {
            double hix = this->f(P_0 + i*h, T, theta);
            double hixh2 = this->f(P_0 + (i+0.5)*h, T, theta);
            double hixh = this->f(P_0 + (i+1)*h, T, theta);
            s += h*((hix + 4*hixh2 + hixh) / 6);
        }
    	return s;
    }

    double VH::dFdP(double p, double T, std::vector<double> theta) {
        return this->f(p, T, theta);
    }

    double VH::dFdT(double p, double T, std::vector<double> theta, std::vector<double> dthetadT) {
        // integrals solved numerically with simpson's rule
        double s = 0.;
        int steps = 20;
        double h = (p-P_0)/steps;  // interval
    
    	// hi
        for (int i = 0; i < steps; i++) 
        {
            double hix = this->dfdT(P_0 + i*h, T, dthetadT);
            double hixh2 = this->dfdT(P_0 + (i+0.5)*h, T, dthetadT);
            double hixh = this->dfdT(P_0 + (i+1)*h, T, dthetadT);
            s += h*((hix + 4*hixh2 + hixh) / 6);
        }
    	return s;
    }

    std::vector<double> VH::dFdxj(double p, double T, std::vector<double> theta, std::vector<double> dthetadxj) {
        
        return {0.};
    }
    */

    double Kihara::w(double r, std::string component)
    {
        // hydrate cage cell potential w(r) = omega(r) [eq. 3.43a]
        double w{ 0. };
        // hard core, soft core radius and potential well depth of guest molecule
        double ai_ = ai[component]; 
        double sigmai = sigma[component];

        // Loop over shell layers of cage m [eq. 4.45]
        for (int l = 0; l < n_shells[phase][cage_index]; l++) 
        {
            double Rn_ = Rn[phase][R1_index + l];
            int zn_ = zn[phase][R1_index + l];
            double delta10 = 1. / 10 * (std::pow((1. - r / Rn_ - ai_ / Rn_), -10.) - std::pow((1. + r / Rn_ - ai_ / Rn_), -10.));
            double delta11 = 1. / 11 * (std::pow((1. - r / Rn_ - ai_ / Rn_), -11.) - std::pow((1. + r / Rn_ - ai_ / Rn_), -11.));
            double delta4 = 1. / 4 * (std::pow((1. - r / Rn_ - ai_ / Rn_), -4.) - std::pow((1. + r / Rn_ - ai_ / Rn_), -4.));
            double delta5 = 1. / 5 * (std::pow((1. - r / Rn_ - ai_ / Rn_), -5.) - std::pow((1. + r / Rn_ - ai_ / Rn_), -5.));
            w += 2.0 * zn_ * (std::pow(sigmai, 12.) / (std::pow(Rn_, 11.) * r) * (delta10 + ai_ / Rn_ * delta11) -
                            std::pow(sigmai, 6.) / (std::pow(Rn_, 5.) * r) * (delta4 + ai_ / Rn_ * delta5));    
        }
        // term in integral for Langmuir constant C_im [eq. 3.42]
        return w;
    } 
    double Kihara::f(double r, double T, std::string component) 
    {
        // hydrate cage cell potential w(r) = omega(r) [eq. 3.43a]
        double w = this->w(r, component);

        return std::exp(-eik[component] / T * w) * std::pow(r, 2);
    }
    double Kihara::F(double T, std::string component) 
    {
        // integrals solved numerically with simpson's rule
        double s = 0.;
        int steps = 20;
        double h = (R1-R0)/steps;  // interval
    
    	// hi
        for (int i = 0; i < steps; i++) 
        {
            double hix = this->f(R0 + i*h, T, component);
            double hixh2 = this->f(R0 + (i+0.5)*h, T, component);
            double hixh = this->f(R0 + (i+1)*h, T, component);
            s += h*((hix + 4*hixh2 + hixh) / 6);
            if (hixh < 1e-200) { break; } // otherwise, integral becomes too large
        }
    	return s;
    }
    double Kihara::dfdT(double r, double T, std::string component) 
    {
        // hydrate cage cell potential w(r) = omega(r) [eq. 3.43a]
        double w = this->w(r, component);

        return std::exp(-eik[component] / T * w) * std::pow(r, 2) * eik[component] * w / std::pow(T, 2);
    }
    double Kihara::d2fdT2(double r, double T, std::string component) 
    {
        // hydrate cage cell potential w(r) = omega(r) [eq. 3.43a]
        double w = this->w(r, component);

        return std::exp(-eik[component] / T * w) * std::pow(r, 2) * (std::pow(eik[component] * w / std::pow(T, 2), 2) 
                                                                    - 2.*eik[component] * w / std::pow(T, 3));
    }
    double Kihara::dFdT(double T, std::string component) 
    {
        // integrals solved numerically with simpson's rule
        double s = 0.;
        int steps = 20;
        double h = (R1-R0)/steps;  // interval
    
    	// hi
        for (int i = 0; i < steps; i++) 
        {
            double hix = this->dfdT(R0 + i*h, T, component);
            double hixh2 = this->dfdT(R0 + (i+0.5)*h, T, component);
            double hixh = this->dfdT(R0 + (i+1)*h, T, component);
            s += h*((hix + 4*hixh2 + hixh) / 6);
            if (this->f(R0 + (i+1)*h, T, component) < 1e-200) { break; } // otherwise, integral becomes too large
        }
    	return s;
    }
    double Kihara::d2FdT2(double T, std::string component) 
    {
        // integrals solved numerically with simpson's rule
        double s = 0.;
        int steps = 20;
        double h = (R1-R0)/steps;  // interval
    
    	// hi
        for (int i = 0; i < steps; i++) 
        {
            double hix = this->d2fdT2(R0 + i*h, T, component);
            double hixh2 = this->d2fdT2(R0 + (i+0.5)*h, T, component);
            double hixh = this->d2fdT2(R0 + (i+1)*h, T, component);
            s += h*((hix + 4*hixh2 + hixh) / 6);
            if (this->f(R0 + (i+1)*h, T, component) < 1e-200) { break; } // otherwise, integral becomes too large
        }
    	return s;
    }
    int Kihara::test_derivatives(double T, std::string component, double tol, bool verbose)
    {
        int error_output = 0;
        double dF = this->dFdT(T, component);
        double d2F = this->d2FdT2(T, component);

        double d, dT{ 1e-5 };
        double F_ = this->F(T-dT, component);
        double F1 = this->F(T+dT, component);
        double dF_num = (F1-F_)/(2*dT);
        double dF_ = this->dFdT(T-dT, component);
        double dF1 = this->dFdT(T+dT, component);
        double d2F_num = (dF1-dF_)/(2*dT);

        d = std::log(std::fabs(dF + 1e-15)) - std::log(std::fabs(dF_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard Kihara dF/dT", {dF, dF_num, d}); error_output++; }
        d = std::log(std::fabs(d2F + 1e-15)) - std::log(std::fabs(d2F_num + 1e-15));
        if (verbose || !(std::fabs(d) < tol)) { print("Ballard Kihara d2F/dT2", {d2F, d2F_num, d}); error_output++; }
        
        return error_output;
    }
} // namespace ballard

Ballard::Ballard(CompData& comp_data, std::string hydrate_type) : VdWP(comp_data, hydrate_type)
{
    zn = ballard::zn[phase];
    n_shells = ballard::n_shells[phase];
    Rn = ballard::Rn[phase];
}

void Ballard::init_PT(double p_, double T_) {
    // Fugacity of water in hydrate phase following modified VdW-P (Ballard, 2002)
    // Initializes all composition-independent parameters:
    // - Ideal gas Gibbs energy of water [eq. 3.1-3.4]
    // - Gibbs energy of water in empty hydrate lattice [eq. 3.47]
    // Each time calculating fugacity of water in hydrate, evaluate:
    // - Contribution of cage occupancy to total energy of hydrate [eq. 3.44]
    // - Activity of water in the hydrate phase [eq. 4.38]
    // - Chemical potential of water in hydrate [eq. 4.35]
    if (p_ != p || T_ != T)
    {
        this->p = p_;
        this->T = T_;

        // Calculate Langmuir constant of each guest k in each cage m
        C_km = this->calc_Ckm();

        // Ideal gas Gibbs energy
        // ideal_gas::IdealGas ig = ideal_gas::IdealGas("H2O");
        ballard::IG ig = ballard::IG("H2O");
        double g_wo = ballard::gw_00/(ballard::R * ballard::T_0); // Gibbs energy of formation of water in ideal gas at reference conditions [constant]
        double h_wo = ig.H(T); // molar enthalpy of formation of water in ideal gas [eq. 3.3]
        g_w0 = g_wo - h_wo; // Gibbs energy of formation of water in ideal gas [eq. 3.2]

        // Gibbs energy of water in empty hydrate lattice
        g_B0 = ballard::g_B0[phase] / (ballard::R * ballard::T_0); // Gibbs energy of formation of empty lattice at reference conditions [constant]
        ballard::HB hb = ballard::HB(phase);
        h_B0 = hb.F(T);   // molar enthalpy of formation of empty lattice [eq. 4.40]
    }
}

void Ballard::init_VT(double, double)
{
	std::cout << "No implementation of volume-based calculations exists for Ballard, aborting.\n";
	exit(1);
}

double Ballard::V(double p_, double T_, std::vector<double>& n_, int start_idx, bool pt)
{
    // Molar volume of hydrate
    (void) n_;
    (void) start_idx;
    (void) pt;
    ballard::VB vb = ballard::VB(phase);
    return vb.f(p_, T_) * (ballard::R * 1E-5 * T_);
}

double Ballard::fw(std::vector<double>& fi) {
    // Calculate fugacity of water in the hydrate
    // - Contribution of cage occupancy to total energy of hydrate [eq. 3.44]
    // - Standard state + change caused by cage filling for:
    //      - Gibbs energy of empty hydrate lattice at T0 [eq. 3.47 + eq. 4.39]
    //      - Enthalpy of empty hydrate lattice at T [eq. 4.40 + eq. 4.39]
    //      - Volume of hydrate at P [eq. 3.47 + eq. 4.41-4.44]
    // - Chemical potential of water in the hydrate [eq. 4.35]

    // Contribution of cage occupancy to total energy of hydrate
    this->f = fi;
    double dmu_H = this->calc_dmuH();

    // Gibbs energy, enthalpy, volume
    ballard::VB vb = ballard::VB(phase);
    // ballard::VH vh = ballard::VH(phase, components);
    
    // double dv = vh.f(ballard::P_0, ballard::T_0, theta_km)-vb.f(ballard::P_0, ballard::T_0); // molar volume difference between real hydrate (vH) and standard hydrate (vB) at reference conditions
    // double dg_w0b = ballard::a[phase]*dv*1e1; // eq. 4.39, used in eq. 4.38
    // double dh_w0b = ballard::b[phase]*dv*1e1; // eq. 4.39, used in eq. 4.38

    g_B = g_B0; // + dg_w0b/(ballard::R * ballard::T_0);
    h_B = h_B0; // + dh_w0b/(ballard::R) * (1./T - 1./ballard::T_0);
    double v_H = vb.F(p, T);
    // double v_H = vh.F(p, T, theta_km);

    // Fugacity of water in hydrate
    double mu_wH = g_B - h_B + v_H + dmu_H; // Chemical potential of water in hydrate [eq. 4.35]
    double f_w0 = 1.; // fugacity of ideal gas at reference pressure
    fi[water_index] = f_w0 * std::exp(mu_wH - g_w0); // eq. 4.47
    return fi[water_index];
}
double Ballard::dfw_dP(std::vector<double>& dfidP) {
    // Derivative of water fugacity w.r.t. P

    // Calculate derivative of mu_wH w.r.t. P
    double ddmu_wH = this->ddmuH_dP(dfidP);

    // Derivatives of Gibbs energy, enthalpy, volume
    ballard::VB vb = ballard::VB(phase);
    // ballard::VH vh = ballard::VH(phase, components);
    
    // g_B =     
    double dg_B = 0.;
    double dh_B = 0.;
    double dv_H = vb.dFdP(p, T);
    // double dv_H = vh.dFdP(p, T, theta_km);
    
    double dmu_wH = dg_B - dh_B + dv_H + ddmu_wH; // Chemical potential of water in hydrate [eq. 4.35]

    // Calculate derivative of fwH w.r.t. P
    dfidP[water_index] = f[water_index] * dmu_wH;
    return dfidP[water_index];
}
double Ballard::dfw_dT(std::vector<double>& dfidT) {
    // Derivative of water fugacity w.r.t. T

    // Calculate derivative of mu_wH w.r.t. T
    double ddmu_wH = this->ddmuH_dT(dfidT);

    // Derivatives of Gibbs energy, enthalpy, volume
    ballard::HB hb = ballard::HB(phase);
    ballard::VB vb = ballard::VB(phase);
    // ballard::VH vh = ballard::VH(phase, components);
    
    // g_B = 
    // double dv = vh.f(ballard::P_0, ballard::T_0, theta_km)-vb.f(ballard::P_0, ballard::T_0); // molar volume difference between real hydrate (vH) and standard hydrate (vB) at reference conditions
    // double dh_w0b = ballard::b[phase]*dv*1e1; // eq. 4.39, used in eq. 4.38  // b in [J/cm3]
    
    double dg_B = 0.;
    
    double dh_B0 = hb.dFdT(T);   // molar enthalpy of formation of empty lattice [eq. 4.40]
    double dh_B = dh_B0; // - dh_w0b/(ballard::R * std::pow(T, 2));

    double dv_H = vb.dFdT(p, T);
    // double dv_H = vh.dFdT(p, T, theta_km, dthetadT);
    
    double dmu_wH = dg_B - dh_B + dv_H + ddmu_wH; // Chemical potential of water in hydrate [eq. 4.35]

    // Derivative of ideal gas Gibbs energy
    ballard::IG ig = ballard::IG("H2O");
    double dh_wo = ig.dHdT(T); // molar enthalpy of formation of water in ideal gas [eq. 3.3]
    double dg_w0 = -dh_wo; // Gibbs energy of formation of water in ideal gas [eq. 3.2]

    // Calculate derivative of fwH w.r.t. T
    dfidT[water_index] = f[water_index] * (dmu_wH - dg_w0);
    return dfidT[water_index];
}
double Ballard::d2fw_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) {
    // Derivative of water fugacity w.r.t. P

    // Calculate derivative of mu_wH w.r.t. P
    double ddmu_wH = this->ddmuH_dP(dfidP);
    double d2dmu_wHdPdT = this->d2dmuH_dPdT(dfidP, dfidT, d2fidPdT);

    // Derivatives of Gibbs energy, enthalpy, volume
    ballard::VB vb = ballard::VB(phase);
    // ballard::VH vh = ballard::VH(phase, components);
    
    // g_B =     
    double dg_B = 0.;
    double dh_B = 0.;
    double dv_H = vb.dFdP(p, T);
    double d2v_H = vb.d2FdPdT(p, T);
    
    double dmu_wH_dP = dg_B - dh_B + dv_H + ddmu_wH; // Chemical potential of water in hydrate [eq. 4.35]
    double d2mu_wH_dPdT = d2v_H + d2dmu_wHdPdT;

    // Calculate derivative of fwH w.r.t. P
    d2fidPdT[water_index] = dfidT[water_index] * dmu_wH_dP + f[water_index] * d2mu_wH_dPdT;
    return d2fidPdT[water_index];
}
double Ballard::d2fw_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2) {
    // Derivative of water fugacity w.r.t. T

    // Calculate derivative of mu_wH w.r.t. T
    double ddmu_wH = this->ddmuH_dT(dfidT);
    double d2dmu_wH = this->d2dmuH_dT2(dfidT, d2fidT2);

    // Derivatives of Gibbs energy, enthalpy, volume
    ballard::HB hb = ballard::HB(phase);
    ballard::VB vb = ballard::VB(phase);
    // ballard::VH vh = ballard::VH(phase, components);
    
    // g_B = 
    // double dv = vh.f(ballard::P_0, ballard::T_0, theta_km)-vb.f(ballard::P_0, ballard::T_0); // molar volume difference between real hydrate (vH) and standard hydrate (vB) at reference conditions
    // double dh_w0b = ballard::b[phase]*dv*1e1; // eq. 4.39, used in eq. 4.38  // b in [J/cm3]
    
    double dg_B = 0.;
    double d2g_B = 0.;
    
    double dh_B0 = hb.dFdT(T);   // molar enthalpy of formation of empty lattice [eq. 4.40]
    double d2h_B0 = hb.d2FdT2(T);
    double dh_B = dh_B0; // - dh_w0b/(ballard::R * std::pow(T, 2));
    double d2h_B = d2h_B0;

    double dv_H = vb.dFdT(p, T);
    double d2v_H = vb.d2FdT2(p, T);
    // double dv_H = vh.dFdT(p, T, theta_km, dthetadT);
    
    double dmu_wH = dg_B - dh_B + dv_H + ddmu_wH; // Chemical potential of water in hydrate [eq. 4.35]
    double d2mu_wH = d2g_B - d2h_B + d2v_H + d2dmu_wH; // Chemical potential of water in hydrate [eq. 4.35]

    // Derivative of ideal gas Gibbs energy
    ballard::IG ig = ballard::IG("H2O");
    double dh_wo = ig.dHdT(T); // molar enthalpy of formation of water in ideal gas [eq. 3.3]
    double d2h_wo = ig.d2HdT2(T);
    double dg_w0 = -dh_wo; // Gibbs energy of formation of water in ideal gas [eq. 3.2]
    double d2g_w0 = -d2h_wo;

    // Calculate derivative of fwH w.r.t. T
    d2fidT2[water_index] = dfidT[water_index] * (dmu_wH - dg_w0) + f[water_index] * (d2mu_wH - d2g_w0);
    return d2fidT2[water_index];
}
std::vector<double> Ballard::dfw_dxj(std::vector<double>& dfidxj) 
{
    // Derivative of water fugacity w.r.t. x_j

    // dfw/dxk = exp(dmu/RT) * d/dxk (dmu/RT)
    std::vector<double> dfwdxk(nc);
    std::vector<double> ddmuHdxk = this->ddmuH_dxj(dfidxj);
    for (int k = 0; k < nc; k++)
    {
        dfwdxk[k] = f[water_index] * ddmuHdxk[k];
    }
    return dfwdxk;
}
std::vector<double> Ballard::d2fw_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj)
{
    // Derivative of water fugacity w.r.t. x_j

    // d2fw/dTdxk = exp(dmu/RT) * d/dxk (dmu/RT)
    std::vector<double> d2fwdTdxk(nc);
    std::vector<double> ddmuHdxk = this->ddmuH_dxj(dfidxj);
    std::vector<double> d2dmuHdTdxk = this->d2dmuH_dTdxj(dfidT, dfidxj, d2fidTdxj);
    for (int k = 0; k < nc; k++)
    {
        d2fwdTdxk[k] = dfidT[water_index] * ddmuHdxk[k] + f[water_index] * d2dmuHdTdxk[k];
    }
    return d2fwdTdxk;
}

std::vector<double> Ballard::calc_Ckm() {
    // Calculate Langmuir constant of guest k in each cage m
    this->C_km = std::vector<double>(nc*n_cages, 0.);
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            R1_index = 0;
            for (int m = 0; m < n_cages; m++) 
            {
                ballard::Kihara kih(ballard::l0, Rn[R1_index]-ballard::ai[this->compdata.components[k]], phase);
                kih.R1_index = R1_index;
                kih.cage_index = m;

                double Cim_int = kih.F(T, this->compdata.components[k]);
                C_km[m*nc + k] = 4 * M_PI / (M_kB * T) * Cim_int * 1e5;
                R1_index += n_shells[m];
            }
        }
    }
    return C_km;
}
std::vector<double> Ballard::dCkm_dP() {
    // Derivatives of Langmuir constant C_km w.r.t. P are zero
    this->dCkmdP = std::vector<double>(nc*n_cages, 0.);
    return dCkmdP;
}
std::vector<double> Ballard::dCkm_dT() {
    // Calculate derivative of Langmuir constant C_km w.r.t. T
    this->dCkmdT = std::vector<double>(nc*n_cages, 0.);

    double invT = 1./T;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            R1_index = 0;
            for (int m = 0; m < n_cages; m++) 
            {
                ballard::Kihara kih(ballard::l0, Rn[R1_index]-ballard::ai[this->compdata.components[k]], phase);
                kih.R1_index = R1_index;
                kih.cage_index = m;

                double ddT_Cim_int = kih.dFdT(T, this->compdata.components[k]);
                dCkmdT[m*nc + k] = (4. * M_PI / M_kB * ddT_Cim_int * 1e5 - C_km[m*nc + k]) * invT;
                R1_index += n_shells[m];
            }
        }
    }
    return dCkmdT;
}
std::vector<double> Ballard::d2Ckm_dPdT() {
    // Second derivatives of Langmuir constant C_km w.r.t. P and T are zero
    this->d2CkmdPdT = std::vector<double>(nc*n_cages, 0.);
    return d2CkmdPdT;
}
std::vector<double> Ballard::d2Ckm_dT2() {
    // Calculate second derivative of Langmuir constant C_km w.r.t. T
    this->d2CkmdT2 = std::vector<double>(nc*n_cages, 0.);

    double invT = 1./T;
    for (int k = 0; k < nc; k++)
    {
        if (k != water_index)
        {
            R1_index = 0;
            for (int m = 0; m < n_cages; m++) 
            {
                ballard::Kihara kih(ballard::l0, Rn[R1_index]-ballard::ai[this->compdata.components[k]], phase);
                kih.R1_index = R1_index;
                kih.cage_index = m;

                double ddT_Cim_int = kih.dFdT(T, this->compdata.components[k]);
                double d2dT2_Cim_int = kih.d2FdT2(T, this->compdata.components[k]);
                d2CkmdT2[m*nc + k] = (4. * M_PI / M_kB * d2dT2_Cim_int * 1e5 - dCkmdT[m*nc + k]) * invT 
                                   - (4. * M_PI / M_kB * ddT_Cim_int * 1e5 - C_km[m*nc + k]) * std::pow(invT, 2);
                R1_index += n_shells[m];
            }
        }
    }
    return d2CkmdT2;
}

int Ballard::derivatives_test(double p_, double T_, std::vector<double>& x_, double tol, bool verbose)
{
    int error_output = VdWP::derivatives_test(p_, T_, x_, tol, verbose);

    ballard::IG ig = ballard::IG("H2O");
    ballard::HB hb = ballard::HB(phase);
    ballard::VB vb = ballard::VB(phase);
    // ballard::VH vh = ballard::VH(phase, components);
    
    error_output += ig.test_derivatives(T_, tol);
    error_output += hb.test_derivatives(T_, tol);
    error_output += vb.test_derivatives(p_, T_, tol);
    // error_output += vh.test_derivatives(p_, T_, tol);

    ballard::Kihara kih(ballard::l0, Rn[0]-ballard::ai["CO2"], phase);
    kih.R1_index = 0;
    kih.cage_index = 0;

    error_output += kih.test_derivatives(T_, "CO2", tol);

    return error_output;
}
