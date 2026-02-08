#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <complex>
#include <memory>

#include "dartsflash/eos/eos_params.hpp"
#include "dartsflash/eos/eos.hpp"
#include "dartsflash/eos/ideal.hpp"
#include "dartsflash/eos/helmholtz/helmholtz.hpp"
#include "dartsflash/eos/helmholtz/cubic.hpp"
#include "dartsflash/eos/iapws/iapws95.hpp"
#include "dartsflash/eos/iapws/iapws_ice.hpp"
#include "dartsflash/eos/aq/aq.hpp"
#include "dartsflash/eos/aq/jager.hpp"
#include "dartsflash/eos/aq/ziabakhsh.hpp"
#include "dartsflash/eos/vdwp/vdwp.hpp"
#include "dartsflash/eos/vdwp/ballard.hpp"
#include "dartsflash/eos/vdwp/munck.hpp"
#include "dartsflash/eos/solid/solid.hpp"

namespace py = pybind11;

template <class EoSBase = EoS> class PyEoS : public EoSBase, public py::trampoline_self_life_support {
public:
    using EoSBase::EoSBase;

    std::unique_ptr<EoS> getCopy() override {
        PYBIND11_OVERRIDE_PURE(
            std::unique_ptr<EoS>, EoS, getCopy,
        );
    }

    EoS::RootFlag is_root_type(bool& is_below_spinodal) override {
        PYBIND11_OVERRIDE(
            EoS::RootFlag, EoS, is_root_type, is_below_spinodal
        );
    }
    EoS::RootSelect select_root(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE(
            EoS::RootSelect, EoS, select_root, n_it
        );
    }

    void init_PT(double p_, double T_) override {
        PYBIND11_OVERRIDE_PURE(
            void, EoS, init_PT, p_, T_
        );
    }
    void solve_PT(std::vector<double>::iterator n_it, bool second_order) override {
        PYBIND11_OVERRIDE_PURE(
            void, EoS, solve_PT, n_it, second_order
        );
    }
    void solve_PT(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override {
        PYBIND11_OVERRIDE(
            void, EoS, solve_PT, p_, T_, n_, start_idx, second_order
        );
    }
    void init_VT(double V_, double T_) override {
        PYBIND11_OVERRIDE_PURE(
            void, EoS, init_VT, V_, T_
        );
    }
    void solve_VT(std::vector<double>::iterator n_it, bool second_order) override {
        PYBIND11_OVERRIDE_PURE(
            void, EoS, solve_VT, n_it, second_order
        );
    }
    void solve_VT(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool second_order=true) override {
        PYBIND11_OVERRIDE(
            void, EoS, solve_VT, V_, T_, n_, start_idx, second_order
        );
    }

    double lnphii(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, EoS, lnphii, i
        );
    }
    double dlnphii_dP(int i) override {
        PYBIND11_OVERRIDE(
            double, EoS, dlnphii_dP, i
        );
    }
	double dlnphii_dT(int i) override {
        PYBIND11_OVERRIDE(
            double, EoS, dlnphii_dT, i
        );
    }
    double dlnphii_dnj(int i, int j) override {
        PYBIND11_OVERRIDE(
            double, EoS, dlnphii_dnj, i, j
        );
    }
    double d2lnphii_dPdT(int i) override {
        PYBIND11_OVERRIDE(
            double, EoS, d2lnphii_dPdT, i
        );
    }
    double d2lnphii_dT2(int i) override {
        PYBIND11_OVERRIDE(
            double, EoS, d2lnphii_dT2, i
        );
    }
    double d2lnphii_dTdnj(int i, int j) override {
        PYBIND11_OVERRIDE(
            double, EoS, d2lnphii_dTdnj, i, j
        );
    }

    std::vector<double> dlnphi_dP() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, dlnphi_dP
        );
    }
    std::vector<double> dlnphi_dT() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, dlnphi_dT
        );
    }
    std::vector<double> dlnphi_dn() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, dlnphi_dn
        );
    }
    std::vector<double> d2lnphi_dPdT() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, d2lnphi_dPdT
        );
    }
    std::vector<double> d2lnphi_dT2() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, d2lnphi_dT2
        );
    }
    std::vector<double> d2lnphi_dTdn() override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, d2lnphi_dTdn
        );
    }

    double cpi(double T_, int i) override {
        PYBIND11_OVERRIDE(
            double, EoS, cpi, T_, i
        );
    }
    double hi(double T_, int i) override {
        PYBIND11_OVERRIDE(
            double, EoS, hi, T_, i
        );
    }
    double si(double X, double T_, int i, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, EoS, si, X, T_, i, pt
        );
    }
    std::vector<double> dSi_dni(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, dSi_dni, X, T_, n_, start_idx, pt
        );
    }

    double Gr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, EoS, Gr, X, T_, n_, start_idx, pt
        );
    }
    double Hr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, EoS, Hr, X, T_, n_, start_idx, pt
        );
    }
    double Sr(double X, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, EoS, Sr, X, T_, n_, start_idx, pt
        );
    }
    // double Ar(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
    //     PYBIND11_OVERRIDE(
    //         double, EoS, Ar, X, T_, n_, start_idx, pt
    //     );
    // }
    // double Ur(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
    //     PYBIND11_OVERRIDE(
    //         double, EoS, Ur, X, T_, n_, start_idx, pt
    //     );
    // }

    std::vector<double> lnphi0(double X, double T_, bool pt=true) override {
        PYBIND11_OVERRIDE(
            std::vector<double>, EoS, lnphi0, X, T_, pt
        );
    }
};

template <class HelmholtzEoSBase = HelmholtzEoS> class PyHelmholtz : public PyEoS<HelmholtzEoSBase> {
public:
    using PyEoS<HelmholtzEoSBase>::PyEoS; // inherit constructors

    EoS::RootFlag is_root_type(bool& is_below_spinodal) override {
        PYBIND11_OVERRIDE(
            EoS::RootFlag, HelmholtzEoS, is_root_type, is_below_spinodal
        );
    }
    EoS::RootSelect select_root(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE(
            EoS::RootSelect, HelmholtzEoS, select_root, n_it
        );
    }
    EoS::RootFlag identify_root(bool& is_below_spinodal) override {
        PYBIND11_OVERRIDE_PURE(
            EoS::RootFlag, HelmholtzEoS, identify_root, is_below_spinodal
        );
    }
    HelmholtzEoS::CriticalPoint critical_point(std::vector<double>& n_) override {
        PYBIND11_OVERRIDE_PURE(
            HelmholtzEoS::CriticalPoint, HelmholtzEoS, critical_point, n_
        );
    }
    bool is_critical(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE_PURE(
            bool, HelmholtzEoS, is_critical, p_, T_, n_, start_idx, pt
        );
    }

    void zeroth_order(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE_PURE(
            void, HelmholtzEoS, zeroth_order, n_it
        );
    }
    void zeroth_order(std::vector<double>::iterator n_it, double V_) override {
        PYBIND11_OVERRIDE_PURE(
            void, HelmholtzEoS, zeroth_order, n_it, V_
        );
    }
    void zeroth_order(double V_) override {
        PYBIND11_OVERRIDE_PURE(
            void, HelmholtzEoS, zeroth_order, V_
        );
    }
    void first_order(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE_PURE(
            void, HelmholtzEoS, first_order, n_it
        );
    }
    void second_order(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE_PURE(
            void, HelmholtzEoS, second_order, n_it
        );
    }
    
    double V() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, V, 
        );
    }
    double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, V, p_, T_, n_, start_idx, pt
        );
    }
    double P(double V_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=false) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, P, V_, T_, n_, start_idx, pt
        );
    }
    std::vector<std::complex<double>> Z() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<std::complex<double>>, HelmholtzEoS, Z, 
        );
    }
    double dZ_dP() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, dZ_dP,
        );
    }
    double d2Z_dP2() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2Z_dP2,
        );
    }

    double lnphii(int i) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, lnphii, i
        );
    }
    double dlnphii_dP(int i) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, dlnphii_dP, i
        );
    }
    double dlnphii_dT(int i) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, dlnphii_dT, i
        );
    }
    double dlnphii_dnj(int i, int j) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, dlnphii_dnj, i, j
        );
    }

    double F() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, F, 
        );
    }
    double dF_dV() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, dF_dV, 
        );
    }
    double dF_dT() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, dF_dT, 
        );
    }
    double dF_dni(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, dF_dni, i
        );
    }
    double d2F_dnidnj(int i, int j) override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dnidnj, i, j 
        );
    }
    double d2F_dTdni(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dTdni, i 
        );
    }
    double d2F_dVdni(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dVdni, i 
        );
    }
    double d2F_dTdV() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dTdV, 
        );
    }
    double d2F_dV2() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dV2, 
        );
    }
    double d2F_dT2() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dT2, 
        );
    }
    double d3F_dV3() override {
        PYBIND11_OVERRIDE_PURE(
            double, HelmholtzEoS, d2F_dT2,
        );
    }

    double Gr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, Gr, X, T_, n_, start_idx, pt
        );
    }
    double Hr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, Hr, X, T_, n_, start_idx, pt
        );
    }
    double Sr(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
        PYBIND11_OVERRIDE(
            double, HelmholtzEoS, Sr, X, T_, n_, start_idx, pt
        );
    }
    // double Ar(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
    //     PYBIND11_OVERRIDE(
    //         double, HelmholtzEoS, Ar, X, T_, n_, start_idx, pt
    //     );
    // }
    // double Ur(double X, double T_, std::vector<double>& n_, int start_idx, bool pt=true) override {
    //     PYBIND11_OVERRIDE(
    //         double, HelmholtzEoS, Ur, X, T_, n_, start_idx, pt
    //     );
    // }
};

template <class CubicEoSBase = CubicEoS> class PyCubic : public PyHelmholtz<CubicEoSBase> {
public:
    using PyHelmholtz<CubicEoSBase>::PyHelmholtz;

    EoS::RootFlag identify_root(bool& is_below_spinodal) override {
        PYBIND11_OVERRIDE(
            EoS::RootFlag, CubicEoS, identify_root, is_below_spinodal
        );
    }
    HelmholtzEoS::CriticalPoint critical_point(std::vector<double>& n_) override {
        PYBIND11_OVERRIDE(
            HelmholtzEoS::CriticalPoint, CubicEoS, critical_point, n_
        );
    }
    bool is_critical(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE(
            bool, CubicEoS, is_critical, p_, T_, n_, start_idx, pt
        );
    }

    void zeroth_order(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE(
            void, CubicEoS, zeroth_order, n_it
        );
    }
    void zeroth_order(std::vector<double>::iterator n_it, double V_) override {
        PYBIND11_OVERRIDE(
            void, CubicEoS, zeroth_order, n_it, V_
        );
    }
    void zeroth_order(double V_) override {
        PYBIND11_OVERRIDE(
            void, CubicEoS, zeroth_order, V_
        );
    }
    void first_order(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE(
            void, CubicEoS, first_order, n_it
        );
    }
    void second_order(std::vector<double>::iterator n_it) override {
        PYBIND11_OVERRIDE(
            void, CubicEoS, second_order, n_it
        );
    }

    double V() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, V, 
        );
    }
    std::vector<std::complex<double>> Z() override {
        PYBIND11_OVERRIDE(
            std::vector<std::complex<double>>, CubicEoS, Z, 
        );
    }
    double dZ_dP() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, dZ_dP,
        );
    }
    double d2Z_dP2() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2Z_dP2,
        );
    }

    double F() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, F, 
        );
    }
    double dF_dV() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, dF_dV, 
        );
    }
    double dF_dT() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, dF_dT, 
        );
    }
    double dF_dni(int i) override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, dF_dni, i
        );
    }
    double d2F_dnidnj(int i, int j) override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2F_dnidnj, i, j 
        );
    }
    double d2F_dTdni(int i) override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2F_dTdni, i 
        );
    }
    double d2F_dVdni(int i) override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2F_dVdni, i 
        );
    }
    double d2F_dTdV() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2F_dTdV, 
        );
    }
    double d2F_dV2() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2F_dV2, 
        );
    }
    double d2F_dT2() override {
        PYBIND11_OVERRIDE(
            double, CubicEoS, d2F_dT2, 
        );
    }
};

template <class VdWPBase = VdWP> class PyVdWP : public PyEoS<VdWPBase> {
public:
    using PyEoS<VdWPBase>::PyEoS;

    double lnphii(int i) override {
        PYBIND11_OVERRIDE(
            double, VdWP, lnphii, i
        );
    }

    double V(double p_, double T_, std::vector<double>& n_, int start_idx=0, bool pt=true) override {
        PYBIND11_OVERRIDE_PURE(
            double, VdWP, V, p_, T_, n_, start_idx, pt
        );
    }

    double fw(std::vector<double>& fi) override {
        PYBIND11_OVERRIDE_PURE(
            double, VdWP, fw, fi
        );
    }
    double dfw_dP(std::vector<double>& dfidP) override {
        PYBIND11_OVERRIDE_PURE(
            double, VdWP, dfw_dP, dfidP
        );
    }
    double dfw_dT(std::vector<double>& dfidT) override {
        PYBIND11_OVERRIDE_PURE(
            double, VdWP, dfw_dT, dfidT
        );
    }
    double d2fw_dPdT(std::vector<double>& dfidP, std::vector<double>& dfidT, std::vector<double>& d2fidPdT) override {
        PYBIND11_OVERRIDE_PURE(
            double, VdWP, d2fw_dPdT, dfidP, dfidT, d2fidPdT
        );
    }
    double d2fw_dT2(std::vector<double>& dfidT, std::vector<double>& d2fidT2) override {
        PYBIND11_OVERRIDE_PURE(
            double, VdWP, d2fw_dT2, dfidT, d2fidT2
        );
    }
    std::vector<double> dfw_dxj(std::vector<double>& dfidxj) override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, dfw_dxj, dfidxj
        );
    }
    std::vector<double> d2fw_dTdxj(std::vector<double>& dfidT, std::vector<double>& dfidxj, std::vector<double>& d2fidTdxj) override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, d2fw_dTdxj, dfidT, dfidxj, d2fidTdxj
        );
    }

    std::vector<double> calc_Ckm() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, calc_Ckm
        );
    }
    std::vector<double> dCkm_dP() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, dCkm_dP
        );
    }
    std::vector<double> dCkm_dT() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, dCkm_dT
        );
    }
    std::vector<double> d2Ckm_dPdT() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, d2Ckm_dPdT
        );
    }
    std::vector<double> d2Ckm_dT2() override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, VdWP, d2Ckm_dT2
        );
    }
};

template <class AQBaseBase = AQBase> class PyAQ : public AQBaseBase, public py::trampoline_self_life_support {
public:
    using AQBaseBase::AQBaseBase;

    std::shared_ptr<AQBase> getCopy() override {
        PYBIND11_OVERRIDE_PURE(
            std::shared_ptr<AQBase>, AQBase, getCopy,
        );
    }

    void init_PT(double p_, double T_, AQEoS::CompType comp_type) override {
        PYBIND11_OVERRIDE_PURE(
            void, AQBase, init_PT, p_, T_, comp_type
        );
    }
    void solve_PT(std::vector<double>& x_, bool second_order, AQEoS::CompType comp_type) override {
        PYBIND11_OVERRIDE_PURE(
            void, AQBase, solve_PT, x_, second_order, comp_type
        );
    }

    double lnphii(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, lnphii, i
        );
    }
    double dlnphii_dP(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, dlnphii_dP, i
        );
    }
    double dlnphii_dT(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, dlnphii_dT, i
        );
    }
    double d2lnphii_dPdT(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, d2lnphii_dPdT, i
        );
    }
    double d2lnphii_dT2(int i) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, d2lnphii_dT2, i
        );
    }
    double dlnphii_dxj(int i, int j) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, dlnphii_dxj, i, j
        );
    }
    std::vector<double> d2lnphii_dTdxj(int i) override {
        PYBIND11_OVERRIDE_PURE(
            std::vector<double>, AQBase, d2lnphii_dTdxj, i
        );
    }

    double lnphi0(double X, double T_, bool pt=true) override {
        PYBIND11_OVERRIDE_PURE(
            double, AQBase, lnphi0, X, T_, pt
        );
    }
};

void pybind_eos(py::module& m)
{
    using namespace pybind11::literals;  // bring in '_a' literal

    // Expose EoSParams class
    py::classh<EoSParams> eos_params(m, "EoSParams", R"pbdoc(
            This is a class that contains all required flash/stability parameters for each EoS object.
            )pbdoc");
    
    eos_params
        .def_readwrite("trial_comps", &EoSParams::trial_comps, "Trial comps associated with EoS object")
        .def_readwrite("root_order", &EoSParams::root_order, "Order of EoS::RootFlag types in FlashResults, default is STABLE (unordered)")
        .def_readwrite("rich_phase_order", &EoSParams::rich_phase_order, "Order of rich phases component idxs in FlashResults, default is empty (unordered)")
        .def_readwrite("rich_phase_composition", &EoSParams::rich_phase_composition, "Composition to be considered rich phase, default is 0.5")
        .def_readwrite("use_gmix", &EoSParams::use_gmix, "Flag to use minimum of gmix for phase split rather than stationary point")
        .def_readwrite("stability_tol", &EoSParams::stability_tol, "Tolerance for stability norm")
        .def_readwrite("stability_switch_tol", &EoSParams::stability_switch_tol, "Tolerance for switch to Newton in stability")
        .def_readwrite("stability_switch_diff", &EoSParams::stability_switch_diff, "If decrease in log(norm) between two SSI iterations is below this number (and tol < switch_tol), switch to Newton - make use of effectiveness of SSI")
        .def_readwrite("stability_line_tol", &EoSParams::stability_line_tol, "Tolerance for line search in stability")
        .def_readwrite("stability_max_iter", &EoSParams::stability_max_iter, "Maximum number of iterations for stability")
        .def_readwrite("stability_line_iter", &EoSParams::stability_line_iter, "Maximum number of iterations for line search in stability")
        
        .def("set_active_components", &EoSParams::set_active_components, R"pbdoc(
            :param idxs: List of active component indices for EoS object
            :type idxs: list
            )pbdoc")
        ;

    // Expose EoS derived classes from EoS base class
    py::class_<EoS, PyEoS<>, py::smart_holder> eos(m, "EoS", R"pbdoc(
            This is a base class for Equations of State (EoS).

            Each EoS child class overrides the methods for:
            - (P,T)-dependent parameters in `init_PT(p, T)`
            - (n)-dependent parameters in `solve_PT(n)`
            - Expressions for `lnphi(i)` and derivatives w.r.t. P, T, nj
            - Expressions for Gibbs free energy `Gr` and enthalpy `Hr`
            )pbdoc");

    py::enum_<EoS::RootFlag>(eos, "RootFlag", "Flag for roots to be selected in EoS")
        .value("STABLE", EoS::RootFlag::STABLE)
        .value("MIN", EoS::RootFlag::MIN)
        .value("MAX", EoS::RootFlag::MAX)
        .export_values()
        ;

    py::enum_<EoS::Property>(eos, "Property", "Enum for property to be calculated from EoS")
        .value("ENTROPY", EoS::Property::ENTROPY)
        .value("GIBBS", EoS::Property::GIBBS)
        .value("ENTHALPY", EoS::Property::ENTHALPY)
        .value("HELMHOLTZ", EoS::Property::HELMHOLTZ)
        .value("INTERNAL_ENERGY", EoS::Property::INTERNAL_ENERGY)
        .export_values()
        ;

    eos.def(py::init<CompData&>(), R"pbdoc(
            This is the constructor of the EoS base class for multicomponent phases.

            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)

        .def("set_eos_range", &EoS::set_eos_range, R"pbdoc(
            Specify composition range for EoS
            
            :param i: Component index
            :type i: int
            :param range: Composition lower and upper bound
            :type range: list
            )pbdoc", "i"_a, "range"_a)
        .def("set_root_flag", &EoS::set_root_flag, R"pbdoc(
            :param flag: Root choice, -1/STABLE) lowest Gibbs energy, 0/MIN) minimum (L), 1/MAX) maximum (V); default is -1
            :type flag: EoS.RootFlag
            )pbdoc", "root_flag"_a)
        .def("get_comp_data", &EoS::get_comp_data, "Getter for CompData object reference")
        .def("is_root_type", py::overload_cast<double, double, std::vector<double>&, bool&, int, bool>(&EoS::is_root_type), R"pbdoc(
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Root type
            :rtype: EoS.RootFlag
            )pbdoc", "p"_a, "T"_a, "n"_a, "is_below_spinodal"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        
        .def("lnphi", py::overload_cast<double, double, std::vector<double>&>(&EoS::lnphi), R"pbdoc(
            :returns: List of lnphi for each component at (P,T,n)
            :rtype: list
            )pbdoc", "p"_a, "T"_a, "n"_a)
        .def("dlnphi_dP", &EoS::dlnphi_dP, R"pbdoc(
            :returns: List of dlnphi/dP for each component at (P,T,n)
            :rtype: list
            )pbdoc")
        .def("dlnphi_dT", &EoS::dlnphi_dT, R"pbdoc(
            :returns: List of dlnphi/dT for each component at (P,T,n)
            :rtype: list
            )pbdoc")

        .def("fugacity", &EoS::fugacity, R"pbdoc(
            Calculate mixture fugacity at (P,T,x)

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param x: Composition
            :type x: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: List of component fugacities
            :rtype: list
            )pbdoc", "p"_a, "T"_a, "x"_a, py::arg("start_idx")=0, py::arg("pt")=true)

        .def("is_convex", py::overload_cast<double, double, std::vector<double>&, int>(&EoS::is_convex), R"pbdoc(
            Function to determine if GE/TPD surface at composition is convex.

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :rtype: bool
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0)
        .def("calc_condition_number", &EoS::calc_condition_number, R"pbdoc(
            Function to determine condition number of Hessian matrix to evaluate curvature of GE/TPD surface at P, T and composition n.

            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :rtype: bool
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0)

        .def("G", &EoS::G, R"pbdoc(
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool
            :returns: Total Gibbs free energy G of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("H", &EoS::H, R"pbdoc(
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool
            :returns: Total enthalpy H of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("S", &EoS::S, R"pbdoc(
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool
            :returns: Total entropy S of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        // .def("A", &EoS::A, R"pbdoc(
        //     :param X: Pressure/Volume
        //     :type X: double
        //     :param T: Temperature
        //     :type T: double
        //     :param n: Composition
        //     :type n: list
        //     :param start_idx: Index of n[0], default is 0
        //     :type start_idx: int
        //     :param pt: Calculate PT-based (true) or VT-based (false) property
        //     :type pt: bool
        //     :returns: Total Helmholtz free energy A of mixture at (P,T,n)/(V,T,n)
        //     :rtype: double
        //     )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        // .def("U", &EoS::U, R"pbdoc(
        //     :param X: Pressure/Volume
        //     :type X: double
        //     :param T: Temperature
        //     :type T: double
        //     :param n: Composition
        //     :type n: list
        //     :param start_idx: Index of n[0], default is 0
        //     :type start_idx: int
        //     :param pt: Calculate PT-based (true) or VT-based (false) property
        //     :type pt: bool
        //     :returns: Total internal energy U of mixture at (P,T,n)/(V,T,n)
        //     :rtype: double
        //     )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)

        // .def("Cv", &EoS::Cv, R"pbdoc(
        //     :param X: Volume
        //     :type X: double
        //     :param T: Temperature
        //     :type T: double
        //     :param n: Composition
        //     :type n: list
        //     :param start_idx: Index of n[0], default is 0
        //     :type start_idx: int
        //     :param pt: Calculate PT-based (true) or VT-based (false) property
        //     :type pt: bool

        //     :returns: Heat capacity at constant volume Cv of mixture at (P,T,n)/(V,T,n)
        //     :rtype: double
        //     )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("Cp", &EoS::Cp, R"pbdoc(
            :param X: Pressure
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Heat capacity at constant pressure Cp of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)

        ;

    py::class_<IdealGas, PyEoS<IdealGas>, EoS, py::smart_holder>(m, "IdealGas", R"pbdoc(
            This class contains component specific data for evaluation of ideal gas properties.
            )pbdoc")
        .def(py::init<CompData&>(), R"pbdoc(
            This is the constructor of the IdealGas class.

            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)

        .def("P", &IdealGas::P, R"pbdoc(
            :param V: Volume
            :type V: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Pressure solver for ideal gas at (V,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("second_order")=false)
        .def("V", &IdealGas::V, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Ideal gas molar volume at (P,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("second_order")=true)
        .def("rho", &IdealGas::rho, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Ideal gas mass density at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("second_order")=true)
        ;

    py::class_<HelmholtzEoS, PyHelmholtz<>, EoS, py::smart_holder>(m, "HelmholtzEoS", R"pbdoc(
            This is a base class for Helmholtz-based EoS. 
            
            For reference, see Michelsen and Mollerup (2007) - Thermodynamic Models: Fundamentals & Computational Aspects.
            )pbdoc")
        .def(py::init<CompData&>(), R"pbdoc(
            This is the constructor of the HelmholtzEoS base class.

            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)

        .def("Z", py::overload_cast<>(&HelmholtzEoS::Z), R"pbdoc(
            :returns: Volume roots of mixture at (P,T,n)
            :rtype: list
            )pbdoc")
        .def("set_preferred_roots", &HelmholtzEoS::set_preferred_roots, R"pbdoc(
            :param i: Component index
            :type i: int
            :param x: Mole fraction of specified component
            :type x: float
            :param root_flag: Root choice, 0/STABLE) lowest Gibbs energy, 1/MIN) minimum (L), 2/MAX) maximum (V); default is 0
            :type root_flag: EoS.RootFlag
            )pbdoc", "i"_a, "x"_a, "root_flag"_a)
        .def("critical_point", &HelmholtzEoS::critical_point, R"pbdoc(
            Determine mixture critical point

            :param n: Composition
            :type n: list

            :returns: HelmholtzEoS::CriticalPoint object
            )pbdoc", "n"_a)
        .def("is_critical", &HelmholtzEoS::is_critical, R"pbdoc(
            Determine if temperature is above or below critical temperature for mixture at (T,n)
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Temperature is critical?
            :rtype: bool
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)

        .def("P", py::overload_cast<double, double, std::vector<double>&, int, bool>(&HelmholtzEoS::P), R"pbdoc(
            :param V: Volume
            :type V: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list            
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Pressure of mixture at (V,T,n)
            :rtype: double
            )pbdoc", "V"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=false)
        .def("V", py::overload_cast<double, double, std::vector<double>&, int, bool>(&HelmholtzEoS::V), R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Molar volume of mixture at (P,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("Z", py::overload_cast<double, double, std::vector<double>&, int, bool>(&HelmholtzEoS::Z), R"pbdoc(
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Compressibility factor Z of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("rho", &HelmholtzEoS::rho, R"pbdoc(
            :param X: Pressure/Volume
            :type X: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Mass density of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "X"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("volume_iterations", &HelmholtzEoS::volume_iterations, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Number of iterations for volume solver at (P,T,n)
            :rtype: int
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)

        .def("Cv", &HelmholtzEoS::Cv, R"pbdoc(
            :param V: Volume
            :type V: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool
            
            :returns: Heat capacity at constant volume Cv of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "V"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)

        .def("vs", &HelmholtzEoS::vs, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool
            
            :returns: Sound speed vs in mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("JT", &HelmholtzEoS::JT, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Joule-Thomson coefficient of mixture at (P,T,n)/(V,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        ;

    py::class_<HelmholtzEoS::CriticalPoint>(m, "CriticalPoint")
        .def_readonly("Pc", &HelmholtzEoS::CriticalPoint::Pc)
        .def_readonly("Tc", &HelmholtzEoS::CriticalPoint::Tc)
        .def_readonly("Vc", &HelmholtzEoS::CriticalPoint::Vc)
        .def_readonly("Zc", &HelmholtzEoS::CriticalPoint::Zc)
        .def("print", &HelmholtzEoS::CriticalPoint::print_point)
        ;

    py::class_<CubicEoS, PyCubic<>, HelmholtzEoS, py::smart_holder> cubic(m, "CubicEoS", R"pbdoc(
            This is a base class for cubic EoS. 
            
            For reference, see Michelsen and Mollerup (2007) - Thermodynamic Models: Fundamentals & Computational Aspects.
            )pbdoc");
    cubic.def(py::init<CompData&, CubicEoS::CubicType, bool>(), R"pbdoc(
            This is the constructor of the HelmholtzEoS base class for predefined cubic parameters. 
            The user can provide `CubicEoS.PR` or `CubicEoS.SRK` in the argument list.

            :param comp_data: Component data
            :type comp_data: CompData
            :param cubic_type: Predefined cubic parameters, `CubicEoS.PR` (default) or `CubicEoS.SRK`
            )pbdoc", "comp_data"_a, "cubic_type"_a, py::arg("volume_shift")=false)
        .def(py::init<CompData&, CubicParams&>(), R"pbdoc(
            This is the constructor of the HelmholtzEoS base class for user-defined cubic parameters. 
            The user can provide a :class:`CubicParams` in the argument list.

            :param comp_data: Component data
            :type comp_data: CompData
            :param cubic_params: User-defined cubic parameters
            )pbdoc", "comp_data"_a, "cubic_params"_a)

        .def("calc_coefficients", py::overload_cast<double, double, std::vector<double>&, int, bool>(&CubicEoS::calc_coefficients), R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :returns: Coefficients of cubic polynomial for mixture at (P,T,n)
            :rtype: list
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        ;

    py::enum_<CubicEoS::CubicType>(cubic, "CubicType", "Predefined Cubic EoS")
        .value("PR", CubicEoS::CubicType::PR)
        .value("SRK", CubicEoS::CubicType::SRK)
        .export_values()
        ;

    py::class_<CubicParams, py::smart_holder>(m, "CubicParams",  R"pbdoc(
            This class allows the user to pass user-defined parameters for CubicEoS.
            )pbdoc")
        .def(py::init<double, double, double, double, std::vector<double>&, CompData&, bool>(), R"pbdoc(
            :param d1: Cubic volumetric dependence of the attractive contribution d1
            :type d1: double
            :param d2: Cubic volumetric dependence of the attractive contribution d2
            :type d2: double
            :param omegaA: Cubic Omega parameter for attractive term a
            :type omegaA: double
            :param omegaB: Cubic Omega parameter for repulsive term b
            :type omegaB: double
            :param kappa: List of cubic linear part of alpha correlations
            :type kappa: list
            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "d1"_a, "d2"_a, "omegaA"_a, "omegaB"_a, "kappa"_a, "comp_data"_a, py::arg("volume_shift")=false)
        ;

    py::class_<IAPWS95, PyHelmholtz<IAPWS95>, HelmholtzEoS, py::smart_holder>(m, "IAPWS95", R"pbdoc(
            This class is an implementation of IAPWS-95 EoS for H2O.
            )pbdoc")
        .def(py::init<CompData&, bool>(), R"pbdoc(
            This is the constructor of the IAPWS95 class.

            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a, "iapws_ideal"_a)
            
        .def("Pd", &IAPWS95::Pd, R"pbdoc(
            :param d: Reduced volume
            :type d: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list

            :returns: Pressure at reduced volume specification (d,T,n)
            :rtype: double
            )pbdoc", "d"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=false)
        .def("Zd", &IAPWS95::Zd, R"pbdoc(
            :param d: Reduced volume
            :type d: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list

            :returns: Compressibility factor Z at reduced volume specification (d,T,n)
            :rtype: double
            )pbdoc", "d"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=false);

    py::class_<AQEoS, PyEoS<AQEoS>, EoS, py::smart_holder> aqeos(m, "AQEoS", R"pbdoc(
            This class is a composition of multiple AQBase child classes.

            It allows the user to describe solvent (water), solutes and ions with different fugacity-activity models.
            )pbdoc");
    aqeos.def(py::init<CompData&>(), R"pbdoc()pbdoc", "comp_data"_a)
        .def(py::init<CompData&, AQEoS::Model>(), R"pbdoc(
            This constructor creates instance of specified AQBase model

            :param comp_data: Component data
            :type comp_data: CompData
            :param model: AQBase model type
            )pbdoc", "comp_data"_a, "model"_a)
        .def(py::init<CompData&, std::map<AQEoS::CompType, AQEoS::Model>&>(), R"pbdoc(
            This constructor creates instance of AQBase models specified in evaluator_map

            :param comp_data: Component data
            :type comp_data: CompData
            :param evaluator_map: Map of [CompType, Model]
            )pbdoc", "comp_data"_a, "evaluator_map"_a)
        // .def(py::init<CompData&, std::map<AQEoS::CompType, AQEoS::Model>&, std::map<AQEoS::Model, std::unique_ptr<AQBase>>&>(), R"pbdoc(
        //     This constructor creates copies of AQBase models passed in evaluators.
            
        //     :param comp_data: Component data
        //     :type comp_data: CompData
        //     :param evaluator_map: Map of [CompType, Model]
        //     :param evaluators: Map of [Model, AQBase*] for evaluation of fugacities
        //     )pbdoc", "comp_data"_a, "evaluator_map"_a, "evaluators"_a)
        ;

    py::enum_<AQEoS::Model>(aqeos, "Model", "Type of AQBase model")
        .value("Ziabakhsh2012", AQEoS::Model::Ziabakhsh2012)
        .value("Jager2003", AQEoS::Model::Jager2003)
        .export_values()
        ;

    py::enum_<AQEoS::CompType>(aqeos, "CompType", "Type of component")
        .value("water", AQEoS::CompType::water)
        .value("solute", AQEoS::CompType::solute)
        .value("ion", AQEoS::CompType::ion)
        .export_values()
        ;

    py::class_<AQBase, PyAQ<>, py::smart_holder>(m, "AQBase", R"pbdoc(
            This is a base class for aqueous phase fugacity-activity models.
            )pbdoc")
        .def(py::init<CompData&>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)
        ;

    py::class_<Ziabakhsh2012, PyAQ<Ziabakhsh2012>, AQBase, py::smart_holder>(m, "Ziabakhsh2012", R"pbdoc(
            This class evaluates Ziabakhsh and Kooi (2012) aqueous phase fugacity-activity model.
            )pbdoc")
        .def(py::init<CompData&>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)
        ;

    py::class_<Jager2003, PyAQ<Jager2003>, AQBase, py::smart_holder>(m, "Jager2003", R"pbdoc(
            This class evaluates Jager (2003) aqueous phase fugacity model.
            )pbdoc")
        .def(py::init<CompData&>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a)
        ;

    py::class_<VdWP, PyVdWP<>, EoS, py::smart_holder>(m, "VdWP", R"pbdoc(
            This is a base class for hydrate Van der Waals-Platteeuw (1959) type EoS.
            )pbdoc")
        .def(py::init<CompData&, std::string>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            :param hydrate_type: Hydrate type (sI, sII, sH)
            :type hydrate_type: str
            )pbdoc", "comp_data"_a, "hydrate_type"_a)

        .def("V", &VdWP::V, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Hydrate molar volume at (P,T,x)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("fw", py::overload_cast<double, double, std::vector<double>&>(&VdWP::fw), R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param f0: List of fugacities of guest molecules
            :type f0: list

            :returns: Fugacity of water in hydrate phase at (P,T,f0)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "f0"_a)
        .def("xH", &VdWP::xH, R"pbdoc(
            :returns: Hydrate composition at equilibrium at (P,T,f0)
            :rtype: list
            )pbdoc")
        ;

    py::class_<Ballard, PyVdWP<Ballard>, VdWP, py::smart_holder>(m, "Ballard", R"pbdoc(
            This class evaluates the Ballard (2002) implementation of VdWP EoS.
            )pbdoc")
        .def(py::init<CompData&, std::string>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            :param hydrate_type: Hydrate type (sI, sII, sH)
            :type hydrate_type: str
            )pbdoc", "comp_data"_a, "hydrate_type"_a)
        ;

    py::class_<Munck, PyVdWP<Munck>, VdWP, py::smart_holder>(m, "Munck", R"pbdoc(
            This class evaluates the Munck (1988) implementation of VdWP EoS.
            )pbdoc")
        .def(py::init<CompData&, std::string>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            :param hydrate_type: Hydrate type (sI, sII, sH)
            :type hydrate_type: str
            )pbdoc", "comp_data"_a, "hydrate_type"_a)
        ;

    py::class_<PureSolid, PyEoS<PureSolid>, EoS, py::smart_holder>(m, "PureSolid", R"pbdoc(
            This class evaluates the Ballard (2002) implementation of a pure solid EoS.
            )pbdoc")
        .def(py::init<CompData&, std::string>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            :param phase: Pure phase
            :type phase: str
            )pbdoc", "comp_data"_a, "phase"_a)
        .def("V", &PureSolid::V, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list

            :returns: Ice molar volume at (P,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a);
    py::class_<IAPWSIce, PyEoS<IAPWSIce>, EoS, py::smart_holder>(m, "IAPWSIce", R"pbdoc(
            This class evaluates the IAPWS (2006) implementation of H2O ice Ih.
            )pbdoc")
        .def(py::init<CompData&, bool>(), R"pbdoc(
            :param comp_data: Component data
            :type comp_data: CompData
            )pbdoc", "comp_data"_a, "iapws_ideal"_a)

        .def("P", &IAPWSIce::P, R"pbdoc(
            :param V: Volume
            :type V: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Pressure solver for Ice at (V,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=false)
        .def("V", &IAPWSIce::V, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Ice molar volume at (P,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        .def("rho", &IAPWSIce::rho, R"pbdoc(
            :param p: Pressure
            :type p: double
            :param T: Temperature
            :type T: double
            :param n: Composition
            :type n: list
            :param start_idx: Index of n[0], default is 0
            :type start_idx: int
            :param pt: Calculate PT-based (true) or VT-based (false) property
            :type pt: bool

            :returns: Ice mass density at (P,T,n)
            :rtype: double
            )pbdoc", "p"_a, "T"_a, "n"_a, py::arg("start_idx")=0, py::arg("pt")=true)
        ;
}
