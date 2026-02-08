//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_STABILITY_STABILITY_H
#define OPENDARTS_FLASH_STABILITY_STABILITY_H
//--------------------------------------------------------------------------

#include <memory>
#include "dartsflash/global/global.hpp"
#include "dartsflash/maths/modifiedcholeskys99.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/trial_phase.hpp"
#include "dartsflash/maths/linesearch.hpp"
#include <Eigen/Dense>

class Stability
{
protected:
    int nc, ni, ns, ns_nonzero;
    double tpd;
    std::vector<int> nonzero_comp;
    std::vector<double> h, lnphi, dlnphidn;
    std::vector<double> lnY, alpha;
    std::vector<TrialPhase> ref_compositions;
    TrialPhase trial_composition;
    FlashParams flash_params;
    Eigen::VectorXd g, ga;
    int error{ 0 }, ssi_iter, newton_iter, switch_back_iter;

public:
    Stability(FlashParams& flashparams);
    virtual ~Stability() = default;

    // Initialize stability test at mixture composition x, set initial guess Y and run test from Y
    void init(std::vector<TrialPhase>& ref_comps);
    void init_gmix(TrialPhase& trial_comp, std::vector<double>& lnphi0);
    int run(TrialPhase& trial_comp, bool gmix_min = false);

    // Getters for solution path and iteration numbers
    int get_ssi_iter() { return this->ssi_iter + this->switch_back_iter; }
    int get_newton_iter() { return this->newton_iter; }
    double calc_condition_number();

protected:
    // SSI and Newton steps
    void perform_ssi();
    void perform_Y();
    void perform_lnY();
    void perform_alpha();

    // Update fugacities, derivatives, tpd and gradient vector
    void update_fugacities(TrialPhase& comp, bool second_order);
    double calc_modtpd();
    void update_g();

    // Calculate norm of gradient vector
    double l2norm() { return g.squaredNorm(); }

    // Construct matrices
    Eigen::VectorXd construct_ga(std::vector<double>& alpha);
    Eigen::MatrixXd construct_U();
    Eigen::MatrixXd construct_Uinv();
    Eigen::MatrixXd construct_PHI();
    Eigen::MatrixXd construct_H(Eigen::MatrixXd& U, Eigen::MatrixXd& PHI);
    Eigen::MatrixXd construct_H(std::vector<double>& alpha);
    Eigen::MatrixXd construct_J(Eigen::MatrixXd& PHI, Eigen::MatrixXd& Uinv);

public:
    double calc_modtpd(TrialPhase& trial_comp);
    int test_matrices();

};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_STABILITY_STABILITY_H
//--------------------------------------------------------------------------
