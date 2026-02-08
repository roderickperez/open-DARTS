//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_PHASESPLIT_BASESPLIT_H
#define OPENDARTS_FLASH_PHASESPLIT_BASESPLIT_H
//--------------------------------------------------------------------------

#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/trial_phase.hpp"
#include "dartsflash/rr/rr.hpp"
#include "dartsflash/stability/stability.hpp"
#include <Eigen/Dense>

class BaseSplit
{
protected:
    int nc, ns, np;
    std::vector<double> z, nu, X, lnK, n_ik;
    std::vector<double> lnphi, dlnphidn;
    std::vector<int> reference_phase, k_idxs, nonzero_comp, roots;
    std::vector<std::string> eos_names;
    std::vector<TrialPhase> trial_compositions;
    double gibbs, norm;
    FlashParams flash_params;
    std::shared_ptr<RR> rr;
    Eigen::VectorXd g;
    int error, ssi_iter, newton_iter, switch_back_iter, loose_iter;

public:
    BaseSplit(FlashParams& flashparams, int np_);
    virtual ~BaseSplit() = default;

    // Run multiphase split at p, T, z with initial guess lnK
    int init(std::vector<double>& z_, std::vector<double>& lnk, std::vector<TrialPhase>& trial_comps, bool set_root_flags = false);
    int run(std::vector<double>& z_, std::vector<double>& lnk, std::vector<TrialPhase>& trial_comps, bool set_root_flags = false);
    int output(std::vector<TrialPhase>& trial_comps);

    // Getters for return variables nu and x
    std::vector<double> get_lnk();
    std::vector<double> getnu() { return this->nu; };
    std::vector<double> getx() { return this->X; };
    double get_gibbs() { return this->gibbs; }
    int get_ssi_iter(){return this->ssi_iter + this->switch_back_iter;}
    int get_newton_iter(){return this->newton_iter;}
    double calc_condition_number();

    // Calculate norm of gradient vector
    double l2norm() { return g.squaredNorm(); }

protected:
    // Solve RR, SSI and Newton steps
    void solve_rr();
    void perform_ssi();
    void perform_nik();
    void perform_lnK();
    void perform_lnK_chol();

    // Phase ordering, for optimal matrix condition number in Newton with mole numbers
    virtual void find_reference_phases() { return; }

    // Update mole numbers, fugacities and derivatives
    void update_fugacities(bool second_order);
    double calc_gibbs();

    // Construct gradient vector and matrices
    void update_g();
    virtual Eigen::MatrixXd construct_U() = 0;
    virtual Eigen::MatrixXd construct_Uinv() = 0;
    virtual Eigen::MatrixXd construct_PHI() = 0;
    virtual Eigen::MatrixXd construct_H(Eigen::MatrixXd& U, Eigen::MatrixXd& PHI);
    virtual Eigen::MatrixXd construct_J(Eigen::MatrixXd& PHI, Eigen::MatrixXd& Uinv);

public:
    virtual int test_matrices();

};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_PHASESPLIT_BASESPLIT_H
//--------------------------------------------------------------------------
