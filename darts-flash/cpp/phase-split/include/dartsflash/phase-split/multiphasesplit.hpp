//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_PHASESPLIT_MULTIPHASESPLIT_H
#define OPENDARTS_FLASH_PHASESPLIT_MULTIPHASESPLIT_H
//--------------------------------------------------------------------------

#include "dartsflash/phase-split/basesplit.hpp"
#include "dartsflash/flash/flash_params.hpp"

class MultiPhaseSplit : public BaseSplit
{
public:
    MultiPhaseSplit(FlashParams& flashparams, int np_);

private:
    // Phase ordering, for optimal matrix condition number in Newton with mole numbers
    void find_reference_phases() override;

    // Construct matrices
    virtual Eigen::MatrixXd construct_U() override;
    virtual Eigen::MatrixXd construct_Uinv() override;
    virtual Eigen::MatrixXd construct_PHI() override;

public:
    virtual int test_matrices() override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_PHASESPLIT_MULTIPHASESPLIT_H
//--------------------------------------------------------------------------
