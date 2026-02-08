//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_PHASESPLIT_TWOPHASESPLIT_H
#define OPENDARTS_FLASH_PHASESPLIT_TWOPHASESPLIT_H
//--------------------------------------------------------------------------

#include "dartsflash/phase-split/basesplit.hpp"
#include "dartsflash/flash/flash_params.hpp"

class TwoPhaseSplit : public BaseSplit
{
public:
    TwoPhaseSplit(FlashParams& flashparams);
    
private:
    virtual Eigen::MatrixXd construct_U() override;
    virtual Eigen::MatrixXd construct_Uinv() override;
    virtual Eigen::MatrixXd construct_PHI() override;
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_PHASESPLIT_TWOPHASESPLIT_H
//--------------------------------------------------------------------------
