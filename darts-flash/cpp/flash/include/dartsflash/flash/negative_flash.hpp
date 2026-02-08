//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_FLASH_NEGATIVEFLASH_H
#define OPENDARTS_FLASH_FLASH_NEGATIVEFLASH_H
//--------------------------------------------------------------------------

#include <vector>
#include "dartsflash/global/global.hpp"
#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/flash/flash_results.hpp"
#include "dartsflash/flash/flash.hpp"
#include "dartsflash/flash/trial_phase.hpp"

class NegativeFlash : public Flash
{
protected:
	std::vector<int> initial_guesses, eos_idxs;

public:
	NegativeFlash(FlashParams& flashparams, const std::vector<std::string>& eos_used, const std::vector<int>& initial_guesses_);

	virtual int evaluate(double p_, double T_, std::vector<double>& z_, bool start_from_feed = true) override;  // multicomponent flash
	virtual int evaluate(double p_, double T_, std::vector<double>& z_, std::shared_ptr<FlashResults> flash_results) override;  // Set initial guess from previous flash results

protected:
	virtual std::vector<double> generate_lnK(std::vector<int>& sp_idxs_) override;
	virtual bool check_negative();
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_FLASH_NEGATIVEFLASH_H
//--------------------------------------------------------------------------
