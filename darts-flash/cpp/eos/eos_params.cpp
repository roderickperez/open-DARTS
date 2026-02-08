#include <cmath>
#include <numeric>
#include <memory>
#include "dartsflash/eos/eos_params.hpp"
#include "dartsflash/maths/maths.hpp"

EoSParams::EoSParams(EoS* eos_ptr, CompData& comp_data)
{
    this->eos = eos_ptr->getCopy();

	this->comp_idxs.resize(comp_data.ns);
	std::iota(comp_idxs.begin(), comp_idxs.end(), 0);

	this->nc = comp_data.nc;
	this->ni = comp_data.ni;
	this->ns = comp_data.ns;
}

void EoSParams::set_active_components(std::vector<int>& idxs)
{
	this->comp_idxs = idxs;

	int NC = this->nc;
	nc = 0; ni = 0;
	for (int i: idxs)
	{
		if (i < NC)
		{
			nc++;
		}
		else
		{
			ni++;
		}
	}
	this->ns = nc + ni;
	return;
}
