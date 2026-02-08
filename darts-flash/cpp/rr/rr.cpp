#include <vector>
#include <numeric>

#include "dartsflash/global/global.hpp"
#include "dartsflash/rr/rr.hpp"

RR::RR(FlashParams& flash_params, int nc_, int np_)
{
    this->nc = nc_; 
    this->np = np_;
    
	this->rr2_tol = flash_params.rr2_tol;
	this->rrn_tol = flash_params.rrn_tol;
	this->loose_tol = (np_ == 2) ? this->rr2_tol * flash_params.rr_loose_tol_multiplier : this->rrn_tol * flash_params.rr_loose_tol_multiplier;
    this->max_iter = flash_params.rr_max_iter;
	this->loose_iter = flash_params.rr_loose_iter;
	this->min_z = flash_params.min_z;
	this->verbose = flash_params.verbose;

    this->z.resize(nc);
    this->K.resize((np-1)*nc);
    this->nu.resize(np);
	this->nonzero_comp.resize(nc);
}

void RR::init(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_)
{
	this->z = z_;
	this->K = K_;
    
    if (!nonzero_comp_.empty()) { this->nonzero_comp = nonzero_comp_; }
	else
	{
		for (int i = 0; i < nc; i++)
		{
			nonzero_comp[i] = (z[i] > min_z) ? 1 : 0;
		}
	}
}

std::vector<double> RR::objective_function(const std::vector<double>& nu_)
{
	// Calculate Rachford-Rice objective function
	std::vector<double> obj(np-1, 0.);

	for (int i = 0; i < nc; i++)
	{
		double m_i = 1.;
		for (int j = 1; j < np; j++)
		{
			m_i += nu_[j] * (K[(j-1)*nc + i] - 1.);
		}

		for (int j = 0; j < np-1; j++)
		{
			obj[j] += z[i] * (K[j*nc + i] - 1.) / m_i;
		}
	}
	return obj;
}

double RR::l2norm()
{
	// Calculate L2-norm of objective function
	std::vector<double> obj = this->objective_function(this->nu);
	return std::sqrt(std::inner_product(obj.begin(), obj.end(), obj.begin(), 0.));
}

std::vector<double> RR::getx()
{
    std::vector<double> x(np*nc);

	bool zero_comp = false;
	for (int i = 0; i < nc; i++) 
	{
		if (nonzero_comp[i])
		{
			double m_i = 1.;
			for (int k = 1; k < np; k++) 
			{
				m_i += nu[k] * (K[(k-1)*nc + i] - 1.);
			}
			x[i] = (m_i > 1e-16) ? z[i] / m_i : z[i] * 1e16;
		}
		else
		{
			zero_comp = true;
			x[i] = 0.;
		}
	}
	
	if (zero_comp)
	{
		double sumx = std::accumulate(x.begin(), x.begin() + nc, 0.);
		double x0 = (sumx < 1.) ? 1.-sumx : min_z;
		for (int i = 0; i < nc; i++)
		{
			if (!nonzero_comp[i])
			{
				x[i] = x0;
			}
		}
	}

    for (int j = 1; j < np; j++) 
	{
    	for (int i = 0; i < nc; i++) 
    	{
	    	x[j*nc + i] = K[(j-1)*nc + i] * x[i];
		}
    }
	return x;
}

int RR::output(int error)
{
	if (error == 1 && this->verbose)
	{
		print("MAX RR Iterations", this->max_iter);
    	print("Norm", this->norm);
	}
    
	// double nu0 = 1. - std::accumulate(nu.begin(), nu.begin() + np-1, 0.);
	// nu.insert(nu.begin(), nu0);

	if ((np == 2 && this->l2norm() > rr2_tol) ||
		(this->l2norm() > rrn_tol))
	{
		return 1;
	}
    return error;
}
