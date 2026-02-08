//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_EOS_AQ_AQ_H
#define OPENDARTS_FLASH_EOS_AQ_AQ_H
//--------------------------------------------------------------------------

#include "dartsflash/eos/eos.hpp"
#include "dartsflash/global/global.hpp"
#include "dartsflash/global/components.hpp"
#include <memory>
#include <map>

class AQBase;

class AQEoS : public EoS
{
public:
	enum class Model : int { Ziabakhsh2012 = 0, Jager2003 };
	enum CompType : int { water = 0, solute, ion };

protected:
	std::vector<double>::iterator n_iterator;

	std::map<CompType, Model> evaluator_map;
	std::map<Model, std::shared_ptr<AQBase>> evaluators;
	std::vector<AQEoS::CompType> comp_types;
	
	std::vector<double> m_s;
	bool constant_salinity;

public:
	AQEoS(CompData& comp_data);
	AQEoS(CompData& comp_data, AQEoS::Model model);
	AQEoS(CompData& comp_data, std::map<AQEoS::CompType, AQEoS::Model>& evaluator_map_);
	// AQEoS(CompData& comp_data, std::map<AQEoS::CompType, AQEoS::Model>& evaluator_map_, std::map<AQEoS::Model, std::unique_ptr<AQBase>>& evaluators_);

	virtual std::unique_ptr<EoS> getCopy() override { return std::make_unique<AQEoS>( *this ); }

	virtual void init_PT(double p_, double T_) override;
	virtual void solve_PT(std::vector<double>::iterator n_it, bool second_order=true) override;
	virtual void init_VT(double V_, double T_) override;
	virtual void solve_VT(std::vector<double>::iterator n_it, bool second_order=true) override;

	virtual double lnphii(int i) override;
	virtual double dlnphii_dP(int i) override;
	virtual double dlnphii_dT(int i) override;
	virtual double dlnphii_dnj(int i, int j) override;
	virtual double d2lnphii_dPdT(int i) override;
	virtual double d2lnphii_dT2(int i) override;
	virtual double d2lnphii_dTdnj(int i, int j) override;

	virtual std::vector<double> dlnphi_dn() override;
	virtual std::vector<double> d2lnphi_dTdn() override;

	// Pure component properties
	virtual std::vector<double> lnphi0(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dP(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> dlnphi0_dT(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dP2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dT2(double X, double T_, bool pt=true) override;
	// virtual std::vector<double> d2lnphi0_dPdT(double X, double T_, bool pt=true) override;

	virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false) override;

protected:
	void species_molality(std::vector<double>& x);
	
};

class AQBase
{
protected:
	CompData compdata;
	int nc, ni, ns, water_index;
	double p, T;
	bool set_molality;
	
	std::vector<std::string> species;
	std::vector<double> x, m_s, dmidxj;
	std::vector<int> charge;

public:
	AQBase(CompData& comp_data);
	virtual ~AQBase() = default;

	AQBase(const AQBase&) = default;

	virtual std::shared_ptr<AQBase> getCopy() = 0;

	virtual void init_PT(double p_, double T_, AQEoS::CompType comp_type) = 0;
	virtual void solve_PT(std::vector<double>& x_, bool second_order, AQEoS::CompType comp_type) = 0;

	virtual double lnphii(int i) = 0;
	virtual double dlnphii_dP(int i) = 0;
	virtual double dlnphii_dT(int i) = 0;
	virtual double dlnphii_dxj(int i, int j) = 0;
	std::vector<double> dlnphii_dxj(int i);
	virtual double d2lnphii_dPdT(int i) = 0;
	virtual double d2lnphii_dT2(int i) = 0;
	virtual std::vector<double> d2lnphii_dTdxj(int i) = 0;

	virtual double lnphi0(double X, double T_, bool pt=true) = 0;
	// virtual double dlnphi0_dP(double X, double T_, bool pt=true) = 0;
	// virtual double dlnphi0_dT(double X, double T_, bool pt=true) = 0;
	// virtual double d2lnphi0_dP2(double X, double T_, bool pt=true) = 0;
	// virtual double d2lnphi0_dT2(double X, double T_, bool pt=true) = 0;
	// virtual double d2lnphi0_dPdT(double X, double T_, bool pt=true) = 0;

	void set_species_molality(std::vector<double>& m) { set_molality = true; this->m_s = m; }
	void set_species_molality();
	double mi(int i);
	double dmi_dxi();
	double dmi_dxw(int i);

	virtual int derivatives_test(double p_, double T_, std::vector<double>& n_, double tol, bool verbose=false);
};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_EOS_AQ_AQ_H
//--------------------------------------------------------------------------
