//--------------------------------------------------------------------------
#ifndef OPENDARTS_FLASH_RR_RR_H
#define OPENDARTS_FLASH_RR_RR_H
//--------------------------------------------------------------------------

#include "dartsflash/flash/flash_params.hpp"
#include "dartsflash/maths/linesearch.hpp"
#include <vector>
#include <Eigen/Dense>

class RR
{
protected:
    int np, nc;
    double norm{ 0. };
	double min_z, rr2_tol, rrn_tol, loose_tol;
    int max_iter, loose_iter;
    std::vector<int> nonzero_comp;
    std::vector<double> z, K, nu;
    bool verbose;

public:
    RR(FlashParams& flash_params, int nc_, int np_);
    virtual ~RR() = default;

    virtual int solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_={}) = 0;
    
    const std::vector<double> &getnu() const { return this->nu; }
    virtual std::vector<double> getx();
    virtual double l2norm();

protected:
    void init(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_);
    std::vector<double> objective_function(const std::vector<double>& nu_);
    int output(int error);
};

// 2-phase Rachford-Rice equation solving with convex transformations (Nichita, 2013)
class RR_EqConvex2 : public RR
{
private:
    std::vector<double> ci, di;
    std::vector<int> k_idxs;

public:
    RR_EqConvex2(FlashParams& flash_params, int nc_);

    virtual int solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_={}) override;
    virtual std::vector<double> getx() override;
    virtual double l2norm() override { return norm; };

private:
    int solve_gh();
    int solve_fgh();
    int output(int error, double a);

    std::vector<int> sort_idxs(std::vector<double> ki);

    // Functions aL, aR, V, F(a), F'(a), G(a), G'(a), H(a) and H'(a)
    double aL(double z1);
    double aR(double zn);
    double V(double a);

    double F(double a);
    double dF(double a);
	double G(double a);
    double G(double a, double f);
    double dG(double a);
	double H(double a);
    double H(double a, double g);
	double dH(double a);
};

/*
// N-phase Rachford-Rice equation solving with negative flash (Iranshahr, 2010)
class RR_EqN : public RR 
{
protected:
    int error_output;
    std::vector<double> f;
    std::vector<double> v_min, v_mid, v_max;
    
public:
    RR_EqN(FlashParams& flash_params, int nc_, int np_);
    
    virtual int solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_={}) override;
    virtual double l2norm() override { return std::sqrt(std::inner_product(f.begin(), f.end(), f.begin(), 0.)); };

protected:
    int bounds(std::vector<double> V_j, int j);
    bool bounded();
    
    int rr_loop(int J);
    double fdf(std::vector<double> V_j, int J); // calculates f_j and df_j/dv_j
};
*/

// N-phase Rachford-Rice minimization with negative flash (Michelsen [1994] and Yan [2012])
class RR_Min : public RR
{
protected:
    std::vector<double> Ei;
    Eigen::VectorXd g;

public:
    RR_Min(FlashParams& flash_params, int nc_, int np_);

    virtual int solve_rr(std::vector<double>& z_, std::vector<double>& K_, const std::vector<int>& nonzero_comp_={}) override;
    virtual std::vector<double> getx() override;

    // Calculate norm of gradient vector
    virtual double l2norm() override { return g.squaredNorm(); }

protected:
    double calc_obj(Eigen::VectorXd& nu_);
    void update_g(std::vector<double>& Ei_inv);

};

//--------------------------------------------------------------------------
#endif // OPENDARTS_FLASH_RR_RR_H
//--------------------------------------------------------------------------
