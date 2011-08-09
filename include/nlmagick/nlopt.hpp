#pragma once
#include <vector>
#include <stdexcept>
#include <new>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace nlopt //TODO rename to nlmagick
{

/** \brief  Enumeration of available algorithms.
 * Naming convention:
 * {G,L}   global or local optimization
 * {N,D}   No derivative, or yes Derivative needed
 * "RAND"  Suffix:  algorithm involves randomness,
 *              so results not repeatable / deterministic.
 * "NOSCAL" Suffix: algorithm *not* scaled to a unit hypercube
 (i.e. sensitive to the units of x)
 */
//typedef nlopt_algorithm OptimAlgorithm;
enum OptimAlgorithm
{
     /* Naming conventions:

        NLOPT_{G/L}{D/N}_*
        = global/local derivative/no-derivative optimization,
              respectively

    *_RAND algorithms involve some randomization.

    *_NOSCAL algorithms are *not* scaled to a unit hypercube
             (i.e. they are sensitive to the units of x)
    */

     NLOPT_GN_DIRECT = 0,
     NLOPT_GN_DIRECT_L,
     NLOPT_GN_DIRECT_L_RAND,
     NLOPT_GN_DIRECT_NOSCAL,
     NLOPT_GN_DIRECT_L_NOSCAL,
     NLOPT_GN_DIRECT_L_RAND_NOSCAL,

     NLOPT_GN_ORIG_DIRECT,
     NLOPT_GN_ORIG_DIRECT_L,

     NLOPT_GD_STOGO,
     NLOPT_GD_STOGO_RAND,

     NLOPT_LD_LBFGS_NOCEDAL,

     NLOPT_LD_LBFGS,

     NLOPT_LN_PRAXIS,

     NLOPT_LD_VAR1,
     NLOPT_LD_VAR2,

     NLOPT_LD_TNEWTON,
     NLOPT_LD_TNEWTON_RESTART,
     NLOPT_LD_TNEWTON_PRECOND,
     NLOPT_LD_TNEWTON_PRECOND_RESTART,

     NLOPT_GN_CRS2_LM,

     NLOPT_GN_MLSL,
     NLOPT_GD_MLSL,
     NLOPT_GN_MLSL_LDS,
     NLOPT_GD_MLSL_LDS,

     NLOPT_LD_MMA,

     NLOPT_LN_COBYLA,

     NLOPT_LN_NEWUOA,
     NLOPT_LN_NEWUOA_BOUND,

     NLOPT_LN_NELDERMEAD,
     NLOPT_LN_SBPLX,

     NLOPT_LN_AUGLAG,
     NLOPT_LD_AUGLAG,
     NLOPT_LN_AUGLAG_EQ,
     NLOPT_LD_AUGLAG_EQ,

     NLOPT_LN_BOBYQA,

     NLOPT_GN_ISRES,

     /* new variants that require local_optimizer to be set,
    not with older constants for backwards compatibility */
     NLOPT_AUGLAG,
     NLOPT_AUGLAG_EQ,
     NLOPT_G_MLSL,
     NLOPT_G_MLSL_LDS,

     NLOPT_LD_SLSQP,

     NLOPT_NUM_ALGORITHMS /* not an algorithm, just the number of them */
};
//typedef nlopt_result OptimResult;
enum OptimResult
{
     NLOPT_FAILURE = -1, /* generic failure code */
     NLOPT_INVALID_ARGS = -2,
     NLOPT_OUT_OF_MEMORY = -3,
     NLOPT_ROUNDOFF_LIMITED = -4,
     NLOPT_FORCED_STOP = -5,
     NLOPT_SUCCESS = 1, /* generic success code */
     NLOPT_STOPVAL_REACHED = 2,
     NLOPT_FTOL_REACHED = 3,
     NLOPT_XTOL_REACHED = 4,
     NLOPT_MAXEVAL_REACHED = 5,
     NLOPT_MAXTIME_REACHED = 6
};
//typedef nlopt_stopping OptimStopCriteria;
struct OptimStopCriteria
{
     unsigned n;
     double minf_max;
     double ftol_rel;
     double ftol_abs;
     double xtol_rel;
     const double *xtol_abs;
     int nevals, maxeval;
     double maxtime, start;
     int *force_stop;
};

/** Client interface for using the optimization framework.
 * User must atleast implement:
 *  evalCostFunction, N, getAlgorithm
 */
class OptimProblem
{
public:

  /** virtual destructor for inheritance.
   *
   */
  virtual ~OptimProblem()
  {
  }

  /** objective function
   * should return f(X), and may fill out G optionally.
   */
  virtual double evalCostFunction(const double* x_vector, double* G) = 0;

  /** Evaluate all inequality constraints.
   * \param results [out] is the size of tol_ne().size(), or M constraints
   * \param x_vector the vector valued parameters, should evaluate inequality at X, size of N
   * \param G this is MxN and is the gradient at each constraint? TODO fill this in
   */
  virtual void evalInequalityConstraints(double *results, const double* x_vector, double* grad_func_at_x)
  {
  }

  /** Evaluate all equality constraints.
   * \param results [out] is the size of tol_e().size(), or M constraints
   * \param x_vector the vector valued parameters, should evaluate equality at X, size of N
   * \param G this is MxN and is the gradient at each constraint? TODO fill this in
   */
  virtual void evalEqualityConstraints(double *results, const double* x_vector, double* G)
  {
  }

  /** Number of free parameters.
   * This is called at registration time,i.e. when the OptimProblem is registered with NLOptCore
   * and must not change!
   */
  virtual size_t N() const = 0;
  
  /** set some display / console output level
    */
  virtual void setVerbosity(int level) 
  { 
  }
  
  /** get the output level
    */
  virtual int getVerbosity() const 
  {
    return 0; 
  }
  /** Should return the desired algorithm.
   * This is called at registration time,i.e. when the OptimProblem is registered with NLOptCore
   * and must not change!
   */
  virtual OptimAlgorithm getAlgorithm() const = 0;

  //*************************************
  // optional interfaces
  //*************************************

  /** inequality tolerances
   * This will be called right before every optimization run.
   * */
  virtual std::vector<double> tol_ne() const
  {
    return std::vector<double>();
  }

  /** equality tolerances
   * This will be called right before every optimization run.
   * */
  virtual std::vector<double> tol_e() const
  {
    return std::vector<double>();
  }

  /** X upper bounds
   * This will be called right before every optimization run.
   * */
  virtual std::vector<double> ub() const
  {
    return std::vector<double>();
  }

  /** X lower bounds
   * This will be called right before every optimization run.
   */
  virtual std::vector<double> lb() const
  {
    return std::vector<double>();
  }

  /** Set the inititial value of the parameters.
   * This will be called right before every optimization run.
   */
  virtual std::vector<double> Xinit() const
  {
    return std::vector<double>(N(), 0);
  }

  /** Should return a filled out struct of stopping criteria.
   * This will be called right before every optimization run.
   */
  virtual OptimStopCriteria getStopCriteria() const
  {
    return GetDefaultStopCriteria();
  }

  /** Mainly for reference, use this to init the stop criteria to useful values.
   */
  static OptimStopCriteria GetDefaultStopCriteria()
  {
    OptimStopCriteria stop_criteria;
    stop_criteria.ftol_abs = 1e-9;
    stop_criteria.ftol_rel = 1e-3;
    stop_criteria.xtol_rel = 1e-3;
    stop_criteria.nevals = 0;
    stop_criteria.maxeval = 1000;
    stop_criteria.maxtime = 60.0;
    stop_criteria.start = 0.0;
    stop_criteria.force_stop = 0;
    stop_criteria.minf_max = 1e9;
    return stop_criteria;
  }

  typedef boost::shared_ptr<OptimProblem> Ptr; //!< Convenience boost pointer type
  typedef boost::shared_ptr<const OptimProblem> ConstPtr; //!< Convenience const boost pointer type
};

/** Optimization engine, used to handle the optimization routines and setup.
 */
class NLOptCore
{
public:
  /** Setup the NLOptCore with the opt problem.
   */
  NLOptCore(OptimProblem::Ptr optprob);
  /** Run optimization, give an initial value if you have one, otherwise, the default initial
   * value will be used.
   */
  void optimize(const double* Xinit = NULL);

  /** Get the result code, will be filled in after a call to optimize
   */
  OptimResult getResult() const;
  
  /** Get the optimal value of the parameters.
   */
  const std::vector<double>& getOptimalVector() const;

  /** Get the function value, given by using the optimal parameters
   */
  double getFunctionValue() const;

private:
  //no copying!
  NLOptCore(const NLOptCore&)
  {
  }
  void operator=(const NLOptCore&)
  {
  }

  struct NLOptCore_impl; // Not defined here, opaque pointer see http://en.wikipedia.org/wiki/Opaque_pointer#C.2B.2B
  boost::shared_ptr<NLOptCore_impl> impl_; // the actual implementation.
};

void printOptimResult( OptimResult result );

} // End Namespace nlopt

