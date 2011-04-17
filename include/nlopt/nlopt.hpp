#ifndef NLOPT_HPP
#define NLOPT_HPP

#include "nlopt.h"

#include <vector>
#include <stdexcept>
#include <new>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <boost/shared_ptr.hpp>

namespace nlopt
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
typedef nlopt_algorithm OptimAlgorithm;
typedef nlopt_result OptimResult;
typedef nlopt_stopping OptimStopCriteria;

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

#endif
