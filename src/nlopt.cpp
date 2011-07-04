#include <nlmagick/nlopt.hpp>
#include <iostream>
#include <stdexcept>
#include <sstream>

namespace nlopt
{
//use this to grab a pointer to the first element in a vector
#define Vp(x) (&(x[0]))
//reinterpret void* to an NLOptCore_impl
#define Thiz(x) ( static_cast<NLOptCore_impl*> (x))

#define Assert(x) ( (x) ? int(0) :  throw std::logic_error("bad") )
struct NLOptCore::NLOptCore_impl
{

  NLOptCore_impl(OptimProblem::Ptr optprob) :
    problem(NULL), opt(optprob)
  {
    setupCore();
  }

  ~NLOptCore_impl()
  {
    cleanup();
  }

  void cleanup()
  {
    nlopt_destroy(problem);
    problem = NULL;
  }

  // 'thiz' : a gangster-rap version of 'this' pointer, more badass.
  // use Thiz(x) to get it from a void*

  /** Objective Function Callback,  minimize F(X) for X having n dimensions
   */
  static double ObjectiveCallback(unsigned n, const double *X, double *G, void *data)
  {
    NLOptCore_impl* thiz = Thiz(data);
    return thiz->opt->evalCostFunction(X, G);
  }

  /** Inequality Constraint Function Callback,  h_i(X) < 0  for i = 1 ... m
   * Might need to fill in $\grad o h_i at X$ if using derivative-based algorithm
   */
  static void InequalityConstraintCallback(unsigned m, double *result, unsigned n, const double *X, double *G, /* NULL if not needed */
  void *data)
  {
    NLOptCore_impl* thiz = Thiz(data);
    thiz->opt->evalInequalityConstraints(result, X, G);
  }

  /** Equality Constraint Function Callback,  g_i(X) = 0  for i = 1 ... m
   * Might need to fill in $\grad o g_i at X$ if using derivative-based algorithm
   */
  static void EqualityConstraintCallback(unsigned m, double *result, unsigned n, const double *X, double *G, /* NULL if not needed */
  void *data)
  {
    NLOptCore_impl* thiz = Thiz(data);
    thiz->opt->evalEqualityConstraints(result, X, G);
  }

  void optimize(const double* _Xinit)
  {
    //pull values from the user defined optimization object
    setupVals();
    //add these to the nlopt problem
    setup_nlopt_criteria();


    if (_Xinit == NULL)
      X = Xinit; //assignment
    else
      X.assign(_Xinit, _Xinit + n); //copy the raw double into X
    fval = opt->evalCostFunction(Vp(X), Vp(G));

    // problem gets flushed somehow !?
    if (problem == NULL)
        throw std::runtime_error("unitialized problem");

    result = nlopt_optimize(problem, Vp(X), &fval);

    if (result <= 0) // the math went wrong (e.g. divide by zero in objective)
    {                // probably not a programming bug
      std::cout << "Warning: optimizer is not happy, result code " << result << std::endl;
      if( NLOPT_INVALID_ARGS == result ) {
        std::cout <<  " check that the initial value was inside lower/upper bounds"
                  <<  " and bounds are not flip-flopped. " << std::endl;
      }
    }
  }

  void setupCore()
  {
    cleanup();
    algorithm = opt->getAlgorithm(); //get the type of algorithm
    n = opt->N(); // number of variables
    if(n < 1) std::logic_error("N must not be less than 1!");
    problem = nlopt_create(algorithm, n);
    result = nlopt_set_min_objective(problem, &NLOptCore_impl::ObjectiveCallback, this);
    Assert( result == NLOPT_SUCCESS );
  }

  void checkVals() const
  {
    if (Xinit.size() != n)
      throw std::logic_error("Xinit().size() needs to be the size of N() in your optimization problem");

    if (!(ub.size() == n || ub.size() == 1 || ub.empty()))
      throw std::logic_error("ub().size() needs to be the size of N() or := 1 in your optimization problem");

    if (!(lb.size() == n || lb.size() == 1 || lb.empty()))
      throw std::logic_error("lb().size() needs to be the size of N() or := 1 in your optimization problem");
  }

  /** \brief grab all of the user defined variables, tolerances, upper bound lower bound, etc...
   */
  void setupVals()
  {
    X = std::vector<double>(n, 0); // state vector to solve for
    G = std::vector<double>(n, 0); // gradient of objective function w.r.t. X
    Xinit = opt->Xinit(); // initial state vector X

    // grab constraint tolerance vectors
    tol_e = opt->tol_e();
    tol_ne = opt->tol_ne();
    ub = opt->ub();
    lb = opt->lb();

    stop_criteria = opt->getStopCriteria();

    fval = 1.0e99; // initialize to huge value that's almost certainly minimizeable

    result = NLOPT_FAILURE;

    checkVals();
  }

  /** setup all of the relevant nlopt criteria, these values come from the setupVals function
   */
  void setup_nlopt_criteria()
  {

    int m_eq = tol_e.size();
    if (m_eq > 0)
    { // set tolerances for equality constraints
      result = nlopt_add_equality_mconstraint(problem, m_eq, &NLOptCore_impl::EqualityConstraintCallback, this,
                                              &(tol_e[0]));
      if (result != NLOPT_SUCCESS)
        std::cout << "result: " << result << std::endl;
      Assert( result == NLOPT_SUCCESS );
    }
    if (tol_ne.size() > 0)
    { // set tolerances for inequality constraints
      result = nlopt_add_inequality_mconstraint(problem, tol_ne.size(), &NLOptCore_impl::InequalityConstraintCallback,
                                                this, &(tol_ne[0]));
      if (result != NLOPT_SUCCESS)
        std::cout << "result: " << result << std::endl;
      Assert( result == NLOPT_SUCCESS );
    }

    if (lb.size() == n)
    { // set lower bound constraints on X
      result = nlopt_set_lower_bounds(problem, Vp(lb));
      Assert( result == NLOPT_SUCCESS );
    }
    else if (lb.size() == 1)
    {
      result = nlopt_set_lower_bounds1(problem, lb[0]);
      Assert( result == NLOPT_SUCCESS );
    }

    if (ub.size() == n)
    { // set upper bound constraints on X
      result = nlopt_set_upper_bounds(problem, Vp(ub));
      Assert( result == NLOPT_SUCCESS );
    }
    else if (ub.size() == 1)
    {
      result = nlopt_set_upper_bounds1(problem, ub[0]);
      Assert( result == NLOPT_SUCCESS );
    }

    // setting up the important ones...
    nlopt_set_ftol_abs(problem, stop_criteria.ftol_abs);
    nlopt_set_ftol_rel(problem, stop_criteria.ftol_rel);
    nlopt_set_maxeval(problem, stop_criteria.maxeval);
    //TODO: Crashes ?? const double* ptr not set right? nlopt_set_xtol_abs(problem, stop_criteria.xtol_abs);
    nlopt_set_xtol_rel(problem, stop_criteria.xtol_rel);
    nlopt_set_maxtime(problem, stop_criteria.maxtime);

  }

  /** C-style pointer to a struct. Must be explicitly deleted.
   *    stores the whole shebang: function pointer set,
   *    xval & grad pointers, current error, iteration count, etc ...
   */
  nlopt_opt problem;
  OptimProblem::Ptr opt;

  /** nl opt struct for stop criteria */
  OptimStopCriteria stop_criteria;

  /** Enum for solver type */
  OptimAlgorithm algorithm;

  /** Enum for stop reason & outcome */
  OptimResult result;
  size_t n;

  //all of these are members so that pointers to them stay valid.
  std::vector<double> X, G, Xinit, tol_e, tol_ne, lb, ub;

  //double* X, *G, *Xinit;
  /** Final value of cost function that we tried to minimze */
  double fval;

};

NLOptCore::NLOptCore(OptimProblem::Ptr optprob) :
  impl_(new NLOptCore_impl(optprob))
{

}

void NLOptCore::optimize(const double* Xinit)
{
  impl_->optimize(Xinit);
}

OptimResult NLOptCore::getResult() const
{
  return impl_->result;
}

double NLOptCore::getFunctionValue() const
{
  return impl_->fval;
}
const std::vector<double>& NLOptCore::getOptimalVector() const
{ // Dangerous! The deriving class is responsible for properly storing this value
  // Otherwise, a stale "X" could exist here!
  return impl_->X;
}

void printOptimResult( OptimResult result )
{
  std::stringstream ss;
  switch( result )
  {
  case  NLOPT_FAILURE:  // =  -1, /* generic failure code */
        ss<<"Generic NLOPT Failure";
        break; 
  case  NLOPT_INVALID_ARGS:  // =  -2,
        ss<<"Invalid arg!?";
        break; 
  case  NLOPT_OUT_OF_MEMORY:  // =  -3,
        ss<<"Out of memory!";
        break; 
  case  NLOPT_ROUNDOFF_LIMITED:  // =  -4,
        ss<<"Reached floating point precision limit before tolerances!";
        break; 
  case  NLOPT_FORCED_STOP:  // =  -5,
        ss<<"Forced Stop (interrupt?)";
        break; 
  case  NLOPT_SUCCESS:  // =  1, /* generic success code */
        ss<<"Success! (generic)";
        break; 
  case  NLOPT_STOPVAL_REACHED:  // =  2,
        ss<<"Check F(X)! Some stop value was reached.";
        break;  
  case  NLOPT_FTOL_REACHED:  // =  3,
        ss<<"Good! Reached Function Tolerance.";
        break; 
  case  NLOPT_XTOL_REACHED:  // =  4,
        ss<<"Good! Reached State-Vector Tolerance.";
        break; 
  case  NLOPT_MAXEVAL_REACHED:  // =  5,
        ss<<"Questionable! Check F(X), max function evals hit. ";
        break; 
  case  NLOPT_MAXTIME_REACHED:  // =  6
        ss<<"Questionable! Check F(X), max run-time hit. ";
        break;
  default:
        ss<<"unknown code passed! ";
        break;
  }
  std::cout << ss.str() << std::endl;  
}



}
