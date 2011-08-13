
#include "nlmagick/nlopt.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <fstream>
#include <iomanip>

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace nlopt;
namespace po = boost::program_options;
using boost::lexical_cast;


class fmincon: public OptimProblem
{
public:
  struct Options {
    string algorithm;
    double vx0;
    double vy0;
    double vmax;
    int    verbosity;

  };

  static int options(int ac, char ** av, Options& opts) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message.")(
          "algorithm,a", po::value<string>(&opts.algorithm)->default_value("NLOPT_LN_SBPLX"),
          "algorithm for solver, such as NLOPT_LN_BOBYQA, NLOPT_LN_SBPLX, NLOPT_LN_COBYLA.")(
          "vx0,x", po::value<double>(&opts.vx0)->default_value(0.0),
          "initial guess: x-velocity")(
          "vy0,y", po::value<double>(&opts.vy0)->default_value(0.0),
          "initial guess: v-velocity")(
          "vmax,m", po::value<double>(&opts.vmax)->default_value(100.0),
          "max abs(...) velocity")(
          "verbose,v", po::value<int>(&opts.verbosity)->default_value(2),
          "verbosity, how much to display crap. between 0 and 3.");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }

    return 0;

  }

public:
  fmincon() {
  }
  virtual ~fmincon() {
  }
  /** Should return a filled out struct of stopping criteria.
     */
  virtual OptimStopCriteria getStopCriteria() const {
    OptimStopCriteria stop_criteria;
    stop_criteria.ftol_abs = 1e-8;
    stop_criteria.ftol_rel = 1e-8;
    stop_criteria.xtol_rel = 1e-8;
    stop_criteria.nevals   = 0;
    stop_criteria.maxeval  = 50000;
    stop_criteria.maxtime  = 5;
    stop_criteria.start    = 0.0;
    stop_criteria.force_stop = 0;
    stop_criteria.minf_max   = 1e9;
    return stop_criteria;
  }


  double evalCostFunction(const double* X, double*)
  {
    double fval = 0.0;
    double x0   = X[0];
    double x1   = X[1];

    fval        = sqrt( x0*x0 + x1*x1 );
    iters++;
    if( fval < fval_best ) {
      fval_best = fval;
      cout << "iter=" << iters << ", fval= " << fval << endl;
    }
    return fval;
  }
  virtual std::vector<double> ub() const
  {
    double c[2] = { input_opts.vmax+input_opts.vx0,input_opts.vmax+input_opts.vy0};
    return std::vector<double>(c,c+2);
  }

  virtual std::vector<double> lb() const
  {
    double c[2] = { -input_opts.vmax+input_opts.vx0,-input_opts.vmax+input_opts.vy0};
    return std::vector<double>(c,c+2);
  }

  virtual size_t N() const {
    return 2; // 2 free params: x,y offsets
  }

  virtual OptimAlgorithm getAlgorithm() const {
    //http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms
    if( input_opts.algorithm.compare("NLOPT_LN_SBPLX") == 0 )
      return NLOPT_LN_SBPLX;
    if( input_opts.algorithm.compare("NLOPT_LN_COBYLA") == 0 ) {
      return NLOPT_LN_COBYLA;
    }
    if( input_opts.algorithm.compare("NLOPT_LN_BOBYQA") == 0 ) {
      return NLOPT_LN_BOBYQA;
    }
    cout << "warning, algorithm string is bogus. using default..." << endl;
    return  NLOPT_LN_SBPLX;
  }

  virtual std::vector<double> Xinit() const
  {
    std::vector<double> x0 = std::vector<double>(2, 0.0);
    x0[0] += input_opts.vx0;
    x0[1] += input_opts.vy0;
    return x0;
  }

  void setup( const Options& opts ) {
    input_opts = opts;
    fval_best = 1e9;
    iters = 0;
  }


public:
  // some internal persistent data
  Options  input_opts;
  double fval_best;
  int iters;
};


int main(int argc, char** argv) {
  fmincon::Options opts;
  if (fmincon::options(argc, argv, opts))
    return 1;

  cout << "running test, minimize || [x;y] ||_2 subject to bound constraints" << endl;

  boost::shared_ptr<fmincon> RT(new fmincon());
  RT->setup( opts );

  // lock and load
  NLOptCore opt_core(RT);

  // launch the nukes
  opt_core.optimize();

  // evaluate body count

  vector<double> optimal_vxvy = opt_core.getOptimalVector();
  double fval_final           = opt_core.getFunctionValue();
  cout << "optimal vector: " << endl;
  cout << optimal_vxvy[0] << ", " << optimal_vxvy[1] << endl;
  cout << "final fval: "  << fval_final << endl;

  // Good => result is 1 or higher
  int ret_code = (opt_core.getResult() > 0 ) ? 0 : 1;
  cout << "return value: " << ret_code << endl;

  return ret_code;

}

