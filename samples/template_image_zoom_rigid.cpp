
#include "opencv_helpers.hpp"
#include <nlopt/nlopt.hpp>
#include <omp.h>


using namespace cv;
using namespace std;
using namespace nlopt;

#define foreach         BOOST_FOREACH
#define reverse_foreach BOOST_REVERSE_FOREACH

namespace po = boost::program_options;
using boost::lexical_cast;


namespace {
  
  struct Options {
    string k_file;
    string focal_ratio;
    string input_img; // what we're solving on. no mask here.
    string template_img;
    string mask_img;  // corresponds to "template". 1 in the planar area we want.
    string out_warp_img;
    string save_solver_imgs_prefix; // save intermediate solver output to files xxxx_000n.png    
    string guess_initial; // a 'csv' file with omega_hat and T initial guesses
    string x_out_final;   // a 'csv' file with omega_hat and T final solved values
    string algorithm;
    
    string lambdaT; // regularizer for translation
    string lambdaW; // regularizer for rotation
    string lambdaF; // regularizer for f-scale change (zoom)
    string alphaIO; // weight for inside-outside disparity
    string alphaIT; // weight for inside-template similarity

    int smoothFilterSize; // cv::Size( n,n ) and sigma = n/3, n/3 for smoothing image
    int maxSolverTime; // seconds
    int vid;        // /dev/videoN
    int verbosity;  // how much crap to display
    int waitKeyLength;
    
    
  };
  
  int options(int ac, char ** av, Options& opts) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message.")(
"solverprefix,p",
po::value<string>(&opts.save_solver_imgs_prefix)->default_value(""),
"solver output intermediate images prefix, saves -v2 display to prefix_000x.png ")(
"intrinsics,K",
po::value<string>(&opts.k_file)->default_value(""),
"The camera intrinsics file, yaml, from opencv calibration, Required. ")(
"inputimage,i", po::value<string>(&opts.input_img)->default_value(""),
"input image where we want to find something's pose. Required / might use webcam otherwise.")(
"templateimage,t", po::value<string>(&opts.template_img),
"template image where pose is identity matrix. Required.")(
"outwarpimage,w", po::value<string>(&opts.out_warp_img)->default_value("warped.jpg"),
"output for warped image. default: warped.jpg")(
"initguess,g", po::value<string>(&opts.guess_initial)->default_value("x"),
"initial guess csv file for params. default: none, all 0's")(
"finalxval,f", po::value<string>(&opts.x_out_final)->default_value("wt_final.out"),
"final output file csv for w,T solved values.")(
"maskimage,m", po::value<string>(&opts.mask_img),
"mask: non-zero ROI in template. Required / might use entire image otherwise. ")(
"video,V", po::value<int>(&opts.vid)->default_value(0),
"Video device number, find video by ls /dev/video*.")(
"algorithm,a", po::value<string>(&opts.algorithm)->default_value("NLOPT_LN_SBPLX"),
"algorithm for solver, such as NLOPT_LN_BOBYQA, NLOPT_LN_SBPLX, NLOPT_LN_COBYLA.")(
"maxTime,T", po::value<int>(&opts.maxSolverTime)->default_value(240),
"max solver run time in seconds.")(
"filterSize,s", po::value<int>(&opts.smoothFilterSize)->default_value(15),
"size of gaussian smoothing window, and sigma = s/3")(
"verbose,v", po::value<int>(&opts.verbosity)->default_value(2),
"verbosity, how much to display crap. between 0 and 3.")(
"waittime,W", po::value<int>(&opts.waitKeyLength)->default_value(5),
"number of milliseconds to wait on display (needs to be higher if running from matlab for example)")(
"lambdaT,A", po::value<string>(&opts.lambdaT)->default_value("0.1"),
"T regularizer weight")(
"lambdaW,R", po::value<string>(&opts.lambdaW)->default_value("0.1"),
"W regularizer weight")(
"lambdaF,Z", po::value<string>(&opts.lambdaF)->default_value("0.1"),
"f-scale regularizer weight")(
"alphaIO,C", po::value<string>(&opts.alphaIO)->default_value("0.1"),
"chan-vese-like weight, in/out disparity")(
"alphaIT,D", po::value<string>(&opts.alphaIT)->default_value("0.1"),
"reverse chan-vese-like weight, in/in-template similarity");
    
    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);
    
    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }
    
    if (!vm.count("intrinsics")) {
      cout << "Must supply a camera calibration file" << "\n";
      cout << desc << endl;
      return 1;
    }
    return 0;
    
  }
  
}

class RigidTransformFitter: public OptimProblem {
public:
  RigidTransformFitter() {
  }
  virtual ~RigidTransformFitter() {
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
    stop_criteria.maxtime  = maxSolverTime;
    stop_criteria.start    = 0.0;
    stop_criteria.force_stop = 0;
    stop_criteria.minf_max   = 1e9;
    return stop_criteria;
  }
  
  void displayOnWarpedTemplate( )
  {
    w_ch[0] = i_ch[0] + warped_mask_border;
    w_ch[1] = i_ch[1] + warped_mask_border_narrow;
    w_ch[2] = i_ch[2] + warped_mask_border;
    merge(w_ch,warped_template);
  }
  
  void   displayCallback(int twait = 0)
  {
    cout << "w_est = " << w_est << ", " << "T_est = " << T_est << endl;
    if( verbosity >= 2 ) {
      displayOnWarpedTemplate();
      waitKey(1);
    }
    
    T_est_float.at<float>(2) +=  1.0;
    stringstream ss;
    ss << "F(x) = " << setprecision(4) << fval_best << ", N-iter = " << iters;
    poseDrawer(warped_template, K, w_est_float, T_est_float, ss.str() );
    
    imshow("warped_template",warped_template);
    if( !solver_prefix.empty() ) {
      stringstream ss;
      ss << solver_prefix << std::setw(4) << std::setfill('0') << this->iters << ".png";
      imwrite( ss.str(), warped_template );
    }
    char key = waitKey(twait);
    cout << "iters: " << iters << endl;
    if( 'q' == key ) {
      cout <<"q was pressed, exiting! " << endl;
      exit(0);
    }
    
  }
  void writeImageCallback(const string& warpname ) {
    displayOnWarpedTemplate();
    Mat outimg = warped_template.clone();
    outimg.convertTo(outimg,CV_8UC3);
    
    T_est_float.at<float>(2) +=  1.0;
    poseDrawer(outimg, K, w_est_float, T_est_float);
    imwrite( warpname, outimg );
  }
  double evalCostFuncBasic( ) {
    double fval = 0.0;
   
    double nnz_projected = (cv::sum(warped_mask)[0] + 1e-2) * (1 / 255.0);      
    static Mat warped_mask_not;
    cv::Scalar mean_rgb_in  = cv::mean(input_img,warped_mask);
    cv::bitwise_not(warped_mask,warped_mask_not);
    cv::Scalar mean_rgb_out = cv::mean(input_img,warped_mask_not);
    
    for(int i = 0; i < (int) w_ch.size();i++)
    { 
      double fval_i       =  (norm(w_ch[i],i_ch[i], cv::NORM_L1,warped_mask))/(nnz_projected);
      fval_i             += -(pow( (mean_rgb_in[i] - mean_rgb_out[i]        ),2.0))*(1.0/255.0)*alpha_in_out;
      fval_i             +=  (pow( (mean_rgb_in[i] - template_mean_rgb_in[i]),2.0))*(1.0/255.0)*alpha_template;
      fval               +=  fval_i;
    }

    fval += lambda_W * norm( w_est - w0 ,NORM_L1) + lambda_T * norm(T_est-T0,NORM_L1); // regularize
    fval += lambda_F * abs( f_est - f0 );
    return fval;
  }
  
  
  double evalCostFunction(const double* X, double*) {
    
    iters++;
    memcpy(w_est.data, X, 3 * sizeof(double));
    memcpy(T_est.data, X + 3, 3 * sizeof(double));
    memcpy(&f_est, X + 6, sizeof(double));
    
    T_est.convertTo(T_est_float,CV_32F);
    w_est.convertTo(w_est_float,CV_32F);
    f_est_float = (float) f_est;
    
    Rodrigues(w_est_float, R_est); //get a rotation matrix
    
    applyPerspectiveWarp();
    double fval = evalCostFuncBasic();
    
    if( fval < fval_best ) { 
      if( verbosity >= 1 ) {
        if( verbosity >= 2 ) {
          displayCallback( input_opts.waitKeyLength );
        }
        if( rand() % 3 == 1 ) {
          cout << "fval: " << fval << ", iters: " << iters
              << ", f-step: " << abs(fval-fval_best) << endl;
        }
      }
      fval_best = fval;
    }
    
    return fval;
  }
  virtual std::vector<double> ub() const
  {
    double c[7] = {CV_PI / 6, CV_PI / 6, CV_PI/6, 0.2, 0.2, 0.2, 20.0 };
    for( int k = 0; k < 7; k++ ) {
      c[k] += xg_input[k];
    }
    return std::vector<double>(c,c+7);
  }
  
  virtual std::vector<double> lb() const
  {
    double c[7] = {-CV_PI / 6, -CV_PI / 6, -CV_PI/6, -0.2, -0.2, -0.2, -20.0 };
    for( int k = 0; k < 7; k++ ) {
      c[k] += xg_input[k];
    }
    return std::vector<double>(c,c+7);
  }
  
  void applyPerspectiveWarp( )
  {
    // Rotation
    double r11 = R_est.at<float>(0,0);        double r12 = R_est.at<float>(0,1);
    double r21 = R_est.at<float>(1,0);        double r22 = R_est.at<float>(1,1);
    double r31 = R_est.at<float>(2,0);        double r32 = R_est.at<float>(2,1);
    
    // Translation
    double tx  = T_est.at<double>(0);
    double ty  = T_est.at<double>(1);
    double tz  = T_est.at<double>(2);
    
    H_est.at<float>(0,0) = r11; H_est.at<float>(0,1) = r12; H_est.at<float>(0,2) = tx;
    H_est.at<float>(1,0) = r21; H_est.at<float>(1,1) = r22; H_est.at<float>(1,2) = ty;
    H_est.at<float>(2,0) = r31; H_est.at<float>(2,1) = r32; H_est.at<float>(2,2) = 1.0+tz;
    
    // Zoom
    Mat Kleft;
    K.copyTo(Kleft);
    Kleft.at<float>(0,0) = f_est_float;
    Kleft.at<float>(1,1) = f_est_float;
    
    H_est =  (Kleft * H_est * K.inv());
    int nthreads, tid;
    nthreads = omp_get_num_threads();
#pragma omp parallel shared(nthreads, tid)
    { // fork some threads, each one does one call to expensive warpPerspective
      tid = omp_get_thread_num();
      if( tid == 0 )
        warpPerspective(template_img, warped_template, H_est,
                        template_img.size(), cv::INTER_LINEAR ,
                        cv::BORDER_CONSTANT, Scalar::all(0));
      else if( tid == 1 )
        warpPerspective(mask_img, warped_mask, H_est,
                        template_img.size(), cv::INTER_LINEAR ,
                        cv::BORDER_CONSTANT, Scalar::all(0));
      else if( tid == 2 )
        warpPerspective(mask_border, warped_mask_border, H_est,
                        template_img.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, Scalar::all(0));
      else if( tid == 3 )
        warpPerspective(mask_border_narrow, warped_mask_border_narrow, H_est,
                        template_img.size(), cv::INTER_LINEAR,
                        cv::BORDER_CONSTANT, Scalar::all(0));
    }
    
    cv::split( warped_template,w_ch );
  }
  
  virtual size_t N() const {
    return 7; //6 free params, 3 for R 3 for T, 1 "f"
  }
  
  virtual OptimAlgorithm getAlgorithm() const {
    //http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms
    if( algorithm.compare("NLOPT_LN_SBPLX") == 0 )
      return NLOPT_LN_SBPLX;
    if( algorithm.compare("NLOPT_LN_COBYLA") == 0 ) {
      return NLOPT_LN_COBYLA;
    }
    if( algorithm.compare("NLOPT_LN_BOBYQA") == 0 ) {
      return NLOPT_LN_BOBYQA;
    }
    cout << "warning, algorithm string is bogus. using default..." << endl;
    return  NLOPT_LN_SBPLX;
  }
  void setup(const vector<Mat>& data) {
    
    // what we're allowed to see: image points and calibration K matrix
    K = data[0].clone();
    input_img    = data[1].clone();
    template_img = data[2].clone();
    mask_img     = data[3].clone();
    input_img.copyTo(input_img_original);
    { // setup image-proc operations
      int fz     = filterSize;
      if( fz > 1 ) {
        if( fz % 2 == 0 ) {
          fz += 1; // must be odd or GaussianBlur fails
        }
        double sig = filterSize / 3.0;
        cv::GaussianBlur( input_img.clone(), input_img, Size(fz,fz),sig,sig );
        cv::GaussianBlur( template_img.clone(), template_img, Size(fz,fz),sig,sig );
      }
      cv::bitwise_and(template_img, mask_img, template_img);
      cv::cvtColor( mask_img.clone(), mask_img, CV_RGB2GRAY );
      mask_img.clone().convertTo(mask_img,CV_8U);
    }
    iters = 0;
    
    T_est = Mat::zeros(3, 1, CV_64F);
    w_est = Mat::zeros(3, 1, CV_64F);
    R_est = Mat::eye(3,3,CV_32F);
    H_est = Mat::zeros(3,3,CV_32F);
    i_ch.resize(3);
    w_ch.resize(3);
    split( input_img, i_ch ); // rgb of input image
    
    nnz_mask_init = cv::sum(mask_img)[0];
    
    
    // compute and store the mean of inside & outside in the template 
    Mat mask_img_not = mask_img.clone();
    cv::bitwise_not(mask_img,mask_img_not);
    template_mean_rgb_in  = cv::mean(template_img,mask_img);
    template_mean_rgb_out = cv::mean(template_img,mask_img_not);
    
    // create border mask for display
    int borderSize = (int) ( mask_img.rows / 100.0 + 0.5 ); 
    borderSize    += ( borderSize % 2 == 0 );
    borderSize     = std::max( 5, borderSize );
    cv::Canny( mask_img, mask_border, 1, 2 );
    mask_border = mask_border * borderSize * 2;
    cv::GaussianBlur(mask_border.clone(), mask_border, Size(borderSize,borderSize),3.0,3.0);
    cv::Canny( mask_border, mask_border_narrow, 1, 2 );
    cv::GaussianBlur(mask_border_narrow.clone(), mask_border_narrow, 
                              Size(borderSize-2,borderSize-2),2.0,2.0);
    mask_border = mask_border - mask_border_narrow;
    
    cout << "verbosity level is: " << verbosity << endl;
    if( verbosity > 0 ) {
      cv::namedWindow("warped_template");
    }
    if( verbosity > 2 ) {
      imshow("border",mask_border); waitKey(0);
      imshow("input_image",input_img);
      imshow("template_image",template_img);
      imshow("mask_image",mask_img);
      cout << "PRESS KEY TO CONTINUE " << endl;
      cv::waitKey(0);
    }
    fval_best = 1e9;
  }
  virtual std::vector<double> Xinit() const
  {
    std::vector<double> x0 = std::vector<double>(N(), 0.0);
    for( int k = 0; k < (int) N(); k++ ) {
      x0[k] = xg_input[k];
    }
    return x0;
  }
  
  static void write_RT_to_csv( const string& csvFile, const vector<double>& WT ) {
    cout << "attempting to write csv file in " << csvFile << endl;
    std::ofstream  data(csvFile.c_str());
    stringstream ss;
    // ss << "wt_out= " << Mat(WT) << ";"; matlab-format (?)
    ss <<WT[0]<< ","<< WT[1]<< ","<<WT[2]<<","<<WT[3]<<","<<WT[4]<<","<<WT[5] << ","<<WT[6];
    data << ss.str() << endl;
    data.close();
  }
  
  static void load_RT_from_csv( const string& csvFile, vector<float>& WT ) {
    cout << "attempting to open csv file in " << csvFile << endl;
    std::ifstream  data(csvFile.c_str());
    std::string line;
    while(std::getline(data,line))
    {
      std::stringstream  lineStream(line);
      std::string        cell;
      int column_index = 0;
      while(std::getline(lineStream,cell,',')) {
        if(column_index > 6) { cerr << "Oh snap, bogus CSV file!" << endl; exit(1); }
        WT[column_index++] = lexical_cast<float>(cell);
      }
    }
    data.close();
  }
  
  void setup( const Options& opts ) {
    cv::Mat K,D;
    cv::Size image_size;
    input_opts = opts;
    
    vector<Mat> data(4); // poltergeist to send to RT fitter
    imread( opts.input_img).convertTo(data[1],CV_8UC3);
    imread( opts.template_img).convertTo(data[2],CV_8UC3);
    imread( opts.mask_img).convertTo(data[3],CV_8UC3);
    
    cout << "reading K from calibration file " << opts.k_file << endl;
    readKfromCalib(K,D,image_size,opts.k_file);
    solver_prefix = opts.save_solver_imgs_prefix;
    
    K.copyTo(data[0]);
    for( int k = 0; k < (int) data.size(); k++ ) {
      if( data[k].empty() ) {
        std::cerr << k << std::endl;
        cerr << "Bogus input: check camera file, input, template, and mask jive!" << endl;
        exit(1);
      }
    }
    
    xg_input = vector<float>(7,0.0);
    w0          = Mat::zeros(3,1,CV_64F);
    T0          = Mat::zeros(3,1,CV_64F);
    xg_input[5] = 0.0; // negative: further away than template
    xg_input[6] = K.at<float>(0,0); 
    if( opts.guess_initial.size() >= 3 ) { // read from arg if it exists
      load_RT_from_csv( opts.guess_initial, xg_input );
    }
    for( int k = 0; k < 3; k++ ) { // store w0,T0,f0 for regularizer costs later 
      w0.at<double>(k) = xg_input[k];
      T0.at<double>(k) = xg_input[k+3];
    }
    f0 = xg_input[6];
  
    cout << "x0 = " << endl << Mat(xg_input) << endl;
    
    verbosity     = opts.verbosity;
    algorithm     = opts.algorithm;
    filterSize    = opts.smoothFilterSize;
    maxSolverTime = (double) opts.maxSolverTime;
    cout << "using solver: " << algorithm << endl;
    setup(data);

    lambda_T       = boost::lexical_cast<double>(opts.lambdaT); 
    lambda_W       = boost::lexical_cast<double>(opts.lambdaW);
    lambda_F       = boost::lexical_cast<double>(opts.lambdaF);
    alpha_in_out   = boost::lexical_cast<double>(opts.alphaIO);
    alpha_template = boost::lexical_cast<double>(opts.alphaIT);
    printf("regularizers: %f, %f, %f, %f \n",lambda_T,lambda_W,alpha_in_out,alpha_template);
    cout << endl;
  }
  
public:
  // some internal persistent data
  
  double lambda_T, lambda_W, lambda_F, alpha_in_out, alpha_template;

  double fval_best;
  int iters,verbosity,filterSize;
  
  Mat K; //camera matrix
  Mat template_img;
  Mat weighted_mask; // TODO: weight the mask by this ... less at the edges!
  Mat mask_img;
  Mat mask_border;
  Mat mask_border_narrow;
  Mat input_img; // gets blurred for image processing convergence 
  Mat input_img_original;  // for final display 
  Mat warped_template; // warp template to match input
  Mat warped_mask_border;
  Mat warped_mask_border_narrow;
  Mat warped_mask;
  Scalar template_mean_rgb_out;
  Scalar template_mean_rgb_in;
  double nnz_mask_init;
  double maxSolverTime;
  double template_mean_cost;
  vector<float> xg_input;  //-g
  vector<float> xf_output; //-f
  vector<Mat> i_ch; // input-image channels
  vector<Mat> w_ch; // warped-template channels
  string algorithm; // what solver to use ,  -a
  string solver_prefix;
  
  // initial inputs
  Mat w0;
  Mat T0;
  double f0;
  
  // solving for these
  Mat T_est;
  Mat w_est;
  double f_est;
  Mat w_est_float;
  Mat T_est_float;
  float f_est_float;
  Mat R_est;
  Mat H_est;
  
  Options  input_opts; 
};

void printBodyCount( NLOptCore& opt_core )
{
  vector<double> optimal_W_and_T = opt_core.getOptimalVector();
  double fval_final = opt_core.getFunctionValue();
  cout << "result code: " << opt_core.getResult() << endl;
  cout << " [w,T] optimal = " << Mat(optimal_W_and_T) << endl;
  cout << " error = " << fval_final << endl;
  
}

int main(int argc, char** argv) {
  Options opts;
  if (options(argc, argv, opts))
    return 1;
  
  omp_set_num_threads(4);
  
  boost::shared_ptr<RigidTransformFitter> RT(new RigidTransformFitter());
  RT->setup( opts );
  
  // lock and load
  NLOptCore opt_core(RT);
  
  // launch the nukes
  opt_core.optimize();
  
  // evaluate body count
  vector<double> optimal_W_and_T = opt_core.getOptimalVector();
  printBodyCount(opt_core);
  RigidTransformFitter::write_RT_to_csv( opts.x_out_final, optimal_W_and_T );
  RT->writeImageCallback(opts.out_warp_img);
  waitKey(5);
  cout << "DONE." << endl;
  return 0;
  
}
