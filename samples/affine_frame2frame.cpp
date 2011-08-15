

#include "nlmagick/nlopt.hpp"

#include <chernobylite/opencv_helpers.hpp>
using namespace chernobylite;

using namespace cv;
using namespace std;
using namespace nlopt;

namespace po = boost::program_options;
using boost::lexical_cast;


class AffineF2F: public OptimProblem
{
public:
  struct Options {
    string k_file;
    string focal_ratio;
    string input_img1;
    string input_img2;
    string out_warp_img;
    string algorithm;
    string directory;

    double vx0;
    double vy0;
    double vmax;
    int    verbosity;

  };

  static int options(int ac, char ** av, Options& opts) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message.")(
          "imageDirectory,d", po::value<string>(&opts.directory)->default_value(""),
          "directory of jpeg xor png images. overrides input A,B. ")(
          "inputimageA,A", po::value<string>(&opts.input_img1)->default_value(""),
          "first image (in time)")(
          "inputimageB,B", po::value<string>(&opts.input_img2)->default_value(""),
          "first image (in time)")(
          "outwarpimage,w", po::value<string>(&opts.out_warp_img)->default_value("warped.jpg"),
          "output for warped image, from 1 to 2.")(
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

    if( (opts.input_img1.empty() || opts.input_img2.empty()) &&
         opts.directory.empty() ) {
      cout << "Error, must either have image0 and image1 OR a directory." << endl;
      cout << desc << endl;
      return 1;
    }

    return 0;

  }

public:
  AffineF2F() {
  }
  virtual ~AffineF2F() {
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

  void   displayCallback(int twait = 0)
  {
    imshow("warped_best",warped_best);
    waitKey(twait);
  }


  double evalCostFunction(const double* X, double*)
  {
    double fval = 0.0;
    double dx = X[0];
    double dy = X[1];
    Af2f.at<double>(0,2) = dx;
    Af2f.at<double>(1,2) = dy;

    applyAffineWarp();

    for(int i = 0; i < (int) w_ch.size();i++)
    {
      cv::absdiff(w_ch[i],i_ch[i],tmp1);
      cv::multiply(tmp1,weights,tmp2);
      double fval_i = (cv::sum(tmp2)[0]);
      fval += fval_i;

    }

    iters++;
    if( (input_opts.verbosity >= 3) && fval < fval_best )
    {
      cout << "dx = " << dx << ", dy = " << dy << ", fval = "
           << fval << ", iters = " << iters << endl;
      fval_best = fval;
      warped_img1.convertTo(warped_best,CV_8UC3,255.0);
      displayCallback(10);
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

  void applyAffineWarp( )
  {
    warpAffine( input_img1, warped_img1, Af2f, input_img1.size(),
                cv::INTER_LINEAR,cv::BORDER_CONSTANT, Scalar::all(0)  );

    cv::split( warped_img1,w_ch );
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
    Mat img1 = imread(opts.input_img1);
    Mat img2 = imread(opts.input_img2);
    img1.convertTo(input_img1,CV_64FC3,1.0/255.0);
    img2.convertTo(input_img2,CV_64FC3,1.0/255.0);
    cv::split(input_img2,i_ch);

    weights = Mat::zeros( img1.rows, img1.cols, CV_64F);
    tmp1    = Mat::zeros( img1.rows, img1.cols, CV_64F);
    tmp2    = Mat::zeros( img1.rows, img1.cols, CV_64F);
    fillWeightsGaussian64(weights,0.25);

    Af2f = Mat::zeros(2,3,CV_64F);
    Af2f.at<double>(0,0) = 1.0;
    Af2f.at<double>(1,1) = 1.0;

    fval_best = 1e9;
    iters = 0;

  }


public:
  // some internal persistent data
  Options  input_opts;

  Mat input_img1;
  Mat input_img2;

  Mat weights;
  Mat warped_img1;
  Mat warped_best;
  Mat Af2f;
  Mat tmp1, tmp2;
  vector<Mat> w_ch, i_ch;
  double fval_best;
  int iters;
};


int main(int argc, char** argv) {
  AffineF2F::Options opts;
  if (AffineF2F::options(argc, argv, opts))
    return 1;

  vector<string> imgList;
  if( !opts.directory.empty() ) {
    cout << "getting files from: " << opts.directory << endl;
    lsFilesOfType(opts.directory.c_str(),".jpg",imgList);
    if( imgList.empty() ) {
      lsFilesOfType(opts.directory.c_str(),".png",imgList);
    }
    cout << "found # of imgs:" << imgList.size() << endl;
    for( int k=0; k<(int)imgList.size(); k++ ) {
      imgList[k] = opts.directory + "/" + imgList[k];
      cout << "adding image to list: " << imgList[k] << endl;
    }
  } else {
    imgList.push_back(opts.input_img1);
    imgList.push_back(opts.input_img2);
  }

  boost::shared_ptr<AffineF2F> RT(new AffineF2F());

  NLOptCore opt_core(RT);

  for( int k=1; k<(int)imgList.size(); k++ )
  {
    string img1 = imgList[k-1];
    string img2 = imgList[k];
    opts.input_img1 = img1;
    opts.input_img2 = img2;

    // lock and load
    RT->setup( opts );

    // launch the nukes
    opt_core.optimize();

    // evaluate body count
    vector<double> optimal_vxvy = opt_core.getOptimalVector();
    cout << Mat(optimal_vxvy) << ";" << endl;

    if( RT->input_opts.verbosity >= 1 )
    {
      int imgW = RT->input_img1.cols;
      int imgH = RT->input_img1.rows;
      double vx= 2*optimal_vxvy[0];
      double vy= 2*optimal_vxvy[1];
      Mat outimg = RT->input_img2.clone();
      Point2f origin( (imgW-1)/2, (imgH-1)/2);
      circle(outimg, origin,1,Scalar(0,0,255),3,CV_AA);
      circle(outimg, origin,1,Scalar(0,255,0),2,CV_AA);
      Point2f vxvy_offset = origin;
      vxvy_offset.x += vx ;
      vxvy_offset.y += vy ;
      circle(outimg, vxvy_offset,1,Scalar(0,0,0),4,CV_AA);
      circle(outimg, vxvy_offset,1,Scalar(200,0,200),3,CV_AA);
      circle(outimg, vxvy_offset,1,Scalar(200,0,0),2,CV_AA);
      line(outimg,origin,vxvy_offset,Scalar(100,200,200,100),2,CV_AA);
      line(outimg,origin,vxvy_offset,Scalar(0,0,0,100),1,CV_AA);
      imshow("from previous to current center. any key to continue. ", outimg );
      char key = 'x';
      if( RT->input_opts.verbosity > 1) {
        key = cv::waitKey(0); // verbose = 2, wait until key
      } else {
        cv::waitKey(100);       // verbose = 1, show fastly
      }
    }

  }
  return 0;

}

