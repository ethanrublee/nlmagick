#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <fstream>

#include <nlopt/nlopt.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pano_core/Camera.h>
#include <pano_core/Blender.h>
#include <pano_core/panoutils.h>

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

using namespace pano;
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
    string input_img; // what we're solving on. no mask here.
    string template_img;
    string mask_img;  // corresponds to "template". 1 in the planar area we want.
    string out_warp_img;
    string guess_initial; // a 'csv' file with omega_hat and T initial guesses
    string x_out_final;   // a 'csv' file with omega_hat and T final solved values
    string algorithm;
    int maxSolverTime; // seconds
    int vid;        // /dev/videoN
    int verbosity;  // how much crap to display
};

int options(int ac, char ** av, Options& opts) {
    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()("help", "Produce help message.")(
            "intrinsics,K",
            po::value<string>(&opts.k_file),
            "The camera intrinsics file, should be yaml and have atleast \"K:...\". Required.")(
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
            "algorithm,a", po::value<string>(&opts.algorithm)->default_value("NLOPT_LN_BOBYQA"),
            "algorithm for solver, such as NLOPT_LN_BOBYQA (default), NLOPT_LN_SBPLX, NLOPT_LN_COBYLA.")(
            "maxTime,T", po::value<int>(&opts.maxSolverTime)->default_value(240),
            "max solver run time in seconds.")(
            "verbose,v", po::value<int>(&opts.verbosity)->default_value(2),
            "verbosity, how much to display crap. between 0 and ~3.");

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
        stop_criteria.ftol_abs = 1e-6;
        stop_criteria.ftol_rel = 1e-6;
        stop_criteria.xtol_rel = 1e-6;
        stop_criteria.nevals   = 0;
        stop_criteria.maxeval  = 50000;
        stop_criteria.maxtime  = maxSolverTime;
        stop_criteria.start    = 0.0;
        stop_criteria.force_stop = 0;
        stop_criteria.minf_max   = 1e9;
        return stop_criteria;
    }
    
    void   displayCallback(int twait = 0)
    {
       cout << "w_est = " << w_est << endl;
       cout << "T_est = " << T_est << endl;
       // cout <<"H_est = \n"<< H_est << endl;
       
       if( verbosity >= 2 ) {
         w_ch[2] = i_ch[2] + warped_mask_border;
         w_ch[0] = w_ch[0]*0.5 + i_ch[0]*0.5;
         i_ch[1].copyTo(w_ch[1]);
         merge(w_ch,warped_template);
       }       
       imshow("warped_template",warped_template);
       if( verbosity > 2 ) {
          imshow("warped_mask",warped_mask);
       }
       waitKey(twait);
       cout << "iters: " << iters << endl;
       waitKey(twait);
      
    }
    void writeImageCallback(const string& warpname ) {
      w_ch[2] = i_ch[2] + warped_mask_border;
      w_ch[0] = w_ch[0]*0.5 + i_ch[0]*0.5;
      i_ch[1].copyTo(w_ch[1]);
      merge(w_ch,warped_template);
      Mat outimg = warped_template.clone();
      outimg.convertTo(outimg,CV_8UC3);
      imwrite( warpname, outimg );
    }
    double evalCostFuncCrazy( ) {
      double fval = 0.0;
      double nnz_projected = cv::sum(warped_mask)[0];
      
      // double wsum = 1.0/(1.0 + nnz_projected ); // non-zero area
      
      cv::Scalar mean_rgb_in  = cv::mean(input_img,warped_mask);
      cvNot( &CvMat(warped_mask.clone()),&CvMat(warped_mask) );
      cv::Scalar mean_rgb_out = cv::mean(input_img,warped_mask);
            
      for(int i = 0; i < (int) w_ch.size();i++)
      {          
        double fval_i =  1.0 / ( (pow( (mean_rgb_in[i] - mean_rgb_out[i]), 2.0 )) + 1e-3 );
        fval_i       +=  abs( (mean_rgb_in[i] - template_mean_rgb_in[i])*1e-3 );
        fval_i       +=  abs( (mean_rgb_out[i] - template_mean_rgb_out[i])*1e-3 );
        fval_i       +=  (norm(w_ch[i],i_ch[i], cv::NORM_L2,warped_mask))/nnz_projected;
        fval         +=fval_i;
      }
      fval += 1e-3 * norm( w_est ); // regularize: shrink omega!
      fval += 1e-3 * norm( T_est ); // regularize: shrink T!

      // slightly ghetto: prevent projection of entire template off-screen
      double pA = abs( 1.0/(nnz_projected / (nnz_mask_init + 1e-8) ) );
      double pB = abs( (nnz_projected / (nnz_mask_init + 1e-8) ) );
      fval     += 1e-1*(pA + pB);
      return fval;
    }
    double evalCostFuncBasic( ) {
      double fval = 0.0;
      double nnz_projected = cv::sum(warped_mask)[0] + 1e-2;      
      
      cv::Scalar mean_rgb_in  = cv::mean(input_img,warped_mask);
      cvNot( &CvMat(warped_mask.clone()),&CvMat(warped_mask) );
      cv::Scalar mean_rgb_out = cv::mean(input_img,warped_mask);
      
      for(int i = 0; i < (int) w_ch.size();i++)
      {          
        double fval_i       =  (norm(w_ch[i],i_ch[i], cv::NORM_L2,warped_mask));
        fval               +=  fval_i;
      }
      fval *= 1 + ( 1e-2*norm( w_est ) + norm(T_est) ); // regularize: shrink omega!
      fval +=  1e-1 * nnz_projected / (input_img.cols * input_img.rows );
      
    }
    

    double evalCostFunction(const double* X, double* /*Gradient-Free!*/ ) {
        iters++;
        
        memcpy(w_est.data, X, 3 * sizeof(double));
        memcpy(T_est.data, X + 3, 3 * sizeof(double));

        Mat w_est_float;
        w_est.convertTo(w_est_float,CV_32F);
        Rodrigues(w_est_float, R_est); //get a rotation matrix
       
        applyPerspectiveWarp();
        double fval = evalCostFuncBasic();
       
        if( fval < fval_best ) { 
          if( verbosity >= 1 ) {
            if( verbosity >= 2 ) {
              displayCallback(5);
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
        double c[6] = {CV_PI / 2, CV_PI / 2, CV_PI/2, 
                       0.25,0.25,5.0  };
        return std::vector<double>(c,c+6);
      }

      virtual std::vector<double> lb() const
      {
        double c[6] = {-CV_PI / 2, -CV_PI / 2, -CV_PI/2,
                        -0.25,-0.25,-0.7 };
        return std::vector<double>(c,c+6);
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
        
        H_est =  (K * H_est * K.inv());


        warpPerspective(template_img, warped_template, H_est,
                        template_img.size(), cv::INTER_LINEAR ,
                        cv::BORDER_CONSTANT, Scalar::all(0));

        warpPerspective(mask_img, warped_mask, H_est,
                        template_img.size(), cv::INTER_LINEAR ,
                        cv::BORDER_CONSTANT, Scalar::all(0));

        warpPerspective(mask_border, warped_mask_border, H_est,
                          template_img.size(), cv::INTER_LINEAR ,
                          cv::BORDER_CONSTANT, Scalar::all(0));
       
        
        cv::split( warped_template,w_ch );
    }
 
    virtual size_t N() const {
        return 6; //6 free params, 3 for R 3 for T, 1 "f"
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
      return  NLOPT_LN_BOBYQA;
    }

    void setup(const vector<Mat>& data) {

        // what we're allowed to see: images and calibration K matrix
        K = data[0].clone();
        input_img    = data[1].clone();
        template_img = data[2].clone();
        mask_img     = data[3].clone();
        
        { // setup image-proc operations
          cv::GaussianBlur( input_img.clone(), input_img, Size(21,21),9.0,9.0 );
          cv::GaussianBlur( template_img.clone(), template_img, Size(21,21),9.0,9.0 );
          cvAnd( &CvMat(template_img.clone()), &CvMat(mask_img), &CvMat(template_img) );
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

        template_mean_rgb_in  = cv::mean(template_img,mask_img);
        Mat mask_img_not = mask_img.clone();
        cvNot( &CvMat(mask_img),&CvMat(mask_img_not) );
        template_mean_rgb_out = cv::mean(template_img,mask_img_not);

        cv::Canny( mask_img, mask_border, 0, 1 ); // create border mask for display
        
        cout << "verbosity level is: " << verbosity << endl;
        if( verbosity > 2 ) {
          imshow("border",mask_border); waitKey(0); 
          imshow("input_image",input_img);
          imshow("template_image",template_img);
          imshow("mask_image",mask_img);
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
        ss <<WT[0]<< ","<< WT[1]<< ","<<WT[2]<<","<<WT[3]<<","<<WT[4]<<","<<WT[5];
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
            if(column_index > 5) { cerr << "Oh snap, bogus CSV file!" << endl; exit(1); }
            WT[column_index++] = lexical_cast<float>(cell);
          }
       }
       data.close();
    }

    void setup( const Options& opts ) {
      Camera camera(opts.k_file);
      vector<Mat> data(4); // poltergeist to send to RT fitter
      data[0]    = camera.K();
      imread( opts.input_img).convertTo(data[1],CV_8UC3);
      imread( opts.template_img).convertTo(data[2],CV_8UC3);
      imread( opts.mask_img).convertTo(data[3],CV_8UC3);

      for( int k = 0; k < (int) data.size(); k++ ) {
        if( data[k].empty() ) {
          cerr << "Bogus input: check camera file, input, template, and mask jive!" << endl;
          exit(1);
        }
      }
      
      
      xg_input = vector<float>(6,0); xg_input[5] = 0.0;
      if( opts.guess_initial.size() >= 3 ) {
        load_RT_from_csv( opts.guess_initial, xg_input );
      }
      cout << "x0 = " << endl << Mat(xg_input) << endl;

      verbosity     = opts.verbosity;
      algorithm     = opts.algorithm;
      maxSolverTime = (double) opts.maxSolverTime;
      
      cout << "using solver: " << algorithm << endl;
      setup(data);
    }

public:
    double fval_best;
    int iters,verbosity;
    // some internal persistent data
    Mat K; //camera matrix
    Mat template_img;
    Mat weighted_mask; // TODO: weight the mask by this ... less at the edges!
    Mat mask_img;
    Mat mask_border;
    Mat input_img;
    Mat warped_template; // warp template to match input
    Mat warped_mask_border;
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
    string algorithm; // what solver to use 

    // solving for these
    Mat T_est;
    Mat w_est;
    Mat R_est;
    Mat H_est;

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
    cout << "DONE." << endl;
    return 0;

}
