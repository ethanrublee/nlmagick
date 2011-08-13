#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <sys/types.h>
#include <dirent.h>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/lexical_cast.hpp>

using std::string;
using std::vector;

template<typename Type>
void fillWeightsGaussian(cv::Mat& weights, Type sigma_squared)
{
  for (int y = 0; y < weights.rows; y++)
  {
    for (int x = 0; x < weights.cols; x++)
    {
      Type y_h = ((Type)y) / (weights.rows - 1.0) - 0.5;
      Type x_h = ((Type)x) / (weights.cols - 1.0) - 0.5;
      x_h *= 2; //x goes from -1 to 1
      y_h *= 2; //y "" ""
      Type val = ((x_h * x_h) + (y_h*y_h))/(2*sigma_squared);
      val = (Type)std::exp(-val);
      weights.at<Type> (y, x) = val;
    }
  }
}

void fillWeightsGaussian32(cv::Mat& weights, float sigma_squared)
{
  fillWeightsGaussian<float>(weights,sigma_squared);
}

void fillWeightsGaussian64(cv::Mat& weights, double sigma_squared)
{
  fillWeightsGaussian<double>(weights,sigma_squared);
}

inline bool readKfromCalib(cv::Mat& K, cv::Mat& distortion, cv::Size & img_size, const std::string& calibfile)
{
  cv::FileStorage fs(calibfile, cv::FileStorage::READ);
  cv::Mat cameramat;
  cv::Mat cameradistortion;
  float width = 0, height = 0;
  if (fs.isOpened())
  {
    cv::read(fs["camera_matrix"], cameramat, cv::Mat());
    cv::read(fs["distortion_coefficients"], cameradistortion, cv::Mat());
    cv::read(fs["image_width"], width, 0);
    cv::read(fs["image_height"], height, 0);

    fs.release();

  }
  else
  {
    throw std::runtime_error("bad calibration!");
  }

  cv::Size _size(width, height);
  img_size = _size;

  cameramat.convertTo(K,CV_32F);
  distortion = cameradistortion;
  return true;
}

inline void poseDrawer(cv::Mat& drawImage, const cv::Mat& K, 
                       const cv::Mat& w, const cv::Mat& t, 
                       const std::string scaleText = std::string(""), int lineThickness=4)
{
  using namespace cv;
  Point3f z(0, 0, -0.25);
  Point3f x(0.25, 0, 0);
  Point3f y(0, -0.25, 0);
  Point3f o(0, 0, 0);
  vector<Point3f> op(4);
  op[1] = x, op[2] = y, op[3] = z, op[0] = o;
  vector<Point2f> ip;

  Mat D = Mat::zeros(4,1,CV_32F);
  projectPoints(Mat(op), w, t, K, D, ip);
  double axes_sz = drawImage.rows / 4.0;
  double zmin    = 5e-2; 
//  ip[1] = ip[0] + (ip[1]- ip[0] ) * ( axes_sz / norm( ip[1] - ip[0] ) );
//  ip[2] = ip[0] + (-ip[2]+ip[0] ) * ( axes_sz / norm( ip[2] - ip[0] ) );
//  ip[3] = ip[0] + (-ip[3]+ip[0] ) * ( (1.0/sqrt(2))*axes_sz / ( zmin + norm( ip[3] - ip[0] ) ) );
  ip[1] = ip[0] + (ip[1]-ip[0] ) * ( axes_sz / norm( ip[1] - ip[0] ) );
  ip[2] = ip[0] + (ip[2]-ip[0] ) * ( axes_sz / norm( ip[2] - ip[0] ) );
  ip[3] = ip[0] + (ip[3]-ip[0] ) * ( (1.0/sqrt(2))*axes_sz / ( zmin + norm( ip[3] - ip[0] ) ) );

  // DRAW AXES LINES  
  vector<Scalar> c(4); //colors
  c[0] = Scalar(255, 255, 255);
  c[1] = Scalar(205, 50, 50);//x
  c[2] = Scalar(100, 200, 0);//y
  c[3] = Scalar(200, 100, 205);//z
  line(drawImage, ip[0], ip[1], c[1],lineThickness,CV_AA);
  line(drawImage, ip[0], ip[2], c[2],lineThickness,CV_AA);
  line(drawImage, ip[0], ip[3], c[3],lineThickness,CV_AA);
 
  if( scaleText.size() > 1 ) 
  { // print some text on the image if desired
    int baseline = 0;
    Size sz = getTextSize(scaleText, CV_FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
    rectangle(drawImage, Point(10, 30 + 5), 
              Point(10, 30) + Point(sz.width, -sz.height - 5), Scalar::all(0), -1);
    putText(drawImage, scaleText, Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 1.0, c[0], 1, CV_AA, false);
  }
  
  // DRAW LETTERS FOR AXES 
  c[1] += Scalar(50,50,50);
  c[2] += Scalar(50,50,50);
  c[3] += Scalar(50,50,50);
  putText(drawImage, "Z", ip[3], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[3], lineThickness, CV_AA, false);
  putText(drawImage, "Y", ip[2], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[2], lineThickness, CV_AA, false);
  putText(drawImage, "X", ip[1], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[1], lineThickness, CV_AA, false);

}


/**  given a "dir" as string and ending extension, put name of files
     into the vector string. vector is sorted lexicographically.
  */
void lsFilesOfType(const char * dir, const string& extension,
        vector<string>& files) {
    files.clear();
    DIR *dp;
    struct dirent *dirp;
    if ((dp = opendir(dir)) == NULL) {
        return;
    }

    while ((dirp = readdir(dp)) != NULL) {
        std::string name(dirp->d_name);
        size_t pos = name.find(extension);
        if (pos != std::string::npos) {
            files.push_back(name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
}
