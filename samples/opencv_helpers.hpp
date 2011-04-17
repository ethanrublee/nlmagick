#pragma once
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <fstream>

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

inline void poseDrawer(cv::Mat& drawImage, const cv::Mat& K, const cv::Mat& w, const cv::Mat& t)
{
  using namespace cv;
  Point3f z(0, 0, 0.25);
  Point3f x(0.25, 0, 0);
  Point3f y(0, 0.25, 0);
  Point3f o(0, 0, 0);
  vector<Point3f> op(4);
  op[1] = x, op[2] = y, op[3] = z, op[0] = o;
  vector<Point2f> ip;

  projectPoints(Mat(op), w, t, K, Mat(), ip);

  vector<Scalar> c(4); //colors
  c[0] = Scalar(255, 255, 255);
  c[1] = Scalar(255, 0, 0);//x
  c[2] = Scalar(0, 255, 0);//y
  c[3] = Scalar(0, 0, 255);//z
  line(drawImage, ip[0], ip[1], c[1]);
  line(drawImage, ip[0], ip[2], c[2]);
  line(drawImage, ip[0], ip[3], c[3]);
  string scaleText = "scale 0.25 meters";
  int baseline = 0;
  Size sz = getTextSize(scaleText, CV_FONT_HERSHEY_SIMPLEX, 1, 1, &baseline);
  rectangle(drawImage, Point(10, 30 + 5), Point(10, 30) + Point(sz.width, -sz.height - 5), Scalar::all(0), -1);
  putText(drawImage, scaleText, Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 1.0, c[0], 1, CV_AA, false);
  putText(drawImage, "Z", ip[3], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[3], 1, CV_AA, false);
  putText(drawImage, "Y", ip[2], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[2], 1, CV_AA, false);
  putText(drawImage, "X", ip[1], CV_FONT_HERSHEY_SIMPLEX, 1.0, c[1], 1, CV_AA, false);

}
