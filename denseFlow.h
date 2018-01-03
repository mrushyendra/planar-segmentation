#ifndef DENSEFLOW_H
#define DENSEFLOW_H

#include "opencv2/opencv.hpp"
#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/optflow.hpp"

double calcMedian(cv::Mat data, cv::Mat exclude, cv::Mat mask);
int denseFlow(char* videoFileName);

#endif
