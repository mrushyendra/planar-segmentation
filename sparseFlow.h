#ifndef SPARSEFLOW_H
#define SPARSEFLOW_H

#include "opencv2/opencv.hpp"
#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/optflow.hpp"

float euclideanDist(Point2f& p, Point2f& q);
int sparseFlow(char* videoFileName);

#endif
