
#include sparseFlow.h
#include denseFlow.h
#include "opencv2/opencv.hpp"
#include "opencv2/ccalib/omnidir.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/optflow.hpp"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char** argv) {
	denseFlow(argv[1]);
	return 0;
}
