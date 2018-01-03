#include "sparseFlow.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::ml;

/**
	Calculates sparse optical flow - optical flow for a collection of points on an image frame from a video
	Moving objects and tall objects should have a greater magnitude of optical flow
*/
int sparseFlow(char* videoFileName) {
	VideoCapture vid(videoFileName);
	int maxImgNum = 1000;
	
	//skip ahead 50 frames
	Mat frame;
	for (int i = 0; i < 50; ++i) {
		vid >> frame;
	}

	vector<Point2f> cornersA;
	
	//creates a grid of points for which the optical flow is calculated
	for (int i = 0; i < frame.rows; ++i) {
		if ((i % 5) == 0) {
			for (int j = 0; j < frame.cols; ++j) {
				if ((j % 5) == 0) {
					Point2f point = Point2f(j, i);
					cornersA.push_back(point);
				}
			}
		}
	}
	
	//properties for the points for which the optical flow is calculated
	struct gridSquare {
		Point2f location = 0;
		Point2f diff = 0;
		float length = 0;
		float angle = 0;
	};

	for (int imgNum = 0; imgNum < maxImgNum; ++imgNum) {
		cout << "yes" << endl;
		Mat imgA, imgB, imgC;
		vid >> imgA;
		vid >> imgB;
		imgC = imgB.clone();

		cvtColor(imgA, imgA, CV_BGR2GRAY);
		cvtColor(imgB, imgB, CV_BGR2GRAY);

		vector<Point2f> cornersB;
		vector<uchar> cornersStatus;
		vector<gridSquare> grid;
		vector<float> errors;
		float maxLength = 0;
		calcOpticalFlowPyrLK(imgA, imgB, cornersA, cornersB, cornersStatus, errors, Size(21, 21), 5);
		
		for (int i = 0; i < (int)cornersStatus.size(); ++i) {
			gridSquare square;
			//ignores points with large calculation errors for the optical flow
			if (!cornersStatus[i] || (cornersB[i].x < 0) || (cornersB[i].y < 0) || (errors[i]>0.5)) {
				grid.push_back(square);
				continue;
			}
			else {
				//draws the optical flow onto the image
				square.location =cornersA[i];
				square.diff = Point2f(cornersB[i].x - cornersA[i].x, cornersB[i].y - cornersA[i].y);
				square.length = euclideanDist(cornersB[i], cornersA[i]);
				square.angle = cvFastArctan(square.diff.y, square.diff.x);
				grid.push_back(square);
				maxLength = max(maxLength, square.length);
				line(imgC, cornersA[i], cornersB[i], Scalar(0, 255, 0), 1);
			}
		}

		Mat gridMat(imgA.rows / 5, imgA.cols / 5, CV_8UC1, Scalar::all(0));
		for (int i = 0; i < (imgA.rows / 5); ++i) {
			for (int j = 0; j < (imgA.cols / 5); ++j) {
				gridMat.at<uchar>(i, j) = saturate_cast<uchar>(80*(grid[((imgA.rows / 5)*i) + j].length));
			}
		}

		imshow("keypoints", imgC);
		waitKey(0);
	}
	return 0;
}
