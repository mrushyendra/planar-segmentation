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

//calculate euclidean distance between any two points
float euclideanDist(Point2f& p, Point2f& q) {
	Point2f diff = p - q;
	return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

//Calculate the median pixel value in the given image, ignoring any pixels in Mat exclude and including pixels specified by Mat mask
double calcMedian(Mat data, Mat exclude, Mat mask) {
	double median = 0;
	vector<float>dataVec;
	for (int i = 0; i < data.rows; i += 1) {
		for (int j = 0; j < data.cols; j += 1) {
			if (mask.at<uchar>(i, j) == 0) {
				continue;
			}
			if (exclude.at<uchar>(i, j) != 1) {
				dataVec.push_back(max(0.0f, data.at<float>(i, j)));
			}
		}
	}
	if (!dataVec.empty()) {
		vector<float>::iterator first = dataVec.begin();
		vector<float>::iterator last = dataVec.end();
		vector<float>::iterator middle = first + ((last - first) / 2);
		nth_element(first, middle, last);
		median = *middle;
	}
	else {
		median = 0;
	}
	return median;
}
int denseFlow();
int sparseFlow();

//usage: ./programName videoFileName
int main(int argc, char** argv) {
	denseFlow(argv[1]);
	return 0;
}

/**
	Calculates dense optical flow between 2 video frames, for all pixels on the frame, .
	Identifies outliers and draws them separately. Outliers are more likely to belong to tall objects because they don't move with a greater magnitude than pixels in the flat areas in the image
*/
int denseFlow(char* videoFileName) {
	VideoCapture vid(videoFileName);

	int frameCount = 0;
	double magnitudeMedian = 0;
	double angleMedian = 0;
	
	while (true) {	
		Mat prevLarge, nextLarge, prev, next, gap;
		vid >> prevLarge; //get an image frame
		for (int count = 0; count < 2; ++count) { //skip2 frames in between
			vid >> gap;
		}
		vid >> nextLarge; //get another image frame
		if (prevLarge.empty() || nextLarge.empty()) {
			break;
		}
		resize(prevLarge, prev, prev.size() / 2, 0.5, 0.5); //resize to smaller image for faster computation
		resize(nextLarge, next, next.size() / 2, 0.5, 0.5);
		Mat prevColor = prev.clone();
	
		cvtColor(prev, prev, CV_BGR2GRAY);
		cvtColor(next, next, CV_BGR2GRAY);
		
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		Mat cannyOutput;
		Canny(prev, cannyOutput, 30, 60);
		findContours(cannyOutput, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
		Mat mask = Mat::zeros(prev.size(), CV_8UC1);
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(255);
			drawContours(mask, contours, i, color, 1, 8, hierarchy, 0, Point());
		}
		imshow("contours", mask);
		waitKey(0);

		Mat flow(prev.size(), CV_32FC2);
		calcOpticalFlowFarneback(prev, next, flow, 0.5, 3, 15, 3, 7, 1.5, 0); //calculate the optical flow of pixels from 1 frame to the next

		Mat flowXY[2];
		split(flow, flowXY);
		Mat magnitudes, angles;
		cartToPolar(flowXY[0], flowXY[1], magnitudes, angles, true); //convert the optical flow values to 2 tables containing the angles and magnitudes of flow for each pixel
		
		Mat outliers(prev.size(), CV_8UC1, Scalar::all(0)); //A mask with all the outlier pixels set to a non-zero value
		for (int iterations = 0; iterations < 2; ++iterations) {
			magnitudeMedian = calcMedian(magnitudes, outliers, mask); //Use the contours detected earlier as a mask. Calculate the median optical flow of pixels on the mask and mark outliers
			angleMedian = calcMedian(angles, outliers, mask); //calculate median value of the change in angle for pixels on the mask and mark outliers 
			//contours are used as a mask because edges provide the most reliable pixels to calculate the optical flow. 
			//On the other hand, the calculation of optical flow for pixels in large uniform areas like the road is unreliable
			for (int i = 0; i < prev.rows; i += 5) {
				for (int j = 0; j < prev.cols; j += 5) {
					if (outliers.at<uchar>(i, j) != 1) {
						if ((abs(magnitudes.at<float>(i, j)) - abs(magnitudeMedian)) > 2.0f) {
							outliers.at<uchar>(i,j) = 1;
						}
					}
				}
			}
		}
		
		//draw the optical flow values
		normalize(magnitudes, magnitudes, 0, 255, NORM_MINMAX);
		for (int i = 0; i < prev.rows; i += 1) {
			for (int j = 0; j < prev.cols; j += 1) {
				Scalar color = Scalar(0, 255, 0);
				Point2f fxy = flow.at<Point2f>(i, j);
				int intensity = magnitudes.at<float>(i, j);
				color = Scalar(0, 0, intensity);
				//if ((abs(magnitudes.at<float>(i, j)) - abs(magnitudeMedian)) > 0.0f) {
				//	int intensity = saturate_cast<int>(255 * ((abs(magnitudes.at<float>(i, j)) - abs(magnitudeMedian)) / 2.0f));
				//	color = Scalar(0, 0, intensity);
					//if ((abs(angles.at<float>(i, j) - angleMedian) > 20.0f)) {
					//	color = Scalar(255, 0, 0);
					//}
				//}
				//else {
				//	int intensity = saturate_cast<int>(255 * (abs(magnitudes.at<float>(i, j) - magnitudeMedian) / 2.0f));
				//	color = Scalar(0, intensity, 0);
				//}
				line(prevColor, Point(j, i), Point(round(j + fxy.x), round(i + fxy.y)), color);
				circle(prevColor, Point(round(j + fxy.x), round(i + fxy.y)), 1, color);
			}
		}
		imshow("optical flow", prevColor);
		waitKey(0);

		Mat prevThres; //convert to black and white
		cvtColor(prevColor, prevThres, CV_BGR2GRAY);
		//adaptiveThreshold(prevThres, prevThres, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);
		threshold(prevThres, prevThres, 0, 255, THRESH_OTSU|THRESH_BINARY); //perform binarization on the optical flow image to find any outliers that have a large magnitude of optical flow
		imshow("optical flow after", prevThres);
		waitKey(0);
		
		vector<vector<Point>> outlierContours; //using the database of outlier points, all the contours are checked if they have any of the outlier points in them
		for (int i = 0; i < prev.rows; i += 5) {
			for (int j = 0; j < prev.cols; j += 5) {
				if (outliers.at<uchar>(i, j) == 1) {
					for (int num = 0; num < contours.size(); ++num) {
							double result = pointPolygonTest(contours[num], Point2f(j, i), false);
							if (result == 0 || result == 1) {
								outlierContours.push_back(contours[num]);
								if (hierarchy[num][2] != -1)
									outlierContours.push_back(contours[hierarchy[num][2]]);
								if (hierarchy[num][3] != -1)
									outlierContours.push_back(contours[hierarchy[num][3]]);
							}
					}
				}
			}
		}

		
		//outlier contours are drawn and displayed
		Mat tallObjectsMask = Mat::zeros(prev.size(), CV_8UC1);
		for (int i = 0; i<outlierContours.size(); i++)
		{
			Scalar color = Scalar(255);
			drawContours(tallObjectsMask, outlierContours, i, color, 1, 8, noArray(), 0, Point());
		}
		imshow("outlier contours", tallObjectsMask);
		waitKey(0);
	
	}
	return 0;
}


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
