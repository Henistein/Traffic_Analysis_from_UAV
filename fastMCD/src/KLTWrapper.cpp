// Copyright (c) 2016 Kwang Moo Yi.
// All rights reserved.

// This  software  is  strictly   for  non-commercial  use  only.  For
// commercial       use,       please        contact       me       at
// kwang.m<dot>yi<AT>gmail<dot>com.   Also,  when  used  for  academic
// purposes, please cite  the paper "Detection of  Moving Objects with
// Non-stationary Cameras in 5.8ms:  Bringing Motion Detection to Your
// Mobile Device,"  Yi et  al, CVPRW 2013  Redistribution and  use for
// non-commercial purposes  in source  and binary forms  are permitted
// provided that  the above  copyright notice  and this  paragraph are
// duplicated  in   all  such   forms  and  that   any  documentation,
// advertising  materials,   and  other  materials  related   to  such
// distribution and use acknowledge that the software was developed by
// the  Perception and  Intelligence Lab,  Seoul National  University.
// The name of the Perception  and Intelligence Lab and Seoul National
// University may not  be used to endorse or  promote products derived
// from this software without specific prior written permission.  THIS
// SOFTWARE IS PROVIDED ``AS IS''  AND WITHOUT ANY WARRANTIES.  USE AT
// YOUR OWN RISK!

#include "KLTWrapper.hpp"


KLTWrapper::KLTWrapper(void)
{
	// For LK funciton in opencv
	win_size = 10;
	//status = 0;
	count = 0;
	flags = 0;

	eig = NULL;
	temp = NULL;
	maskimg = NULL;
}

KLTWrapper::~KLTWrapper(void)
{
	eig->release();
	temp->release();
	maskimg->release();
}

void KLTWrapper::Init(Mat imgGray)
{
	int ni = imgGray.cols;
	int nj = imgGray.rows;

	// Allocate Maximum possible + some more for safety
	MAX_COUNT = (float (ni) / float (GRID_SIZE_W) + 1.0)*(float (nj) / float (GRID_SIZE_H) + 1.0);

	// Pre-allocate
	image = Mat(Size(imgGray.cols, imgGray.rows), CV_8U, 3); 
	imgPrevGray = Mat(Size(imgGray.cols, imgGray.rows), CV_8U, 1);


	flags = 0;

	if (eig != NULL) {
		eig->release();
		temp->release();
		maskimg->release();
	}

	eig = new Mat(Size(imgGray.cols, imgGray.rows), CV_8U, 1);
	temp = new Mat(Size(imgGray.cols, imgGray.rows), CV_8U, 1);
	maskimg = new Mat(Size(imgGray.cols, imgGray.rows), CV_8U, 1);

	// Gen mask
	BYTE *pMask = (BYTE *) maskimg->data;
	int widthStep = maskimg->step;
	for (int j = 0; j < nj; ++j) {
		for (int i = 0; i < ni; ++i) {
			pMask[i + j * widthStep] = (i >= ni / 5) && (i <= ni * 4 / 5) && (j >= nj / 5) && (j <= nj * 4 / 5) ? (BYTE) 255 : (BYTE) 255;
		}
	}

	// Init homography
	homoMat = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
}

void KLTWrapper::InitFeatures(Mat imgGray)
{
	/* automatic initialization */
	double quality = 0.01;
	double min_distance = 10;

	int ni = imgGray.cols;
	int nj = imgGray.rows;


	if (pointsPrev.empty())
	{
	//pointsPrev.clear();
		for (int i = 0; i < ni / GRID_SIZE_W - 1; ++i) {
			for (int j = 0; j < nj / GRID_SIZE_H - 1; ++j) {
				pointsPrev.push_back(Vec2f(i * GRID_SIZE_W + GRID_SIZE_W / 2, j * GRID_SIZE_H + GRID_SIZE_H / 2));
				//pointsCurrent.push_back(Vec2f(i * GRID_SIZE_W + GRID_SIZE_W / 2, j * GRID_SIZE_H + GRID_SIZE_H / 2));
			}
		}
	}

	SwapData(imgGray);
}

void KLTWrapper::RunTrack(Mat imgGray, Mat prevGray)
{
	int i, k = 0;
	int* nMatch = (int*)alloca(sizeof(int) * MAX_COUNT);

	if (!prevGray.empty()) {
		//imgPrevGray->copyTo(*prevGray);
		prevGray = imgPrevGray;
	} else {
		flags = 0;
	}

	//memset(image->data, 0, image->rows * image->cols);
	if (pointsPrev.size()) {

		//goodFeaturesToTrack(*prevGray, points0, 100, 0.3, 7, Mat(), 7, false, 0.04);

		//std::vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 20, 0.03);
		calcOpticalFlowPyrLK(prevGray, imgGray, pointsPrev, pointsCurrent, status, noArray(), Size(15, 15), 2, criteria, flags);
		//calcOpticalFlowPyrLK(*prevGray, *imgGray, points[0], points[1], status, noArray(), Size(15, 15), 2, criteria, flags);

		//flags |= 1;
		for (i = k = 0; i < status.size(); i++) {
			if (!status[i]) {
				continue;
			}

			nMatch[k++] = i;
		}
		count = k;
	}

	if (k >= 10) {
		// Make homography matrix with correspondences
		MakeHomoGraphy(nMatch, count);
		//homoMat = findHomography(points0, points1, RANSAC, 1);
	} else {
		homoMat = (Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	}

	//Mat result;
	//warpPerspective(*imgGray, result, homoMat, Size(imgGray->cols, imgGray->rows));
	//imshow("warped", result);
	
	InitFeatures(imgGray);
}

void KLTWrapper::SwapData(Mat imgGray)
{
	imgGray.copyTo(imgPrevGray);
	//cv::swap(*prev_pyramid, *pyramid);
	//std::swap(pointsCurrent, pointsPrev);
}

void KLTWrapper::GetHomography(Mat& pmatH)
{
	pmatH = homoMat;
}

void KLTWrapper::MakeHomoGraphy(int *pnMatch, int nCnt)
{
	vector<Point2f> pt1;
	vector<Point2f> pt2;
	for (int i = 0; i < nCnt; ++i)
	{
		pt1.push_back(pointsCurrent[pnMatch[i]]);
		pt2.push_back(pointsPrev[pnMatch[i]]);

	}
	homoMat = findHomography(pt1, pt2, RANSAC, 1);
}
