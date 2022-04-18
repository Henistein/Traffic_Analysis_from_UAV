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

#pragma once

#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/calib3d.hpp>

#define GRID_SIZE_W (32)
#define GRID_SIZE_H (24)

using namespace std;
using namespace cv;

typedef unsigned char BYTE;

class KLTWrapper {
 private:
	Mat* eig;
	Mat* temp;
	Mat* maskimg;

	// For LK
	Mat image;
	Mat imgPrevGray;
	int win_size;
	int MAX_COUNT;
	std::vector<Point2f> pointsPrev, pointsCurrent;
	//char *status;
	vector<uchar> status;
	int count;
	int flags;

	// For Homography Matrix
	Mat homoMat;
 private:
	void SwapData(Mat imgGray);
	void MakeHomoGraphy(int *pnMatch, int nCnt);

 public:
	 KLTWrapper(void);
	~KLTWrapper(void);

	void Init(Mat imgGray);
	void InitFeatures(Mat imgGray);
	void RunTrack(Mat imgGray, Mat prevGray);	// with MakeHomography
	void GetHomography(Mat& pmatH);
};
