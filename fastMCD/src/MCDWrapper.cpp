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

#ifndef	_MCDWRAPPER_CPP_
#define	_MCDWRAPPER_CPP_

#include <time.h>
#include <winsock.h>

#include <ctime>
#include <cstring>
#include "MCDWrapper.hpp"
#include "params.hpp"



#if defined _WIN32 || defined _WIN64

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

struct timezone
{
	int  tz_minuteswest; /* minutes W of Greenwich */
	int  tz_dsttime;     /* type of dst correction */
};

int gettimeofday(struct timeval* tv, struct timezone* tz)
{
	FILETIME ft;
	unsigned __int64 tmpres = 0;
	static int tzflag;

	if (NULL != tv)
	{
		GetSystemTimeAsFileTime(&ft);

		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;

		/*converting file time to unix epoch*/
		tmpres /= 10;  /*convert into microseconds*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS;
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	}

	if (NULL != tz)
	{
		if (!tzflag)
		{
			_tzset();
			tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	}

	return 0;
}

#else
#include <sys/time.h>
#endif

MCDWrapper::MCDWrapper()
{
}

MCDWrapper::~MCDWrapper()
{
}

void
 MCDWrapper::Init(Mat frame_gray)
{

	frm_cnt = 0;

	//// Allocate
	imgGray = Mat(Size(frame_gray.cols, frame_gray.rows), CV_8U, 1);
	imgGrayPrev = Mat(Size(frame_gray.cols, frame_gray.rows), CV_8U, 1);
	//imgGaussLarge = new Mat(Size(in_imgIpl->cols, in_imgIpl->rows), CV_8U, 1);
	//imgGaussSmall = new Mat(Size(in_imgIpl->cols, in_imgIpl->rows), CV_8U, 1);
	//imgDOG = new Mat(Size(in_imgIpl->cols, in_imgIpl->rows), CV_8U, 1);

	//detect_img = new Mat(Size(in_imgIpl->cols, in_imgIpl->rows), CV_8U, 1);
	fgMask = Mat(Size(frame_gray.cols, frame_gray.rows), CV_8UC1, Scalar(0));

	//TODO directly retrieve imgIpl (change to Mat later)

	// Smoothing using median filter
	medianBlur(frame_gray, imgGray, 5);

	m_LucasKanade.Init(imgGray);
	BGModel.init(&imgGray);

	imgGray.copyTo(imgGrayPrev);
}

Mat MCDWrapper::Run(Mat frame_gray)
{

	frm_cnt++;

	timeval tic, toc, tic_total, toc_total;
	float rt_preProc;	// pre Processing time
	float rt_motionComp;	// motion Compensation time
	float rt_modelUpdate;	// model update time
	float rt_total;		// Background Subtraction time

	//--TIME START
	gettimeofday(&tic, NULL);
	// Smmothign using median filter
	medianBlur(frame_gray, imgGray, 5);

	//--TIME END
	gettimeofday(&toc, NULL);
	rt_preProc = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//--TIME START
	gettimeofday(&tic, NULL);
	// Calculate Backward homography
	// Get H
	Mat homoMat;
	m_LucasKanade.RunTrack(imgGray, imgGrayPrev);
	m_LucasKanade.GetHomography(homoMat);


	double h[9];
	h[0] = homoMat.at<double>(0, 0);
	h[1] = homoMat.at<double>(0, 1);
	h[2] = homoMat.at<double>(0, 2);
	h[3] = homoMat.at<double>(1, 0);
	h[4] = homoMat.at<double>(1, 1);
	h[5] = homoMat.at<double>(1, 2);
	h[6] = homoMat.at<double>(2, 0);
	h[7] = homoMat.at<double>(2, 1);
	h[8] = homoMat.at<double>(2, 2);

	BGModel.motionCompensate(h);

	//--TIME END
	gettimeofday(&toc, NULL);
	rt_motionComp = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	//--TIME START
	gettimeofday(&tic, NULL);
	// Update BG Model and Detect
	//BGModel.update(detect_img);
	fgMask.setTo(Scalar(0));
	BGModel.update(imgGray.data, imgGray.step, fgMask.data, fgMask.step);

	//imshow("mask", fgMask);

	//--TIME END
	gettimeofday(&toc, NULL);
	rt_modelUpdate = (float)(toc.tv_usec - tic.tv_usec) / 1000.0;

	rt_total = rt_preProc + rt_motionComp + rt_modelUpdate;

	// Debug display of individual maps
	// cv::Mat mean = cv::Mat(BGModel.modelHeight, BGModel.modelWidth, CV_32F, BGModel.m_Mean[0]);
	// cv::imshow("mean",mean/255.0);
	// cv::Mat var = cv::Mat(BGModel.modelHeight, BGModel.modelWidth, CV_32F, BGModel.m_Var[0]);
	// cv::imshow("var",var/255.0);
	// cv::Mat age = cv::Mat(BGModel.modelHeight, BGModel.modelWidth, CV_32F, BGModel.m_Age[0]);
	// cv::imshow("age",age/255.0);

	//////////////////////////////////////////////////////////////////////////
	// Debug Output
	//for (int i = 0; i < 100; ++i) {
	//	printf("\b");
	//}
	//printf("PP: %.2f(ms)\tOF: %.2f(ms)\tBGM: %.2f(ms)\tTotal time: \t%.2f(ms)", MAX(0.0, rt_preProc), MAX(0.0, rt_motionComp), MAX(0.0, rt_modelUpdate), MAX(0.0, rt_total));

	// Uncomment this block if you want to save runtime to txt
	// if(rt_preProc >= 0 && rt_motionComp >= 0 && rt_modelUpdate >= 0 && rt_total >= 0){
	//      FILE* fileRunTime = fopen("runtime.txt", "a");
	//      fprintf(fileRunTime, "%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n", rt_preProc, rt_motionComp, rt_modelUpdate, 0.0, rt_total);
	//      fclose(fileRunTime);
	// }

	imgGray.copyTo(imgGrayPrev);
	waitKey(10);

	//imshow("imgGray", *imgGray);
	//imshow("imgGrayPrev", *imgGrayPrev);
	return fgMask;
}

#endif				// _MCDWRAPPER_CPP_
