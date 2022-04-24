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

#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include "main.hpp"


int main(int argc, char *argv[])
{

	string infile_name = argv[1];

  /*
	for (size_t index = 0; index < argc; index++)
	{
		if (0 == strcmp("-v", argv[index])) {
			if ((++index) < argc)
				infile_name = argv[index];
		}
	}
  */


	/************************************************************************/
	/*  Initialize Variables                                                */
	/************************************************************************/

	// wrapper class for mcd
	MCDWrapper *mcdwrapper = new MCDWrapper();

	// OPEN CV VARIABLES
	std::string window_name = "OUTPUT";
	Mat frame, frame_gray, fg, frame_copy;
	Mat* edge;
	Mat hsv, hsv_mask;

	std::cout << "opening file: " << infile_name << std::endl;
	// Initialize capture
	VideoCapture *pInVideo = new VideoCapture(infile_name);

	// Output window to be displayed
	namedWindow(window_name, WINDOW_AUTOSIZE);

	// Reset capture position
	pInVideo->set(CAP_PROP_POS_FRAMES, 0);

	// Init frame number and exit condition
	int frame_num = 1;
  int fd, img_size;
	bool bRun = true;
  const char *myfifo = "fifo";
  mkfifo(myfifo, 0666);

	/************************************************************************/
	/*  The main process loop                                               */
	/************************************************************************/
	while (bRun == true && pInVideo->isOpened()) {	// the main loop

		double start = getTickCount();

		// Extract Frame (do decoding or other work)
		if (!pInVideo->read(frame))
			break;

		//resize(frame, frame, Size(960, 540));
		cvtColor(frame, frame_gray, COLOR_RGB2GRAY);

		if (frame_num == 1) {

			// Init the wrapper for first frame
			mcdwrapper->Init(frame_gray);

		} 
		else {

			// Run detection
			fg = mcdwrapper->Run(frame_gray);

		}
		
		double elapsed_time = ((double)getTickCount() - start) / getTickFrequency();
		//float fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		//float fps = 100.0 / elapsed_time;

		putText(frame, std::to_string(frame_num), Point2d(10, 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
		putText(frame, std::to_string(elapsed_time), Point2d(100, 20), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
		
		if (fg.data)
		{
			//cvtColor(fg, fg, COLOR_GRAY2RGB);
			Mat zeros = Mat(fg.rows, fg.cols, CV_8UC1, Scalar(0));
			vector<Mat> channels;
			channels.push_back(fg);
			channels.push_back(zeros);
			channels.push_back(zeros);

			Mat mask;
			merge(channels, mask);
			bitwise_or(frame, mask, frame);
		}	

    img_size = frame.total() * frame.elemSize();
    fd = open(myfifo, O_WRONLY); 
    write(fd, frame.data, img_size);
    close(fd);
		//imshow(window_name, frame);
		//waitKey(10);
		
		//KeyBoard Process
		//int k = waitKey(1);
    /*
		if ('q' == k)
		{
			break;
		}
    */
		++frame_num;

	}

	pInVideo->release();

	return 0;
}
