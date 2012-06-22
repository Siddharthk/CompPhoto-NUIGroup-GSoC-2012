/*
#########################  Mixed Poisson Blending ############################

Copyright (C) 2012 Siddharth Kherada
Copyright (C) 2006-2012 Natural User Interface Group

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details. "

##############################################################################
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "mixedpoisson.h"

using namespace std;
using namespace cv;

IplImage* img0, * img1, * img2, * subimg, *result;
CvPoint point;
int drag = 0;
int destx, desty;

void drawImage(IplImage* target, IplImage* source, int x, int y) 
{
	for (int ix=0; ix<source->width; ix++) 
	{
		for (int iy=0; iy<source->height; iy++) 
		{
			int r = cvGet2D(source, iy, ix).val[2];
			int g = cvGet2D(source, iy, ix).val[1];
			int b = cvGet2D(source, iy, ix).val[0];
			CvScalar bgr = cvScalar(b, g, r);
			cvSet2D(target, iy+y, ix+x, bgr);
		}
	}

}
void mouseHandler(int event, int x, int y, int flags, void* param)
{


	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		point = cvPoint(x, y);
		drag  = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE && drag)
	{
		img1 = cvCloneImage(img0);

		cvRectangle(img1,point,cvPoint(x, y),CV_RGB(255, 0, 0),1, 8, 0);

		cvShowImage("Source", img1);
	}

	if (event == CV_EVENT_LBUTTONUP && drag)
	{
		img1 = cvCloneImage(img0);

		cvSetImageROI(img1,cvRect(point.x,point.y,x - point.x,y - point.y));

		subimg = cvCreateImage(cvGetSize(img1), img1->depth, img1->nChannels);	

		cvCopy(img1, subimg, NULL);

		cvNamedWindow("ROI",1);
		cvShowImage("ROI", subimg);
		cvWaitKey(0);
		cvDestroyWindow("ROI");
		cvResetImageROI(img1);
		cvShowImage("Source", img1);
		drag = 0;
	}

	if (event == CV_EVENT_RBUTTONUP)
	{
		cvShowImage("Source", img0);
		drag = 0;
	}
}


void mouseHandler1(int event, int x, int y, int flags, void* param)
{
	IplImage *im, *im1;

	im1 = cvCloneImage(img2);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		point = cvPoint(x, y);

		cvRectangle(im1,cvPoint(x, y),cvPoint(x+subimg->width,y+subimg->height),CV_RGB(255, 0, 0),1, 8, 0);

		destx = x;
		desty = y;

		cvShowImage("Destination", im1);
	}
	if (event == CV_EVENT_RBUTTONUP)
	{
		if(destx+subimg->width > img2->width || desty+subimg->height > img2->height)
		{
			cout << "Index out of range" << endl;
			exit(0);
		}

		drawImage(im1,subimg,destx,desty);
		result = poisson_blend(img2,subimg,desty,destx);

		////////// save blended result ////////////////////

		cvSaveImage("Output.jpg",result);
		cvSaveImage("cutpaste.jpg",im1);

		cvNamedWindow("Image cloned",1);
		cvShowImage("Image cloned", result);
		cvWaitKey(0);
		cvDestroyWindow("Image cloned");
	}
}

int main(int argc, char** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "Usage: %s <source image> <destination image>\n", argv[0]);
		return 1;
	}

	img0 = cvLoadImage(argv[1]);

	img2 = cvLoadImage(argv[2]);

	//////////// source image ///////////////////

	cvNamedWindow("Source", 1);
	cvSetMouseCallback("Source", mouseHandler, NULL);
	cvShowImage("Source", img0);

	/////////// destination image ///////////////

	cvNamedWindow("Destination", 1);
	cvSetMouseCallback("Destination", mouseHandler1, NULL);
	cvShowImage("Destination",img2);

	cvWaitKey(0);
	cvDestroyWindow("Source");
	cvDestroyWindow("Destination");

	cvReleaseImage(&img0);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);

	return 0;
}
