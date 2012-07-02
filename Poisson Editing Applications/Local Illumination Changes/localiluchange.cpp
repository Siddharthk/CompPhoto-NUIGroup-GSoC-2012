#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "poisson.h"

using namespace std;
using namespace cv;

IplImage* img0, * img1, * img2, * subimg, *result;
CvPoint point;
int drag = 0;
int destx, desty;

int flag=0;

void mouseHandler(int event, int x, int y, int flags, void* param)
{


	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		point = cvPoint(x, y);
		destx = point.x;
		desty = point.y;
		drag  = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE && drag)
	{
		if(flag == 1)
		{
			cvShowImage("Source", img0);
		}
		else
		{

			img1 = cvCloneImage(img0);

			cvRectangle(img1,point,cvPoint(x, y),CV_RGB(255, 0, 0),1, 8, 0);

			cvShowImage("Source", img1);
		}
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
		drag = 0;
		cvShowImage("Source", img0);

		result = poisson_blend(img0,subimg,desty,destx);
		cvSaveImage("Output.jpg",result);
		cvNamedWindow("Image cloned",1);
		cvShowImage("Image cloned", result);
		cvWaitKey(0);
		cvDestroyWindow("Image cloned");
	}
}

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		fprintf(stderr, "Usage: %s <source image>\n", argv[0]);
		return 1;
	}

	img0 = cvLoadImage(argv[1]);

	//////////// source image ///////////////////

	cvNamedWindow("Source", 1);
	cvSetMouseCallback("Source", mouseHandler, NULL);
	cvShowImage("Source", img0);

	cvWaitKey(0);
	cvDestroyWindow("Source");

	cvReleaseImage(&img0);
	cvReleaseImage(&img1);

	return 0;
}
