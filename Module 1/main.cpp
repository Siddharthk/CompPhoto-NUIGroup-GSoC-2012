/*
############################ Poisson Image Editing ##################################

Copyright (C) 2012 Siddharth Kherada
Copyright (C) 2006-2012 Natural User Interface Group

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
Lesser General Public License for more details.

#####################################################################################
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "poisson.h"

using namespace std;
using namespace cv;

IplImage *img0, *img1, *img2, *subimg, *result;
CvPoint point;
int drag = 0;
int flag = 0;
int destx, desty;
int num;
char src[50];
char dest[50];
float alpha,beta,red,green,blue;


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
		if(num == 1)
		{
			Normal_Blending obj;
			result = obj.normal_blend(img2,subimg,desty,destx);
		}
		if(num == 2)
		{
			Mixed_Blending obj;
			result = obj.mixed_blend(img2,subimg,desty,destx);
		}
		if(num == 5)
		{
			Mono_trans obj;
			result = obj.monochrome_transfer(img2,subimg,desty,destx);
		}

		////////// save blended result ////////////////////

		cvSaveImage("Output.jpg",result);
		cvSaveImage("cutpaste.jpg",im1);

		cvNamedWindow("Image cloned",1);
		cvShowImage("Image cloned", result);
		cvWaitKey(0);
		cvDestroyWindow("Image cloned");
	}
}
void mouseHandler2(int event, int x, int y, int flags, void* param)
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

		if(num == 3)
		{
			Local_color_change obj;
			result = obj.color_change(img0,subimg,desty,destx,red,green,blue);
		}
		if(num == 4)
		{
			Local_illum_change obj;
			result = obj.illum_change(img0,subimg,desty,destx,alpha,beta);
		}

		cvSaveImage("Output.jpg",result);
		cvNamedWindow("Image cloned",1);
		cvShowImage("Image cloned", result);
		cvWaitKey(0);
		cvDestroyWindow("Image cloned");
	}
}

void checkfile(char *file)
{
	while(1)
	{
		printf("Enter %s Image: ",file);
		if(!strcmp(file,"Source"))
			cin >> src;
		else if(!strcmp(file,"Destination"))
			cin >> dest;

		if(access( src, F_OK ) != -1 ) 
		{
			break;
		}
		else 
		{
			printf("Image doesn't exist\n");
		}
	}
}

int main(int argc, char** argv)
{

	cout << " Poisson Image Editing" << endl;
	cout << "-----------------------" << endl;
	cout << "Options: " << endl;
	cout << endl;
	cout << "1) Poisson Blending " << endl;
	cout << "2) Mixed Poisson Blending " << endl;
	cout << "3) Local Color Change " << endl;
	cout << "4) Local Illumination Change " << endl;
	cout << "5) Monochrome Transfer " << endl;
	cout << "6) Texture Flattening " << endl;

	cout << endl;

	cout << "Press number 1-6 to choose from above techniques: ";
	cin >> num;
	cout << endl;

	char s[]="Source";
	char d[]="Destination";

	if(num == 1)
	{

		checkfile(s);
		checkfile(d);

		img0 = cvLoadImage(src);

		img2 = cvLoadImage(dest);
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
	}
	else if(num == 2)
	{
		checkfile(s);
		checkfile(d);

		img0 = cvLoadImage(src);

		img2 = cvLoadImage(dest);
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
	}
	else if(num == 3)
	{
		checkfile(s);

		cout << "Enter RGB values: " << endl;
		cout << "Red: ";
		cin >> red;
		
		cout << "Green: ";
		cin >> green;
		
		cout << "Blue: ";
		cin >> blue;

		img0 = cvLoadImage(src);

		//////////// source image ///////////////////

		cvNamedWindow("Source", 1);
		cvSetMouseCallback("Source", mouseHandler2, NULL);
		cvShowImage("Source", img0);

		cvWaitKey(0);
		cvDestroyWindow("Source");

		cvReleaseImage(&img0);
	}
	else if(num == 4)
	{
		checkfile(s);

		cout << "alpha: ";
		cin >> alpha;

		cout << "beta: ";
		cin >> beta;

		img0 = cvLoadImage(src);

		//////////// source image ///////////////////

		cvNamedWindow("Source", 1);
		cvSetMouseCallback("Source", mouseHandler2, NULL);
		cvShowImage("Source", img0);

		cvWaitKey(0);
		cvDestroyWindow("Source");

		cvReleaseImage(&img0);
	}
	else if(num == 5)
	{
		checkfile(s);
		checkfile(d);

		img0 = cvLoadImage(src);

		img2 = cvLoadImage(dest);
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
	}
	else if(num == 6)
	{
		checkfile(s);

		img0 = cvLoadImage(src);
	
		Texture_flat obj;
		result = obj.tex_flattening(img0);

		cvSaveImage("Output.jpg",result);
		cvNamedWindow("Image cloned",1);
		cvShowImage("Image cloned", result);
		cvWaitKey(0);
		cvDestroyWindow("Image cloned");

		cvWaitKey(0);
		cvDestroyWindow("Source");

		cvReleaseImage(&img0);
	}

	return 0;
}
