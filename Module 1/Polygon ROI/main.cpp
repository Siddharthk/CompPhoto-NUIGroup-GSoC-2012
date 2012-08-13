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
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "poisson.h"

using namespace std;
using namespace cv;

IplImage *img0, *img1, *img2, *result, *res, *res1, *final, *final1;

CvPoint point;
int drag = 0;
int destx, desty;

int numpts = 50;
CvPoint* pts = new CvPoint[50];
CvPoint* pts1 = new CvPoint[50];
CvPoint* pts2 = new CvPoint[50];

int s = 0;
int flag = 0;
int flag1 = 0;

int minx,miny,maxx,maxy,lenx,leny;
int minxd,minyd,maxxd,maxyd,lenxd,lenyd;

int channel,num;

char src[50];
char dest[50];

float alpha,beta;

float red, green, blue;

void mouseHandler(int event, int x, int y, int flags, void* param)
{

	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		if(flag1 == 0)
		{
			if(s==0)
				img1 = cvCloneImage(img0);
			point = cvPoint(x, y);
			cvCircle(img1,point,2,CV_RGB(255, 0, 0),-1, 8, 0);
			pts[s] = point;
			s++;
			drag  = 1;
			if(s>1)
				cvLine(img1,pts[s-2], point, cvScalar(0, 0, 255, 0), 2, CV_AA, 0);

			cvShowImage("Source", img1);
		}
	}


	if (event == CV_EVENT_LBUTTONUP && drag)
	{
		cvShowImage("Source", img1);
		drag = 0;
	}
	if (event == CV_EVENT_RBUTTONDOWN)
	{
		flag1 = 1;
		img1 = cvCloneImage(img0);
		for(int i = s; i < numpts ; i++)
			pts[i] = point;
		
		if(s!=0)
		{
			cvPolyLine( img1, &pts, &numpts,1, 1, CV_RGB(0,0,0), 2, CV_AA, 0);
		}

		for(int i=0;i<s;i++)
		{
			minx = min(minx,pts[i].x);
			maxx = max(maxx,pts[i].x);
			miny = min(miny,pts[i].y);
			maxy = max(maxy,pts[i].y);
		}
		lenx = maxx - minx;
		leny = maxy - miny;

		cvShowImage("Source", img1);
	}

	if (event == CV_EVENT_RBUTTONUP)
	{
		flag = s;

		cvFillPoly(res1, &pts, &numpts, 1, CV_RGB(255, 255, 255), CV_AA, 0);

		cvAnd(img0, img0, final,res1);

		cvNamedWindow("mask",1);
		cvShowImage("mask", final);
		cvSaveImage("mask.jpg",final);
		cvShowImage("Source", img1);
		if(num == 3)
		{
			Local_color_change obj;
			result = obj.color_change(img0,final,res1,red,green,blue);

			cvSaveImage("Output.jpg",result);
			cvNamedWindow("Blended Image",1);
			cvShowImage("Blended Image", result);
			cvWaitKey(0);
			cvDestroyWindow("Blended Image");
		}
		else if(num == 4)
		{
			Local_illum_change obj;
			result = obj.illum_change(img0,final,res1,alpha,beta);

			cvSaveImage("Output.jpg",result);
			cvNamedWindow("Blended Image",1);
			cvShowImage("Blended Image", result);
			cvWaitKey(0);
			cvDestroyWindow("Blended Image");

		}

	}
	if (event == CV_EVENT_MBUTTONDOWN)
	{
		for(int i = 0; i < numpts ; i++)
		{
			pts[i].x=0;
			pts[i].y=0;
		}
		s = 0;
		flag1 = 0;
		cvShowImage("Source", img0);
		drag = 0;
	}
}


void mouseHandler1(int event, int x, int y, int flags, void* param)
{


	IplImage *im1;
	
	im1 = cvCloneImage(img2);
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		if(flag1 == 1)
		{
			point = cvPoint(x, y);

			for(int i =0; i < numpts;i++)
				pts1[i] = pts[i];
		
			int tempx;
			int tempy;
			for(int i =0; i < flag; i++)
			{
				tempx = pts1[i+1].x - pts1[i].x;
				tempy = pts1[i+1].y - pts1[i].y;
				if(i==0)
				{
					pts2[i+1].x = point.x + tempx;
					pts2[i+1].y = point.y + tempy;
				}
				else if(i>0)
				{
					pts2[i+1].x = pts2[i].x + tempx;
					pts2[i+1].y = pts2[i].y + tempy;
				}

			}	
	
			for(int i=flag;i<numpts;i++)
				pts2[i] = pts2[flag-1]; 

			pts2[0] = point;

			cvPolyLine( im1, &pts2, &numpts,1, 1, CV_RGB(255,0,0), 2, CV_AA, 0);

			destx = x;
			desty = y;

			cvShowImage("Destination", im1);
		}
	}
	if (event == CV_EVENT_RBUTTONUP)
	{
		for(int i=0;i<flag;i++)
		{
			minxd = min(minxd,pts2[i].x);
			maxxd = max(maxxd,pts2[i].x);
			minyd = min(minyd,pts2[i].y);
			maxyd = max(maxyd,pts2[i].y);
		}

		if(maxxd > im1->width || maxyd > im1->height || minxd < 0 || minyd < 0)
		{
			cout << "Index out of range" << endl;
			exit(0);
		}

		int k,l;
		for(int i=miny, k=minyd;i<(miny+leny);i++,k++)
			for(int j=minx,l=minxd ;j<(minx+lenx);j++,l++)
			{
				for(int c=0;c<channel;c++)
				{
					CV_IMAGE_ELEM(final1,uchar,k,l*channel+c) = CV_IMAGE_ELEM(final,uchar,i,j*channel+c);
				}
			}
			

		cvFillPoly(res, &pts2, &numpts, 1, CV_RGB(255, 255, 255), CV_AA, 0);

		if(num == 1 || num == 2 || num == 5)
		{
			Normal_Blending obj;
			result = obj.normal_blend(img2,final1,res,num);
		}
	
		cvZero(res);
		for(int i = 0; i < flag ; i++)
		{
			pts2[i].x=0;
			pts2[i].y=0;
		}
		
		minxd = 100000; minyd = 100000; maxxd = -100000; maxyd = -100000;

		////////// save blended result ////////////////////

		cvSaveImage("Output.jpg",result);
		cvNamedWindow("Blended Image",1);
		cvShowImage("Blended Image", result);
		cvWaitKey(0);
		cvDestroyWindow("Blended Image");
	}
	
	cvReleaseImage(&im1);
}

void checkfile(char *file)
{
	while(1)
	{
		printf("Enter %s Image: ",file);
		if(!strcmp(file,"Source"))
		{
			cin >> src;
			if(access( src, F_OK ) != -1 )
			{
				break;
			}
			else
			{
				printf("Image doesn't exist\n");
			}
		}
		else if(!strcmp(file,"Destination"))
		{
			cin >> dest;

			if(access( dest, F_OK ) != -1 )
			{	
				break;
			}
			else
			{
				printf("Image doesn't exist\n");
			}

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

	minx = 100000; miny = 100000; maxx = -100000; maxy = -100000;

	minxd = 100000; minyd = 100000; maxxd = -100000; maxyd = -100000;


	if(num == 1 || num == 2 || num == 5)
	{

		checkfile(s);
		checkfile(d);

		img0 = cvLoadImage(src);

		img2 = cvLoadImage(dest);

		channel = img0->nChannels;

		res = cvCreateImage(cvGetSize(img2), 8, 1);
		res1 = cvCreateImage(cvGetSize(img0), 8, 1);
		final = cvCreateImage(cvGetSize(img0), 8, 3);
		final1 = cvCreateImage(cvGetSize(img2), 8, 3);
		cvZero(res1);
		cvZero(final);
		cvZero(final1);
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

		res1 = cvCreateImage(cvGetSize(img0), 8, 1);
		final = cvCreateImage(cvGetSize(img0), 8, 3);
		cvZero(res1);
		cvZero(final);

		//////////// source image ///////////////////

		cvNamedWindow("Source", 1);
		cvSetMouseCallback("Source", mouseHandler, NULL);
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

		res1 = cvCreateImage(cvGetSize(img0), 8, 1);
		final = cvCreateImage(cvGetSize(img0), 8, 3);
		cvZero(res1);
		cvZero(final);
		
		//////////// source image ///////////////////

		cvNamedWindow("Source", 1);
		cvSetMouseCallback("Source", mouseHandler, NULL);
		cvShowImage("Source", img0);

		cvWaitKey(0);
		cvDestroyWindow("Source");

		cvReleaseImage(&img0);
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
	
	cvReleaseImage(&res);
	cvReleaseImage(&res1);
	cvReleaseImage(&final);
	cvReleaseImage(&final1);
	cvReleaseImage(&result);

	return 0;
}
