/*
############################ Domain Transform Edge-Aware Filter ##################################

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

##################################################################################################
*/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "math.h"

#include "domain_filter.h"

using namespace std;
using namespace cv;

void checkfile(char *src)
{
	while(1)
	{
		printf("Enter Image: ");

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
}

int main(int argc, char* argv[])
{
	int num;

	cout << endl;
	cout << " Domain aware edge filter" << endl;
	cout << "--------------------------" << endl;

	cout << "Options: " << endl;
	cout << endl;

	cout << "1) Edge Preserve Smoothing" << endl;
	cout << "2) Detail Enhancement" << endl;
	cout << endl;

	cout << "Press number 1-6 to choose from above techniques: ";

	cin >> num;

	cout << endl;

	char src[50];

	checkfile(src);
	float sigma_s, sigma_r;


	IplImage *I = cvLoadImage(src);

	
	int h = I->height;
	int w = I->width;
	int channel = I->nChannels;
	
	
	Mat res = Mat(h,w,CV_32FC1);

	Domain_Filter obj1;
	obj1.display("Original Image",I);

	if(num == 1)
	{
		sigma_s = 60;
		sigma_r = .45; 

		cout << endl;
		cout << "sigma_s(default val: 60): ";
		cin >> sigma_s;

		cout << "sigma_r(default val: .45): ";
		cin >> sigma_r;

		IplImage *img = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		cvConvertScale(I,img,1.0/255.0,0.0);

		Recursive_Filter obj;
		obj.RecursiveFilter(img, res, sigma_s, sigma_r);

		imshow("Filtered Image",res);

		convertScaleAbs(res, res, 255,0);

		imwrite("result.jpg",res);

	}
	else if(num == 2)
	{
		float factor;

		factor = 3.0;
		sigma_s = 10;
		sigma_r = .15; 

		cout << endl;
		cout << "sigma_s(default val: 10): ";
		cin >> sigma_s;

		cout << "sigma_r(default val: .15): ";
		cin >> sigma_r;

		cout << "Scale Factor(default val: 3): ";
		cin >> factor;;

		IplImage *img = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		cvConvertScale(I,img,1.0/255.0,0.0);

		IplImage *result = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );

		IplImage *lab = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 3 );
		IplImage *l_channel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
		IplImage *a_channel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
		IplImage *b_channel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
		cvCvtColor(img,lab,CV_BGR2Lab);

		IplImage *L = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
		IplImage *final = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 3 );

		cvCvtPixToPlane(lab, l_channel, a_channel, b_channel,0);

		cvConvertScale(l_channel,L,1.0/255.0,0.0);


		Recursive_Filter obj;
		
		obj.RecursiveFilter(L, res, sigma_s, sigma_r);

		Mat detail = Mat(h,w,CV_32FC1);

		for(int i = 0; i < h; i++)
			for(int j = 0; j < w; j++)
				detail.at<float>(i,j) = CV_IMAGE_ELEM(L,float,i,j) - res.at<float>(i,j);



		for(int i = 0; i < h; i++)
			for(int j = 0; j < w; j++)
				CV_IMAGE_ELEM(L,float,i,j) = res.at<float>(i,j) + factor*detail.at<float>(i,j);

		cvConvertScale(L,l_channel,255,0);

		for(int i = 0; i < h; i++)
			for(int j = 0; j < w; j++)
			{
				CV_IMAGE_ELEM(lab,float,i,j*channel+0) = CV_IMAGE_ELEM(l_channel,float,i,j);
				CV_IMAGE_ELEM(lab,float,i,j*channel+1) = CV_IMAGE_ELEM(a_channel,float,i,j);
				CV_IMAGE_ELEM(lab,float,i,j*channel+2) = CV_IMAGE_ELEM(b_channel,float,i,j);
			}

		cvCvtColor(lab,result,CV_Lab2BGR);


		Domain_Filter obj2;

		obj2.display("Detail enhance",result);
		cvConvertScale(result,final,255,0);

		cvSaveImage("Output.jpg",final);
	}


	cvReleaseImage(&I);
	cvWaitKey(0);

	return 0;

}
