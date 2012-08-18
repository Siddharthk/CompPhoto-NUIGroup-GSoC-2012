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

void checkfile(char *src, int flag)
{
	while(1)
	{
		if(flag == 1)
		{
			printf("Enter Joint Image: ");
		}
		else
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
	int num,type;

	int flag = 0;

	cout << endl;
	cout << " Domain aware edge filter" << endl;
	cout << "--------------------------" << endl;

	cout << "Options: " << endl;
	cout << endl;

	cout << "1) Edge Preserve Smoothing" << endl;
	cout << "   -> Using Normalized convolution Filter" << endl;
	cout << "   -> Using Recursive Filter" << endl;
	cout << "2) Detail Enhancement" << endl;
	cout << "3) Pencil sketch/Color Pencil Drawing" << endl;
	cout << "4) Stylization" << endl;
	cout << "5) Clip-Art Compression Artifact Removal" << endl;
	cout << "6) Depth of Field" << endl;
	cout << "7) Edge Enhancement" << endl;
	cout << endl;

	cout << "Press number 1-7 to choose from above techniques: ";

	cin >> num;

	if(num == 1)
	{
		cout << endl;
		cout << "Press 1 for Normalized Convolution Filter and 2 for Recursive Filter: ";

		cin >> type;

	}

	cout << endl;

	char src[50];

	checkfile(src,flag);
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

		if(type == 1)
		{
			NC_Filter obj;
			obj.Normalized_Conv_Filter(img, res, sigma_s, sigma_r);
		}
		else if(type == 2)
		{
			Recursive_Filter obj;
			obj.RecursiveFilter(img, res, sigma_s, sigma_r);
		}

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
		cin >> factor;

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
	else if(num == 3)
	{
		float shade_factor;

		shade_factor = .05;
		sigma_s = 60;
		sigma_r = .07;

		cout << endl;
		cout << "sigma_s(default val: 60): ";
		cin >> sigma_s;

		cout << "sigma_r(default val: .07): ";
		cin >> sigma_r;

		cout << "Shade_Factor(Range: .01 to .1): ";
		cin >> shade_factor;

		IplImage *img = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		cvConvertScale(I,img,1.0/255.0,0.0);

		Pencil_Filter obj;

		obj.Pencil_Sketch(img, sigma_s, sigma_r, shade_factor);


	}
	else if(num == 4)
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

		int h = img->height;
		int w = img->width;
		int channel = img->nChannels;

		Mat res = Mat(h,w,CV_32FC3);

		NC_Filter obj;
		obj.Normalized_Conv_Filter(img, res, sigma_s, sigma_r);

		vector <Mat> planes;
		split(res, planes);

		Mat magXR = Mat(h, w, CV_32FC1);
		Mat magYR = Mat(h, w, CV_32FC1);

		Mat magXG = Mat(h, w, CV_32FC1);
		Mat magYG = Mat(h, w, CV_32FC1);

		Mat magXB = Mat(h, w, CV_32FC1);
		Mat magYB = Mat(h, w, CV_32FC1);

		Sobel(planes[0], magXR, CV_32FC1, 1, 0, 3);
		Sobel(planes[0], magYR, CV_32FC1, 0, 1, 3);

		Sobel(planes[1], magXG, CV_32FC1, 1, 0, 3);
		Sobel(planes[1], magYG, CV_32FC1, 0, 1, 3);

		Sobel(planes[2], magXB, CV_32FC1, 1, 0, 3);
		Sobel(planes[2], magYB, CV_32FC1, 0, 1, 3);

		Mat magx = Mat(h,w,CV_32FC1);
		Mat magy = Mat(h,w,CV_32FC1);

		Mat mag1 = Mat(h,w,CV_32FC1);
		Mat mag2 = Mat(h,w,CV_32FC1);
		Mat mag3 = Mat(h,w,CV_32FC1);


		magnitude(magXR,magYR,mag1);
		magnitude(magXG,magYG,mag2);
		magnitude(magXB,magYB,mag3);

		Mat magnitude = Mat(h,w,CV_32FC1);

		for(int i =0;i < h;i++)
			for(int j=0;j<w;j++)
			{
				magnitude.at<float>(i,j) = mag1.at<float>(i,j) + mag2.at<float>(i,j) + mag3.at<float>(i,j);
			}


		for(int i =0;i < h;i++)
			for(int j=0;j<w;j++)
			{
				magnitude.at<float>(i,j) = 1.0 -  magnitude.at<float>(i,j);
			}


		Mat stylized = Mat(h,w,CV_32FC3);

		for(int i =0;i < h;i++)
			for(int j=0;j<w;j++)
				for(int c=0;c<channel;c++)
				{
					stylized.at<float>(i,j*channel + c) = res.at<float>(i,j*channel + c) * magnitude.at<float>(i,j);
				}

		IplImage *final  = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		IplImage *final8 = cvCreateImage(cvGetSize(I), 8, 3 );

		for(int i =0;i < h;i++)
			for(int j=0;j<w;j++)
				for(int c=0;c<channel;c++)
				{
					CV_IMAGE_ELEM(final,float,i,j*channel+c) = stylized.at<float>(i,j*channel+c);
				}


		cvConvertScale(final,final8,255.0,0.0);

		cvSaveImage("Stylized.jpg",final8);

		imshow("stylized",stylized);

	}
	else if(num == 5)
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


		NC_Filter obj;
		obj.Normalized_Conv_Filter(img, res, sigma_s, sigma_r);

		imshow("Filtered Image",res);

		convertScaleAbs(res, res, 255,0);

		imwrite("result.jpg",res);

	}
	else if(num == 6)
	{

		flag = 1;
		checkfile(src,flag);

		IplImage *I1 = cvLoadImage(src);

		IplImage *img = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		cvConvertScale(I,img,1.0/255.0,0.0);

		IplImage *img1 = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		cvConvertScale(I1,img1,1.0/255.0,0.0);

		float sigma_s = 20;
		float sigma_r = .2;

		cout << "sigma_s(default val: 20): ";
		cin >> sigma_s;

		cout << "sigma_r(default val: .2): ";
		cin >> sigma_r;

		DOF_Filter obj;
		obj.Depth_of_field(img, sigma_s, sigma_r, img1);

	}
	else if(num == 7)
	{

		IplImage *img  = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
		cvConvertScale(I,img,1.0/255.0,0.0);

		Mat orig(img);

		float sigma_s = 60;
		float sigma_r = .45;

		cout << endl;
		cout << "sigma_s(default val: 60): ";
		cin >> sigma_s;

		cout << "sigma_r(default val: .3): ";
		cin >> sigma_r;


		int h = img->height;
		int w = img->width;
		int channel = img->nChannels;

		Mat res = Mat(h,w,CV_32FC3);
		Mat magnitude = Mat(h,w,CV_32FC1);

		Mat mag8 = Mat(h,w,CV_32FC1);

		Edge_Enhance obj;
		
		obj.find_magnitude(orig,magnitude);

		convertScaleAbs(magnitude, mag8, 255.0,0);

		imwrite("Before.jpg",mag8);

		imshow("Original Edge",magnitude);


		obj.Normalized_Conv_Filter(img, res, sigma_s, sigma_r);

		obj.find_magnitude(res,magnitude);

		convertScaleAbs(magnitude, mag8, 255.0,0);

		imwrite("After.jpg",mag8);

		imshow("Edge enhance",magnitude);


	}


	cvReleaseImage(&I);
	cvWaitKey(0);

	return 0;

}
