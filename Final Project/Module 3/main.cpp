/*
############################ Color2Gray Filter #####################################

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
#include "math.h"
#include <vector>
#include <limits>

#include "contrast_preserve.h"

using namespace std;
using namespace cv;

void checkfile(char *src)
{
	while(1)
	{
		cout << endl;
		printf("Enter Color Image: ");

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


int main(int argc,char* argv[])
{
	cout << endl;
	cout << " Color2Gray Filter" << endl;
	cout << "-------------------" << endl;

	char src[50];
	
	checkfile(src);

	IplImage *I  = cvLoadImage(src);

	float sigma = .02;
	int maxIter = 10;
	int iterCount = 0;
	
	if(I->nChannels !=3)
	{
		printf("Input RGB Image\n");
		exit(0);
	}

	Decolor obj;

	obj.display("Original Image",I);

	int h1 = I->height;
	int w1 = I->width;

	IplImage *dst = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 ); 
	IplImage *final1 = cvCreateImage(cvGetSize(I), 8, 3 ); 

	double sizefactor;

	IplImage *img;
	
	if((h1+w1) > 900)
	{
		sizefactor = (double)900/(h1+w1);
		IplImage *Im = cvCreateImage(cvSize(obj.rounding(w1*sizefactor) , obj.rounding(h1*sizefactor)),I->depth, I->nChannels);
		cvResize(I, Im);
		img = cvCreateImage(cvGetSize(Im), IPL_DEPTH_32F, 3 ); 
		cvConvertScale(Im,img,1.0/255.0,0.0);
		cvReleaseImage(&Im);
	}
	else
	{
		img = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 ); 
		cvConvertScale(I,img,1.0/255.0,0.0);
	}

	int h = img->height;
	int w = img->width;
	int channel = img->nChannels;

	IplImage *orig_gray = cvCreateImage(cvGetSize(I), 8, 1 );

	cvCvtColor(I,orig_gray,CV_BGR2GRAY);
	obj.display("OpenCV Gray Image",orig_gray);

	cvSaveImage("orig_gray.jpg",orig_gray);
	
	obj.init();

	vector <double> Cg;
	vector < vector <double> > polyGrad;
	vector < vector <double> > bc;
	vector < vector < int > > comb;

	vector <double> alf;

	obj.grad_system(img,polyGrad,Cg,comb);
	obj.weak_order(img,alf);

	Mat Mt = Mat(polyGrad.size(),polyGrad[0].size(), CV_32FC1);
	obj.wei_update_matrix(polyGrad,Cg,Mt);

	vector <double> wei;
	obj.wei_inti(comb,wei);


	//////////////////////////////// main loop starting ////////////////////////////////////////


	while (iterCount < maxIter)
	{
		iterCount +=1;

		vector <double> G_pos;
		vector <double> G_neg;

		vector <double> temp;
		vector <double> temp1;

		double val = 0.0;
		for(int i=0;i< polyGrad[0].size();i++)
		{
			val = 0.0;
			for(int j =0;j<polyGrad.size();j++)
				val = val + (polyGrad[j][i] * wei[j]);
			temp.push_back(val - Cg[i]);
			temp1.push_back(val + Cg[i]);
		}

		double ans = 0.0;
		double ans1 = 0.0;
		for(int i =0;i<alf.size();i++)
		{
			ans = ((1 + alf[i])/2) * exp((-1.0 * 0.5 * pow(temp[i],2))/pow(sigma,2));
			ans1 =((1 - alf[i])/2) * exp((-1.0 * 0.5 * pow(temp1[i],2))/pow(sigma,2));
			G_pos.push_back(ans);
			G_neg.push_back(ans1);
		}

		vector <double> EXPsum;
		vector <double> EXPterm;

		for(int i = 0;i<G_pos.size();i++)
			EXPsum.push_back(G_pos[i]+G_neg[i]);


		vector <double> temp2;

		for(int i=0;i<EXPsum.size();i++)
		{
			if(EXPsum[i] == 0)
				temp2.push_back(1.0);
			else
				temp2.push_back(0.0);
		}

		for(int i =0; i < G_pos.size();i++)
			EXPterm.push_back((G_pos[i] - G_neg[i])/(EXPsum[i] + temp2[i]));

		
		double val1 = 0.0;
		vector <double> wei1;

		for(int i=0;i< polyGrad.size();i++)
		{
			val1 = 0.0;
			for(int j =0;j<polyGrad[0].size();j++)
			{
				val1 = val1 + (Mt.at<float>(i,j) * EXPterm[j]);
			}
			wei1.push_back(val1);
		}

		for(int i =0;i<wei.size();i++)
			wei[i] = wei1[i];

		G_pos.clear();
		G_neg.clear();
		temp.clear();
		temp1.clear();
		EXPsum.clear();
		EXPterm.clear();
		temp2.clear();
		wei1.clear();
	}

	IplImage *Gray = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
	obj.grayImContruct(wei, img, Gray);

	IplImage* O = cvCreateImage(cvGetSize(img),8,1);
	cvConvertScale(Gray,O,255);
	cvSaveImage("Grayimage.jpg",O);

	obj.display("Contrast Preserved Gray Image",O);

	////////////////////////////////////////       Contrast Boosting   /////////////////////////////////////////////////////
	
	IplImage *lab = cvCreateImage(cvGetSize(img), 8, 3 );
	IplImage *contrast = cvCreateImage(cvGetSize(img), 8, 3 );
	IplImage *l_channel = cvCreateImage(cvGetSize(img), 8, 1 ); 
	IplImage *a_channel = cvCreateImage(cvGetSize(img), 8, 1 ); 
	IplImage *b_channel = cvCreateImage(cvGetSize(img), 8, 1 ); 
	
	cvCvtColor(I,lab,CV_BGR2Lab);

	cvCvtPixToPlane(lab, l_channel, a_channel, b_channel,0);

	for(int i =0;i<h1;i++)
		for(int j=0;j<w1;j++)
		{
			CV_IMAGE_ELEM(l_channel,uchar,i,j) = 255.0*CV_IMAGE_ELEM(Gray,float,i,j);
		}

	for(int i =0;i<h1;i++)
		for(int j=0;j<w1;j++)
		{
			CV_IMAGE_ELEM(lab,uchar,i,j*3+0) = CV_IMAGE_ELEM(l_channel,uchar,i,j);
			CV_IMAGE_ELEM(lab,uchar,i,j*3+1) = CV_IMAGE_ELEM(a_channel,uchar,i,j);
			CV_IMAGE_ELEM(lab,uchar,i,j*3+2) = CV_IMAGE_ELEM(b_channel,uchar,i,j);
		}

	cvCvtColor(lab,contrast,CV_Lab2BGR);

	obj.display("Color boost",contrast);

	cvSaveImage("Colorboost.jpg",contrast);
	
	cvReleaseImage(&I);
	cvReleaseImage(&final1);
	cvReleaseImage(&dst);
	cvReleaseImage(&lab);
	cvReleaseImage(&contrast);
	cvReleaseImage(&l_channel);
	cvReleaseImage(&a_channel);
	cvReleaseImage(&b_channel);
	cvReleaseImage(&O);
	cvReleaseImage(&Gray);

	Cg.clear();
	polyGrad.clear();
	bc.clear();
	comb.clear();
	alf.clear();
	wei.clear();

	cvWaitKey(0);
}


