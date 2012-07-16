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


void Domain_Filter::display(const char* name, IplImage *img)
{
	cvNamedWindow(name);
	cvShowImage(name,img);
}

void Domain_Filter::diffx(const IplImage *img, IplImage *temp)
{
	int h = img->height;
	int w = img->width;
	int channel = img->nChannels;

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w-1; j++)
		{
			for(int c =0; c < channel; c++)
			{
				CV_IMAGE_ELEM(temp,float,i,j*channel+c) = 
					CV_IMAGE_ELEM(img,float,i,(j+1)*channel+c) - CV_IMAGE_ELEM(img,float,i,j*channel+c);
			}
		}
}

void Domain_Filter::diffy(const IplImage *img, IplImage *temp)
{
	int h = img->height;
	int w = img->width;
	int channel = img->nChannels;

	for(int i = 0; i < h-1; i++)
		for(int j = 0; j < w; j++)
		{
			for(int c =0; c < channel; c++)
			{
				CV_IMAGE_ELEM(temp,float,i,j*channel+c) = 
					CV_IMAGE_ELEM(img,float,(i+1),j*channel+c) - CV_IMAGE_ELEM(img,float,i,j*channel+c);
			}
		}
}

void Recursive_Filter::compute(Mat &O, Mat &horiz, float sigma_h)
{

	float a;

	int h = O.rows;
	int w = O.cols;
	int channel = O.channels();

	a = exp(-sqrt(2) / sigma_h);

	Mat temp = Mat(h,w,CV_32FC3);

	for(int i =0; i < h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				temp.at<float>(i,j*channel+c) = O.at<float>(i,j*channel+c);


	Mat V = Mat(h,w,CV_32FC1);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			V.at<float>(i,j) = pow(a,horiz.at<float>(i,j));


	for(int i=0; i<h; i++)
	{
		for(int j =1; j < w; j++)
		{
			for(int c = 0; c<channel; c++)
			{
				temp.at<float>(i,j*channel+c) = temp.at<float>(i,j*channel+c) + 
					(temp.at<float>(i,(j-1)*channel+c) - temp.at<float>(i,j*channel+c)) * V.at<float>(i,j);
			}
		}
	}
				
	for(int i=0; i<h; i++)
	{
		for(int j =w-2; j >= 0; j--)
		{
			for(int c = 0; c<channel; c++)
			{
				temp.at<float>(i,j*channel+c) = temp.at<float>(i,j*channel+c) +
					(temp.at<float>(i,(j+1)*channel+c) - temp.at<float>(i,j*channel+c))*V.at<float>(i,j+1);
			}
		}
	}


	for(int i =0; i < h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				O.at<float>(i,j*channel+c) = temp.at<float>(i,j*channel+c);


}

void Recursive_Filter::RecursiveFilter(const IplImage *img, float sigma_s, float sigma_r)
{

	int iterations = 3;
	int h = img->height;
	int w = img->width;
	int channel = img->nChannels;

	////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////


	IplImage *derivx = cvCreateImage(cvSize(w-1,h), IPL_DEPTH_32F, channel );
	IplImage *derivy = cvCreateImage(cvSize(w,h-1), IPL_DEPTH_32F, channel );

	cvZero(derivx);
	cvZero(derivy);

	diffx(img,derivx);
	diffy(img,derivy);

	IplImage *distx = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1 );
	IplImage *disty = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1 );

	cvZero(distx);
	cvZero(disty);

	//////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////


	int k;
	for(int i = 0; i < h; i++)
		for(int j = 0,k=1; j < w-1; j++,k++)
			for(int c = 0; c < channel; c++)
			{
				CV_IMAGE_ELEM(distx,float,i,k) = 
					CV_IMAGE_ELEM(distx,float,i,k) + abs(CV_IMAGE_ELEM(derivx,float,i,j*channel+c));
			}

	for(int i = 0,k=1; i < h-1; i++,k++)
		for(int j = 0; j < w; j++)
			for(int c = 0; c < channel; c++)
			{
				CV_IMAGE_ELEM(disty,float,k,j) = 
					CV_IMAGE_ELEM(disty,float,k,j) + abs(CV_IMAGE_ELEM(derivy,float,i,j*channel+c));
			}

	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

	Mat horiz = Mat(h,w,CV_32FC1);
	Mat vert = Mat(h,w,CV_32FC1);


	Mat final = Mat(h,w,CV_32FC3);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
				horiz.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(distx,float,i,j);
				vert.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(disty,float,i,j);
		}

	Mat vert_t = vert.t();  

	float sigma_h = sigma_s;

	Mat O;
	Mat O_t;

	if(channel == 3)
	{
		O   = Mat(h,w,CV_32FC3);
		O_t = Mat(w,h,CV_32FC3);
	}
	else if (channel == 1)
	{
		O   = Mat(h,w,CV_32FC1);
		O_t = Mat(w,h,CV_32FC1);
	}


	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				O.at<float>(i,j*channel+c) =  CV_IMAGE_ELEM(img,float,i,j*channel+c);


	for(int i=0;i<iterations;i++)
	{
		sigma_h = sigma_s * sqrt(3) * pow(2,(iterations - (i+1))) / sqrt(pow(4,iterations) -1);

		compute(O, horiz, sigma_h);

		O_t = O.t();

		compute(O_t, vert_t, sigma_h);
	
		O = O_t.t();

	}


	imshow("Filtered Image",O);

	convertScaleAbs(O, final, 255,0);

	imwrite("result.jpg",final);

}

