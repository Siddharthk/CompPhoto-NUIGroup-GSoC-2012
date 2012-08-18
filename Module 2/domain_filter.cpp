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

double myinf = std::numeric_limits<double>::infinity();

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

void Edge_Enhance::getGradientx( const Mat &img, Mat &gx)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				gx.at<float>(i,j*channel+c) =
					img.at<float>(i,(j+1)*channel+c) - img.at<float>(i,j*channel+c);
			}
}
void Edge_Enhance::getGradienty( const Mat &img, Mat &gy)
{
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				gy.at<float>(i,j*channel+c) =
					img.at<float>(i+1,j*channel+c) - img.at<float>(i,j*channel+c);

			}
}

void Edge_Enhance::find_magnitude(Mat &img, Mat &mag)
{

	int h = img.rows;
	int w = img.cols;

	vector <Mat> planes;
	split(img, planes);

	Mat magXR = Mat(h, w, CV_32FC1);
	Mat magYR = Mat(h, w, CV_32FC1);

	Mat magXG = Mat(h, w, CV_32FC1);
	Mat magYG = Mat(h, w, CV_32FC1);

	Mat magXB = Mat(h, w, CV_32FC1);
	Mat magYB = Mat(h, w, CV_32FC1);

	getGradientx(planes[0], magXR);
	getGradienty(planes[0], magYR);

	getGradientx(planes[1], magXG);
	getGradienty(planes[1], magYG);

	getGradientx(planes[2], magXR);
	getGradienty(planes[2], magYR);

	Mat magx = Mat(h,w,CV_32FC1);
	Mat magy = Mat(h,w,CV_32FC1);

	Mat mag1 = Mat(h,w,CV_32FC1);
	Mat mag2 = Mat(h,w,CV_32FC1);
	Mat mag3 = Mat(h,w,CV_32FC1);

	magnitude(magXR,magYR,mag1);
	magnitude(magXG,magYG,mag2);
	magnitude(magXB,magYB,mag3);

	for(int i =0;i < h;i++)
		for(int j=0;j<w;j++)
		{
			mag.at<float>(i,j) = mag1.at<float>(i,j) + mag2.at<float>(i,j) + mag3.at<float>(i,j);
		}


	for(int i =0;i < h;i++)
		for(int j=0;j<w;j++)
		{
			mag.at<float>(i,j) = 1.0 -  mag.at<float>(i,j);
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

void Pencil_Filter::compute(Mat &O, Mat &horiz, Mat &pencil, float radius)
{

	int h = O.rows;
	int w = O.cols;
	int channel = O.channels();

	Mat lower_pos = Mat(h,w,CV_32FC1);
	Mat upper_pos = Mat(h,w,CV_32FC1);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			lower_pos.at<float>(i,j) = horiz.at<float>(i,j) - radius;
			upper_pos.at<float>(i,j) = horiz.at<float>(i,j) + radius;
		}

	Mat lower_idx = Mat::zeros(h,w,CV_32FC1);
	Mat upper_idx = Mat::zeros(h,w,CV_32FC1);

	Mat domain_row = Mat::zeros(1,w+1,CV_32FC1);

	for(int i=0;i<h;i++)
	{
		for(int j=0;j<w;j++)
			domain_row.at<float>(0,j) = horiz.at<float>(i,j);
		domain_row.at<float>(0,w) = myinf;

		Mat lower_pos_row = Mat::zeros(1,w,CV_32FC1);
		Mat upper_pos_row = Mat::zeros(1,w,CV_32FC1);

		for(int j=0;j<w;j++)
		{
			lower_pos_row.at<float>(0,j) = lower_pos.at<float>(i,j);
			upper_pos_row.at<float>(0,j) = upper_pos.at<float>(i,j);
		}
		Mat temp_lower_idx = Mat::zeros(1,w,CV_32FC1);
		Mat temp_upper_idx = Mat::zeros(1,w,CV_32FC1);

		for(int j=0;j<w;j++)
		{
			if(domain_row.at<float>(0,j) > lower_pos_row.at<float>(0,0))
			{
				temp_lower_idx.at<float>(0,0) = j;
				break;
			}
		}
		for(int j=0;j<w;j++)
		{
			if(domain_row.at<float>(0,j) > upper_pos_row.at<float>(0,0))
			{
				temp_upper_idx.at<float>(0,0) = j;
				break;
			}
		}

		int temp = 0;
		for(int j=1;j<w;j++)
		{
			int count=0;
			for(int k=temp_lower_idx.at<float>(0,j-1);k<w+1;k++)
			{
				if(domain_row.at<float>(0,k) > lower_pos_row.at<float>(0,j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_lower_idx.at<float>(0,j) = temp_lower_idx.at<float>(0,j-1) + temp;

			count = 0;
			for(int k=temp_upper_idx.at<float>(0,j-1);k<w+1;k++)
			{


				if(domain_row.at<float>(0,k) > upper_pos_row.at<float>(0,j))
				{
					temp = count;
					break;
				}
				count++;
			}

			temp_upper_idx.at<float>(0,j) = temp_upper_idx.at<float>(0,j-1) + temp;
		}

		for(int j=0;j<w;j++)
		{
			lower_idx.at<float>(i,j) = temp_lower_idx.at<float>(0,j) + 1;
			upper_idx.at<float>(i,j) = temp_upper_idx.at<float>(0,j) + 1;
		}


	}

	///////////////////////////////////////////////////// Pencil drawing ////////////////////////////////////////////////////

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			pencil.at<float>(i,j) = upper_idx.at<float>(i,j) - lower_idx.at<float>(i,j);
	/////////////////////////////////////////////////////   calculate box filter /////////////////////////////////////////////////


	Mat box_filter = Mat::zeros(h,w+1,CV_32FC3);


	for(int i = 0; i < h; i++)
	{
		box_filter.at<float>(i,1*channel+0) = O.at<float>(i,0*channel+0);
		box_filter.at<float>(i,1*channel+1) = O.at<float>(i,0*channel+1);
		box_filter.at<float>(i,1*channel+2) = O.at<float>(i,0*channel+2);
		for(int j = 2; j < w+1; j++)
		{
			for(int c=0;c<channel;c++)
				box_filter.at<float>(i,j*channel+c) = O.at<float>(i,(j-1)*channel+c) + box_filter.at<float>(i,(j-1)*channel+c);
		}
	}

	Mat indices = Mat::zeros(h,w,CV_32FC1);
	Mat final =   Mat::zeros(h,w,CV_32FC3);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			indices.at<float>(i,j) = i+1;

	Mat a = Mat::zeros(h,w,CV_32FC1);
	Mat b = Mat::zeros(h,w,CV_32FC1);

	for(int c=0;c<channel;c++)
	{
		Mat flag = Mat::ones(h,w,CV_32FC1);
		
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
				flag.at<float>(i,j) = (c+1)*flag.at<float>(i,j);
		
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
			{
				a.at<float>(i,j) = (flag.at<float>(i,j) - 1) * h * (w+1) + (lower_idx.at<float>(i,j) - 1) * h + indices.at<float>(i,j);
				b.at<float>(i,j) = (flag.at<float>(i,j) - 1) * h * (w+1) + (upper_idx.at<float>(i,j) - 1) * h + indices.at<float>(i,j);

			}

		int p,q,r,rem;
		int p1,q1,r1,rem1;

		for(int i=0;i<h;i++)
		{
			for(int j=0;j<w;j++)
			{

				r = b.at<float>(i,j)/(h*(w+1));
				rem = b.at<float>(i,j) - r*h*(w+1);
				q = rem/h;
				p = rem - q*h;

				r1 = a.at<float>(i,j)/(h*(w+1));
				rem1 = a.at<float>(i,j) - r1*h*(w+1);
				q1 = rem1/h;
				p1 = rem1 - q1*h;

				final.at<float>(i,j*channel+2-c) = (box_filter.at<float>(p-1,q*channel+(2-r)) - box_filter.at<float>(p1-1,q1*channel+(2-r1)))
					/(upper_idx.at<float>(i,j) - lower_idx.at<float>(i,j));
			}
		}

	}
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				O.at<float>(i,j*channel+c) = final.at<float>(i,j*channel+c);



}

void NC_Filter::compute(Mat &F, Mat &dHdx, float radius)
{

	int h = F.rows;
	int w = F.cols;
	int channel = F.channels();

	Mat lpos = Mat(h,w,CV_32FC1);
	Mat upos = Mat(h,w,CV_32FC1);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			lpos.at<float>(i,j) = dHdx.at<float>(i,j) - radius;
			upos.at<float>(i,j) = dHdx.at<float>(i,j) + radius;
		}

	Mat lidx = Mat::zeros(h,w,CV_32FC1);
	Mat uidx = Mat::zeros(h,w,CV_32FC1);

	Mat domain_row = Mat::zeros(1,w+1,CV_32FC1);

	for(int i=0;i<h;i++)
	{
		for(int j=0;j<w;j++)
			domain_row.at<float>(0,j) = dHdx.at<float>(i,j);
		domain_row.at<float>(0,w) = myinf;

		Mat lpos_row = Mat::zeros(1,w,CV_32FC1);
		Mat upos_row = Mat::zeros(1,w,CV_32FC1);

		for(int j=0;j<w;j++)
		{
			lpos_row.at<float>(0,j) = lpos.at<float>(i,j);
			upos_row.at<float>(0,j) = upos.at<float>(i,j);
		}

		Mat local_lidx = Mat::zeros(1,w,CV_32FC1);
		Mat local_uidx = Mat::zeros(1,w,CV_32FC1);

		for(int j=0;j<w;j++)
		{
			if(domain_row.at<float>(0,j) > lpos_row.at<float>(0,0))
			{
				local_lidx.at<float>(0,0) = j;
				break;
			}
		}
		for(int j=0;j<w;j++)
		{
			if(domain_row.at<float>(0,j) > upos_row.at<float>(0,0))
			{
				local_uidx.at<float>(0,0) = j;
				break;
			}
		}

		int temp = 0;
		for(int j=1;j<w;j++)
		{
			int count=0;
			for(int k=local_lidx.at<float>(0,j-1);k<w+1;k++)
			{
				if(domain_row.at<float>(0,k) > lpos_row.at<float>(0,j))
				{
					temp = count;
					break;
				}
				count++;
			}

			local_lidx.at<float>(0,j) = local_lidx.at<float>(0,j-1) + temp;

			count = 0;
			for(int k=local_uidx.at<float>(0,j-1);k<w+1;k++)
			{


				if(domain_row.at<float>(0,k) > upos_row.at<float>(0,j))
				{
					temp = count;
					break;
				}
				count++;
			}

			local_uidx.at<float>(0,j) = local_uidx.at<float>(0,j-1) + temp;
		}

		for(int j=0;j<w;j++)
		{
			lidx.at<float>(i,j) = local_lidx.at<float>(0,j) + 1;
			uidx.at<float>(i,j) = local_uidx.at<float>(0,j) + 1;
		}


	}

	/////////////////////////////////////////////////////   calculate box filter /////////////////////////////////////////////////


	Mat box_filter = Mat::zeros(h,w+1,CV_32FC3);

	for(int i = 0; i < h; i++)
	{
		box_filter.at<float>(i,1*channel+0) = F.at<float>(i,0*channel+0);
		box_filter.at<float>(i,1*channel+1) = F.at<float>(i,0*channel+1);
		box_filter.at<float>(i,1*channel+2) = F.at<float>(i,0*channel+2);
		for(int j = 2; j < w+1; j++)
		{
			for(int c=0;c<channel;c++)
				box_filter.at<float>(i,j*channel+c) = F.at<float>(i,(j-1)*channel+c) + box_filter.at<float>(i,(j-1)*channel+c);
		}
	}

	Mat indices = Mat::zeros(h,w,CV_32FC1);
	Mat final =   Mat::zeros(h,w,CV_32FC3);

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			indices.at<float>(i,j) = i+1;

	Mat a = Mat::zeros(h,w,CV_32FC1);
	Mat b = Mat::zeros(h,w,CV_32FC1);

	for(int c=0;c<channel;c++)
	{
		Mat flag = Mat::ones(h,w,CV_32FC1);
		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
				flag.at<float>(i,j) = (c+1)*flag.at<float>(i,j);

		for(int i=0;i<h;i++)
			for(int j=0;j<w;j++)
			{
				a.at<float>(i,j) = (flag.at<float>(i,j) - 1) * h * (w+1) + (lidx.at<float>(i,j) - 1) * h + indices.at<float>(i,j);
				b.at<float>(i,j) = (flag.at<float>(i,j) - 1) * h * (w+1) + (uidx.at<float>(i,j) - 1) * h + indices.at<float>(i,j);

			}

		int p,q,r,rem;
		int p1,q1,r1,rem1;

		for(int i=0;i<h;i++)
		{
			for(int j=0;j<w;j++)
			{

				r = b.at<float>(i,j)/(h*(w+1));
				rem = b.at<float>(i,j) - r*h*(w+1);
				q = rem/h;
				p = rem - q*h;
				if(q==0)
				{
					p=h;
					q=w;
					r=r-1;
				}
				if(p==0)
				{
					p=h;
					q=q-1;
				}
						

				r1 = a.at<float>(i,j)/(h*(w+1));
				rem1 = a.at<float>(i,j) - r1*h*(w+1);
				q1 = rem1/h;
				p1 = rem1 - q1*h;
				if(p1==0)
				{
					p1=h;
					q1=q1-1;
				}


				final.at<float>(i,j*channel+2-c) = (box_filter.at<float>(p-1,q*channel+(2-r)) - box_filter.at<float>(p1-1,q1*channel+(2-r1)))
					/(uidx.at<float>(i,j) - lidx.at<float>(i,j));
			}
		}
	}

	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
				F.at<float>(i,j*channel+c) = final.at<float>(i,j*channel+c);

}



void Recursive_Filter::RecursiveFilter(const IplImage *img, Mat &res, float sigma_s, float sigma_r)
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


	for(int  i=0;i<h;i++)
		for(int  j=0;j<w;j++)
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

	res = O.clone();

}

void Pencil_Filter::Pencil_Sketch(const IplImage *img, float sigma_s, float sigma_r, float shade_factor)
{

	int no_of_iter = 3;
	int h = img->height;
	int w = img->width;
	int channel = img->nChannels;

	////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////


	IplImage *derivx = cvCreateImage(cvSize(w-1,h), IPL_DEPTH_32F, 3 );
	IplImage *derivy = cvCreateImage(cvSize(w,h-1), IPL_DEPTH_32F, 3 );

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

	/////////////////////// convert to YCBCR model for color pencil drawing //////////////////////////////////////////////////////


	IplImage *color_sketch = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 3 );
	IplImage *color_res = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 3 );
	cvCvtColor(img,color_sketch,CV_BGR2YCrCb);

	IplImage *Y_channel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
	IplImage *U_channel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );
	IplImage *V_channel = cvCreateImage(cvGetSize(img), IPL_DEPTH_32F, 1 );



	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

	Mat horiz = Mat(h,w,CV_32FC1);
	Mat vert = Mat(h,w,CV_32FC1);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			horiz.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(distx,float,i,j);
			vert.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(disty,float,i,j);
		}


	//////////////////////////////////////////////// Integrate ////////////////////////////////////////////////////////////////////

	Mat ct_H = Mat(h,w,CV_32FC1);
	Mat ct_V = Mat(h,w,CV_32FC1);



	for(int i = 0; i < h; i++)
	{
		ct_H.at<float>(i,0) = horiz.at<float>(i,0);
		for(int j = 1; j < w; j++)
		{
			ct_H.at<float>(i,j) = horiz.at<float>(i,j) + ct_H.at<float>(i,j-1);
		}
	}

	for(int j = 0; j < w; j++)
	{
		ct_V.at<float>(0,j) = vert.at<float>(0,j);
		for(int i = 1; i < h; i++)
		{
			ct_V.at<float>(i,j) = vert.at<float>(i,j) + ct_V.at<float>(i-1,j);
		}
	}

	//// Transpose
	Mat vert_t = ct_V.t();

	float sigma_h = sigma_s;

	Mat O = Mat(h,w,CV_32FC3);
	Mat penx = Mat(h,w,CV_32FC1);

	Mat pen_res = Mat::zeros(h,w,CV_32FC1);
	Mat sketch = Mat(h,w,CV_32FC1);

	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				O.at<float>(i,j*channel+c) =  CV_IMAGE_ELEM(img,float,i,j*channel+c);

	Mat O_t = Mat(w,h,CV_32FC3);
	Mat peny = Mat(w,h,CV_32FC1);

	Mat peny_t;

	float radius;

	for(int i=0;i<no_of_iter;i++)
	{
		sigma_h = sigma_s * sqrt(3) * pow(2,(no_of_iter - (i+1))) / sqrt(pow(4,no_of_iter) -1);

		radius = sqrt(3) * sigma_h;

		compute(O, ct_H, penx, radius);

		O_t = O.t();

		compute(O_t, vert_t, peny, radius);

		O = O_t.t();

		peny_t = peny.t();

		for(int k=0;k<h;k++)
			for(int j=0;j<w;j++)
				pen_res.at<float>(k,j) = (shade_factor * (penx.at<float>(k,j) + peny_t.at<float>(k,j)));

		if(i==0)
		{
			imshow("pencil_sketch", pen_res);

			convertScaleAbs(pen_res, sketch, 255,0);

			imwrite("pencil_sketch.jpg",sketch);

			cvCvtPixToPlane(color_sketch, Y_channel, U_channel, V_channel,0);

			for(int k=0;k<h;k++)
				for(int j=0;j<w;j++)
					CV_IMAGE_ELEM(Y_channel,float,k,j) = pen_res.at<float>(k,j);

			cvMerge(Y_channel,U_channel,V_channel,0,color_sketch);
			cvCvtColor(color_sketch,color_res,CV_YCrCb2BGR);

			Domain_Filter obj;
			obj.display("color_pencil_sketch",color_res);

			cvConvertScale(color_res,color_res,255,0.0);

			cvSaveImage("color_pencil_sketch.jpg",color_res);
		}


	}

}
void NC_Filter::Normalized_Conv_Filter(const IplImage *img, Mat &res, float sigma_s, float sigma_r)
{

	int no_of_iter = 3;
	int h = img->height;
	int w = img->width;
	int channel = img->nChannels;

	////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////


	IplImage *dIcdx = cvCreateImage(cvSize(w-1,h), IPL_DEPTH_32F, 3 );
	IplImage *dIcdy = cvCreateImage(cvSize(w,h-1), IPL_DEPTH_32F, 3 );

	cvZero(dIcdx);
	cvZero(dIcdy);

	diffx(img,dIcdx);
	diffy(img,dIcdy);

	IplImage *dIdx = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1 );
	IplImage *dIdy = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1 );

	cvZero(dIdx);
	cvZero(dIdy);

	//////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////


	int k;
	for(int i = 0; i < h; i++)
		for(int j = 0,k=1; j < w-1; j++,k++)
			for(int c = 0; c < channel; c++)
			{
				CV_IMAGE_ELEM(dIdx,float,i,k) =
					CV_IMAGE_ELEM(dIdx,float,i,k) + abs(CV_IMAGE_ELEM(dIcdx,float,i,j*channel+c));
			}

	for(int i = 0,k=1; i < h-1; i++,k++)
		for(int j = 0; j < w; j++)
			for(int c = 0; c < channel; c++)
			{
				CV_IMAGE_ELEM(dIdy,float,k,j) =
					CV_IMAGE_ELEM(dIdy,float,k,j) + abs(CV_IMAGE_ELEM(dIcdy,float,i,j*channel+c));
			}

	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

	Mat dHdx = Mat(h,w,CV_32FC1);
	Mat dVdy = Mat(h,w,CV_32FC1);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			dHdx.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(dIdx,float,i,j);
			dVdy.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(dIdy,float,i,j);
		}


	//////////////////////////////////////////////// Integrate ////////////////////////////////////////////////////////////////////

	Mat ct_H = Mat(h,w,CV_32FC1);
	Mat ct_V = Mat(h,w,CV_32FC1);



	for(int i = 0; i < h; i++)
	{
		ct_H.at<float>(i,0) = dHdx.at<float>(i,0);
		for(int j = 1; j < w; j++)
		{
			ct_H.at<float>(i,j) = dHdx.at<float>(i,j) + ct_H.at<float>(i,j-1);
		}
	}

	for(int j = 0; j < w; j++)
	{
		ct_V.at<float>(0,j) = dVdy.at<float>(0,j);
		for(int i = 1; i < h; i++)
		{
			ct_V.at<float>(i,j) = dVdy.at<float>(i,j) + ct_V.at<float>(i-1,j);
		}
	}

	//// Transpose
	Mat dVdy_t = ct_V.t();

	float sigma_h = sigma_s;

	Mat F = Mat(h,w,CV_32FC3);

	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				F.at<float>(i,j*channel+c) =  CV_IMAGE_ELEM(img,float,i,j*channel+c);

	Mat F_t = Mat(w,h,CV_32FC3);

	float radius;

	for(int i=0;i<no_of_iter;i++)
	{
		sigma_h = sigma_s * sqrt(3) * pow(2,(no_of_iter - (i+1))) / sqrt(pow(4,no_of_iter) -1);

		radius = sqrt(3) * sigma_h;

		compute(F, ct_H, radius);

		F_t = F.t();

		compute(F_t, dVdy_t, radius);

		F = F_t.t();

	}

	res = F.clone();
}

void DOF_Filter::Depth_of_field(const IplImage *img, float sigma_s, float sigma_r, IplImage *img1)
{

	int no_of_iter = 3;
	int h = img1->height;
	int w = img1->width;
	int channel = img1->nChannels;

	////////////////////////////////////     horizontal and vertical partial derivatives /////////////////////////////////


	IplImage *dIcdx = cvCreateImage(cvSize(w-1,h), IPL_DEPTH_32F, 3 );
	IplImage *dIcdy = cvCreateImage(cvSize(w,h-1), IPL_DEPTH_32F, 3 );

	cvZero(dIcdx);
	cvZero(dIcdy);

	diffx(img1,dIcdx);
	diffy(img1,dIcdy);

	IplImage *dIdx = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1 );
	IplImage *dIdy = cvCreateImage(cvSize(w,h), IPL_DEPTH_32F, 1 );

	cvZero(dIdx);
	cvZero(dIdy);

	//////////////////////// Compute the l1-norm distance of neighbor pixels ////////////////////////////////////////////////


	int k;
	for(int i = 0; i < h; i++)
		for(int j = 0,k=1; j < w-1; j++,k++)
			for(int c = 0; c < channel; c++)
			{
				CV_IMAGE_ELEM(dIdx,float,i,k) =
					CV_IMAGE_ELEM(dIdx,float,i,k) + abs(CV_IMAGE_ELEM(dIcdx,float,i,j*channel+c));
			}

	for(int i = 0,k=1; i < h-1; i++,k++)
		for(int j = 0; j < w; j++)
			for(int c = 0; c < channel; c++)
			{
				CV_IMAGE_ELEM(dIdy,float,k,j) =
					CV_IMAGE_ELEM(dIdy,float,k,j) + abs(CV_IMAGE_ELEM(dIcdy,float,i,j*channel+c));
			}

	////////////////////// Compute the derivatives of the horizontal and vertical domain transforms. /////////////////////////////

	Mat dHdx = Mat(h,w,CV_32FC1);
	Mat dVdy = Mat(h,w,CV_32FC1);

	for(int i = 0; i < h; i++)
		for(int j = 0; j < w; j++)
		{
			dHdx.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(dIdx,float,i,j);
			dVdy.at<float>(i,j) = (float) 1.0 + (sigma_s/sigma_r) * CV_IMAGE_ELEM(dIdy,float,i,j);
		}


	//////////////////////////////////////////////// Integrate ////////////////////////////////////////////////////////////////////

	Mat ct_H = Mat(h,w,CV_32FC1);
	Mat ct_V = Mat(h,w,CV_32FC1);



	for(int i = 0; i < h; i++)
	{
		ct_H.at<float>(i,0) = dHdx.at<float>(i,0);
		for(int j = 1; j < w; j++)
		{
			ct_H.at<float>(i,j) = dHdx.at<float>(i,j) + ct_H.at<float>(i,j-1);
		}
	}

	for(int j = 0; j < w; j++)
	{
		ct_V.at<float>(0,j) = dVdy.at<float>(0,j);
		for(int i = 1; i < h; i++)
		{
			ct_V.at<float>(i,j) = dVdy.at<float>(i,j) + ct_V.at<float>(i-1,j);
		}
	}

	//// Transpose
	Mat dVdy_t = ct_V.t();

	float sigma_h = sigma_s;

	Mat F = Mat(h,w,CV_32FC3);

	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				F.at<float>(i,j*channel+c) =  CV_IMAGE_ELEM(img,float,i,j*channel+c);

	Mat F_t = Mat(w,h,CV_32FC3);

	float radius;

	for(int i=0;i<no_of_iter;i++)
	{
		sigma_h = sigma_s * sqrt(3) * pow(2,(no_of_iter - (i+1))) / sqrt(pow(4,no_of_iter) -1);

		radius = sqrt(3) * sigma_h;

		compute(F, ct_H, radius);

		F_t = F.t();

		compute(F_t, dVdy_t, radius);

		F = F_t.t();

	}

	Mat res = Mat(h,w,CV_32FC3);
	Mat final = Mat(h,w,CV_32FC3);

	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				res.at<float>(i,j*channel+c) =  F.at<float>(i,j*channel+c);

	for(int  i =0;i<h;i++)
		for(int  j =0;j<w;j++)
			for(int c=0;c<channel;c++)
				if(CV_IMAGE_ELEM(img1,float,i,j*channel+c) > .5)
					CV_IMAGE_ELEM(img1,float,i,j*channel+c) = 1.0;

	for(int i =0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;c++)
			{
				if(CV_IMAGE_ELEM(img1,float,i,j*channel+c) == 1.0)
				{
					res.at<float>(i,j*channel+c) = CV_IMAGE_ELEM(img,float,i,j*channel+c);
				}
			}


	imshow("Depth of Field", res);

	convertScaleAbs(res, final, 255,0);

	imwrite("result.jpg",final);

}


