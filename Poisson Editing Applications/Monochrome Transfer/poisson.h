#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#include "math.h"

using namespace std;
using namespace cv;

#define pi 3.1416

void display(const char *name, IplImage *img)
{
	cvNamedWindow(name);
	cvShowImage(name,img);
}

void getGradientx( const IplImage *img, IplImage *gx)
{
	int w = img->width;
	int h = img->height;
	int channel = img->nChannels;

	cvZero( gx );
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(gx,float,i,j*channel+c) =
					(float)CV_IMAGE_ELEM(img,uchar,i,(j+1)*channel+c) - (float)CV_IMAGE_ELEM(img,uchar,i,j*channel+c);
			}
}
void getGradienty( const IplImage *img, IplImage *gy)
{
	int w = img->width;
	int h = img->height;
	int channel = img->nChannels;

	cvZero( gy );
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(gy,float,i,j*channel+c) =
					(float)CV_IMAGE_ELEM(img,uchar,(i+1),j*channel+c) - (float)CV_IMAGE_ELEM(img,uchar,i,j*channel+c);
					
			}
}
void lapx( const IplImage *img, IplImage *gxx)
{
	int w = img->width;
	int h = img->height;
	int channel = img->nChannels;

	cvZero( gxx );
	for(int i=0;i<h;i++)
		for(int j=0;j<w-1;j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(gxx,float,i,(j+1)*channel+c) =
						(float)CV_IMAGE_ELEM(img,float,i,(j+1)*channel+c) - (float)CV_IMAGE_ELEM(img,float,i,j*channel+c);
			}
}
void lapy( const IplImage *img, IplImage *gyy)
{
	int w = img->width;
	int h = img->height;
	int channel = img->nChannels;

	cvZero( gyy );
	for(int i=0;i<h-1;i++)
		for(int j=0;j<w;j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(gyy,float,i+1,j*channel+c) =
					(float)CV_IMAGE_ELEM(img,float,(i+1),j*channel+c) - (float)CV_IMAGE_ELEM(img,float,i,j*channel+c);
					
			}
}

void dst(double *gtest, double *gfinal,int h,int w)
{

	int k,r,z;
	unsigned long int idx;

	Mat temp = Mat(2*h+2,1,CV_32F);
	Mat res  = Mat(h,1,CV_32F);

	int p=0;
	for(int i=0;i<w;i++)
	{
		temp.at<float>(0,0) = 0.0;
		
		for(int j=0,r=1;j<h;j++,r++)
		{
			idx = j*w+i;
			temp.at<float>(r,0) = gtest[idx];
		}

		temp.at<float>(h+1,0)=0.0;

		for(int j=h-1, r=h+2;j>=0;j--,r++)
		{
			idx = j*w+i;
			temp.at<float>(r,0) = -1*gtest[idx];
		}
		
		Mat planes[] = {Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F)};

		Mat complex1;
		merge(planes, 2, complex1);

		dft(complex1,complex1,0,0);

		Mat planes1[] = {Mat::zeros(complex1.size(), CV_32F), Mat::zeros(complex1.size(), CV_32F)};
		
		// planes1[0] = Re(DFT(I)), planes1[1] = Im(DFT(I))
		split(complex1, planes1); 

		std::complex<double> two_i = std::sqrt(std::complex<double>(-1));

		double fac = -2*imag(two_i);

		for(int c=1,z=0;c<h+1;c++,z++)
		{
			res.at<float>(z,0) = planes1[1].at<float>(c,0)/fac;
		}

		for(int q=0,z=0;q<h;q++,z++)
		{
			idx = q*w+p;
			gfinal[idx] =  res.at<float>(z,0);
		}
		p++;
	}

}

void idst(double *gtest, double *gfinal,int h,int w)
{
	int nn = h+1;
	unsigned long int idx;
	dst(gtest,gfinal,h,w);
	for(int  i= 0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			idx = i*w + j;
			gfinal[idx] = (double) (2*gfinal[idx])/nn;
		}

}
void transpose(double *mat, double *mat_t,int h,int w)
{

	Mat tmp = Mat(h,w,CV_32FC1);
	int p =0;
	unsigned long int idx; 
	for(int i = 0 ; i < h;i++)
	{
		for(int j = 0 ; j < w; j++)
		{

			idx = i*(w) + j;
			tmp.at<float>(i,j) = mat[idx];
		}
	}
	Mat tmp_t = tmp.t();

	for(int i = 0;i < tmp_t.size().height; i++)
		for(int j=0;j<tmp_t.size().width;j++)
		{
			idx = i*tmp_t.size().width + j;
			mat_t[idx] = tmp_t.at<float>(i,j);
		}

}
void poisson_solver(const IplImage *img, IplImage *gxx , IplImage *gyy, Mat &result)
{

	int w = img->width;
	int h = img->height;
	int channel = img->nChannels;

	unsigned long int idx,idx1;

	IplImage *lap  = cvCreateImage(cvGetSize(img), 32, 1);

	for(int i =0;i<h;i++)
		for(int j=0;j<w;j++)
			CV_IMAGE_ELEM(lap,float,i,j)=CV_IMAGE_ELEM(gyy,float,i,j)+CV_IMAGE_ELEM(gxx,float,i,j);

	Mat bound(img);

	for(int i =1;i<h-1;i++)
		for(int j=1;j<w-1;j++)
		{
			bound.at<uchar>(i,j) = 0.0;
		}
	
	double *f_bp = new double[h*w];


	for(int i =1;i<h-1;i++)
		for(int j=1;j<w-1;j++)
		{
			idx=i*w + j;
			f_bp[idx] = -4*(int)bound.at<uchar>(i,j) + (int)bound.at<uchar>(i,(j+1)) + (int)bound.at<uchar>(i,(j-1))
					+ (int)bound.at<uchar>(i-1,j) + (int)bound.at<uchar>(i+1,j);
		}
	

	Mat diff = Mat(h,w,CV_32FC1);
	for(int i =0;i<h;i++)
	{
		for(int j=0;j<w;j++)
		{
			idx = i*w+j;
			diff.at<float>(i,j) = (CV_IMAGE_ELEM(lap,float,i,j) - f_bp[idx]);
		}
	}
	double *gtest = new double[(h-2)*(w-2)];
	for(int i = 0 ; i < h-2;i++)
	{
		for(int j = 0 ; j < w-2; j++)
		{
			idx = i*(w-2) + j;
			gtest[idx] = diff.at<float>(i+1,j+1);
			
		}
	}
	///////////////////////////////////////////////////// Find DST  /////////////////////////////////////////////////////

	double *gfinal = new double[(h-2)*(w-2)];
	double *gfinal_t = new double[(h-2)*(w-2)];
	double *denom = new double[(h-2)*(w-2)];
	double *f3 = new double[(h-2)*(w-2)];
	double *f3_t = new double[(h-2)*(w-2)];
	double *img_d = new double[(h)*(w)];

	dst(gtest,gfinal,h-2,w-2);

	transpose(gfinal,gfinal_t,h-2,w-2);

	dst(gfinal_t,gfinal,w-2,h-2);

	transpose(gfinal,gfinal_t,w-2,h-2);

	int cx=1;
	int cy=1;

	for(int i = 0 ; i < w-2;i++,cy++)
	{
		for(int j = 0,cx = 1; j < h-2; j++,cx++)
		{
			idx = j*(w-2) + i;
			denom[idx] = (float) 2*cos(pi*cy/( (double) (w-1))) - 2 + 2*cos(pi*cx/((double) (h-1))) - 2;
			
		}
	}

	for(idx = 0 ; idx < (w-2)*(h-2) ;idx++)
	{
		gfinal_t[idx] = gfinal_t[idx]/denom[idx];
	}


	idst(gfinal_t,f3,h-2,w-2);

	transpose(f3,f3_t,h-2,w-2);

	idst(f3_t,f3,w-2,h-2);

	transpose(f3,f3_t,w-2,h-2);

	for(int i = 0 ; i < h;i++)
	{
		for(int j = 0 ; j < w; j++)
		{
			idx = i*w + j;
			img_d[idx] = (double)CV_IMAGE_ELEM(img,uchar,i,j);	
		}
	}
	for(int i = 1 ; i < h-1;i++)
	{
		for(int j = 1 ; j < w-1; j++)
		{
			idx = i*w + j;
			img_d[idx] = 0.0;	
		}
	}
	int id1,id2;
	for(int i = 1,id1=0 ; i < h-1;i++,id1++)
	{
		for(int j = 1,id2=0 ; j < w-1; j++,id2++)
		{
			idx = i*w + j;
			idx1= id1*(w-2) + id2;
			img_d[idx] = f3_t[idx1];	
		}
	}
	
	for(int i = 0 ; i < h;i++)
	{
		for(int j = 0 ; j < w; j++)
		{
			idx = i*w + j;
			if(img_d[idx] < 0.0)
				result.at<uchar>(i,j) = 0;
			else if(img_d[idx] > 255.0)
				result.at<uchar>(i,j) = 255.0;
			else
				result.at<uchar>(i,j) = img_d[idx];	
		}
	}

}

IplImage *poisson_blend(IplImage *I, IplImage *mask, int posx, int posy)
{

	unsigned long int idx;

	if(I->nChannels < 3)
	{
		printf("Enter RGB image\n");
		exit(0);
	}

	IplImage *grx  = cvCreateImage(cvGetSize(I), 32, 3);
	IplImage *gry  = cvCreateImage(cvGetSize(I), 32, 3);

	IplImage *sgx  = cvCreateImage(cvGetSize(mask), 32, 3);
	IplImage *sgy  = cvCreateImage(cvGetSize(mask), 32, 3);

	IplImage *S    = cvCreateImage(cvGetSize(I), 8, 3);
	IplImage *ero  = cvCreateImage(cvGetSize(I), 8, 3);
	IplImage *res  = cvCreateImage(cvGetSize(I), 8, 3);

	cvZero(S);
	cvZero(res);


	IplImage *O    = cvCreateImage(cvGetSize(I), 8, 3);
	IplImage *error= cvCreateImage(cvGetSize(I), 8, 3);


	int w = I->width;
	int h = I->height;
	int channel = I->nChannels;

	int w1 = mask->width;
	int h1 = mask->height;
	int channel1 = mask->nChannels;

	getGradientx(I,grx);
	getGradienty(I,gry);


	IplImage *gray = cvCreateImage( cvGetSize(mask), 8, 1 );
	IplImage *gray8 = cvCreateImage( cvGetSize(mask), 8, 3 );
	cvCvtColor(mask, gray, CV_BGR2GRAY );

	cvMerge(gray,gray,gray,0,gray8);

	getGradientx(gray8,sgx);
	getGradienty(gray8,sgy);

	for(int i=posx, ii =0;i<posx+h1;i++,ii++)
		for(int j=0,jj=posy;j<w1;j++,jj++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(S,uchar,i,jj*channel+c) = 255;
			}

	IplImage* bmaskx = cvCreateImage(cvGetSize(ero),32,3);
	cvConvertScale(S,bmaskx,1.0/255.0,0.0);

	IplImage* bmasky = cvCreateImage(cvGetSize(ero),32,3);
	cvConvertScale(S,bmasky,1.0/255.0,0.0);

	for(int i=posx, ii =0;i<posx+h1;i++,ii++)
		for(int j=0,jj=posy;j<w1;j++,jj++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(bmaskx,float,i,jj*channel+c) = CV_IMAGE_ELEM(sgx,float,ii,j*channel+c);
				CV_IMAGE_ELEM(bmasky,float,i,jj*channel+c) = CV_IMAGE_ELEM(sgy,float,ii,j*channel+c);
			}

	cvErode(S,ero,NULL,1);

	IplImage* smask = cvCreateImage(cvGetSize(ero),32,3);
	cvConvertScale(ero,smask,1.0/255.0,0.0);

	IplImage* srx32 = cvCreateImage(cvGetSize(res),32,3);
	cvConvertScale(res,srx32,1.0/255.0,0.0);

	IplImage* sry32 = cvCreateImage(cvGetSize(res),32,3);
	cvConvertScale(res,sry32,1.0/255.0,0.0);

	for(int i=0;i < h; i++)
		for(int j=0; j < w; j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(srx32,float,i,j*channel+c) =
					(CV_IMAGE_ELEM(bmaskx,float,i,j*channel+c)*CV_IMAGE_ELEM(smask,float,i,j*channel+c));
				CV_IMAGE_ELEM(sry32,float,i,j*channel+c) =
					(CV_IMAGE_ELEM(bmasky,float,i,j*channel+c)*CV_IMAGE_ELEM(smask,float,i,j*channel+c));
			}

	cvNot(ero,ero);

	IplImage* smask1 = cvCreateImage(cvGetSize(ero),32,3);
	cvConvertScale(ero,smask1,1.0/255.0,0.0);

	IplImage* grx32 = cvCreateImage(cvGetSize(res),32,3);
	cvConvertScale(res,grx32,1.0/255.0,0.0);

	IplImage* gry32 = cvCreateImage(cvGetSize(res),32,3);
	cvConvertScale(res,gry32,1.0/255.0,0.0);

	for(int i=0;i < h; i++)
		for(int j=0; j < w; j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(grx32,float,i,j*channel+c) =
					(CV_IMAGE_ELEM(grx,float,i,j*channel+c)*CV_IMAGE_ELEM(smask1,float,i,j*channel+c));
				CV_IMAGE_ELEM(gry32,float,i,j*channel+c) =
					(CV_IMAGE_ELEM(gry,float,i,j*channel+c)*CV_IMAGE_ELEM(smask1,float,i,j*channel+c));
			}


	IplImage* fx = cvCreateImage(cvGetSize(res),32,3);
	IplImage* fy = cvCreateImage(cvGetSize(res),32,3);

	for(int i=0;i < h; i++)
		for(int j=0; j < w; j++)
			for(int c=0;c<channel;++c)
			{
				CV_IMAGE_ELEM(fx,float,i,j*channel+c) =
					(CV_IMAGE_ELEM(grx32,float,i,j*channel+c)+CV_IMAGE_ELEM(srx32,float,i,j*channel+c));
				CV_IMAGE_ELEM(fy,float,i,j*channel+c) =
					(CV_IMAGE_ELEM(gry32,float,i,j*channel+c)+CV_IMAGE_ELEM(sry32,float,i,j*channel+c));
			}


	IplImage *gxx  = cvCreateImage(cvGetSize(I), 32, 3);
	IplImage *gyy  = cvCreateImage(cvGetSize(I), 32, 3);

	lapx(fx,gxx);
	lapy(fy,gyy);

	IplImage *rx_channel = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1 );
	IplImage *gx_channel = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1 );
	IplImage *bx_channel = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1 );

	cvCvtPixToPlane(gxx, rx_channel, gx_channel, bx_channel,0);

	IplImage *ry_channel = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1 );
	IplImage *gy_channel = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1 );
	IplImage *by_channel = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 1 );

	cvCvtPixToPlane(gyy, ry_channel, gy_channel, by_channel,0);
	
	IplImage *r_channel = cvCreateImage(cvGetSize(I), 8, 1 );
	IplImage *g_channel = cvCreateImage(cvGetSize(I), 8, 1 );
	IplImage *b_channel = cvCreateImage(cvGetSize(I), 8, 1 );

	cvCvtPixToPlane(I, r_channel, g_channel, b_channel,0);

	Mat resultr = Mat(h,w,CV_8UC1);
	Mat resultg = Mat(h,w,CV_8UC1);
	Mat resultb = Mat(h,w,CV_8UC1);

	clock_t tic = clock();

	poisson_solver(r_channel,rx_channel, ry_channel,resultr);
	poisson_solver(g_channel,gx_channel, gy_channel,resultg);
	poisson_solver(b_channel,bx_channel, by_channel,resultb);

	clock_t toc = clock();

	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

	IplImage *final = cvCreateImage(cvGetSize(I), 8, 3 );
	
	for(int i=0;i<h;i++)
		for(int j=0;j<w;j++)
		{
			CV_IMAGE_ELEM(final,uchar,i,j*3+0) = resultr.at<uchar>(i,j);
			CV_IMAGE_ELEM(final,uchar,i,j*3+1) = resultg.at<uchar>(i,j);
			CV_IMAGE_ELEM(final,uchar,i,j*3+2) = resultb.at<uchar>(i,j);
		}

	return final;

}

