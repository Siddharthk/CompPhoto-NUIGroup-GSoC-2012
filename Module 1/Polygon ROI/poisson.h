#include <iostream>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

using namespace std;
using namespace cv;

class Blending
{
	public:
		void display(const char *name, IplImage *img);
		void getGradientx(const IplImage *img, IplImage *gx);
		void getGradienty(const IplImage *img, IplImage *gy);
		void lapx(const IplImage *img, IplImage *gxx);
		void lapy(const IplImage *img, IplImage *gyy);
		void dst(double *gtest, double *gfinal,int h,int w);
		void idst(double *gtest, double *gfinal,int h,int w);
		void transpose(double *mat, double *mat_t,int h,int w);
		void poisson_solver(const IplImage *img, IplImage *gxx , IplImage *gyy, Mat &result);
};

class Normal_Blending : public Blending
{
	public:
		IplImage *normal_blend(IplImage *I, IplImage *mask, IplImage *wmask, int num);
};

class Local_color_change : public Blending
{
	public:
		IplImage *color_change(IplImage *I, IplImage *mask, IplImage *wmask, float red, float green, float blue);
};

class Local_illum_change : public Blending
{
	public:
		IplImage *illum_change(IplImage *I, IplImage *mask, IplImage *wmask, float alpha, float beta);
};

class Texture_flat : public Blending
{
	public:
		IplImage *tex_flattening(IplImage *I);
};

