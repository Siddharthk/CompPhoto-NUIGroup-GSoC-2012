#include <iostream>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

using namespace std;
using namespace cv;

class Blending
{
	public:
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
		IplImage *normal_blend(IplImage *I, IplImage *mask, int posx, int posy);
};

class Mixed_Blending : public Blending
{
	public:
		IplImage *mixed_blend(IplImage *I, IplImage *mask, int posx, int posy);
};

class Local_color_change : public Blending
{
	public:
		IplImage *color_change(IplImage *I, IplImage *mask, int posx, int posy, float red, float green, float blue);
};

class Local_illum_change : public Blending
{
	public:
		IplImage *illum_change(IplImage *I, IplImage *mask, int posx, int posy, float alpha, float beta);
};

class Mono_trans : public Blending
{
	public:
		IplImage *monochrome_transfer(IplImage *I, IplImage *mask, int posx, int posy);
};

class Texture_flat : public Blending
{
	public:
		IplImage *tex_flattening(IplImage *I);
};

