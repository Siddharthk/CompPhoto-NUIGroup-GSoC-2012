#include <iostream>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

using namespace std;
using namespace cv;

class Domain_Filter
{
	public:
		void display(const char* name, IplImage *img);
		void diffx(const IplImage *img, IplImage *temp);
		void diffy(const IplImage *img, IplImage *temp);
};

class Recursive_Filter : public Domain_Filter
{
	public:
		void compute(Mat &O, Mat &horiz, float sigma_h);
		void RecursiveFilter(const IplImage *img, Mat &res, float sigma_s, float sigma_r);

};
class NC_Filter : public Domain_Filter
{
	public:
		void compute(Mat &O, Mat &horiz, float radius);
		void Normalized_Conv_Filter(const IplImage *img, Mat &res, float sigma_s, float sigma_r);

};
class Pencil_Filter : public Domain_Filter
{
	public:
		void compute(Mat &O, Mat &horiz, Mat &pencil, float radius);
		void Pencil_Sketch(const IplImage *img, float sigma_s, float sigma_r, float shade_factor);

};

class DOF_Filter : public NC_Filter
{
	public:
		void Depth_of_field(const IplImage *img, float sigma_s, float sigma_r, IplImage *img1);

};

class Edge_Enhance : public NC_Filter
{
	public:
		void getGradientx( const Mat &img, Mat &gx);
		void getGradienty( const Mat &img, Mat &gy);
		void find_magnitude(Mat &img, Mat &mag);

};


