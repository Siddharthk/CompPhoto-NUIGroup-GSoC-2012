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
		void compute(Mat &O, Mat &dHdx, float sigma_h);
		void RecursiveFilter(const IplImage *img, float sigma_s, float sigma_r);
		

};



