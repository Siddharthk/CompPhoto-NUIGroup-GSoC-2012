#include <iostream>
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"

using namespace std;
using namespace cv;

class Decolor 
{
	public:
		Mat kernel; 
		Mat kernel1;
		void init();
		int rounding(double a);
		void display(const char *name, IplImage *img);
		vector<double> product(vector < vector<int> > &comb, vector <double> &initRGB);
		void singleChannelGradx(const Mat &img, Mat& dest);
		void singleChannelGrady(const Mat &img, Mat& dest);
		void gradvector(const Mat &img, vector <double> &grad);
		void colorGrad(IplImage *img, vector <double> &Cg);
		void add_vector(vector < vector <int> > &comb, int r,int g,int b);
		void add_to_vector_poly(vector < vector <double> > &polyGrad, vector <double> &curGrad);
		void weak_order(IplImage* img, vector <double> &alf);
		void grad_system(IplImage* img, vector < vector < double > > &polyGrad, vector < double > &Cg, vector < vector <int> >& comb);
		void wei_update_matrix(vector < vector <double> > &poly, vector <double> &Cg, Mat &X);
		void wei_inti(vector < vector <int> > &comb, vector <double> &wei);
		void grayImContruct(vector <double> &wei, IplImage *img, IplImage *Gray);

};

