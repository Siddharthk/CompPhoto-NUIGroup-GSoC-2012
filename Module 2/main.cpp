/*
############################ Domain Aware Edge Filter ##################################

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

########################################################################################
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

void checkfile(char *src)
{
	while(1)
	{
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
	int num;

	cout << endl;
	cout << " Domain aware edge filter" << endl;
	cout << "--------------------------" << endl;

	cout << "Options: " << endl;
	cout << endl;

	cout << "1) Edge Preserve Smoothing" << endl;
	cout << endl;

	cout << "Press number 1-6 to choose from above techniques: ";

	cin >> num;

	cout << endl;

	char src[50];

	checkfile(src);
	float sigma_s, sigma_r;
	
	sigma_s = 60;
	sigma_r = .45; 

	cout << endl;
	cout << "sigma_s(default val: 60): ";
	cin >> sigma_s;

	cout << "sigma_r(default val: .45): ";
	cin >> sigma_r;

	IplImage *I = cvLoadImage(src);

	Domain_Filter obj1;
	obj1.display("Original Image",I);

	IplImage *img = cvCreateImage(cvGetSize(I), IPL_DEPTH_32F, 3 );
	cvConvertScale(I,img,1.0/255.0,0.0);


	Recursive_Filter obj;
	obj.RecursiveFilter(img, sigma_s, sigma_r);

	cvReleaseImage(&I);
	cvWaitKey(0);

	return 0;

}
