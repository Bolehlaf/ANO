// DIP.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#define PI 3.14159265

using namespace std;

void tresholding(cv::Mat& img) {
	uchar p;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			p = img.at<uchar>(y, x);
			if (p <= 120) {
				img.at<uchar>(y, x) = 0;
			}
			else {
				img.at<uchar>(y, x) = 255;
			}
		}
	}
}

void indexing(cv::Mat& img, cv::Mat& indexes, int x, int y, int &index) {
	indexes.at<uchar>(y, x) = index;

	if (img.at<uchar>(y - 1, x) == 255 && indexes.at<uchar>(y - 1, x) != index) {
		indexing(img, indexes, x, y - 1, index);
	}
	if (img.at<uchar>(y - 1, x - 1) == 255 && indexes.at<uchar>(y - 1, x - 1) != index) {
		indexing(img, indexes, x - 1, y - 1, index);
	}
	if (img.at<uchar>(y, x - 1) == 255 && indexes.at<uchar>(y, x - 1) != index) {
		indexing(img, indexes, x - 1, y, index);
	}
	if (img.at<uchar>(y - 1, x + 1) == 255 && indexes.at<uchar>(y - 1, x + 1) != index) {
		indexing(img, indexes, x + 1, y - 1, index);
	}
	if (img.at<uchar>(y, x + 1) == 255 && indexes.at<uchar>(y, x + 1) != index) {
		indexing(img, indexes, x + 1, y, index);
	}
	if (img.at<uchar>(y + 1, x + 1) == 255 && indexes.at<uchar>(y + 1, x + 1) != index) {
		indexing(img, indexes, x + 1, y + 1, index);
	}
	if (img.at<uchar>(y + 1, x) == 255 && indexes.at<uchar>(y + 1, x) != index) {
		indexing(img, indexes, x, y + 1, index);
	}
	if (img.at<uchar>(y + 1, x - 1) == 255 && indexes.at<uchar>(y + 1, x - 1) != index) {
		indexing(img, indexes, x - 1, y + 1, index);
	}
	else {
		return;
	}
}

class Object
{
public:
	int id;
	int area;
	int perimeter;
	int xt;
	int yt;
	string type;
	double F1;
	double F2;
	double F3;
	Object(int id);
	void toString();
	~Object();

private:
	
};

Object::Object(int id)
{
	this->id = id;
	this->F1 = 0;
	this->F2 = 0;
	this->F3 = this->area = this->perimeter = this->xt = this->yt = 0;
	this->type = "unknown";
}

Object::~Object()
{
}

void Object::toString() {
	cout << "Id: " << this->id << endl;
	cout << "area: " << this->area << endl;
	cout << "perimeter: " << this->perimeter << endl;
	cout << "xt: " << this->xt << endl;
	cout << "yt: " << this->yt << endl;
	cout << "Feature 1: " << this->F1 << endl;
	cout << "Feature 2: " << this->F2 << endl;
	cout << "Feature 3: " << this->F3 << endl;
	cout << "type: " << this->type << endl;
}

int moment(cv::Mat indexes, int p, int q, int index) {

	int m = 0;
	for (int y = 0; y < indexes.rows; y++) {
		for (int x = 0; x < indexes.cols; x++) {
			if (indexes.at<uchar>(y, x) == index)
				m += pow(x, p)*pow(y, q);
		}
	}

	return m;
}

bool outerPoint(cv::Mat indexes, int index, int x, int y) {
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			if (i == j == 0)
				continue;
			else if (indexes.at<uchar>(y - i, x - j) != index)
				return true;
		}
	}

	return false;
}

int perimeter(cv::Mat indexes, int index) {

	int perimeter = 0;
	for (int y = 0; y < indexes.rows; y++) {
		for (int x = 0; x < indexes.cols; x++) {
			if (indexes.at<uchar>(y, x) == index && outerPoint(indexes, index, x, y)) {
				perimeter++;
			}
		}
	}

	return perimeter;
}

double mi(cv::Mat indexes, int p, int q, int index) {

	int area = moment(indexes, 0, 0, index);
	double xt = (double)moment(indexes, 1, 0, index) / (double)area;
	double yt = (double)moment(indexes, 0, 1, index) / (double)area;

	double m = 0;

	for (int y = 0; y < indexes.rows; y++) {
		for (int x = 0; x < indexes.cols; x++) {
			if (indexes.at<uchar>(y, x) == index) {
				m += pow(x - xt, p)*pow(y - yt, q);
			}
		}
	}

	return m;
}



double f1(cv::Mat indexes, int index) {

	double p = perimeter(indexes, index);
	double area = mi(indexes, 0, 0, index);
	return (p*p) / (100 * area);
}

double f2(cv::Mat indexes, int index) {

	double min = 0.5 * (mi(indexes, 2, 0, index) + mi(indexes, 0, 2, index)) - 0.5 * sqrt(4 * pow(mi(indexes, 1, 1, index), 2) + pow(mi(indexes, 2, 0, index) - mi(indexes, 0, 2, index), 2));
	double max = 0.5 * (mi(indexes, 2, 0, index) + mi(indexes, 0, 2, index)) + 0.5 * sqrt(4 * pow(mi(indexes, 1, 1, index), 2) + pow(mi(indexes, 2, 0, index) - mi(indexes, 0, 2, index), 2));
	return min / max;
}

//0-left, 1-right, 2-down, 3-up
int extremes(cv::Mat indexes, int index) {
	int extreme[4][2] = { 0 };
	extreme[0][0] = INT_MAX;
	extreme[2][1] = INT_MAX;
	for (int y = 0; y < indexes.rows; y++) {
		for (int x = 0; x < indexes.cols; x++) {
			if (indexes.at<uchar>(y, x) == index) {
				if (x < extreme[0][0]) {
					extreme[0][0] = x;
					extreme[0][1] = y;
				}
				if (x > extreme[1][0]) {
					extreme[1][0] = x;
					extreme[1][1] = y;
				}
				if (y < extreme[2][1]) {
					extreme[2][0] = x;
					extreme[2][1] = y;
				}
				if (y > extreme[3][1]) {
					extreme[3][0] = x;
					extreme[3][1] = y;
				}
			}
		}
	}

	if ((extreme[1][0] - extreme[0][0]) > (extreme[3][1] - extreme[2][1]))
		return (extreme[1][0] - extreme[0][0])*(extreme[1][0] - extreme[0][0]);
	else
		return (extreme[3][1] - extreme[2][1])*(extreme[3][1] - extreme[2][1]);
}

int rotate(cv::Mat& img, int index, int xt, int yt, int angle) {
	cv::Mat copy (cv::Size(img.cols, img.rows+100), CV_8UC1, cv::Scalar(0));
	int rx;
	int ry;
	int area;
	for (int y = 0; y < img.rows; y++) {
		for (int x = 0; x < img.cols; x++) {
			if (img.at<uchar>(y, x) == index) {
				rx = xt + cos(angle * PI / 180.0)*(x - xt) - sin(angle * PI / 180.0)*(y - yt);
				ry = yt + sin(angle * PI / 180.0)*(x - xt) + cos(angle * PI / 180.0)*(y - yt);
				copy.at<uchar>(ry, rx) = index;
			}
		}
	}

	area = extremes(copy, index);
	return area;
}

int min(int *arr, int length) {
	int min = arr[0];
	for (int i = 1; i < length; i++) {
		if (arr[i] < min) {
			min = arr[i];
		}
	}
	return min;
}

int max(int *arr, int length) {
	int max = arr[0];
	for (int i = 1; i < length; i++) {
		if (arr[i] > max) {
			max = arr[i];
		}
	}
	return max;
}

double f3(cv::Mat indexes, int index) {
	/*double p = perimeter(indexes, index);
	double area = mi(indexes, 0, 0, index);
	return (p*p) / (area);*/
	double area = mi(indexes, 0, 0, index);
	int xt = moment(indexes, 1, 0, index) / area;
	int yt = moment(indexes, 0, 1, index) / area;
	int *areas = new int[36];

	for (int i = 0; i < 180; i += 5) {
		areas[i/5] = rotate(indexes, index, xt, yt, i);
	}
	int numerator = min(areas, 36);
	int denominator = max(areas, 36);
	return numerator/(double)denominator;
}

void groups(list<Object> listOfObjects, double** avg) {
	int scale = 300;
	list<Object>::iterator it;
	cv::Mat graph(cv::Size(scale, scale), CV_8UC1, cv::Scalar(0));

	for (it = listOfObjects.begin(); it != listOfObjects.end(); it++) {
		int x = it->F1 * scale;
		int y = it->F2 * scale;
		if (it->type != "circle")
			graph.at<uchar>(y, x) = 100;

		if (it->type == "square") {
			avg[0][0] += it->F1 * scale;
			avg[0][1] += it->F2 * scale;
			avg[0][2] += it->F3 * scale;
		}
		else if (it->type == "star") {
			avg[1][0] += it->F1 * scale;
			avg[1][1] += it->F2 * scale;
			avg[1][2] += it->F3 * scale;
		}
		else if (it->type == "rectangle") {
			avg[2][0] += it->F1 * scale;
			avg[2][1] += it->F2 * scale;
			avg[2][2] += it->F3 * scale;
		}
		else {
			avg[3][0] += it->F1 * scale;
			avg[3][1] += it->F2 * scale;
			avg[3][2] += it->F3 * scale;
		}
	}

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			avg[i][j] /= 4;
		}
	}
	
	graph.at<uchar>((int)avg[0][1], (int)avg[0][0]) = 255;
	graph.at<uchar>((int)avg[1][1], (int)avg[1][0]) = 255;
	graph.at<uchar>((int)avg[2][1], (int)avg[2][0]) = 255;

	cv::imshow("groups", graph);
	//cv::waitKey(0);
}

void labeling(cv::Mat &indexes, int objectCount, double **avg) {
	list<Object> listOfObjects;
	list<Object>::iterator it;
	int index = 10;
	int incrementation = 10;
	int scale = 300;

	for (int i = 0; i < objectCount; i++) {
		Object o(i);
		o.F1 = f1(indexes, index);
		o.F2 = f2(indexes, index);
		o.F3 = f3(indexes, index);

		double x = o.F1 * scale;
		double y = o.F2 * scale;
		double z = o.F3 * scale;

		cout << endl;
		double range[4];
		range[0] = sqrt(pow(avg[0][0] - x, 2) + pow(avg[0][1] - y, 2) + pow(avg[0][2] - z, 2));
		range[1] = sqrt(pow(avg[1][0] - x, 2) + pow(avg[1][1] - y, 2) + pow(avg[1][2] - z, 2));
		range[2] = sqrt(pow(avg[2][0] - x, 2) + pow(avg[2][1] - y, 2) + pow(avg[2][2] - z, 2));
		range[3] = sqrt(pow(avg[3][0] - x, 2) + pow(avg[3][1] - y, 2) + pow(avg[3][2] - z, 2));

		int shape = 0;
		double min = range[shape];
		if (range[1] < min) {
			shape = 1;
			min = range[shape];
		}
		if (range[2] < min) {
			shape = 2;
			min = range[shape];
		}
		if (range[3] < min) {
			shape = 3;
			min = range[shape];
		}

		switch (shape) {
		case 0:
			printf("Object %d je ctverec\n", i+1);
			o.type = "square";
			break;
		case 1:
			printf("Object %d je hvezda\n", i+1);
			o.type = "star";
			break;
		case 2:
			printf("Object %d je obdelnik\n", i+1);
			o.type = "rectangle";
			break;
		case 3:
			printf("Object %d je kruh\n", i + 1);
			o.type = "circle";
			break;
		}

		listOfObjects.push_back(o);
		index += incrementation;
	}
}

double distance(double x1, double y1, double x2, double y2) {
	double d = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
	return d;
}

void kmeans(int objectCount, list<Object> listOfObjects) {

	const int k = 3;
	int scale = 300;
	double means[2][k];
	int meansCount[k] = { 0 };
	for (int i = 0; i < k; i++) {
		means[0][i] = rand() % scale;
		means[1][i] = rand() % scale;
	}


	list<Object>::iterator it;
	int objectMean[12];
	double min = 0;
	bool treshold = true;
	int steps = 0;

	while (treshold && steps < 20) {
		treshold = false;
		for (it = listOfObjects.begin(); it != listOfObjects.end(); it++) {
			double x = it->F1 * scale;
			double y = it->F2 * scale;

			min = distance(x, y, means[0][0], means[1][0]);
			objectMean[it->id] = 0;
			for (int j = 1; j < k; j++) {
				if (min > distance(x, y, means[0][j], means[1][j])) {
					min = distance(x, y, means[0][j], means[1][j]);
					objectMean[it->id] = j;
				}
				meansCount[objectMean[it->id]]++;
			}
		}

		double oldmeans[2][k];
		for (int i = 0; i < k; i++) {
			oldmeans[0][i] = means[0][i];
			oldmeans[1][i] = means[1][i];
		}

		for (int i = 0; i < k; i++) {
			means[0][i] = 0;
			means[1][i] = 0;
		}

		for (it = listOfObjects.begin(); it != listOfObjects.end(); it++) {

			for (int j = 0; j < k; j++) {
				if (objectMean[it->id] == j) {
					means[0][j] += it->F1 * scale;
					means[1][j] += it->F2 * scale;
				}
			}

		}

		for (int i = 0; i < k; i++) {
			means[0][i] /= meansCount[i];
			means[1][i] /= meansCount[i];
		}

		double minx;
		double miny;
		double d;
		double tmp;
		double roz = 0;

		for (int i = 0; i < k; i++) {
			minx = scale;
			miny = scale;
			d = distance(minx, miny, means[0][i], means[1][i]);
			for (it = listOfObjects.begin(); it != listOfObjects.end(); it++) {
				minx = it->F1 * scale;
				miny = it->F2 * scale;
				tmp = distance(minx, miny, means[0][i], means[1][i]);
				roz = abs(d - tmp);

				if (tmp < d && objectMean[it->id] == i) {
					d = tmp;
					means[0][i] = minx;
					means[1][i] = miny;
				}
			}
		}
		
		for (int i = 0; i < k; i++) {
			if (oldmeans[0][i] != means[0][i])
				treshold = true;
			if (oldmeans[1][i] != means[1][i])
				treshold = true;
		}
		steps++;
	}
	cv::Mat kmeans(cv::Size(scale, scale), CV_8UC1, cv::Scalar(0));

	for (it = listOfObjects.begin(); it != listOfObjects.end(); it++) {
		double x = it->F1 * scale;
		double y = it->F2 * scale;
		kmeans.at<uchar>(y, x) = 100;
	}

	for (int i = 0; i < 3; i++) {
		kmeans.at<uchar>(means[1][i], means[0][i]) = 255;
		kmeans.at<uchar>(means[1][i] + 1, means[0][i] + 1) = 255;
		kmeans.at<uchar>(means[1][i] + 1, means[0][i] - 1) = 255;
		kmeans.at<uchar>(means[1][i] - 1, means[0][i] + 1) = 255;
		kmeans.at<uchar>(means[1][i] - 1, means[0][i] - 1) = 255;
	}

	cv::imshow("kmeans", kmeans);
}

int main()
{
	cv::Mat imgTrain = cv::imread("images/train.png", CV_LOAD_IMAGE_GRAYSCALE);
	tresholding(imgTrain);

	cv::Mat indexes(cv::Size(imgTrain.cols, imgTrain.rows), CV_8UC1, cv::Scalar(0));
	int incrementation = 10;
	int index = 10;

	for (int y = 0; y < imgTrain.rows; y++) {
		for (int x = 0; x < imgTrain.cols; x++) {
			if (imgTrain.at<uchar>(y, x) == 255 && indexes.at<uchar>(y, x) == 0) {
				indexing(imgTrain, indexes, x, y, index);
				index += incrementation;
			}
		}
	}

	cv::imshow("indexing", indexes);
	list<Object> listOfObjects;
	list<Object>::iterator it;
	int objectCount = (index - incrementation) / incrementation;
	index = 10;

	for (int i = 0; i < objectCount; i++) {
		Object o(i);
		o.area = moment(indexes, 0, 0, index);
		o.perimeter = perimeter(indexes, index);
		o.xt = moment(indexes, 1, 0, index) / o.area;
		o.yt = moment(indexes, 0, 1, index) / o.area;
		o.F1 = f1(indexes, index);
		o.F2 = f2(indexes, index);
		o.F3 = f3(indexes, index);

		if (i < 4) {
			o.type = "square";
		}
		else if (i < 8) {
			o.type = "star";
		}
		else if (i < 12) {
			o.type = "rectangle";
		}
		else {
			o.type = "circle";
		}

		listOfObjects.push_back(o);
		index += incrementation;
	}

	double **averages = new double*[4];
	for (int i = 0; i < 4; i++) {
		averages[i] = new double[3];
	}

	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 3; j++) {
			averages[i][j] = 0;
		}
	}

	groups(listOfObjects, averages);

	for (it = listOfObjects.begin(); it != listOfObjects.end(); it++) {
		it->toString();
		cout << endl;
	}
	
	cv::Mat test = cv::imread("images/test02.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (test.empty()) {
		printf("Unable to read input file (%s, %d).", __FILE__, __LINE__);
	}
	tresholding(test);

	cv::Mat testIndexes(cv::Size(test.cols, test.rows), CV_8UC1, cv::Scalar(0));
	index = 10;
	for (int y = 0; y < test.rows; y++) {
		for (int x = 0; x < test.cols; x++) {
			if (test.at<uchar>(y, x) == 255 && testIndexes.at<uchar>(y, x) == 0) {
				indexing(test, testIndexes, x, y, index);
				index += 10;
			}
		}
	}
	cv::imshow("img", testIndexes);
	int testObjectCount = (index - incrementation) / incrementation;
	
	labeling(testIndexes, testObjectCount, averages);
	
	kmeans(objectCount, listOfObjects);
	
	cv::waitKey(0); // wait until keypressed

	return 0;
}
