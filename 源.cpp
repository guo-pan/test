#include<opencv2\core\core.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\highgui\highgui.hpp>
#include"main.h"
#include<cmath>
#include<iostream>
using namespace std;
using namespace cv;
class h_f
{
private:
	int h_s[1];//项的数量
	float h_r[2];//像素的最小和最大值
	const float*ranges[1];
	int channnels[1];//只用到一个通道
public:
	 h_f()//构造函数说明和定义在一起
	{
			h_s[0] = 256;
			h_r[0] = 0.0;
		   h_r[1] = 255.0;
		   ranges[0] = h_r;
			channnels[0] = 0;
	}
	 ~h_f(){}
	cv::MatND getHistogram(const cv::Mat &image)
	{
		cv::MatND hist;
		cv::calcHist(&image, 1, channnels, cv::Mat(),hist,1,h_s,ranges);
		return hist;
	}
};
struct area
{
	double a, b, c;
	double area_a()
	{
		double s = (a + b + c) / 2;
		return sqrt(s*(s-a)*(s-b)*(s-c));
	}

};
class contour_1
{
public:
	cv::Point cot_point;
	int lable;
public:
	contour_1()
	{
		lable = 10000;
	}
	~contour_1()
	{}
};
void blur_mean(cv::Mat &image, cv::Mat &result,int num)
{
	
	cv::Mat kernel(num, num, CV_32F, cv::Scalar(0));
	int a = num*num;
//	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	for (size_t i = 0; i < num; i++)
	{
		for (size_t j = 0; j < num; j++)
		{
			kernel.at<float>(i, j) = 1.0 / a;
		}
	}
	/*kernel.at<float>(0, 1) = 1.0 / 9;
	kernel.at<float>(1, 1) = 1.0 / 9;
	kernel.at<float>(2, 1) = 1.0 / 9;
	kernel.at<float>(0, 0) = 1.0 / 9;
	kernel.at<float>(1, 0) = 1.0 / 9;
	kernel.at<float>(2, 0) = 1.0 / 9;
	kernel.at<float>(2, 2) = 1.0 / 9;
	kernel.at<float>(1, 2) = 1.0 / 9;
	kernel.at<float>(0, 2) = 1.0 / 9;*/
	cv::filter2D(image, result, image.depth(), kernel);
}
void sharpen2D(cv::Mat &image,cv::Mat &result)
{
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;
	cv::filter2D(image, result, image.depth(), kernel);
}
void sobel_x(cv::Mat &image, cv::Mat &result)
{
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 1.0;
	kernel.at<float>(0, 0) = -1.0;
	kernel.at<float>(0, 2) = 1.0;
	kernel.at<float>(1, 0) = -2.0;
	kernel.at<float>(1, 2) = 2.0;
	kernel.at<float>(2, 0) = -1.0;
	kernel.at<float>(2, 2) = 1.0;
	cv::filter2D(image, result, image.depth(), kernel);
}
void sobel_y(cv::Mat &image, cv::Mat &result)
{
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 1.0;
	kernel.at<float>(0, 0) = -1.0;
	kernel.at<float>(2, 0) = 1.0;
	kernel.at<float>(0, 1) = -2.0;
	kernel.at<float>(2, 1) = 2.0;
	kernel.at<float>(0, 2) = -1.0;
	kernel.at<float>(2, 2) = 1.0;
	cv::filter2D(image, result, image.depth(), kernel);
}
void r_y(cv::Mat &image, cv::Mat &result)
{
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 0.0;
	kernel.at<float>(0, 0) = 0.0;
	kernel.at<float>(0, 2) = 0.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = 1.0;
	kernel.at<float>(2, 0) = 0.0;
	kernel.at<float>(2, 2) = 0.0;
	cv::filter2D(image, result, image.depth(), kernel);
}
void c_x(cv::Mat &image, cv::Mat &result)
{
	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 0.0;
	kernel.at<float>(0, 0) = 0.0;
	kernel.at<float>(2, 0) = 0.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(2, 1) = 1.0;
	kernel.at<float>(0, 2) = 0.0;
	kernel.at<float>(2, 2) = 0.0;
	cv::filter2D(image, result, image.depth(), kernel);
}
void houghline(cv::Mat &image)
{
	cv::Mat img_1;
	cv::cvtColor(image,img_1,COLOR_GRAY2BGR);
	std::vector<cv::Vec4i>lines;
	cv::HoughLinesP(image,lines,1,CV_PI/180,700,1000,10000);
	for (size_t i = 0; i < lines.size();i++)
	{
		cv::Vec4i l = lines[i];
		cv::line(image, Point(l[0], l[1]), Point(l[2], l[3]), 160, 1, 0);
	}
}
void lable(cv::Mat &image)//需要改善
{
	int *c_s = new int[300000];
	std::vector<cv::Point> S[10000];
	int a_lable = 1;
	cv::Mat img_lable = image;
	for (size_t i = 0; i < img_lable.rows; i++)
	{
		uchar *p = img_lable.ptr<uchar>(i);
		for (size_t j = 0; j < img_lable.cols; j++)
		{
			if (i == 1)//判断第一行，给第一行贴标签
			{
				if (p[j] == 0)
				{
					if (j == 0)
					{
						//	contour[i*img_lable.rows + j].lable = a_lable;
						c_s[i*img_lable.rows + j] = a_lable;
						//	contour[i*img_lable.rows + j].cot_point = cv::Point(j,i);		
						S[a_lable].push_back(cv::Point(j, i));
						a_lable++;
						continue;
					}
					//	contour.push_back();
					if (j>1 && p[j - 1] == 0)
					{
						//contour[i*img_lable.rows + j].lable = contour[i*img_lable.rows + j - 1].lable;
						c_s[i*img_lable.rows + j] = c_s[i*img_lable.rows + j - 1];
						//contour[i*img_lable.rows + j].cot_point = cv::Point(j, i);
						S[c_s[i*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
						continue;
					}
					if (j > 1 && p[j - 1] != 0)
					{
						c_s[i*img_lable.rows + j] = a_lable;
						S[a_lable].push_back(cv::Point(j, i));
						a_lable++;
						continue;
					}
				}
			}
			if (i != 1)
			{
				uchar *p1 = img_lable.ptr<uchar>(i - 1);
				if (p[j] == 0)
				{
					if (j == 0)
					{
						if (p1[j] == 0)
						{
							c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j];
							S[c_s[(i - 1)*img_lable.rows + j]].push_back(cv::Point(j, i));
							continue;
						}
						if (p1[j + 1] == 0)
						{
							c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j + 1];
							S[c_s[(i - 1)*img_lable.rows + j + 1]].push_back(cv::Point(j, i));
							continue;
						}
						if (p1[j] != 0 && p1[j + 1] != 0)
						{
							c_s[i*img_lable.rows + j] = a_lable;
							S[a_lable].push_back(cv::Point(j, i));
							a_lable++;
							continue;
						}
					}
					if (j == (img_lable.cols - 1))
					{
						if (p1[j] != 0 && p1[j + 1] != 0)
						{
							c_s[i*img_lable.rows + j] = a_lable;
							S[a_lable].push_back(cv::Point(j, i));
							a_lable++;
							continue;
						}
						else
						{
							c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j];
							S[c_s[(i - 1)*img_lable.rows + j]].push_back(cv::Point(j, i));
							continue;
						}
					}
					if ((j != 0) && (j != (img_lable.cols - 1)))
					{
						if (p1[j] != 0 && p1[j + 1] != 0 && p1[j - 1] != 0 && p[j - 1] != 0)
						{
							c_s[i*img_lable.rows + j] = a_lable;
							S[a_lable].push_back(cv::Point(j, i));
							a_lable++;
							continue;
						}
						if (p1[j] == 0)
						{
							c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j];
							S[c_s[(i - 1)*img_lable.rows + j]].push_back(cv::Point(j, i));
							continue;
						}
						if ((p1[j] != 0) && (p1[j + 1] != 0) && (p1[j - 1] == 0 || p[j - 1] == 0))//上，右上！=1；
						{
							if (p1[j - 1] == 0)
							{
								c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j - 1];
								S[c_s[(i - 1)*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
								continue;
							}
							if (p[j - 1] == 0)
							{
								c_s[i*img_lable.rows + j] = c_s[i*img_lable.rows + j - 1];
								//contour[i*img_lable.rows + j].cot_point = cv::Point(j, i);
								S[c_s[i*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
								continue;
							}
						}
						if ((p1[j] != 0) && (p1[j + 1] == 0) && (p1[j - 1] == 0 || p[j - 1] == 0))//
						{
							if (p1[j - 1] == 0)//判断左上和右上
							{
								if (c_s[(i - 1)*img_lable.rows + j - 1] == c_s[(i - 1)*img_lable.rows + j + 1])
								{
									c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j - 1];
									S[c_s[(i - 1)*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
									continue;
								}
								if (c_s[(i - 1)*img_lable.rows + j - 1] > c_s[(i - 1)*img_lable.rows + j + 1])
								{
									c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j + 1];
									for (size_t i1 = 0; i1 < S[c_s[(i - 1)*img_lable.rows + j - 1]].size(); i1++)
									{
										c_s[S[c_s[(i - 1)*img_lable.rows + j - 1]][i1].x + S[c_s[(i - 1)*img_lable.rows + j - 1]][i1].y*img_lable.rows] = c_s[(i - 1)*img_lable.rows + j + 1];
										//	S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y*img_lable.rows+S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y
										S[c_s[(i - 1)*img_lable.rows + j + 1]].push_back(S[c_s[(i - 1)*img_lable.rows + j - 1]][i1]);
									}
									S[c_s[(i - 1)*img_lable.rows + j + 1]].push_back(cv::Point(j, i));
									continue;
								}
								if (c_s[(i - 1)*img_lable.rows + j - 1] < c_s[(i - 1)*img_lable.rows + j + 1])
								{
									c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j - 1];
									for (size_t i1 = 0; i1 < S[c_s[(i - 1)*img_lable.rows + j + 1]].size(); i1++)
									{
										c_s[S[c_s[(i - 1)*img_lable.rows + j + 1]][i1].x + S[c_s[(i - 1)*img_lable.rows + j + 1]][i1].y*img_lable.rows] = c_s[(i - 1)*img_lable.rows + j - 1];
										//	S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y*img_lable.rows+S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y
										S[c_s[(i - 1)*img_lable.rows + j - 1]].push_back(S[c_s[(i - 1)*img_lable.rows + j + 1]][i1]);
									}
									S[c_s[(i - 1)*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
									continue;
								}
							}
							if (p[j - 1] == 0)//判断左上和右上
							{
								if (c_s[(i)*img_lable.rows + j - 1] == c_s[(i - 1)*img_lable.rows + j + 1])
								{
									c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j - 1];
									S[c_s[(i - 1)*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
									continue;
								}
								if (c_s[(i)*img_lable.rows + j - 1] > c_s[(i - 1)*img_lable.rows + j + 1])
								{
									c_s[i*img_lable.rows + j] = c_s[(i - 1)*img_lable.rows + j + 1];
									for (size_t i1 = 0; i1 < S[c_s[(i)*img_lable.rows + j - 1]].size(); i1++)
									{
										c_s[S[c_s[(i)*img_lable.rows + j - 1]][i1].x + S[c_s[(i)*img_lable.rows + j - 1]][i1].y*img_lable.rows] = c_s[(i - 1)*img_lable.rows + j + 1];
										//	S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y*img_lable.rows+S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y
										S[c_s[(i - 1)*img_lable.rows + j + 1]].push_back(S[c_s[(i)*img_lable.rows + j - 1]][i1]);
									}
									S[c_s[(i - 1)*img_lable.rows + j + 1]].push_back(cv::Point(j, i));
									continue;
								}
								if (c_s[(i)*img_lable.rows + j - 1] < c_s[(i - 1)*img_lable.rows + j + 1])
								{
									c_s[i*img_lable.rows + j] = c_s[(i)*img_lable.rows + j - 1];
									for (size_t i1 = 0; i1 < S[c_s[(i - 1)*img_lable.rows + j + 1]].size(); i1++)
									{
										c_s[S[c_s[(i - 1)*img_lable.rows + j + 1]][i1].x + S[c_s[(i - 1)*img_lable.rows + j + 1]][i1].y*img_lable.rows] = c_s[(i)*img_lable.rows + j - 1];
										//	S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y*img_lable.rows+S[contour[(i - 1)*img_lable.rows + j - 1].lable][i1].y
										S[c_s[(i)*img_lable.rows + j - 1]].push_back(S[c_s[(i - 1)*img_lable.rows + j + 1]][i1]);
									}
									S[c_s[(i)*img_lable.rows + j - 1]].push_back(cv::Point(j, i));
									continue;
								}
							}
						}
					}
				}
			}

		}
	}
	delete[] c_s;
}
int main()
{
	cv::Mat img_or(2000, 2000, CV_8UC1, cv::Scalar(0));
	int d = 300;
	for (int i1 = 0; i1 < 4; i1++)
	{
		for (int i = 0; i < img_or.rows; i++)
		{
			for (int j = 100+400 * i1; j <400 * i1+500; j++)
			{
				if ((j-400*i1)<300)
				{
					img_or.at<uchar>(i, j) = 255;
				}
				else 	img_or.at<uchar>(i, j) = 0;
			}
		}
	}
	/*cv::Mat image_x=cv::imread("G:\\cout1.bmp");
	cv::Mat image_x1 = cv::imread("G:\\out4.bmp");*/
	//cv::Rect ss1; 
	//cv::Mat image_x1 = image_x.clone();
	//cv::floodFill(image_x, Point(224, 748), 0, &ss1, 7, 7);
	//int *a = new int[1000000];
	//std::vector<cv::Point>counter212[10000];
	////std::vector<cv::Point>counter212[10000];
	//a[0] = 1;
	/*counter212[a[0]].push_back(cv::Point(2, 3));
	cv::Point p4 = counter212[a[0]][0];
	delete[] a;*/
//	delete[] counter212;
	cv::Mat img1 = cv::imread("G:\\img3.bmp", 0);
	cv::Mat img2, img3, img4;
	cv::Canny(img1, img2, 50, 100, 3);
	std::vector<std::vector<cv::Point>>point_counter;//双层
	std::vector<cv::Point>cou;
	std::vector<cv::Point>cou1;
	std::vector<cv::Point>cou2[2];
	cou.push_back(cv::Point(2,3));//创建两个
	cou.push_back(cv::Point(3, 3));
	cou1.push_back(cv::Point(5, 3));
	cou2[0].push_back(cv::Point(2, 3));//创建两个
	cou2[0].push_back(cv::Point(3, 3));
	cou2[1].push_back(cv::Point(5, 3));
	//point_counter[0].push_back(cv::Point(1, 2));
	point_counter.push_back(cou);
	point_counter.push_back(cou1);
	cv::Point p = point_counter[0][0];
	cv::Point p1 = point_counter[1][0];
	cv::Point p2 = cou2[0][0];
	cv::Point p3 = cou2[1][0];
	area m;
	//cin >> m.a >> m.b >> m.c;
	//cout << "三角形面积" << m.area_a() << endl;
	cv::Mat image1 = cv::imread("G:\\2.bmp",0);
	cv::Mat image1_f = image1.clone();
	cv::Rect ss;
	cv::floodFill(image1_f,Point(110,100),140,&ss,3,3);
	cv::Mat img_x, img_y;
	c_x(image1, img_x);
	r_y(image1, img_y);
	cv::Mat img_1 = cv::imread("G:\\img1.bmp",0);
	cv::Mat image1_1 = cv::imread("G:\\2.bmp", 1);
	cv::imshow("b", image1);
	cv::Mat image2, image3, image4, image5, image6,image7,image8,image9,image10,image11;
	blur_mean(image1_1, image8,5);//均值滤波
	cv::blur(image1_1, image7,cv::Size(5,5));//opencv中自带均值滤波；
	cv::Mat gauss = cv::getGaussianKernel(9,0,CV_32F); 
	cv::filter2D(image1_1, image9, image1_1.depth(), gauss);
	cv::GaussianBlur(image1_1, image10,cv::Size(3,3),0,0);//高斯滤波
	cv::bilateralFilter(img_1, image11, 25, 25 * 2, 25 / 2);
	cv::Mat a_img = img_1 - image11 + img_1;
	cv::medianBlur(image1, image3,5);//中值滤波
	sobel_x(image1, image4);//sobel_x滤波
	sobel_y(image1, image5);//sobel_y滤波
	cv::addWeighted(image4, 0.5, image5,0.5,0,image6);
	sharpen2D(image1,image2);//laplacian滤波
	
	houghline(img2);//hough变换
	//cv::cvtColor(img2);
	cv::threshold(img2,img3,128,255,cv::THRESH_BINARY_INV);
	cv::imshow("a",image2);
	//h_f h;
	MyClass h;
	cv::MatND histo = h.getHistogram(image1);
	for (int i = 0; i < 256; i++)
	cout << "Value" << i << "=" << histo.at<float>(i) << endl;
	cv::Mat img_lable_1 = cv::imread("G://学习文件//光电图像处理//常用图像//31.bmp",0);
	//cv::vector<contour_1>contour[10000000];
	//contour_1   contour[300000];//存储像素点的标签   
	cv::waitKey();
}