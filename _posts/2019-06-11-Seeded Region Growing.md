---
layout: post
title:  "Seeded Region Growing with codes"
date:   2019-06-11
excerpt: "A brief illustration about seeded region growing."
tag:
- opencv
- computer vision
comments: true
---

## Seeded Region Growing

**Boundary Constraint:** 

1. The growing points must stay in image plane or ROI.

2. The grown point, whether is successful or not, can not be calculate again.

**Growing Constraint:**

 The growing points should satisfy some necessary condition according the function of algorithm. Growing constraint  

eg: 
\\[ 
abs \mid gray(firstPoint) - gray(nextPoint)\mid < threshold
\\]

```c++
#include <opencv2/opencv.hpp>
#include <stack>
cv::Mat regionGrow(const cv::Mat &src, const cv::Point seed, int throld);
bool growingConstraint(const cv::Mat &src, cv::Point currentPoint, cv::Point nextPoint, int throld);

//Region Growing Method
cv::Mat regionGrow(const cv::Mat &src, const cv::Point seed, int throld)
{
	cv::Mat grayImg;
	cv::cvtColor(src, grayImg, cv::COLOR_RGB2GRAY);

	cv::Mat resultImg = cv::Mat::zeros(grayImg.size(), CV_8UC1);
	if (seed.x < 0 || seed.y < 0)
		return resultImg;

	resultImg.at<uchar>(seed.y, seed.x) = 255;
	const int growDirection[8][2] = 
	{ {-1,-1}, {0,-1}, {1,-1}, {1,0}, {1,1}, {0,1}, {-1,1}, {-1,0} };
	std::stack<cv::Point> grownPoints;
	grownPoints.push(seed);

	while (!grownPoints.empty())
	{
		cv::Point currentPoint = grownPoints.top();
		grownPoints.pop();
		for (int i = 0; i < 8; ++i)
		{
			//The position of next point
			cv::Point nextPoint(currentPoint.x + growDirection[i][0], 
                                currentPoint.y + growDirection[i][1]);
			//Growing constraint and Boundary constraint
			if (nextPoint.x >= 0 && nextPoint.x < grayImg.cols 
                && nextPoint.y >= 0 && nextPoint.y < grayImg.rows
				&& growingConstraint(grayImg, currentPoint, nextPoint, throld) 
				&& resultImg.at<uchar>(nextPoint.y, nextPoint.x) == 0)
			{
				resultImg.at<uchar>(nextPoint.y, nextPoint.x) = 255;
				grownPoints.push(nextPoint);
			}
		}
	}
	return resultImg;

}

//Growing Constraint
bool growingConstraint(const cv::Mat &src, 
                       cv::Point currentPoint, 
                       cv::Point nextPoint, int throld)
{
	int diff = abs(src.at<uchar>(currentPoint.y, currentPoint.x) 
                   - src.at<uchar>(nextPoint.y, nextPoint.x));
	if (diff < throld)
		return true;
	return false;

}

int main()
{
	cv::Mat img1 = cv::imread("test.png");
	cv::namedWindow("original image", cv::WINDOW_NORMAL);
	cv::imshow("original image", img1);

	cv::Mat img2 = regionGrow(img, cv::Point(30, 180), 10);
	cv::namedWindow("resulted image", cv::WINDOW_NORMAL);
	cv::imshow("resulted image", img2);

	cv::waitKey(0);

}
```

<video src="..\images\2019-06-11\SRG.mp4" width="75%" height="75%"></video>



