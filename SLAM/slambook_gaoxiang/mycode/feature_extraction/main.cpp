#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    if(argc != 3)
    {
        cout<<"--usage: ./feature_extraction 1.png 2.png";
        return 1;
    }

    // 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    // 初始化
    vector<KeyPoint> keypoint_1, keypoint_2;
    Mat descriptor_1, descriptor_2;

    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 第一步：计算FAST角点位置
    detector->detect(img_1,keypoint_1);
    detector->detect(img_2,keypoint_2);

    //第二步：根据角点位置计算BEIEF描述子
    descriptor->compute(img_1,keypoint_1,descriptor_1);
    descriptor->compute(img_2,keypoint_2,descriptor_2);

    // 显示角点
    Mat outimg_1, outimg_2;
    drawKeypoints(img_1,keypoint_1,outimg_1);
    drawKeypoints(img_2,keypoint_2,outimg_2);
    imshow("ORB features of img_1",outimg_1);
    imshow("ORB features of img_2",outimg_2);

    // 第三步：将两幅图像的描述子用Hamming距离进行匹配
    vector<DMatch> matches;
    matcher->match(descriptor_1,descriptor_2,matches);

    // 第四步：筛选匹配点对，距离范围0-10000
    double min_dist=10000, max_dist=0;

    for (int i = 0; i < descriptor_1.rows; i++)
    {
        double dist=matches[i].distance;
        if (dist<min_dist)
            min_dist=dist;
        if (dist>max_dist)
            max_dist=dist;
    }

    printf("Max dist: %f \n",max_dist);
    printf("Min dist: %f \n",min_dist);


    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    vector<DMatch> good_matches;
    for (int i = 0; i < descriptor_1.rows; i++)
    {
        if (matches[i].distance<=max(2*min_dist,30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }

    // 第五步：绘制匹配结果
    Mat outimg_match;
    Mat outimg_goodmatch;

    drawMatches(img_1,keypoint_1,img_2,keypoint_2,matches,outimg_match);
    drawMatches(img_1,keypoint_1,img_2,keypoint_2,good_matches,outimg_goodmatch);

    imshow("matched points", outimg_match);
    imshow("matched points after optimition",outimg_goodmatch);

    waitKey(0);

    return (0);
}