#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat& img_1,
        const Mat& img_2,
        vector<KeyPoint>& keypoint_1,
        vector<KeyPoint>& keypoint_2,
        vector<DMatch>& matches);

Point2d pixel2cam(const Point2d& p,const Mat& K);

void pose_estimation_2d2d(
        vector<KeyPoint> keypoints_1,
        vector<KeyPoint> keypoints_2,
        vector<DMatch> matches,
        Mat& R,
        Mat& t);

int main(int argc, char** argv)
{
    if ( argc != 3 )
    {
        cout<<"usage: pose_estimation_2d2d img1 img2"<<endl;
        return 1;
    }

    //-- 读取图像，argv[1]就是1.png，argv[2]就是2.png
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    // keypoints_1, keypoints_2是特征点，matches是匹配点
    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;

    // 调用find_feature_matches()函数找到特征点和匹配点
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    //-- 调用pose_estimation_2d2d()函数估计两张图像间运动
    Mat R,t;
    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

    //-- 验证E=t^R*scale
    Mat t_x = ( Mat_<double> ( 3,3 ) <<
                                     0,                      -t.at<double> ( 2,0 ),     t.at<double> ( 1,0 ),
            t.at<double> ( 2,0 ),      0,                      -t.at<double> ( 0,0 ),
            -t.at<double> ( 1,0 ),     t.at<double> ( 0,0 ),      0 );

    cout<<"t^R="<<endl<<t_x*R<<endl;

    //-- 验证对极约束
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    for ( DMatch m: matches )
    {
        Point2d pt1 = pixel2cam ( keypoints_1[ m.queryIdx ].pt, K );
        Mat y1 = ( Mat_<double> ( 3,1 ) << pt1.x, pt1.y, 1 );
        Point2d pt2 = pixel2cam ( keypoints_2[ m.trainIdx ].pt, K );
        Mat y2 = ( Mat_<double> ( 3,1 ) << pt2.x, pt2.y, 1 );
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }

    return 0;
}


void find_feature_matches(
        const Mat& img_1,
        const Mat& img_2,
        vector<KeyPoint>& keypoint_1,
        vector<KeyPoint>& keypoint_2,
        vector<DMatch>& matches)
{
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
    vector<DMatch> match;
    matcher->match(descriptor_1,descriptor_2,matches);

    // 第四步：筛选匹配点对，距离范围0-10000
    double min_dist=10000, max_dist=0;

    for (int i = 0; i < descriptor_1.rows; i++)
    {
        double dist=match[i].distance;
        if (dist<min_dist)
            min_dist=dist;
        if (dist>max_dist)
            max_dist=dist;
    }

    printf("Max dist: %f \n",max_dist);
    printf("Min dist: %f \n",min_dist);


    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptor_1.rows; i++)
    {
        if (match[i].distance<=max(2*min_dist,30.0))
        {
            matches.push_back(match[i]);
        }
    }

    // 第五步：绘制匹配结果
    Mat outimg_match;
    Mat outimg_goodmatch;

    drawMatches(img_1,keypoint_1,img_2,keypoint_2,match,outimg_match);
    drawMatches(img_1,keypoint_1,img_2,keypoint_2,match,outimg_goodmatch);

    imshow("matched points", outimg_match);
    imshow("matched points after optimition",outimg_goodmatch);

}

Point2d pixel2cam(const Point2d& p,const Mat& K)
{
    return Point2d((p.x-K.at<double>(0,2))/K.at<double>(0,0),
                   (p.y-K.at<double>(1,2))/K.at<double>(1,2));
}

void pose_estimation_2d2d(
        vector<KeyPoint> keypoints_1,
        vector<KeyPoint> keypoints_2,
        vector<DMatch> matches,
        Mat& R,
        Mat& t
        )
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );


    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    // 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix=findFundamentalMat(points1,points2,CV_FM_8POINT);
    cout<<"fundamental matrix is: "<<endl<<fundamental_matrix<<endl;


    // 计算本质矩阵
    Point2d principal_point(325.1,249.7);    // 相机光心
    double focal_length=521;    // 相机焦距
    Mat essiential_matrix;
    essiential_matrix=findEssentialMat(points1,points2,focal_length,principal_point);
    cout<<"essiential matrix is : "<<endl<<essiential_matrix<<endl;


    // 计算单应矩阵
    Mat homography_matrix;
    homography_matrix=findHomography(points1,points2,RANSAC);
    cout<<"homography matrix is: "<<endl<<homography_matrix<<endl;


    // 从本质矩阵中恢复出旋转和平移向量
    recoverPose(essiential_matrix,points1,points2,R,t,focal_length,principal_point);
    cout<<"rotation matrix R is : "<<endl<<R<<endl;
    cout<<"translation matrix t is : "<<endl<<t<<endl;
}
