#include <time.h>
#include <math.h>
#include <cstdlib>
#include <stdio.h>
#include <iomanip>
#include <iostream>
#include <signal.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <libfreenect2/logger.h>
#include <libfreenect2/registration.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/packet_pipeline.h>
#include <libfreenect2/frame_listener_impl.h>

using namespace cv;
using namespace std;
enum Processor { cl, gl, cpu };

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

int main(int argc, char *argv[])
{
    //! 为了调用方便,接下面变量定义为全局变量
    libfreenect2::Freenect2 freenect2;
    libfreenect2::Freenect2Device *dev = nullptr;
    libfreenect2::PacketPipeline *pipeline = nullptr;

    //! 检测kinect是否连接成功，并初始化传感器
    if(freenect2.enumerateDevices() == 0)
    {
        std::cout << "no device connected!" << std::endl;
        return -1;
    }

    //! 获得传感器序列号
    string serial = freenect2.getDefaultDeviceSerialNumber();
    if(serial == "")
        return -1;
    cout<<"The serial number is :"<<serial<<endl;

    //! 选择处理器：cpu、gl、cl。测试发现选择cpu处理帧率极低
    int depthProcessor = Processor::cl;
    if(depthProcessor == Processor::cpu)
    {
        if(!pipeline)
        {
            pipeline = new libfreenect2::CpuPacketPipeline();
        }
    }

    else if (depthProcessor == Processor::gl)
    {
#ifdef LIBFREENECT2_WITH_OPENGL_SUPPORT
        if(!pipeline)
            pipeline = new libfreenect2::OpenGLPacketPipeline();
#else
        std::cout << "OpenGL pipeline is not supported!" << std::endl;
#endif
    }

    else if (depthProcessor == Processor::cl)
    {
#ifdef LIBFREENECT2_WITH_OPENCL_SUPPORT
        if(!pipeline)
            pipeline = new libfreenect2::OpenCLPacketPipeline();
#else
        std::cout << "OpenCL pipeline is not supported!" << std::endl;
#endif
    }

    //! 打开设备
    if(pipeline)
    {
        dev = freenect2.openDevice(serial, pipeline);
    }
    else
    {
        dev = freenect2.openDevice(serial);
    }


    if(dev == 0)
    {
        std::cout << "failure opening device!" << std::endl;
        return -1;
    }


    //! 设置listener接收Color、Depth、Ir信息流
    libfreenect2::SyncMultiFrameListener listener(libfreenect2::Frame::Color |libfreenect2::Frame::Depth |libfreenect2::Frame::Ir);
    libfreenect2::FrameMap frames;
    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);

    //! 开始处理信息流
    dev->start();
    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

    //! 变量值设置
    libfreenect2::Registration* registration = new libfreenect2::Registration(dev->getIrCameraParams(), dev->getColorCameraParams());
    libfreenect2::Frame undistorted(512, 424, 4), registered(512, 424, 4), depth2rgb(1920, 1080 + 2, 4);
    Mat rgbmat, depthmat, irmat, depthmatUndistorted, rgbd, rgbd2;

    float x, y, z, color;

    //! 创建一个显示点云的窗口
    pcl::visualization::CloudViewer viewer ("Point Cloud Viewer");

    //! 循环接收信息流(Esc键退出)
    while(!(waitKey(1)==27))
    {
        listener.waitForNewFrame(frames);
        libfreenect2::Frame *rgb = frames[libfreenect2::Frame::Color];
        libfreenect2::Frame *ir = frames[libfreenect2::Frame::Ir];
        libfreenect2::Frame *depth = frames[libfreenect2::Frame::Depth];

        //! 使用opencv：定义rgbmat、irmat、depthmat
        Mat rgbmat = cv::Mat(rgb->height, rgb->width, CV_8UC4, rgb->data);
        Mat irmat = cv::Mat(ir->height, ir->width, CV_32FC1, ir->data);
        Mat depthmat =cv::Mat(depth->height, depth->width, CV_32FC1, depth->data);

        PointCloud::Ptr cloud ( new PointCloud ); //使用智能指针，创建一个空点云。这种指针用完会自动释放。
        for (int m = 0;  m < 512 ; m++)
        {
            for (int n = 0 ; n < 424 ; n++)
            {
                PointT p;
                registration->getPointXYZRGB(&undistorted, &registered, n, m, x, y, z, color);
                const uint8_t *c = reinterpret_cast<uint8_t*>(&color);
                uint8_t b = c[0];
                uint8_t g = c[1];
                uint8_t r = c[2];
                if (z<1.0 && y<0.2)  //暂时先通过限定xyz来除去不需要的点(其中z的最小值应大于0.5m，一般取1m左右)，后续可以使用点云分割
                {
                    p.z = -z;
                    p.x = x;
                    p.y = -y;
                    p.b = b;
                    p.g = g;
                    p.r = r;
                }
                cloud->points.push_back( p );
            }
        }

        viewer.showCloud (cloud);

        //! imshow显示信息流
        cv::imshow("RGB", rgbmat);
        cv::imshow("IR", irmat / 4500.0f);
        cv::imshow("Depth", depthmat / 4500.0f);

        // //! [registration]
         registration->apply(rgb, depth, &undistorted, &registered, true, &depth2rgb);
        // //! [registration]

        // cv::Mat(undistorted.height, undistorted.width, CV_32FC1, undistorted.data).copytTo(depthmatUndistorted);
        // cv::Mat(registered.height, registered.width, CV_8UC4, registered.data).copyTo(rgbd);
        // cv::Mat(depth2rgb.height, depth2rgb.width, CV_32FC1, depth2rgb.data).copyTo(rgbd2);


        // cv::imshow("undistorted", depthmatUndistorted / 4500.0f);
        // cv::imshow("registered", rgbd);
        // cv::imshow("depth2RGB", rgbd2 / 4500.0f);


        listener.release(frames);
    }


    //! 结束数据处理
    dev->stop();

    //! 关闭设备
    dev->close();

    delete registration;

    std::cout << "Goodbye World!" << std::endl;
    return 0;
}
