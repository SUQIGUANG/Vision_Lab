/*
PCL中实现欧式聚类提取。
对三维点云组成的场景进行分割
桌子平面上的点云团　使用　欧式聚类的算法　kd树搜索　对点云聚类分割

1. 半径滤波(统计学滤波)删除离群点　体素格下采样等
2. 采样一致找到桌面（平面）或者除去滤波
3. 提取除去平面内点的　外点　（桌上的物体就自然成了一个个的浮空点云团）
4. 欧式聚类　提取出我们想要识别的东西

基于欧式距离的分割和基于区域生长的分割本质上都是用区分邻里关系远近来完成的。
由于点云数据提供了更高维度的数据，故有很多信息可以提取获得。欧几里得算法使用邻居
之间距离作为判定标准，而区域生长算法则利用了法线，曲率，颜色等信息来判断点云是否应该聚成一类。


*/

#include <pcl/ModelCoefficients.h>//模型系数
#include <pcl/point_types.h>//点云基本类型
#include <pcl/io/pcd_io.h>//io
#include <pcl/filters/extract_indices.h>//根据索引提取点云
#include <pcl/filters/voxel_grid.h>//体素格下采样
#include <pcl/features/normal_3d.h>//点云法线特征
#include <pcl/kdtree/kdtree.h>//kd树搜索算法
#include <pcl/sample_consensus/method_types.h>//采样方法
#include <pcl/sample_consensus/model_types.h>//采样模型
#include <pcl/segmentation/sac_segmentation.h>//随机采用分割
#include <pcl/segmentation/extract_clusters.h>//欧式聚类分割
#include <pcl/visualization/pcl_visualizer.h> // 可视化

/******************************************************************************
 打开点云数据，并对点云进行滤波重采样预处理，然后采用平面分割模型对点云进行分割处理
 提取出点云中所有在平面上的点集，并将其存盘
******************************************************************************/
int main (int argc, char** argv)
{
  // 读取桌面场景点云
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    reader.read ("../../Filtering/table_scene_lms400.pcd", *cloud);
    std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl;

    // 初始化滤波器
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);

    // 体素格滤波下采样，设置叶子（网格）大小为1cm×1cm×1cm
    vg.setInputCloud (cloud);
    vg.setLeafSize (0.01f, 0.01f, 0.01f);
    vg.filter (*cloud_filtered);

    // 输出滤波后点云数量
    std::cout << "PointCloud after filtering has: " <<
	       	     cloud_filtered->points.size ()     <<
		         " data points." << std::endl;

    // 创建平面模型分割的对象并设置参数
    // SACSegmentation：Sample Consensus采样一致性分割
    // PointIndices：点云索引库
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);  //系数
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::PCDWriter writer;

    seg.setOptimizeCoefficients (true);        //Set to true if a coefficient refinement is required.
    seg.setModelType (pcl::SACMODEL_PLANE);    //分割模型设定：平面模型
    seg.setMethodType (pcl::SAC_RANSAC);       //SAC方法设定：随机采样一致性。除此之外还可以用
                                               //SAC_LMEDS（最小平方中值）、SAC_MSAC、SAC_RRANSAC等
    seg.setMaxIterations (100);                //最大的迭代的次数
    seg.setDistanceThreshold (0.02);           //设置符合模型的内点　阀值

    int i=0, nr_points = (int) cloud_filtered->points.size ();//下采样前点云数量

    //模型分割直到剩余点云数量在30%以上，确保模型点云较好
    while (cloud_filtered->points.size () > 0.3 * nr_points)
    {
      // Segment the largest planar component from the remaining cloud
      seg.setInputCloud (cloud_filtered);
      seg.segment (*inliers, *coefficients);//分割

      if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

      pcl::ExtractIndices<pcl::PointXYZ> extract;  //按索引提取点云
      extract.setInputCloud (cloud_filtered);
      extract.setIndices (inliers);                //提取符合平面模型的内点
      extract.setNegative (false);

      // 平面模型内点
      extract.filter (*cloud_plane);
      std::cout << "PointCloud representing the planar component: "
              << cloud_plane->points.size ()
              << " data points." << std::endl;

      // 移去平面内点，提取剩余点云
      extract.setNegative (true);
      extract.filter (*cloud_f);
      *cloud_filtered = *cloud_f;//剩余点云
    }

    // 桌子平面上的点云团使用欧式聚类的算法、kd树搜索，对点云聚类分割
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);  // Kdtree作为点云搜索方法
    tree->setInputCloud (cloud_filtered);             // 桌子平面上其他的点云
    std::vector<pcl::PointIndices> cluster_indices;   // 定义点云聚类索引
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;// 欧式聚类对象
    ec.setClusterTolerance (0.02);                    // 设置近邻搜索的搜索半径为2cm
    ec.setMinClusterSize (100);                       // 设置一个聚类需要的最少的点数目为100
    ec.setMaxClusterSize (25000);                     // 设置一个聚类需要的最大点数目为25000
    ec.setSearchMethod (tree);                        // 设置点云的搜索机制
    ec.setInputCloud (cloud_filtered);
    ec.extract (cluster_indices);                     // 从点云中提取聚类，并将点云索引保存在cluster_indices中


    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_all (new pcl::PointCloud<pcl::PointXYZ>);

    //迭代访问点云索引cluster_indices,直到分割处所有聚类
    int j = 0;
    std::vector<pcl::PointIndices>::const_iterator it;

    for (it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
        std::vector<int>::const_iterator pit;

        for (pit = it->indices.begin (); pit != it->indices.end (); ++pit)
    	    cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //获取每一个点云团　的　点

        cloud_cluster->width = cloud_cluster->points.size ();
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;

        // 输出聚类后的点云大小
        std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;

        // 输出聚类后的点云文件到文件夹
        std::stringstream ss;
        ss << "cloud_cluster_" << j << ".pcd";
        writer.write<pcl::PointXYZ> (ss.str (), *cloud_cluster, false); //*
        j++;

        *cloud_cluster_all += *cloud_cluster;
    }

    pcl::io::savePCDFileASCII("cloud_cluster_all.pcd", *cloud_cluster_all);


    // 3D点云显示 绿色
    pcl::visualization::PCLVisualizer viewer ("Clustering Viewer");
    viewer.setBackgroundColor (0, 0, 0);//背景颜色　白色
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters ();

    // 平面上的点云　红色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_plane_handler(cloud_plane, 255, 0, 0);
    viewer.addPointCloud (cloud_plane, cloud_plane_handler, "plan point");//添加点云
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "plan point");

    // 桌上的物体就自然成了一个个的浮空点云团　绿色
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_cluster_handler(cloud_cluster_all, 0, 255, 0);
    viewer.addPointCloud (cloud_cluster_all, cloud_cluster_handler, "cloud_cluster point");//添加点云
    viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud_cluster point");

      while (!viewer.wasStopped())
      {
          viewer.spinOnce(100);
          boost::this_thread::sleep(boost::posix_time::microseconds(100000));
      }

    return (0);
}





