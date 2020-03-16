#include <cuda_renderer/renderer.h>
#include <hist_prioritize/hist.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <Eigen/Dense>
#include <angles/angles.h>
#include <ros/ros.h>

#include <pcl/io/ply_io.h>
#include <pcl/common/io.h>
#include <pcl/io/ascii_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/common/common.h>
#include <pcl/conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/PolygonMesh.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

// #include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#endif

static std::string prefix = "/media/jessy/Data/dataset/models/008_pudding_box/";

class Pose
{
    public:
        Pose(double x, double y, double z, double roll, double pitch, double yaw);
        Eigen::Isometry3d GetTransform() const;
    
        double x_ = 0.0;
        double y_ = 0.0;
        double z_ = 0.0;
        double roll_ = 0.0;
        double pitch_ = 0.0;
        double yaw_ = 0.0;
};
Eigen::Isometry3d Pose::GetTransform() const {
  const Eigen::AngleAxisd roll_angle(roll_, Eigen::Vector3d::UnitX());
  const Eigen::AngleAxisd pitch_angle(pitch_, Eigen::Vector3d::UnitY());
  const Eigen::AngleAxisd yaw_angle(yaw_, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond quaternion;
  quaternion = yaw_angle * pitch_angle * roll_angle;
  quaternion.normalize();
  const Eigen::Isometry3d transform(Eigen::Translation3d(x_, y_, z_) * quaternion);
  return transform;
}
Pose::Pose(double x, double y, double z, double roll, double pitch,
                   double yaw) : x_(x), y_(y), z_(z),
  roll_(angles::normalize_angle_positive(roll)),
  pitch_(angles::normalize_angle_positive(pitch)),
  yaw_(angles::normalize_angle_positive(yaw)) {
};



class color_only
{
private:
  ros::NodeHandle n;
  ros::Subscriber sub;

public:

  color_only();
  
  std::vector<cuda_renderer::Model> models;
  std::vector<float> mode_trans;
  float table_height;
  cv::Mat cam_intrinsic;
  Eigen::Matrix4d cam_intrinsic_eigen;
  Eigen::Isometry3d cam_to_world_;
  Eigen::Matrix4d cam_matrix;
  int width;
  int height;
  cv::Mat background_image;
  cv::Mat origin_image;
  cv::Mat cv_input_color_image;

  float x_min,x_max,y_min,y_max;
  float res,theta_res;
  cuda_renderer::Model::mat4x4 proj_mat;
  std::vector<cuda_renderer::Model::mat4x4> trans_mat;
  std::vector<int> estimate_score;
  std::vector<Pose> Pose_list;
  // for hist comparison
  std::vector<int> hist_total;
  std::vector<float> gpu_bb;
  std::vector<std::vector<float> > gpu_cam_m;
  // std::vector<cuda_renderer::Model::mat4x4> Predict(cv::Mat observed);
  void imageCallback(const sensor_msgs::ImagePtr& msg);
  void setinput();
  ~color_only();
  
};