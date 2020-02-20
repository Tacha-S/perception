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
#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#endif

static std::string prefix = "/media/jessy/Data/dataset/models/008_pudding_box/";

cuda_renderer::Model model(prefix+"textured.ply");
float table_height;
float kCameraFX=768.1605834960938;
float kCameraFY=768.1605834960938;
float kCameraCX=480;
float kCameraCY=270;
cv::Mat cam_intrinsic=(cv::Mat_<float>(3,3) << kCameraFX, 0.0, kCameraCX, 0.0, kCameraFY, kCameraCY, 0.0, 0.0, 1.0);
Eigen::Matrix4d cam_intrinsic_eigen;
Eigen::Isometry3d cam_to_world_;
Eigen::Matrix4d cam_matrix;
int width = 960;
int height = 540;
cv::Mat background_image;
cv::Mat origin_image;
cv::Mat cv_input_color_image;


float x_min,x_max,y_min,y_max;
float res,theta_res;
cuda_renderer::Model::mat4x4 proj_mat = cuda_renderer::compute_proj(cam_intrinsic, width, height);
std::vector<cuda_renderer::Model::mat4x4> trans_mat;

std::vector<int> estimate_score;
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

std::vector<Pose> Pose_list;