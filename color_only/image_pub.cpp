#include <ros/ros.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_publisher");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<sensor_msgs::Image>("color_only_image", 1);
  ros::Rate loop_rate(1);
  for(int i=23; i <25; i ++){
    std::string num = std::to_string(i);
    std::string a = "/media/jessy/Data/dataset/Zed/NewMap_OBJ4_2/0000"+num+".left.png";
    cv::Mat image = cv::imread(a, CV_LOAD_IMAGE_COLOR);
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image).toImageMsg();
    msg->header.frame_id = num; 
    pub.publish(msg);
    ros::spinOnce();
    loop_rate.sleep();
  }
  

  // ros::Rate loop_rate(5);
  // // while (nh.ok()) {
  //   pub.publish(msg);
  //   ros::spinOnce();
  //   loop_rate.sleep();
  // }
}