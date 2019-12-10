#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"


using namespace cv;
int *difffilter(const cv::Mat& input,const cv::Mat& input1, cv::Mat& output);
int main (int argc, char* argv[])
{
    try
    {
        cv::Mat src_host = cv::imread("/home/jessy/cv/000000.left.png", cv::IMREAD_COLOR);
        cv::Mat src_host1 = cv::imread("/home/jessy/cv/output_color_image.png", cv::IMREAD_COLOR);
        cv::Mat hsv;
        cv::Mat hsv1;
        cv::cvtColor(src_host,hsv,cv::COLOR_BGR2HSV);
        cv::cvtColor(src_host1,hsv1,cv::COLOR_BGR2HSV);
        cv::Mat destiny = cv::Mat::zeros( src_host.size(), CV_8UC1);
        difffilter(hsv,hsv1,destiny);

        //std::cout << "M = "<< std::endl << " "  << destiny << std::endl << std::endl;
        double cost;
        cost = sum(destiny)[0];
        std::cout<<cost;
        //cv::imshow("Input",src_host);
        //cv::imshow("Output",destiny);
        
        //Wait for key press
        //cv::waitKey();
    }
    catch(const cv::Exception& ex)
    {
        std::cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}
// cv::Mat hsv;
    // cv::Mat hsv1;
    // cv::cvtColor(*cv_color_image,hsv,cv::COLOR_BGR2HSV);
    // cv::cvtColor(cv_input_color_image,hsv1,cv::COLOR_BGR2HSV);
    // cv::Mat destiny = cv::Mat::zeros( cv_color_image->size(), CV_8UC1);
    // difffilter(hsv,hsv1,destiny);
    // cost = sum(destiny)[0];
    