#include "./include/hist_prioritize/hist.h"
#include <chrono>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#ifdef CUDA_ON
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#endif

using namespace cv;
int main(int argc, char const *argv[])
{

    const size_t width = 960; const size_t height = 540;

    cv::Mat src_host = cv::imread("/home/jessy/cv/000000.left.png", cv::IMREAD_COLOR);
    std::vector<uint8_t> r_v;
    std::vector<uint8_t> g_v;
    std::vector<uint8_t> b_v;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b elem = src_host.at<cv::Vec3b>(y, x);
            r_v.push_back(elem[2]);
            g_v.push_back(elem[1]);
            b_v.push_back(elem[0]);
            
        }
    }
    // std::cout << "M = "<< std::endl << " "  << lab << std::endl << std::endl;
    std::vector<std::vector<uint8_t>> observed;
    observed.push_back(r_v);
    observed.push_back(g_v);
    observed.push_back(b_v);
    float x_min = 0;
    float x_max = 1;
    float y_min = 0;
    float y_max = 1;
    float theta_min = 0;
    float theta_max = 3.14;
    float trans_res = 0.3;
    float angle_res = 1;
    // std::vector<int> res = hist_prioritize::compare_hist(x_min,x_max,
    //                                                       y_min,y_max,
    //                                                       theta_min,theta_max,
    //                                                       trans_res, angle_res,
    //                                                       observed);
    return 0;


}