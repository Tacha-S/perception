#pragma once

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include "math.h"

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

#define USE_TREE 0
#define USE_CLUTTER 0
#define THREADS_PER_BLOCK 256
#define POINT_DIM 3

namespace cuda_renderer {

class Model{
public:
    Model();
    ~Model();

    Model(const std::string & fileName);

    const struct aiScene* scene;
    void LoadModel(const std::string & fileName);

    struct int3 {
        size_t v0;
        size_t v1;
        size_t v2;
    };

    struct ROI{
        size_t x;
        size_t y;
        size_t width;
        size_t height;
    };

    struct float3{
        float x;
        float y;
        float z;
        friend std::ostream& operator<<(std::ostream& os, const float3& dt)
        {
            os << dt.x << '\t' << dt.y << '\t' << dt.z << std::endl;
            return os;
        }
    };
    struct Triangle{
        float3 v0;
        float3 v1;
        float3 v2;
        int3 color;

        friend std::ostream& operator<<(std::ostream& os, const Triangle& dt)
        {
            os << dt.v0 << dt.v1 << dt.v2;
            return os;
        }
    };
    struct mat4x4{
        float a0=1; float a1=0; float a2=0; float a3=0;
        float b0=0; float b1=1; float b2=0; float b3=0;
        float c0=0; float c1=0; float c2=1; float c3=0;
        float d0=0; float d1=0; float d2=0; float d3=1;

        void t(){
            float temp;
            temp = a1; a1=b0; b0=temp;
            temp = a2; a2=c0; c0=temp;
            temp = a3; a3=d0; d0=temp;
            temp = b2; b2=c1; c1=temp;
            temp = b3; b3=d1; d1=temp;
            temp = c3; c3=d2; d2=temp;
        }
        void init_from_eigen(const Eigen::Matrix4d& pose_in_cam, int scale_factor){
            // scale factor is to convert to cm for rendering
            a0 = pose_in_cam(0,0)*scale_factor;
            a1 = pose_in_cam(0,1)*scale_factor;
            a2 = pose_in_cam(0,2)*scale_factor;
            a3 = pose_in_cam(0,3)*scale_factor;
            b0 = pose_in_cam(1,0)*scale_factor;
            b1 = pose_in_cam(1,1)*scale_factor;
            b2 = pose_in_cam(1,2)*scale_factor;
            b3 = pose_in_cam(1,3)*scale_factor;
            c0 = pose_in_cam(2,0)*scale_factor;
            c1 = pose_in_cam(2,1)*scale_factor;
            c2 = pose_in_cam(2,2)*scale_factor;
            c3 = pose_in_cam(2,3)*scale_factor;
            d0 = pose_in_cam(3,0);
            d1 = pose_in_cam(3,1);
            d2 = pose_in_cam(3,2);
            d3 = pose_in_cam(3,3);
        }
        void init_from_cv(const cv::Mat& pose){ // so stupid
            assert(pose.type() == CV_32F);

            a0 = pose.at<float>(0, 0); a1 = pose.at<float>(0, 1);
            a2 = pose.at<float>(0, 2); a3 = pose.at<float>(0, 3);

            b0 = pose.at<float>(1, 0); b1 = pose.at<float>(1, 1);
            b2 = pose.at<float>(1, 2); b3 = pose.at<float>(1, 3);

            c0 = pose.at<float>(2, 0); c1 = pose.at<float>(2, 1);
            c2 = pose.at<float>(2, 2); c3 = pose.at<float>(2, 3);

            d0 = pose.at<float>(3, 0); d1 = pose.at<float>(3, 1);
            d2 = pose.at<float>(3, 2); d3 = pose.at<float>(3, 3);
        }

        void init_from_ptr(const float* data){
            a0 = data[0]; a1 = data[1]; a2 = data[2]; a3 = data[3];
            b0 = data[4]; b1 = data[5]; b2 = data[6]; b3 = data[7];
            c0 = data[8]; c1 = data[9]; c2 = data[10]; c3 = data[11];
            d0 = data[12]; d1 = data[13]; d2 = data[14]; d3 = data[15];
        }

        void init_from_ptr(const float* R, const float* t){
            a0 = R[0]; a1 = R[1]; a2 = R[2];  a3 = t[0];
            b0 = R[3]; b1 = R[4]; b2 = R[5];  b3 = t[1];
            c0 = R[6]; c1 = R[7]; c2 = R[8];  c3 = t[2];
        }

        void init_from_cv(const cv::Mat& R, const cv::Mat& t){
            assert(R.type() == CV_32F);
            assert(t.type() == CV_32F);

            a0 = R.at<float>(0, 0)*100; a1 = R.at<float>(0, 1)*100;
            a2 = R.at<float>(0, 2)*100; a3 = t.at<float>(0, 0);

            b0 = R.at<float>(1, 0)*100; b1 = R.at<float>(1, 1)*100;
            b2 = R.at<float>(1, 2)*100; b3 = t.at<float>(1, 0);

            c0 = R.at<float>(2, 0)*100; c1 = R.at<float>(2, 1)*100;
            c2 = R.at<float>(2, 2)*100; c3 = t.at<float>(2, 0);

            d0 = 0; d1 = 0;
            d2 = 0; d3 = 1;
        }

        void print(){
            std::cout<<a0<<", "<<a1<<", "<<a2<<", "<<a3<<"\n"
                <<b0<<", "<<b1<<", "<<b2<<", "<<b3<<"\n"
                <<c0<<", "<<c1<<", "<<c2<<", "<<c3<<"\n"
                <<d0<<", "<<d1<<", "<<d2<<", "<<d3<<"\n";
        }
    };

    // wanted data
    std::vector<Triangle> tris;
    std::vector<float3> vertices;
    std::vector<int3> faces;
    aiVector3D bbox_min, bbox_max;

    void recursive_render(const struct aiScene *sc, const struct aiNode* nd, aiMatrix4x4 m = aiMatrix4x4());

    static float3 mat_mul_vec(const aiMatrix4x4& mat, const aiVector3D& vec);

    void get_bounding_box_for_node(const aiNode* nd, aiVector3D& min, aiVector3D& max, aiMatrix4x4* trafo) const;
    void get_bounding_box(aiVector3D& min, aiVector3D& max) const;
};

}