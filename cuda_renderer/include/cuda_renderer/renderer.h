#pragma once

#ifdef CUDA_ON
// cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#else
// invalidate cuda macro
#define __device__
#define __host__

#endif

// load ply
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <opencv/cv.h>
#include <opencv2/core/core.hpp>
#include <opencv/highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/core/core.hpp>
#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include "math.h"
#include <chrono>
#include "cuda_renderer/model.h"
// #include <fast_gicp/gicp/fast_gicp_cuda.hpp>


namespace cuda_renderer {

// #ifdef CUDA_ON
// // thrust device vector can't be used in cpp by design
// // same codes in cuda renderer,
// // because we don't want these two related to each other
// template <typename T>
// class device_vector_holder{
// public:
//     T* __gpu_memory;
//     size_t __size;
//     bool valid = false;
//     device_vector_holder(){}
//     device_vector_holder(size_t size);
//     device_vector_holder(size_t size, T init);
//     ~device_vector_holder();

//     T* data(){return __gpu_memory;}
//     thrust::device_ptr<T> data_thr(){return thrust::device_ptr<T>(__gpu_memory);}
//     T* begin(){return __gpu_memory;}
//     thrust::device_ptr<T> begin_thr(){return thrust::device_ptr<T>(__gpu_memory);}
//     T* end(){return __gpu_memory + __size;}
//     thrust::device_ptr<T> end_thr(){return thrust::device_ptr<T>(__gpu_memory + __size);}

//     size_t size(){return __size;}

//     void __malloc(size_t size);
//     void __free();
// };

// extern template class device_vector_holder<int>;
// extern template class device_vector_holder<Model::Triangle>;
// #endif

// #ifdef CUDA_ON
//     using Int_holder = device_vector_holder<int>;
// #else
//     using Int_holder = std::vector<int>;
// #endif
// std::vector<int> compute_rgbd_cost(
//     const std::vector<std::vector<uint8_t>> input_color,
//     std::vector<int32_t> input_depth,
//     const std::vector<std::vector<uint8_t>> observed_color,
//     std::vector<int32_t> observed_depth,
//     size_t height, size_t width, size_t num_rendered
// );
// std::vector<int> compute_cost(const std::vector<std::vector<uint8_t>> input,const std::vector<std::vector<uint8_t>> observed,size_t height,size_t width,size_t num_rendered) ;
// std::vector<Model::mat4x4> mat_to_compact_4x4(const std::vector<cv::Mat>& poses);
Model::mat4x4 compute_proj(const cv::Mat& K, int width, int height, float near=10, float far=10000);


//roi: directly crop while rendering, expected to save much time & space
// std::vector<int32_t> render_cpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
//                             size_t width, size_t height, const Model::mat4x4& proj_mat,
//                                 const Model::ROI roi= {0, 0, 0, 0});

// #ifdef CUDA_ON
// device_vector_holder<int> render_cuda_multi(
//                             const std::vector<Model::Triangle>& tris,
//                             const std::vector<Model::mat4x4>& poses,
//                             const std::vector<int> pose_model_map,
//                             const std::vector<int> tris_model_count,
//                             size_t width, size_t height, const Model::mat4x4& proj_mat,
//                             const std::vector<int32_t>& source_result_depth,
//                             const std::vector<std::vector<uint8_t>>& source_result_color,
//                             std::vector<int32_t>& result_depth, 
//                             std::vector<std::vector<uint8_t>>& result_color,
//                             std::vector<int>& pose_occluded,
//                             int single_result_image,
//                             std::vector<int>& pose_occluded_other,
//                             std::vector<float>& clutter_cost,
//                             const std::vector<uint8_t>& source_mask_label,
//                             const std::vector<int>& pose_segmentation_label);

// device_vector_holder<int> render_cuda(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
//                             size_t width, size_t height, const Model::mat4x4& proj_mat, 
//                             std::vector<int32_t>& result_depth, std::vector<std::vector<uint8_t>>& result_color,
//                             const Model::ROI roi= {0, 0, 0, 0});
// // triangles in gpu side
// std::vector<int32_t> render_cuda(device_vector_holder<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
//                             size_t width, size_t height, const Model::mat4x4& proj_mat,
//                                  const Model::ROI roi= {0, 0, 0, 0});


// device_vector_holder<int> render_cuda_keep_in_gpu(const std::vector<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
//                             size_t width, size_t height, const Model::mat4x4& proj_mat,
//                                                        const Model::ROI roi= {0, 0, 0, 0});
// // triangles in gpu side
// device_vector_holder<int> render_cuda_keep_in_gpu(device_vector_holder<Model::Triangle>& tris,const std::vector<Model::mat4x4>& poses,
//                             size_t width, size_t height, const Model::mat4x4& proj_mat,
                                                            //    const Model::ROI roi= {0, 0, 0, 0});

bool depth2cloud_global(const std::vector<int32_t>& depth_data,
                        const std::vector<std::vector<uint8_t>> &color_data,
                        Eigen::Vector3f* &result_cloud_eigen,
                        float *&result_cloud,
                        uint8_t *&result_cloud_color,
                        int *&dc_index,
                        int &point_num,
                        int *&cloud_pose_map,
                        int *&result_observed_cloud_label,
                        const int width,
                        const int height,
                        const int num_poses,
                        const std::vector<int>& pose_occluded,
                        const float kCameraCX,
                        const float kCameraCY,
                        const float kCameraFX,
                        const float kCameraFY,
                        const float depth_factor,
                        const int stride,
                        const int point_dim,
                        const std::vector<uint8_t>& label_mask_data = std::vector<uint8_t>(),
                        const std::vector<double>& observed_cloud_bounds = std::vector<double>(),
                        const Eigen::Matrix4f *camera_transform = NULL);

// bool compute_rgbd_cost(
//     float &sensor_resolution,
//     float* knn_dist,
//     int* knn_index,
//     int* poses_occluded,
//     int* cloud_pose_map,
//     float* result_observed_cloud,
//     uint8_t* result_observed_cloud_color,
//     float* result_rendered_cloud,
//     uint8_t* result_rendered_cloud_color,
//     int rendered_cloud_point_num,
//     int observed_cloud_point_num,
//     int num_poses,
//     float* &rendered_cost,
//     std::vector<float> pose_observed_points_total,
//     float* &observed_cost,
//     int* pose_segmentation_label,
//     int* result_observed_cloud_label,
//     int cost_type,
//     bool calculate_observed_cost);

// void render_cuda_multi_unified_old(
//         const std::string stage, 
//         const std::vector<Model::Triangle>& tris,
//         const std::vector<Model::mat4x4>& poses,
//         const std::vector<int> pose_model_map,
//         const std::vector<int> tris_model_count,
//         size_t width, size_t height, const Model::mat4x4& proj_mat,
//         const std::vector<int32_t>& source_depth,
//         const std::vector<std::vector<uint8_t>>& source_color,
//         int single_result_image,
//         std::vector<float>& clutter_cost,
//         const std::vector<uint8_t>& source_mask_label,
//         const std::vector<int>& pose_segmentation_label,
//         int stride,
//         int point_dim,
//         int depth_factor,
//         float kCameraCX,
//         float kCameraCY,
//         float kCameraFX,
//         float kCameraFY,
//         float* observed_depth,
//         uint8_t* observed_color,
//         int observed_point_num,
//         // Cost calculation specific stuff
//         std::vector<float> pose_observed_points_total,
//         int* result_observed_cloud_label,
//         int cost_type,
//         bool calculate_observed_cost,
//         float sensor_resolution,
//         float color_distance_threshold,
//         float occlusion_threshold,
//         //// Outputs
//         std::vector<int32_t>& result_depth, 
//         std::vector<std::vector<uint8_t>>& result_color,
//         float* &result_cloud,
//         uint8_t* &result_cloud_color,
//         int& result_cloud_point_num,
//         int* &result_cloud_pose_map,
//         int* &result_dc_index,
//         // Costs
//         float* &rendered_cost,
//         float* &observed_cost,
//         float* &points_diff_cost,
//         double& peak_memory_usage);

void render_cuda_multi_unified(
        const std::string stage, 
        const std::vector<Model::Triangle>& tris,
        const std::vector<Model::mat4x4>& poses,
        const std::vector<int> pose_model_map,
        const std::vector<int> tris_model_count,
        size_t width, size_t height, const Model::mat4x4& proj_mat,
        const std::vector<int32_t>& source_depth,
        const std::vector<std::vector<uint8_t>>& source_color,
        int single_result_image,
        std::vector<float>& clutter_cost,
        const std::vector<uint8_t>& source_mask_label,
        const std::vector<int>& pose_segmentation_label,
        int stride,
        int point_dim,
        int depth_factor,
        float kCameraCX,
        float kCameraCY,
        float kCameraFX,
        float kCameraFY,
        float* observed_depth,
        Eigen::Vector3f* observed_depth_eigen,
        uint8_t* observed_color,
        int observed_point_num,
        // Cost calculation specific stuff
        std::vector<float> pose_observed_points_total,
        int* result_observed_cloud_label,
        int cost_type,
        bool calculate_observed_cost,
        float sensor_resolution,
        float color_distance_threshold,
        float occlusion_threshold,
        bool do_icp,
        //// Outputs
        std::vector<int32_t>& result_depth, 
        std::vector<std::vector<uint8_t>>& result_color,
        float* &result_cloud,
        uint8_t* &result_cloud_color,
        int& result_cloud_point_num,
        int* &result_cloud_pose_map,
        int* &result_dc_index,
        // ICP stuff
        std::vector<Model::mat4x4>& adjusted_poses,
        // Costs
        float* &rendered_cost,
        float* &observed_cost,
        float* &points_diff_cost,
        gpu_stats& stats);

// #endif

// render: results keep in gpu or cpu side
// template<typename ...Params>
// Int_holder render(Params&&...params)
// {
// #ifdef CUDA_ON
//     return cuda_renderer::render_cuda_keep_in_gpu(std::forward<Params>(params)...);
// #else
//     return cuda_renderer::render_cpu(std::forward<Params>(params)...);
// #endif
// }

// // render host: always in cpu side
// template<typename ...Params>
// std::vector<int32_t> render_host(Params&&...params)
// {
// #ifdef CUDA_ON
//     return cuda_renderer::render_cuda(std::forward<Params>(params)...);
// #else
//     return cuda_renderer::render_cpu(std::forward<Params>(params)...);
// #endif
// }

//low_level
// namespace normal_functor{  // similar to thrust
//     __host__ __device__ inline
//     Model::float3 minus(const Model::float3& one, const Model::float3& the_other)
//     {
//         return {
//             one.x - the_other.x,
//             one.y - the_other.y,
//             one.z - the_other.z
//         };
//     }
//     __host__ __device__ inline
//     Model::float3 cross(const Model::float3& one, const Model::float3& the_other)
//     {
//         return {
//             one.y*the_other.z - one.z*the_other.y,
//             one.z*the_other.x - one.x*the_other.z,
//             one.x*the_other.y - one.y*the_other.x
//         };
//     }
//     __host__ __device__ inline
//     Model::float3 normalized(const Model::float3& one)
//     {
//         float norm = std::sqrt(one.x*one.x+one.y*one.y+one.z*one.z);
//         return {
//           one.x/norm,
//           one.y/norm,
//           one.z/norm
//         };
//     }

//     __host__ __device__ inline
//     Model::float3 get_normal(const Model::Triangle& dev_tri)
//     {
// //      return normalized(cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v1, dev_tri.v0)));

//       // no need for normalizing?
//       return (cross(minus(dev_tri.v1, dev_tri.v0), minus(dev_tri.v2, dev_tri.v0)));
//     }

//     __host__ __device__ inline
//     bool is_back(const Model::Triangle& dev_tri){
//         return normal_functor::get_normal(dev_tri).z < 0;
//     }
// };

// __host__ __device__ inline
// Model::float3 mat_mul_v(const Model::mat4x4& tran, const Model::float3& v){
//     return {
//         tran.a0*v.x + tran.a1*v.y + tran.a2*v.z + tran.a3,
//         tran.b0*v.x + tran.b1*v.y + tran.b2*v.z + tran.b3,
//         tran.c0*v.x + tran.c1*v.y + tran.c2*v.z + tran.c3,
//     };
// }

// __host__ __device__ inline
// Model::Triangle transform_triangle(const Model::Triangle& dev_tri, const Model::mat4x4& tran){
//     return {
//         mat_mul_v(tran, (dev_tri.v0)),
//         mat_mul_v(tran, (dev_tri.v1)),
//         mat_mul_v(tran, (dev_tri.v2)),
//         dev_tri.color
//     };
// }

// __host__ __device__ inline
// float calculateSignedArea(float* A, float* B, float* C){
//     return 0.5f*((C[0]-A[0])*(B[1]-A[1]) - (B[0]-A[0])*(C[1]-A[1]));
// }

// __host__ __device__ inline
// Model::float3 barycentric(float* A, float* B, float* C, size_t* P) {

//     float float_P[2] = {float(P[0]), float(P[1])};

//     auto base_inv = 1/calculateSignedArea(A, B, C);
//     auto beta = calculateSignedArea(A, float_P, C)*base_inv;
//     auto gamma = calculateSignedArea(A, B, float_P)*base_inv;

//     return {
//         1.0f-beta-gamma,
//         beta,
//         gamma,
//     };
// }

// __host__ __device__ inline
// float std__max(float a, float b){return (a>b)? a: b;};
// __host__ __device__ inline
// float std__min(float a, float b){return (a<b)? a: b;};
}
