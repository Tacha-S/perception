#ifndef CUDA_COMPUTE_POINT_CLOUDS_CUH
#define CUDA_COMPUTE_POINT_CLOUDS_CUH
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <Eigen/Core>

#include "cuda_renderer/model.h"
#include "cuda_renderer/cuda/utils.cuh"

namespace cuda_renderer {
namespace image_to_cloud {
    __device__ void transform_point(int x, int y, int32_t depth,
        float kCameraCX, float kCameraCY, float kCameraFX, float kCameraFY, float depth_factor,
        const Eigen::Matrix4f* camera_transform,
        float &x_pcd, float &y_pcd, float &z_pcd)
    {
        // depth factor here basically converts from cm depth to value in m
        z_pcd = static_cast<float>(depth)/depth_factor;
        x_pcd = (static_cast<float>(x) - kCameraCX)/kCameraFX * z_pcd;
        y_pcd = (static_cast<float>(y) - kCameraCY)/kCameraFY * z_pcd;
        // printf("kCameraCX:%f,kCameraFX:%f, kCameraCY:%f, kCameraFY:%f\n", kCameraCX,kCameraFX,kCameraCY, kCameraFY);

        if (camera_transform != NULL)
        {
            Eigen::Matrix<float, 3, 1> pt (x_pcd, y_pcd, z_pcd);
            Eigen::Vector3f world_point = camera_transform->block<3,3>(0,0) * pt;
            world_point += camera_transform->block<3,1>(0,3);
            z_pcd = world_point[2];
            y_pcd = world_point[1];
            x_pcd = world_point[0];
            // printf("x:%d,y:%d, x_pcd:%f, y_pcd:%f, z_pcd:%f\n", x,y,x_pcd, y_pcd, z_pcd);
        }
    }

    __global__ void depth_to_mask(
        const int32_t* depth, int* mask, int width, int height, int stride, int num_poses, const int* pose_occluded, const uint8_t* label_mask_data,
        float kCameraCX, float kCameraCY, float kCameraFX, float kCameraFY, float depth_factor,
        const double* observed_cloud_bounds, const Eigen::Matrix4f* camera_transform)
    {
        /**
         * Creates a mask corresponding to valid depth points by using the depth data
         * Optionally also filters the point clouds based on bounds given in world frame for 3-Dof
         *
        */
        int n = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/(width/stride));
        
        // threadid in x corresponds to width of image
        int x = (blockIdx.x * blockDim.x + threadIdx.x)%(width/stride);

        //thread id in block in y direction corresponds to height of image
        int y = blockIdx.y*blockDim.y + threadIdx.y;
        // int y = (blockIdx.y*blockDim.y + threadIdx.y)%(height/stride);
        x = x*stride;
        y = y*stride;
        if(x >= width) return;
        if(y >= height) return;
        if (n >= num_poses) return;
        uint32_t idx_depth = n * width * height + x + y*width;
        uint32_t idx_mask = n * width * height + x + y*width;
    
        // if(depth[idx_depth] > 0 && !pose_occluded[n]) 
        if(depth[idx_depth] > 0) 
        {
            if (label_mask_data == NULL && camera_transform == NULL && observed_cloud_bounds == NULL)
            {
                // No label mask provided, so just create mask based on valid depth values
                mask[idx_mask] = 1;
            }
            else if (label_mask_data != NULL)
            {
                // Use the label mask provided to create point cloud of only objects of interest
                if (label_mask_data[idx_depth] > 0)
                {
                    mask[idx_mask] = 1;
                }
            }
            else if (camera_transform != NULL && observed_cloud_bounds != NULL)
            {
                // Filter point cloud for 3Dof based on bounds given 
                float x_pcd, y_pcd, z_pcd;
                transform_point(x, y, depth[idx_depth], kCameraCX, kCameraCY, kCameraFX, kCameraFY,
                                depth_factor, camera_transform, x_pcd, y_pcd, z_pcd);

                if (x_pcd > (float) observed_cloud_bounds[0] || x_pcd < (float) observed_cloud_bounds[1]) return;
                if (y_pcd > (float) observed_cloud_bounds[2] || y_pcd < (float) observed_cloud_bounds[3]) return;
                if (z_pcd > (float) observed_cloud_bounds[4] || z_pcd < (float) observed_cloud_bounds[5]) return;
                
                mask[idx_mask] = 1;
            }
        }
    }
    
    __global__ void depth_to_2d_cloud(
        const int32_t* depth, const uint8_t* r_in, const uint8_t* g_in, const uint8_t* b_in, float* cloud, float* cloud_1d, 
        Eigen::Vector3f* cloud_eigen,
        size_t cloud_pitch, 
        uint8_t* cloud_color, int cloud_rendered_cloud_point_num, int* mask, int width, int height, 
        float kCameraCX, float kCameraCY, float kCameraFX, float kCameraFY, float depth_factor,
        int stride, int num_poses, int* cloud_pose_map, const uint8_t* label_mask_data,  int* cloud_mask_label,
        const double* observed_cloud_bounds, const Eigen::Matrix4f* camera_transform, const int* pose_segmentation_label_ptr)
    {
        /**
         * Creates a point cloud by combining a mask corresponding to valid depth pixels and depth data using the camera params
         * Optionally also records the correct color of the points and their mask label
        */
        int n = (int)floorf((blockIdx.x * blockDim.x + threadIdx.x)/(width/stride));
        int x = (blockIdx.x * blockDim.x + threadIdx.x)%(width/stride);
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // uint32_t x = blockIdx.x*blockDim.x + threadIdx.x;
        // uint32_t y = blockIdx.y*blockDim.y + threadIdx.y;
        x = x*stride;
        y = y*stride;
        if(x >= width) return;
        if(y >= height) return;
        if(n >= num_poses) return;
        uint32_t idx_depth = n * width * height + x + y*width;
    
        // Need to check if pixel is valid here so that some invalid depth doesn't get written
        // Previously was happening in getgravityalignedpointcloud in search_env.cpp
        if(depth[idx_depth] <= 0) return;
        // 6-Dof invalid pixel case
        if (label_mask_data != NULL)
        {
            if(label_mask_data[idx_depth] <= 0) return;
        }
        float x_pcd, y_pcd, z_pcd;
        // printf("depth:%d\n", depth[idx_depth]);
        // 3-Dof invalid pixel case, transform to table frame and check bounds       
        if (camera_transform != NULL && observed_cloud_bounds != NULL) 
        {
            transform_point(x, y, depth[idx_depth], kCameraCX, kCameraCY, kCameraFX, kCameraFY,
                depth_factor, camera_transform, x_pcd, y_pcd, z_pcd);
            if (x_pcd > (float) observed_cloud_bounds[0] || x_pcd < (float) observed_cloud_bounds[1]) return;
            if (y_pcd > (float) observed_cloud_bounds[2] || y_pcd < (float) observed_cloud_bounds[3]) return;
            if (z_pcd > (float) observed_cloud_bounds[4] || z_pcd < (float) observed_cloud_bounds[5]) return;
        }

        // Get actual point which should be in camera frame itself
        transform_point(x, y, depth[idx_depth], kCameraCX, kCameraCY, kCameraFX, kCameraFY,
            depth_factor, NULL, x_pcd, y_pcd, z_pcd);
        // printf("kCameraCX:%f,kCameraFX:%f, kCameraCY:%f, kCameraCY:%f\n", kCameraCX,kCameraFX,kCameraCY, y_pcd, z_pcd);
        
        uint32_t idx_mask = n * width * height + x + y*width;
        int cloud_idx = mask[idx_mask];
        float* row_0 = (float *)((char*)cloud + 0 * cloud_pitch);
        float* row_1 = (float *)((char*)cloud + 1 * cloud_pitch);
        float* row_2 = (float *)((char*)cloud + 2 * cloud_pitch);
        row_0[cloud_idx] = x_pcd;
        row_1[cloud_idx] = y_pcd;
        row_2[cloud_idx] = z_pcd;
    
        cloud_1d[cloud_idx + 0*cloud_rendered_cloud_point_num] = x_pcd;
        cloud_1d[cloud_idx + 1*cloud_rendered_cloud_point_num] = y_pcd;
        cloud_1d[cloud_idx + 2*cloud_rendered_cloud_point_num] = z_pcd;
        
        cloud_eigen[cloud_idx](0) = x_pcd;
        cloud_eigen[cloud_idx](1) = y_pcd;
        cloud_eigen[cloud_idx](2) = z_pcd;

        cloud_color[cloud_idx + 0*cloud_rendered_cloud_point_num] = r_in[idx_depth];
        cloud_color[cloud_idx + 1*cloud_rendered_cloud_point_num] = g_in[idx_depth];
        cloud_color[cloud_idx + 2*cloud_rendered_cloud_point_num] = b_in[idx_depth];
    
        cloud_pose_map[cloud_idx] = n;
        if (label_mask_data != NULL)
        {
            // Creating label from image mask - for observed point cloud
            // Do -1 to make it start from 0
            cloud_mask_label[cloud_idx] = label_mask_data[idx_depth] - 1;
        }
        else if (pose_segmentation_label_ptr != NULL)
        {
            // Creating label from pose data - for rendered point cloud
            cloud_mask_label[cloud_idx] = pose_segmentation_label_ptr[n];
        }
        
        // printf("cloud_idx:%d\n", pose_segmentation_label_ptr[n]);    
        // cloud[3*cloud_idx + 0] = x_pcd;
        // cloud[3*cloud_idx + 1] = y_pcd;
        // cloud[3*cloud_idx + 2] = z_pcd;
    }
        
}

void compute_point_clouds(const thrust::device_vector<int32_t>& device_depth_int,
                          const thrust::device_vector<uint8_t>& device_red_int,
                          const thrust::device_vector<uint8_t>& device_green_int,
                          const thrust::device_vector<uint8_t>& device_blue_int,
                          const int    num_poses,
                          const size_t width,
                          const size_t height,
                          const float  kCameraCX, 
                          const float  kCameraCY, 
                          const float  kCameraFX, 
                          const float  kCameraFY, 
                          const float  depth_factor,
                          const int    stride,
                          const thrust::device_vector<int>&      device_pose_occluded,
                          float* &result_2d_point_cloud,
                          size_t& query_pitch_in_bytes,
                          thrust::device_vector<Eigen::Vector3f>& result_cloud_eigen,
                          thrust::device_vector<float>&   result_point_cloud,
                          thrust::device_vector<uint8_t>& result_point_cloud_color,
                          int& result_cloud_point_count,
                          thrust::device_vector<int>&     result_dc_index,
                          thrust::device_vector<int>&     result_cloud_pose_map,
                          thrust::device_vector<int>&     result_cloud_label,
                          gpu_stats& stats,
                          const Eigen::Matrix4f* camera_transform = NULL,
                          const thrust::device_vector<uint8_t>&  device_image_label   = thrust::device_vector<uint8_t>(0),
                          const thrust::device_vector<double>&   device_observed_cloud_bounds = thrust::device_vector<double>(0),
                          const thrust::device_vector<int>&      pose_segmentation_label = thrust::device_vector<int>(0)
                          ) {
    
    printf("compute_point_clouds()\n");
    printf("device_image_label empty : %d\n", device_image_label.empty());
    printf("device_image_label size : %d\n", device_image_label.size());
    printf("device_observed_cloud_bounds empty : %d\n", device_observed_cloud_bounds.empty());
    printf("device_observed_cloud_bounds size : %d\n", device_observed_cloud_bounds.size());
    printf("Num poses : %d\n", num_poses);
    const int32_t* depth_data      = thrust::raw_pointer_cast(device_depth_int.data());  
    const uint8_t* red_in          = thrust::raw_pointer_cast(device_red_int.data());  
    const uint8_t* green_in        = thrust::raw_pointer_cast(device_green_int.data());  
    const uint8_t* blue_in         = thrust::raw_pointer_cast(device_blue_int.data());  
    const int* poses_occluded            = thrust::raw_pointer_cast(device_pose_occluded.data());
    const uint8_t* image_label           = device_image_label.empty() ? NULL : thrust::raw_pointer_cast(device_image_label.data());
    const double*  observed_cloud_bounds = device_observed_cloud_bounds.empty() ? NULL : thrust::raw_pointer_cast(device_observed_cloud_bounds.data());
    const int* pose_segmentation_label_ptr = pose_segmentation_label.empty() ? NULL : thrust::raw_pointer_cast(pose_segmentation_label.data());
    
    // thrust::copy(
    //     device_red_int.begin(),
    //     device_red_int.end(), 
    //     std::ostream_iterator<uint8_t>(std::cout, " ")
    // );
    // printf("\n");

    // if (device_image_label.size() > 0)
    // {
    //     printf("Using segementation labels to create point cloud\n");
    //     image_label = thrust::raw_pointer_cast(device_image_label.data());
    // }
    // if (device_observed_cloud_bounds.size() > 0)
    // {
    //     // std::cout << observed_cloud_bounds[0] << " " << observed_cloud_bounds[1] << std::endl;
    //     printf("Using filter bounds to create point cloud\n");
    //     observed_cloud_bounds = thrust::raw_pointer_cast(device_observed_cloud_bounds.data()); 
    //     // printf("x_min : %f, x_max : %f\n", observed_cloud_bounds[1], observed_cloud_bounds[0]);
    //     // printf("y_min : %f, y_max : %f\n", observed_cloud_bounds[3], observed_cloud_bounds[2]);
    //     // printf("z_min : %f, z_max : %f\n", observed_cloud_bounds[5], observed_cloud_bounds[4]);
    // }
    // if (device_pose_occluded.size() > 0) 
    // {
    //     poses_occluded = thrust::raw_pointer_cast(device_pose_occluded.data()); 
    // }
    // if (camera_transform != NULL)
    // {
    //     printf("Using camera transform to transform point cloud to world frame\n");
    // //     std::cout << *camera_transform << std::endl; 
    // //     cudaMalloc(&camera_transform_cuda, sizeof(Eigen::Matrix4f));
    // //     cudaMemcpy(camera_transform_cuda, camera_transform, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
    // }
    // thrust::device_vector<int> mask(width*height*num_poses, 0);
    // result_dc_index.clear();
    result_dc_index.resize(width*height*num_poses, 0);
    int* mask_ptr = thrust::raw_pointer_cast(result_dc_index.data());

    dim3 threadsPerBlock(16, 16);
    assert(width % stride == 0);
    dim3 numBlocks((width/stride * num_poses + threadsPerBlock.x - 1)/threadsPerBlock.x, 
                    (height/stride + threadsPerBlock.y - 1)/threadsPerBlock.y);
    image_to_cloud::depth_to_mask<<<numBlocks, threadsPerBlock>>>(depth_data,
                                                                    mask_ptr,
                                                                    width, 
                                                                    height, 
                                                                    stride, 
                                                                    num_poses,
                                                                    poses_occluded,
                                                                    image_label,
                                                                    kCameraCX, 
                                                                    kCameraCY, 
                                                                    kCameraFX, 
                                                                    kCameraFY, 
                                                                    depth_factor,
                                                                    observed_cloud_bounds, 
                                                                    camera_transform);
    
    int mask_back_temp = result_dc_index.back();
    thrust::exclusive_scan(result_dc_index.begin(), result_dc_index.end(), result_dc_index.begin(), 0); // in-place scan
    result_cloud_point_count = result_dc_index.back() + mask_back_temp;
    printf("Actual points in all clouds : %d\n", result_cloud_point_count);

    cudaMallocPitch(&result_2d_point_cloud,   &query_pitch_in_bytes,   result_cloud_point_count * sizeof(float), POINT_DIM);
    
    // result_cloud_eigen.clear();
    // result_point_cloud.clear();
    // result_point_cloud_color.clear();
    // result_cloud_pose_map.clear();

    result_cloud_eigen.resize(result_cloud_point_count, Eigen::Vector3f::Constant(0));
    result_point_cloud.resize(POINT_DIM * result_cloud_point_count, 0);
    result_point_cloud_color.resize(POINT_DIM * result_cloud_point_count, 0);
    result_cloud_pose_map.resize(result_cloud_point_count, 0);
    
    Eigen::Vector3f* cloud_eigen = thrust::raw_pointer_cast(result_cloud_eigen.data());
    float* cloud_1d = thrust::raw_pointer_cast(result_point_cloud.data());
    uint8_t* cloud_color = thrust::raw_pointer_cast(result_point_cloud_color.data());
    int* cloud_pose_map = thrust::raw_pointer_cast(result_cloud_pose_map.data());

    // Assign output cloud segmentation label if needed
    int* cloud_mask_label = NULL;
    if (!device_image_label.empty() || !pose_segmentation_label.empty())
    {
        // result_cloud_label.clear();
        result_cloud_label.resize(result_cloud_point_count, 0);
        cloud_mask_label = thrust::raw_pointer_cast(result_cloud_label.data());
    }
    stats.peak_memory_usage = std::max(print_cuda_memory_usage(), stats.peak_memory_usage);
    image_to_cloud::depth_to_2d_cloud<<<numBlocks, threadsPerBlock>>>(depth_data, 
                                                      red_in, 
                                                      green_in, 
                                                      blue_in,
                                                      result_2d_point_cloud, 
                                                      cloud_1d,
                                                      cloud_eigen,
                                                      query_pitch_in_bytes, 
                                                      cloud_color, 
                                                      result_cloud_point_count, 
                                                      mask_ptr, 
                                                      width, 
                                                      height, 
                                                      kCameraCX, 
                                                      kCameraCY, 
                                                      kCameraFX, 
                                                      kCameraFY, 
                                                      depth_factor, 
                                                      stride, 
                                                      num_poses, 
                                                      cloud_pose_map,
                                                      image_label, 
                                                      cloud_mask_label, 
                                                      observed_cloud_bounds, 
                                                      camera_transform,
                                                      pose_segmentation_label_ptr);
    // result_point_cloud.assign(cuda_cloud, cuda_cloud + POINT_DIM * result_cloud_point_count);
    // thrust::copy(
    //     result_point_cloud.begin(),
    //     result_point_cloud.end(), 
    //     std::ostream_iterator<float>(std::cout, " ")
    // );
    // printf("\nColor Cloud : \n");
    // thrust::copy(
    //     result_point_cloud_color.begin(),
    //     result_point_cloud_color.end(), 
    //     std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    // thrust::copy(
    //     result_dc_index.begin(),
    //     result_dc_index.end(), 
    //     std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    printf("compute_point_clouds() done\n");
}
}

#endif