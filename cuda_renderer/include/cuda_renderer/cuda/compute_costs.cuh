#ifndef CUDA_COMPUTE_COSTS_CUH
#define CUDA_COMPUTE_COSTS_CUH
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "cuda_renderer/model.h"
#include "cuda_renderer/cuda/utils.cuh"

#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)

namespace cuda_renderer {
namespace cost_computation {
    struct cost_percentage_functor{

        cost_percentage_functor(){}

        __host__ __device__
        float operator()(const float& x, const float& y) const
        {
            if (y == 0)
            {
                return -1;
            }
            else
            {
                return x/y;
            }
        }
    };

    struct cost_multiplier_functor{

        cost_multiplier_functor(){}

        __host__ __device__
        float operator()(const float& x, const float& y) const
        {
            if (x == -1)
            {
                return -1;
            }
            else
            {
                return x*y;
            }
        }
    };

     __device__ void rgb2lab(uint8_t rr,uint8_t gg, uint8_t bbb, float* lab){
        double r = rr / 255.0;
        double g = gg / 255.0;
        double b = bbb / 255.0;
        double x;
        double y;
        double z;
        r = ((r > 0.04045) ? pow((r + 0.055) / 1.055, 2.4) : (r / 12.92)) * 100.0;
        g = ((g > 0.04045) ? pow((g + 0.055) / 1.055, 2.4) : (g / 12.92)) * 100.0;
        b = ((b > 0.04045) ? pow((b + 0.055) / 1.055, 2.4) : (b / 12.92)) * 100.0;

        x = r*0.4124564 + g*0.3575761 + b*0.1804375;
        y = r*0.2126729 + g*0.7151522 + b*0.0721750;
        z = r*0.0193339 + g*0.1191920 + b*0.9503041;

        x = x / 95.047;
        y = y / 100.00;
        z = z / 108.883;

        x = (x > 0.008856) ? cbrt(x) : (7.787 * x + 16.0 / 116.0);
        y = (y > 0.008856) ? cbrt(y) : (7.787 * y + 16.0 / 116.0);
        z = (z > 0.008856) ? cbrt(z) : (7.787 * z + 16.0 / 116.0);
        float l,a,bb;

        l = (116.0 * y) - 16;
        a = 500 * (x - y);
        bb = 200 * (y - z);

        lab[0] = l;
        lab[1] = a;
        lab[2] = bb;
    }

    __device__ double color_distance(float l1,float a1,float b1,
                        float l2,float a2,float b2){
        double eps = 1e-5;
        double c1 = sqrtf(SQR(a1) + SQR(b1));
        double c2 = sqrtf(SQR(a2) + SQR(b2));
        double meanC = (c1 + c2) / 2.0;
        double meanC7 = POW7(meanC);

        double g = 0.5*(1 - sqrtf(meanC7 / (meanC7 + 6103515625.))); // 0.5*(1-sqrt(meanC^7/(meanC^7+25^7)))
        double a1p = a1 * (1 + g);
        double a2p = a2 * (1 + g);

        c1 = sqrtf(SQR(a1p) + SQR(b1));
        c2 = sqrtf(SQR(a2p) + SQR(b2));
        double h1 = fmodf(atan2f(b1, a1p) + 2*M_PI, 2*M_PI);
        double h2 = fmodf(atan2f(b2, a2p) + 2*M_PI, 2*M_PI);

        // compute deltaL, deltaC, deltaH
        double deltaL = l2 - l1;
        double deltaC = c2 - c1;
        double deltah;

        if (c1*c2 < eps) {
            deltah = 0;
        }
        if (std::abs(h2 - h1) <= M_PI) {
            deltah = h2 - h1;
        }
        else if (h2 > h1) {
            deltah = h2 - h1 - 2* M_PI;
        }
        else {
            deltah = h2 - h1 + 2 * M_PI;
        }

        double deltaH = 2 * sqrtf(c1*c2)*sinf(deltah / 2);

        // calculate CIEDE2000
        double meanL = (l1 + l2) / 2;
        meanC = (c1 + c2) / 2.0;
        meanC7 = POW7(meanC);
        double meanH;

        if (c1*c2 < eps) {
            meanH = h1 + h2;
        }
        if (std::abs(h1 - h2) <= M_PI + eps) {
            meanH = (h1 + h2) / 2;
        }
        else if (h1 + h2 < 2*M_PI) {
            meanH = (h1 + h2 + 2*M_PI) / 2;
        }
        else {
            meanH = (h1 + h2 - 2*M_PI) / 2;
        }

        double T = 1
            - 0.17*cosf(meanH - DegToRad(30))
            + 0.24*cosf(2 * meanH)
            + 0.32*cosf(3 * meanH + DegToRad(6))
            - 0.2*cosf(4 * meanH - DegToRad(63));
        double sl = 1 + (0.015*SQR(meanL - 50)) / sqrtf(20 + SQR(meanL - 50));
        double sc = 1 + 0.045*meanC;
        double sh = 1 + 0.015*meanC*T;
        double rc = 2 * sqrtf(meanC7 / (meanC7 + 6103515625.));
        double rt = -sinf(DegToRad(60 * expf(-SQR((RadToDeg(meanH) - 275) / 25)))) * rc;

        double cur_dist = sqrtf(SQR(deltaL / sl) + SQR(deltaC / sc) + SQR(deltaH / sh) + rt * deltaC / sc * deltaH / sh);
        return cur_dist;
    }
    
    __global__ void compute_render_cost(
        const float* cuda_knn_dist,
        const int* cuda_knn_index,
        const int* cuda_cloud_pose_map,
        const int* cuda_poses_occluded,
        float* cuda_rendered_cost,
        const const float sensor_resolution,
        const int rendered_cloud_point_num,
        const int observed_cloud_point_num,
        float* cuda_pose_point_num,
        const uint8_t* rendered_cloud_color,
        const uint8_t* observed_cloud_color,
        uint8_t* cuda_observed_explained,
        const int* pose_segmentation_label,
        const int* result_observed_cloud_label,
        int type,
        const float color_distance_threshold)
    {
        /**
        * Params -
        * @cuda_knn_dist : distance to nn from knn library
        * @cuda_knn_index : index of nn in observed cloud from knn library
        * @cuda_cloud_pose_map : the pose corresponding to every point in cloud
        * @*_cloud_color : color values of clouds, to compare rgb cost of NNs
        * Returns :
        * @cuda_pose_point_num : Number of points in each rendered pose
        */
        size_t point_index = blockIdx.x*blockDim.x + threadIdx.x;
        if(point_index >= rendered_cloud_point_num) return;

        int pose_index = cuda_cloud_pose_map[point_index];
        // printf("pose index : %d\n", pose_index);
        int o_point_index = cuda_knn_index[point_index];
        if (cuda_poses_occluded[pose_index])
        {
            cuda_rendered_cost[pose_index] = -1;
        }
        else
        {
            // count total number of points in this pose for normalization later
            atomicAdd(&cuda_pose_point_num[pose_index], 1);
            // float camera_z = rendered_cloud[point_index + 2 * rendered_cloud_point_num];
            // float cost = 10 * camera_z;
            float cost = 1.0;
            // printf("KKN distance : %f\n", cuda_knn_dist[point_index]);
            if (cuda_knn_dist[point_index] > sensor_resolution)
            {
                atomicAdd(&cuda_rendered_cost[pose_index], cost);
            }
            else
            {
                // compute color cost
                // printf("%d, %d\n", pose_segmentation_label[pose_index], result_observed_cloud_label[o_point_index]);
                uint8_t red2  = rendered_cloud_color[point_index + 2*rendered_cloud_point_num];
                uint8_t green2  = rendered_cloud_color[point_index + 1*rendered_cloud_point_num];
                uint8_t blue2  = rendered_cloud_color[point_index + 0*rendered_cloud_point_num];

                uint8_t red1  = observed_cloud_color[o_point_index + 2*observed_cloud_point_num];
                uint8_t green1  = observed_cloud_color[o_point_index + 1*observed_cloud_point_num];
                uint8_t blue1  = observed_cloud_color[o_point_index + 0*observed_cloud_point_num];

                if (type == 1)
                {
                    
                    float lab2[3];
                    rgb2lab(red2,green2,blue2,lab2);
                    float lab1[3];
                    rgb2lab(red1,green1,blue1,lab1);
                    double cur_dist = color_distance(lab1[0],lab1[1],lab1[2],lab2[0],lab2[1],lab2[2]);
                    // printf("color distance :%f\n", cur_dist);
                    if(cur_dist > color_distance_threshold){
                        // add to render cost if color doesnt match
                        atomicAdd(&cuda_rendered_cost[pose_index], cost);
                    }
                    else {
                        // the point is explained, so mark corresponding observed point explained
                        // atomicOr(cuda_observed_explained[o_point_index], 1);
                        cuda_observed_explained[o_point_index + pose_index * observed_cloud_point_num] = 1;
                    }
                }
                else if (type == 0) {
                    // the point is explained, so mark corresponding observed point explained
                    // atomicOr(cuda_observed_explained[o_point_index], 1);
                    cuda_observed_explained[o_point_index + pose_index * observed_cloud_point_num] = 1;
                }
                else if (type == 2) {
                    // printf("pose_segmentation_label :%d, result_observed_cloud_label %d\n", 
                    //     pose_segmentation_label[pose_index], result_observed_cloud_label[o_point_index]);
                    // if (pose_segmentation_label[pose_index] != result_observed_cloud_label[o_point_index])
                    // {
                    //     // the euclidean distance is fine, but segmentation labels dont match
                    //     atomicAdd(&cuda_rendered_cost[pose_index], cost);
                    // }
                    // else
                    // {
                        // the point is explained, so mark corresponding observed point explained
                        // atomicOr(cuda_observed_explained[o_point_index], 1);
                        // float lab2[3];
                        // rgb2lab(red2,green2,blue2,lab2);

                        // float lab1[3];
                        // rgb2lab(red1,green1,blue1,lab1);

                        // double cur_dist = color_distance(lab1[0],lab1[1],lab1[2],lab2[0],lab2[1],lab2[2]);
                        // if(cur_dist > 30)
                        //     atomicAdd(&cuda_rendered_cost[pose_index], cost);
                        // else
                        cuda_observed_explained[o_point_index + pose_index * observed_cloud_point_num] = 1;
                    // }
                }
            }
        }
    }
    __global__ void compute_observed_cost(
        int num_poses,
        int observed_cloud_point_num,
        uint8_t* cuda_observed_explained,
        float* observed_total_explained)
    {
        /*
         * @observed_cloud_point_num - number of points in each pose in observed scene
         * @cuda_observed_explained - binary value indicating whether given point is explained or not based on distance
         */
        size_t point_index = blockIdx.x*blockDim.x + threadIdx.x;
        if(point_index >= num_poses * observed_cloud_point_num) return;

        size_t pose_index = point_index/observed_cloud_point_num;
        atomicAdd(&observed_total_explained[pose_index], (float) cuda_observed_explained[point_index]);
        // printf("%d\n", cuda_observed_explained[point_index]);
    }
}

void compute_costs(const int   num_images,
                        const int   cost_type,
                        const bool  calculate_observed_cost,
                        const float sensor_resolution,
                        const float color_distance_threshold,
                        const thrust::device_vector<uint8_t>& observed_cloud_color,
                        const thrust::device_vector<int>&     observed_cloud_label,
                        const int observed_cloud_point_count,
                        const thrust::device_vector<uint8_t>& rendered_cloud_color,
                        const thrust::device_vector<int>&     rendered_cloud_pose_map,
                        const thrust::device_vector<int>&     rendered_poses_occluded,
                        const thrust::device_vector<int>&     rendered_poses_label,
                        const thrust::device_vector<float> rendered_poses_observed_points_total,
                        const int rendered_cloud_point_count,
                        const thrust::device_vector<float>& k_distances,
                        const thrust::device_vector<int>&   k_indices,
                        thrust::device_vector<float>& cuda_rendered_cost_vec,
                        thrust::device_vector<float>& cuda_observed_cost_vec,
                        thrust::device_vector<float>& cuda_pose_points_diff_cost_vec,
                        gpu_stats& stats                        
                        ) {
        /*
        * rendered_poses_observed_points_total - number of observed points in cylinder volume or in segmentation for given pose
        * 
        */
        printf("compute_costs()\n");
        cuda_rendered_cost_vec.resize(num_images, 0);
        float* cuda_rendered_cost = thrust::raw_pointer_cast(cuda_rendered_cost_vec.data());
        thrust::device_vector<float> cuda_pose_point_num_vec(num_images, 0);
        float* cuda_pose_point_num = thrust::raw_pointer_cast(cuda_pose_point_num_vec.data());
        thrust::device_vector<float> cuda_rendered_explained_vec(num_images, 0);

        // Points in observed that get explained by render
        thrust::device_vector<uint8_t> cuda_observed_explained_vec(num_images * observed_cloud_point_count, 0);
        uint8_t* cuda_observed_explained = thrust::raw_pointer_cast(cuda_observed_explained_vec.data());

        // Copy segmentation label of observed if required
        const int* cuda_observed_cloud_label = (cost_type == 2) ? thrust::raw_pointer_cast(observed_cloud_label.data()) : NULL;
        // peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);

        const uint8_t* cuda_observed_cloud_color = thrust::raw_pointer_cast(observed_cloud_color.data());
        const float* dist_dev = thrust::raw_pointer_cast(k_distances.data());
        const int* index_dev = thrust::raw_pointer_cast(k_indices.data());
        const int* cuda_cloud_pose_map = thrust::raw_pointer_cast(rendered_cloud_pose_map.data());
        const int* device_pose_occluded_vec = thrust::raw_pointer_cast(rendered_poses_occluded.data());
        const uint8_t* cuda_cloud_color = thrust::raw_pointer_cast(rendered_cloud_color.data());
        const int* device_pose_segmentation_label_vec = thrust::raw_pointer_cast(rendered_poses_label.data());
        
        stats.peak_memory_usage = std::max(print_cuda_memory_usage(), stats.peak_memory_usage);

        dim3 numBlocksR((rendered_cloud_point_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
        cost_computation::compute_render_cost<<<numBlocksR, THREADS_PER_BLOCK>>>(
            dist_dev,
            index_dev,
            cuda_cloud_pose_map,
            device_pose_occluded_vec,
            cuda_rendered_cost,
            sensor_resolution,
            rendered_cloud_point_count,
            observed_cloud_point_count,
            cuda_pose_point_num, // Can be 0 if that pose had no points in it
            cuda_cloud_color,
            cuda_observed_cloud_color,
            cuda_observed_explained,
            device_pose_segmentation_label_vec,
            cuda_observed_cloud_label,
            cost_type,
            color_distance_threshold);
        
        thrust::device_vector<float> percentage_multiplier_val(num_images, 100);
        // Convert cost to percentage out of 100
        thrust::transform(
            cuda_pose_point_num_vec.begin(), cuda_pose_point_num_vec.end(), 
            cuda_rendered_cost_vec.begin(), cuda_rendered_explained_vec.begin(), 
            thrust::minus<float>()
        );
        thrust::transform(
            cuda_rendered_cost_vec.begin(), cuda_rendered_cost_vec.end(), 
            cuda_pose_point_num_vec.begin(), cuda_rendered_cost_vec.begin(), 
            cost_computation::cost_percentage_functor()
        );
        thrust::transform(
            cuda_rendered_cost_vec.begin(), cuda_rendered_cost_vec.end(), 
            percentage_multiplier_val.begin(), cuda_rendered_cost_vec.begin(), 
            cost_computation::cost_multiplier_functor()
        );
        // printf("Rendered cost\n");
        // thrust::copy(
        //     cuda_rendered_cost_vec.begin(),
        //     cuda_rendered_cost_vec.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );

        // if (calculate_observed_cost && cost_type == 2)
        if (calculate_observed_cost)
        {
            thrust::device_vector<float> cuda_pose_observed_explained_vec(num_images, 0);
            float* cuda_pose_observed_explained = thrust::raw_pointer_cast(cuda_pose_observed_explained_vec.data());
            cuda_pose_points_diff_cost_vec.resize(num_images, 0);

            // peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);
        
            dim3 numBlocksO((num_images * observed_cloud_point_count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, 1);
            //// Calculate the number of explained points in every pose, by adding
            stats.peak_memory_usage = std::max(print_cuda_memory_usage(), stats.peak_memory_usage);

            cost_computation::compute_observed_cost<<<numBlocksO, THREADS_PER_BLOCK>>>(
                num_images,
                observed_cloud_point_count,
                cuda_observed_explained,
                cuda_pose_observed_explained
            );
            
            //// Get difference of explained points between rendered and observed
            thrust::transform(
                cuda_rendered_explained_vec.begin(), cuda_rendered_explained_vec.end(), 
                cuda_pose_observed_explained_vec.begin(), cuda_pose_points_diff_cost_vec.begin(), 
                thrust::minus<float>()
            );
            // printf("Point diff\n");
            // thrust::copy(
            //     cuda_pose_points_diff_cost_vec.begin(),
            //     cuda_pose_points_diff_cost_vec.end(), 
            //     std::ostream_iterator<int>(std::cout, " ")
            // );
            
            // Subtract total observed points for each pose with explained points for each pose
            // thrust::device_vector<float> cuda_pose_observed_points_total_vec = pose_observed_points_total;
            cuda_observed_cost_vec.resize(num_images, 0);
            thrust::transform(
                rendered_poses_observed_points_total.begin(), rendered_poses_observed_points_total.end(), 
                cuda_pose_observed_explained_vec.begin(), cuda_observed_cost_vec.begin(), 
                thrust::minus<float>()
            );

            // printf("Observed explained\n");
            // thrust::copy(
            //     cuda_pose_observed_points_total_vec.begin(),
            //     cuda_pose_observed_points_total_vec.end(), 
            //     std::ostream_iterator<int>(std::cout, " ")
            // );
            // Divide by total points
            thrust::transform(
                cuda_observed_cost_vec.begin(), cuda_observed_cost_vec.end(), 
                rendered_poses_observed_points_total.begin(), cuda_observed_cost_vec.begin(), 
                thrust::divides<float>()
            );

            // Multiply by 100
            thrust::transform(
                cuda_observed_cost_vec.begin(), cuda_observed_cost_vec.end(), 
                percentage_multiplier_val.begin(), cuda_observed_cost_vec.begin(), 
                thrust::multiplies<float>()
            );

            // printf("Observed cost\n");
            // thrust::copy(
            //     cuda_observed_cost_vec.begin(),
            //     cuda_observed_cost_vec.end(), 
            //     std::ostream_iterator<int>(std::cout, " ")
            // );

        }
        printf("compute_costs() done\n");
}
}

#endif