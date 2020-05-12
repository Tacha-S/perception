#ifndef CUDA_RENDERER_IMAGE_RENDERER_CUH
#define CUDA_RENDERER_IMAGE_RENDERER_CUH
#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

#include "cuda_renderer/model.h"
#include "cuda_renderer/utils.cuh"

namespace cuda_renderer {
namespace image_renderer {

        __host__ __device__ inline
        float std__max(float a, float b){return (a>b)? a: b;};
        
        __host__ __device__ inline
        float std__min(float a, float b){return (a<b)? a: b;};

        __host__ __device__ inline
        Model::float3 mat_mul_v(const Model::mat4x4& tran, const Model::float3& v){
            return {
                tran.a0*v.x + tran.a1*v.y + tran.a2*v.z + tran.a3,
                tran.b0*v.x + tran.b1*v.y + tran.b2*v.z + tran.b3,
                tran.c0*v.x + tran.c1*v.y + tran.c2*v.z + tran.c3,
            };
        }

        __host__ __device__ inline
        Model::Triangle transform_triangle(const Model::Triangle& dev_tri, const Model::mat4x4& tran){
            return {
                mat_mul_v(tran, (dev_tri.v0)),
                mat_mul_v(tran, (dev_tri.v1)),
                mat_mul_v(tran, (dev_tri.v2)),
                dev_tri.color
            };
        }

        __host__ __device__ inline
        float calculateSignedArea(float* A, float* B, float* C){
            return 0.5f*((C[0]-A[0])*(B[1]-A[1]) - (B[0]-A[0])*(C[1]-A[1]));
        }

        __host__ __device__ Model::float3 barycentric(float* A, float* B, float* C, size_t* P) {

            float float_P[2] = {float(P[0]), float(P[1])};

            auto base_inv = 1/calculateSignedArea(A, B, C);
            auto beta = calculateSignedArea(A, float_P, C)*base_inv;
            auto gamma = calculateSignedArea(A, B, float_P)*base_inv;

            return {
                1.0f-beta-gamma,
                beta,
                gamma,
            };
        }

        __device__ void rasterization_with_source(const Model::Triangle dev_tri, Model::float3 last_row,
                                                int32_t* depth_entry, size_t width, size_t height,
                                                const Model::ROI roi, 
                                                uint8_t* red_entry,uint8_t* green_entry,uint8_t* blue_entry,
                                                const int32_t* source_depth_entry,
                                                const uint8_t* source_red_entry,
                                                const uint8_t* source_green_entry,
                                                const uint8_t* source_blue_entry,
                                                int* pose_occluded_entry,
                                                int32_t* lock_entry,
                                                int* pose_occluded_other_entry,
                                                float* pose_clutter_points_entry,
                                                float* pose_total_points_entry,
                                                const uint8_t* source_label_entry,
                                                const int* pose_segmentation_label_entry,
                                                bool use_segmentation_label,
                                                float occlusion_threshold) {
                                                // float* l_entry,float* a_entry,float* b_entry){
            // refer to tiny renderer
            // https://github.com/ssloy/tinyrenderer/blob/master/our_gl.cpp
            float pts2[3][2];

            // viewport transform(0, 0, width, height)
            pts2[0][0] = dev_tri.v0.x/last_row.x*width/2.0f+width/2.0f; pts2[0][1] = dev_tri.v0.y/last_row.x*height/2.0f+height/2.0f;
            pts2[1][0] = dev_tri.v1.x/last_row.y*width/2.0f+width/2.0f; pts2[1][1] = dev_tri.v1.y/last_row.y*height/2.0f+height/2.0f;
            pts2[2][0] = dev_tri.v2.x/last_row.z*width/2.0f+width/2.0f; pts2[2][1] = dev_tri.v2.y/last_row.z*height/2.0f+height/2.0f;

            float bboxmin[2] = {FLT_MAX,  FLT_MAX};
            float bboxmax[2] = {-FLT_MAX, -FLT_MAX};

            float clamp_max[2] = {float(width-1), float(height-1)};
            float clamp_min[2] = {0, 0};

            size_t real_width = width;
            if(roi.width > 0 && roi.height > 0){  // depth will be flipped
                clamp_min[0] = roi.x;
                clamp_min[1] = height-1 - (roi.y + roi.height - 1);
                clamp_max[0] = (roi.x + roi.width) - 1;
                clamp_max[1] = height-1 - roi.y;
                real_width = roi.width;
            }


            for (int i=0; i<3; i++) {
                for (int j=0; j<2; j++) {
                    bboxmin[j] = std__max(clamp_min[j], std__min(bboxmin[j], pts2[i][j]));
                    bboxmax[j] = std__min(clamp_max[j], std__max(bboxmax[j], pts2[i][j]));
                }
            }

            size_t P[2];
            for(P[1] = size_t(bboxmin[1]+0.5f); P[1]<=bboxmax[1]; P[1] += 1){
                for(P[0] = size_t(bboxmin[0]+0.5f); P[0]<=bboxmax[0]; P[0] += 1){
                    Model::float3 bc_screen  = barycentric(pts2[0], pts2[1], pts2[2], P);

                    if (bc_screen.x<-0.0f || bc_screen.y<-0.0f || bc_screen.z<-0.0f ||
                            bc_screen.x>1.0f || bc_screen.y>1.0f || bc_screen.z>1.0f ) continue;

                    Model::float3 bc_over_z = {bc_screen.x/last_row.x, bc_screen.y/last_row.y, bc_screen.z/last_row.z};

                    // refer to https://en.wikibooks.org/wiki/Cg_Programming/Rasterization, Perspectively Correct Interpolation
        //            float frag_depth = (dev_tri.v0.z * bc_over_z.x + dev_tri.v1.z * bc_over_z.y + dev_tri.v2.z * bc_over_z.z)
        //                    /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

                    // this seems better
                    float frag_depth = (bc_screen.x + bc_screen.y + bc_screen.z)
                            /(bc_over_z.x + bc_over_z.y + bc_over_z.z);

                    size_t x_to_write = (P[0] + roi.x);
                    size_t y_to_write = (height-1 - P[1] - roi.y);
                    int32_t curr_depth = int32_t(frag_depth/**1000*/ + 0.5f);
                    // printf("x:%d, y:%d, depth:%d\n", x_to_write, y_to_write, curr_depth);
                    int32_t& depth_to_write = depth_entry[x_to_write+y_to_write*real_width];
                    const int32_t& source_depth = source_depth_entry[x_to_write+y_to_write*real_width];
                    uint8_t source_red = source_red_entry[x_to_write+y_to_write*real_width];
                    uint8_t source_green = source_green_entry[x_to_write+y_to_write*real_width];
                    uint8_t source_blue = source_blue_entry[x_to_write+y_to_write*real_width];
                    uint8_t source_label = 0;
                    if (use_segmentation_label == true)
                        source_label = source_label_entry[x_to_write+y_to_write*real_width];

                    // if(depth_to_write > curr_depth){
                    //     red_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v0);
                    //     green_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v1);
                    //     blue_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v2);
                    // }
                    // atomicMin(&depth_to_write, curr_depth);
                    bool wait = true;
                    while(wait){
                        if(0 == atomicExch(&lock_entry[x_to_write+y_to_write*real_width], 1)){
                            if(curr_depth < depth_entry[x_to_write+y_to_write*real_width]){
                                // occluding an existing point of same object
                                depth_entry[x_to_write+y_to_write*real_width] = curr_depth;
                                red_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v0);
                                green_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v1);
                                blue_entry[x_to_write+y_to_write*real_width] = (uint8_t)(dev_tri.color.v2);
                            }
                            lock_entry[x_to_write+y_to_write*real_width] = 0;
                            wait = false;
                        }
                    }
                    // 1.0 is 1cm occlusion threshold
                    int32_t& new_depth = depth_entry[x_to_write+y_to_write*real_width];
                    if ((use_segmentation_label == false && abs(new_depth - source_depth) > occlusion_threshold) ||
                        (use_segmentation_label == true && 
                        *pose_segmentation_label_entry != source_label && abs(new_depth - source_depth) > 0.5))
                    {
                        // printf("%d, %d\n", *pose_segmentation_label_entry, source_label);
                        // printf("%d, %d\n", source_depth, curr_depth);
                        if(new_depth > source_depth && source_depth > 0){
                            // when we are rendering at x,y where source pixel is also present at depth closer to camera
                            // valid condition as source occludes render
                            // if (false)
                            // {
                            //     // add source pixels
                            //     red_entry[x_to_write+y_to_write*real_width] = source_red;
                            //     green_entry[x_to_write+y_to_write*real_width] = source_green;
                            //     blue_entry[x_to_write+y_to_write*real_width] = source_blue;
                            //     atomicMin(&new_depth, source_depth);
                            // }
                            // else
                            // {
                                // add black
                                red_entry[x_to_write+y_to_write*real_width] = 0;
                                green_entry[x_to_write+y_to_write*real_width] = 0;
                                blue_entry[x_to_write+y_to_write*real_width] = 0;
                                atomicMax(&new_depth, INT_MAX);
                                if (USE_TREE)
                                    atomicOr(pose_occluded_other_entry, 1);
                                if (USE_CLUTTER)
                                {
                                    if (source_depth <=  new_depth - 5)
                                    {
                                        atomicAdd(pose_clutter_points_entry, 1);
                                    }
                                }
                        }
                        // invalid condition where source pixel is behind and we are rendering a pixel at same x,y with lesser depth 
                        else if(new_depth <= source_depth && source_depth > 0){
                            // invalid as render occludes source
                            if (USE_TREE)
                                atomicOr(pose_occluded_entry, 1);
                            // printf("Occlusion\n");
                        }
                    }
                    if (USE_CLUTTER)
                        atomicAdd(pose_total_points_entry, 1);

                }
            }
        }

        __global__ void render_triangle_multi(
            const Model::Triangle* device_tris_ptr, const size_t device_tris_size,
            const Model::mat4x4* device_poses_ptr, size_t device_poses_size,
            int32_t* depth_image_vec, size_t width, size_t height,
            const int* device_pose_model_map_ptr, int* device_tris_model_count_low_ptr,  
            int* device_tris_model_count_high_ptr,
            const Model::mat4x4 proj_mat, 
            const Model::ROI roi,
            uint8_t* red_image_vec,uint8_t* green_image_vec,uint8_t* blue_image_vec,
            const int32_t* device_source_depth_vec,
            const uint8_t* device_source_red_vec,
            const uint8_t* device_source_green_vec,
            const uint8_t* device_source_blue_vec,
            int* pose_occluded_vec,
            const int device_single_result_image,
            int32_t* lock_int_vec,
            int* pose_occluded_other_vec,
            float* pose_clutter_points_vec, 
            float* pose_total_points_vec,
            const uint8_t* device_source_mask_label_vec,
            int* pose_segmentation_label_vec,
            bool use_segmentation_label,
            const float occlusion_threshold
        ) {
            size_t pose_i = blockIdx.y;
            int model_id = device_pose_model_map_ptr[pose_i];
            size_t tri_i = blockIdx.x*blockDim.x + threadIdx.x;

            if(tri_i>=device_tris_size) return;

            if (!(tri_i < device_tris_model_count_high_ptr[model_id] && tri_i >= device_tris_model_count_low_ptr[model_id]))
                return; 

            size_t real_width = width;
            size_t real_height = height;
            if(roi.width > 0 && roi.height > 0){
                real_width = roi.width;
                real_height = roi.height;
            }
            int32_t* depth_entry;
            int32_t* lock_entry;
            uint8_t* red_entry;
            uint8_t* green_entry;
            uint8_t* blue_entry;
            int* pose_occluded_entry;
            int* pose_occluded_other_entry;
            float* pose_clutter_points_entry;
            float* pose_total_points_entry;
            int* pose_segmentation_label_entry = NULL;
            // printf("device_single_result_image:%d\n",device_single_result_image);
            if (device_single_result_image)
            {
                depth_entry = depth_image_vec; //length: width*height 32bits int
                red_entry = red_image_vec;
                green_entry = green_image_vec;
                blue_entry = blue_image_vec;
                pose_occluded_entry = pose_occluded_vec;
                lock_entry = lock_int_vec;
                pose_occluded_other_entry = pose_occluded_other_vec;
                pose_clutter_points_entry = pose_clutter_points_vec;
                pose_total_points_entry = pose_total_points_vec;
                if (pose_segmentation_label_vec != NULL)
                    pose_segmentation_label_entry = pose_segmentation_label_vec;
            }
            else
            {
                depth_entry = depth_image_vec + pose_i*real_width*real_height; //length: width*height 32bits int
                lock_entry = lock_int_vec + pose_i*real_width*real_height;
                red_entry = red_image_vec + pose_i*real_width*real_height;
                green_entry = green_image_vec + pose_i*real_width*real_height;
                blue_entry = blue_image_vec + pose_i*real_width*real_height;
                pose_occluded_entry = pose_occluded_vec + pose_i;
                pose_occluded_other_entry = pose_occluded_other_vec + pose_i;
                pose_clutter_points_entry = pose_clutter_points_vec + pose_i;
                pose_total_points_entry = pose_total_points_vec + pose_i;
                if (pose_segmentation_label_vec != NULL)
                    pose_segmentation_label_entry = pose_segmentation_label_vec + pose_i;
            }
            

            const Model::mat4x4* pose_entry = device_poses_ptr + pose_i; // length: 16 32bits float
            const Model::Triangle* tri_entry = device_tris_ptr + tri_i; // length: 9 32bits float

            // model transform
            Model::Triangle local_tri = transform_triangle(*tri_entry, *pose_entry);

            // assume last column of projection matrix is  0 0 1 0
            Model::float3 last_row = {
                local_tri.v0.z,
                local_tri.v1.z,
                local_tri.v2.z
            };
            // projection transform
            local_tri = transform_triangle(local_tri, proj_mat);

            rasterization_with_source(
                local_tri, last_row, depth_entry, width, height, roi,
                red_entry,green_entry,blue_entry,
                device_source_depth_vec,
                device_source_red_vec, device_source_green_vec, device_source_blue_vec,
                pose_occluded_entry,
                lock_entry,
                pose_occluded_other_entry,
                pose_clutter_points_entry,
                pose_total_points_entry,
                device_source_mask_label_vec,
                pose_segmentation_label_entry,
                use_segmentation_label,
                occlusion_threshold);
        }

        
    struct max2zero_functor_renderer{

        max2zero_functor_renderer(){}

        __host__ __device__
        int32_t operator()(const int32_t& x) const
        {
            return (x==INT_MAX)? 0: x;
        }
    };
}

void image_render(const thrust::device_vector<Model::Triangle>& device_tris,
                    const thrust::device_vector<Model::mat4x4>& device_poses,
                    const thrust::device_vector<int>& device_pose_model_map,
                    const thrust::device_vector<int>& device_tris_model_count,
                    const thrust::device_vector<int32_t> device_source_depth,
                    const thrust::device_vector<uint8_t> device_source_color_red,
                    const thrust::device_vector<uint8_t> device_source_color_green,
                    const thrust::device_vector<uint8_t> device_source_color_blue,
                    const thrust::device_vector<uint8_t> device_source_mask_label,
                    thrust::device_vector<int> device_pose_segmentation_label,
                    const int num_images,
                    const size_t width,
                    const size_t height,
                    const Model::mat4x4& proj_mat,
                    const float occlusion_threshold,
                    const int device_single_result_image,
                    thrust::device_vector<int>& device_pose_occluded,
                    thrust::device_vector<int>& device_pose_occluded_other,
                    thrust::device_vector<float>& device_pose_clutter_points,
                    thrust::device_vector<float>& device_pose_total_points,
                    thrust::device_vector<int32_t>& device_depth_int,
                    thrust::device_vector<uint8_t>& device_red_int,
                    thrust::device_vector<uint8_t>& device_green_int,
                    thrust::device_vector<uint8_t>& device_blue_int,
                    gpu_stats& stats) {
        
        printf("image_render()\n");
        /*
        *   Render image with occlusion, if segmentation label present, only do occlusion from different label (use seg label of pose and pixel label of source)
        *   If not seg label, do occlusion with entire source image (3dof)
        *   Both done with some threshold so that point doesnt occlude itself
        */
        
        const Model::ROI roi= {0, 0, 0, 0};
        // Create upper and lower limits for model triangles
        thrust::device_vector<int> device_tris_model_count_low = device_tris_model_count;
        thrust::device_vector<int> device_tris_model_count_high = device_tris_model_count;
        thrust::exclusive_scan(
            device_tris_model_count_low.begin(), device_tris_model_count_low.end(), 
            device_tris_model_count_low.begin(), 0
        ); // in-place scan
        thrust::inclusive_scan(
            device_tris_model_count_high.begin(), device_tris_model_count_high.end(), 
            device_tris_model_count_high.begin()
        ); // in-place scan
        printf("Number of triangles : %d\n", device_tris.size());
        printf("Number of poses : %d\n", num_images);

        // Create output vectors 
        device_pose_occluded.resize(num_images, 0);
        device_pose_occluded_other.resize(num_images, 0);
        device_pose_clutter_points.resize(num_images, 0);
        device_pose_total_points.resize(num_images, 0);

        thrust::device_vector<int32_t> device_lock_int(num_images*width*height, 0);
        device_depth_int.clear();
        device_red_int.clear();
        device_green_int.clear();
        device_blue_int.clear();
        device_depth_int.resize(num_images*width*height, INT_MAX);
        device_red_int.resize(num_images*width*height, 0);
        device_green_int.resize(num_images*width*height, 0);
        device_blue_int.resize(num_images*width*height, 0);

        // Create pointers for Kernel
        const Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
        const Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());

        //// Mapping each pose to model
        const int* device_pose_model_map_ptr = thrust::raw_pointer_cast(device_pose_model_map.data());

        //// Mapping each model to triangle range
        int* device_tris_model_count_low_ptr = thrust::raw_pointer_cast(device_tris_model_count_low.data());
        int* device_tris_model_count_high_ptr = thrust::raw_pointer_cast(device_tris_model_count_high.data());

        int* device_pose_occluded_vec = thrust::raw_pointer_cast(device_pose_occluded.data());
        int* device_pose_occluded_other_vec = thrust::raw_pointer_cast(device_pose_occluded_other.data());
        float* device_pose_clutter_points_vec = thrust::raw_pointer_cast(device_pose_clutter_points.data());
        float* device_pose_total_points_vec = thrust::raw_pointer_cast(device_pose_total_points.data());
        
        // Assign data on source image used as input for occlusion
        const int32_t* device_source_depth_vec = thrust::raw_pointer_cast(device_source_depth.data());
        const uint8_t* device_source_red_vec = thrust::raw_pointer_cast(device_source_color_red.data());
        const uint8_t* device_source_green_vec = thrust::raw_pointer_cast(device_source_color_green.data());
        const uint8_t* device_source_blue_vec = thrust::raw_pointer_cast(device_source_color_blue.data());
        
        // Pixel wise segmentation label data of every pixel in source image
        const uint8_t* device_source_mask_label_vec = thrust::raw_pointer_cast(device_source_mask_label.data());
        // Segmentation label used when available to do occlusion only from another label
        int* device_pose_segmentation_label_vec = thrust::raw_pointer_cast(device_pose_segmentation_label.data());
        bool use_segmentation_label = false;

        if (device_pose_segmentation_label.size() > 0)
        {
            //// 6-Dof case, segmentation label between pose and source image pixel would be compared for occlusion checking
            use_segmentation_label = true ;
        }
        printf("use_segmentation_label : %d\n", use_segmentation_label);

        // Assign output data
        int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());
        int32_t* lock_int_vec = thrust::raw_pointer_cast(device_lock_int.data());
        uint8_t* red_image_vec = thrust::raw_pointer_cast(device_red_int.data());
        uint8_t* green_image_vec = thrust::raw_pointer_cast(device_green_int.data());
        uint8_t* blue_image_vec = thrust::raw_pointer_cast(device_blue_int.data());

        stats.peak_memory_usage = std::max(print_cuda_memory_usage(), stats.peak_memory_usage);

        dim3 numBlocks((device_tris.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, num_images);
        image_renderer::render_triangle_multi<<<numBlocks, THREADS_PER_BLOCK>>>(device_tris_ptr, device_tris.size(),
                                                        device_poses_ptr, num_images,
                                                        depth_image_vec, width, height, 
                                                        device_pose_model_map_ptr, device_tris_model_count_low_ptr,
                                                        device_tris_model_count_high_ptr,
                                                        proj_mat, roi,
                                                        red_image_vec,green_image_vec,blue_image_vec,
                                                        device_source_depth_vec,
                                                        device_source_red_vec, device_source_green_vec, device_source_blue_vec,
                                                        device_pose_occluded_vec,
                                                        device_single_result_image,
                                                        lock_int_vec,
                                                        device_pose_occluded_other_vec,
                                                        device_pose_clutter_points_vec,
                                                        device_pose_total_points_vec,
                                                        device_source_mask_label_vec,
                                                        device_pose_segmentation_label_vec,
                                                        use_segmentation_label,
                                                        occlusion_threshold);

        thrust::transform(device_depth_int.begin(), device_depth_int.end(), 
                          device_depth_int.begin(), image_renderer::max2zero_functor_renderer());
        thrust::transform(device_red_int.begin(), device_red_int.end(),
                          device_red_int.begin(), image_renderer::max2zero_functor_renderer());
        thrust::transform(device_green_int.begin(), device_green_int.end(),
                          device_green_int.begin(), image_renderer::max2zero_functor_renderer());
        thrust::transform(device_blue_int.begin(), device_blue_int.end(),
                          device_blue_int.begin(), image_renderer::max2zero_functor_renderer());
        if (USE_CLUTTER)
        {
            // printf("Pose Clutter Ratio\n");
            thrust::transform(
                device_pose_clutter_points.begin(), device_pose_clutter_points.end(), 
                device_pose_total_points.begin(), device_pose_clutter_points.begin(), 
                thrust::divides<float>()
            );
            thrust::device_vector<float> rendered_multiplier_val(num_images, 100);
            thrust::transform(
                device_pose_clutter_points.begin(), device_pose_clutter_points.end(), 
                rendered_multiplier_val.begin(), device_pose_clutter_points.begin(), 
                thrust::multiplies<float>()
            );
            // thrust::copy(device_pose_clutter_points.begin(), device_pose_clutter_points.end(), clutter_cost.begin());
            // thrust::copy(
            //     device_pose_clutter_points.begin(),
            //     device_pose_clutter_points.end(), 
            //     std::ostream_iterator<float>(std::cout, " ")
            // );
            // printf("\n");
        }
        printf("image_render() done\n");
}
}

#endif