#include "cuda_renderer/common.cuh"
#include "cuda_renderer/image_renderer.cuh"
#include "cuda_renderer/compute_point_clouds.cuh"
#include "cuda_renderer/compute_costs.cuh"
#include "cuda_renderer/renderer.h"
#include "cuda_renderer/utils.cuh"
#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <fast_gicp/cuda/fast_gicp_cuda.cuh>

// #include <math.h> 
#include "cuda_fp16.h"
// #include <numeric> 
#define SQR(x) ((x)*(x))
#define POW2(x) SQR(x)
#define POW3(x) ((x)*(x)*(x))
#define POW4(x) (POW2(x)*POW2(x))
#define POW7(x) (POW3(x)*POW3(x)*(x))
#define DegToRad(x) ((x)*M_PI/180)
#define RadToDeg(x) ((x)/M_PI*180)
#define USE_TREE 0
#define USE_CLUTTER 0

namespace cuda_renderer {
    static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
    {
        if(err!=cudaSuccess)
        {
            fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
            std::cin.get();
            exit(EXIT_FAILURE);
        }
    }
    #define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
    {
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }


    template<typename T>
    device_vector_holder<T>::~device_vector_holder(){
        __free();
    }

    template<typename T>
    void device_vector_holder<T>::__free(){
        if(valid){
            cudaFree(__gpu_memory);
            valid = false;
            __size = 0;
        }
    }

    template<typename T>
    device_vector_holder<T>::device_vector_holder(size_t size_, T init)
    {
        __malloc(size_);
        thrust::fill(begin_thr(), end_thr(), init);
    }

    template<typename T>
    void device_vector_holder<T>::__malloc(size_t size_){
        if(valid) __free();
        cudaMalloc((void**)&__gpu_memory, size_ * sizeof(T));
        __size = size_;
        valid = true;
    }

    template<typename T>
    device_vector_holder<T>::device_vector_holder(size_t size_){
        __malloc(size_);
    }

    template class device_vector_holder<int>;

    

    struct max2zero_functor{

        max2zero_functor(){}

        __host__ __device__
        int32_t operator()(const int32_t& x) const
        {
            return (x==INT_MAX)? 0: x;
        }
    };

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

    __device__
    void rasterization_with_source(const Model::Triangle dev_tri, Model::float3 last_row,
                                            int32_t* depth_entry, size_t width, size_t height,
                                            const Model::ROI roi, 
                                            uint8_t* red_entry,uint8_t* green_entry,uint8_t* blue_entry,
                                            int32_t* source_depth_entry,
                                            uint8_t* source_red_entry,uint8_t* source_green_entry,uint8_t* source_blue_entry,
                                            int* pose_occluded_entry,
                                            int32_t* lock_entry,
                                            int* pose_occluded_other_entry,
                                            float* pose_clutter_points_entry,
                                            float* pose_total_points_entry,
                                            uint8_t* source_label_entry,
                                            int* pose_segmentation_label_entry,
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
                int32_t& source_depth = source_depth_entry[x_to_write+y_to_write*real_width];
                uint8_t source_red = source_red_entry[x_to_write+y_to_write*real_width];
                uint8_t source_green = source_green_entry[x_to_write+y_to_write*real_width];
                uint8_t source_blue = source_blue_entry[x_to_write+y_to_write*real_width];
                uint8_t source_label = source_label_entry[x_to_write+y_to_write*real_width];

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
                                    Model::Triangle* device_tris_ptr, size_t device_tris_size,
                                    Model::mat4x4* device_poses_ptr, size_t device_poses_size,
                                    int32_t* depth_image_vec, size_t width, size_t height,
                                    int* device_pose_model_map_ptr, int* device_tris_model_count_low_ptr,  
                                    int* device_tris_model_count_high_ptr,
                                    const Model::mat4x4 proj_mat, const Model::ROI roi,
                                    uint8_t* red_image_vec,uint8_t* green_image_vec,uint8_t* blue_image_vec,
                                    int32_t* device_source_depth_vec,
                                    uint8_t* device_source_red_vec,uint8_t* device_source_green_vec,uint8_t* device_source_blue_vec,
                                    int* pose_occluded_vec,
                                    int* device_single_result_image,
                                    int32_t* lock_int_vec,
                                    int* pose_occluded_other_vec,
                                    float* pose_clutter_points_vec, 
                                    float* pose_total_points_vec,
                                    uint8_t* device_source_mask_label_vec,
                                    int* pose_segmentation_label_vec,
                                    bool use_segmentation_label,
                                    float occlusion_threshold) {
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
        if (*device_single_result_image)
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
        

        Model::mat4x4* pose_entry = device_poses_ptr + pose_i; // length: 16 32bits float
        Model::Triangle* tri_entry = device_tris_ptr + tri_i; // length: 9 32bits float

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

        // rasterization(local_tri, last_row, depth_entry, width, height, roi,red_entry,green_entry,blue_entry);
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



    device_vector_holder<int> render_cuda_multi(
                                const std::vector<Model::Triangle>& tris,
                                const std::vector<Model::mat4x4>& poses,
                                const std::vector<int> pose_model_map,
                                const std::vector<int> tris_model_count,
                                size_t width, size_t height, const Model::mat4x4& proj_mat,
                                const std::vector<int32_t>& source_depth,
                                const std::vector<std::vector<uint8_t>>& source_color,
                                std::vector<int32_t>& result_depth, 
                                std::vector<std::vector<uint8_t>>& result_color,
                                std::vector<int>& pose_occluded,
                                int single_result_image,
                                std::vector<int>& pose_occluded_other,
                                std::vector<float>& clutter_cost,
                                const std::vector<uint8_t>& source_mask_label,
                                const std::vector<int>& pose_segmentation_label) {

        // Create device inputs
        int* device_single_result_image;
        cudaMalloc((void**)&device_single_result_image, sizeof(int));
        cudaMemcpy(device_single_result_image, &single_result_image, sizeof(int), cudaMemcpyHostToDevice);
        int num_images;
        if (single_result_image)
        {
            num_images = 1;
        }
        else
        {
            num_images = poses.size();
        }
        const Model::ROI roi= {0, 0, 0, 0};
        const size_t threadsPerBlock = 256;
        // std::cout <<tris[0].color.v1;
        thrust::device_vector<Model::Triangle> device_tris = tris;
        thrust::device_vector<Model::mat4x4> device_poses = poses;
        thrust::device_vector<int> device_tris_model_count_low = tris_model_count;
        thrust::device_vector<int> device_tris_model_count_high = tris_model_count;
        thrust::device_vector<int> device_pose_model_map = pose_model_map;
        thrust::device_vector<int> device_pose_segmentation_label = pose_segmentation_label;

        thrust::device_vector<int32_t> device_source_depth = source_depth;
        thrust::device_vector<uint8_t> device_source_color_red = source_color[0];
        thrust::device_vector<uint8_t> device_source_color_green = source_color[1];
        thrust::device_vector<uint8_t> device_source_color_blue = source_color[2];
        thrust::device_vector<uint8_t> device_source_mask_label = source_mask_label;

        // thrust::copy(
        //     device_tris_model_count.begin(),
        //     device_tris_model_count.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        printf("\nPose segmentation label : \n");
        thrust::copy(
            device_pose_segmentation_label.begin(),
            device_pose_segmentation_label.end(), 
            std::ostream_iterator<int>(std::cout, " ")
        );
        thrust::exclusive_scan(
            device_tris_model_count_low.begin(), device_tris_model_count_low.end(), 
            device_tris_model_count_low.begin(), 0
        ); // in-place scan
        thrust::inclusive_scan(
            device_tris_model_count_high.begin(), device_tris_model_count_high.end(), 
            device_tris_model_count_high.begin()
        ); // in-place scan
        // thrust::copy(
        //     device_tris_model_count_low.begin(),
        //     device_tris_model_count_low.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        // printf("\n");
        // thrust::copy(
        //     device_tris_model_count_high.begin(),
        //     device_tris_model_count_high.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        // printf("\n");
        // thrust::copy(
        //     device_pose_model_map.begin(),
        //     device_pose_model_map.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        printf("\nNumber of triangles : %d\n", tris.size());
        printf("Number of poses : %d\n", num_images);

        size_t real_width = width;
        size_t real_height = height;

        // atomic min only support int32
        
        // Create device outputs
        thrust::device_vector<int> device_pose_occluded(num_images, 0);
        thrust::device_vector<int> device_pose_occluded_other(num_images, 0);
        thrust::device_vector<float> device_pose_clutter_points(num_images, 0);
        thrust::device_vector<float> device_pose_total_points(num_images, 0);

        device_vector_holder<int32_t> device_depth_int(num_images*real_width*real_height, INT_MAX);
        // thrust::device_vector<int32_t> device_depth_int(poses.size()*real_width*real_height, INT_MAX);
        thrust::device_vector<int32_t> device_lock_int(num_images*real_width*real_height, 0);
        thrust::device_vector<uint8_t> device_red_int(num_images*real_width*real_height, 0);
        thrust::device_vector<uint8_t> device_green_int(num_images*real_width*real_height, 0);
        thrust::device_vector<uint8_t> device_blue_int(num_images*real_width*real_height, 0);

    
        Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
        Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());

        // Mapping each pose to model
        int* device_pose_model_map_ptr = thrust::raw_pointer_cast(device_pose_model_map.data());

        // Mapping each model to triangle range
        int* device_tris_model_count_low_ptr = thrust::raw_pointer_cast(device_tris_model_count_low.data());
        int* device_tris_model_count_high_ptr = thrust::raw_pointer_cast(device_tris_model_count_high.data());
        // int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());

        int* device_pose_occluded_vec = thrust::raw_pointer_cast(device_pose_occluded.data());
        int* device_pose_occluded_other_vec = thrust::raw_pointer_cast(device_pose_occluded_other.data());
        float* device_pose_clutter_points_vec = thrust::raw_pointer_cast(device_pose_clutter_points.data());
        float* device_pose_total_points_vec = thrust::raw_pointer_cast(device_pose_total_points.data());
        int* device_pose_segmentation_label_vec = thrust::raw_pointer_cast(device_pose_segmentation_label.data());
        bool use_segmentation_label = false;
        if (device_pose_segmentation_label.size() > 0)
            use_segmentation_label = true ;

        int32_t* device_source_depth_vec = thrust::raw_pointer_cast(device_source_depth.data());
        uint8_t* device_source_red_vec = thrust::raw_pointer_cast(device_source_color_red.data());
        uint8_t* device_source_green_vec = thrust::raw_pointer_cast(device_source_color_green.data());
        uint8_t* device_source_blue_vec = thrust::raw_pointer_cast(device_source_color_blue.data());
        uint8_t* device_source_mask_label_vec = thrust::raw_pointer_cast(device_source_mask_label.data());

        int32_t* depth_image_vec = device_depth_int.data();
        int32_t* lock_int_vec = thrust::raw_pointer_cast(device_lock_int.data());
        uint8_t* red_image_vec = thrust::raw_pointer_cast(device_red_int.data());
        uint8_t* green_image_vec = thrust::raw_pointer_cast(device_green_int.data());
        uint8_t* blue_image_vec = thrust::raw_pointer_cast(device_blue_int.data());

        // Initialize rendered images with source images
        dim3 block(16,16);
        dim3 grid((real_width*num_images + block.x - 1)/block.x, (real_height + block.y - 1)/block.y);
        // copy_source_to_render<<<grid,block>>>(red_image_vec,green_image_vec,blue_image_vec,
        //                             depth_image_vec,
        //                             device_source_red_vec, device_source_green_vec, device_source_blue_vec,
        //                             device_source_depth_vec,
        //                             width,height,num_images);
        // cudaDeviceSynchronize();

        // Render all poses
        dim3 numBlocks((tris.size() + threadsPerBlock - 1) / threadsPerBlock, poses.size());
        render_triangle_multi<<<numBlocks, threadsPerBlock>>>(device_tris_ptr, tris.size(),
                                                        device_poses_ptr, poses.size(),
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
                                                        1.0);
        // cudaDeviceSynchronize();
        // Objects occluding other objects already in the scene
        if (USE_TREE)
        {
            printf("Pose Occlusions\n");
            thrust::copy(
                device_pose_occluded.begin(),
                device_pose_occluded.end(), 
                std::ostream_iterator<int>(std::cout, " ")
            );
            printf("\n");
            thrust::copy(device_pose_occluded.begin(), device_pose_occluded.end(), pose_occluded.begin());
            
            // Objects occluded by existing objects in the scene, need to do ICP again for these
            printf("Pose Occlusions Other\n");
            thrust::copy(
                device_pose_occluded_other.begin(),
                device_pose_occluded_other.end(), 
                std::ostream_iterator<int>(std::cout, " ")
            );
            printf("\n");
            thrust::copy(device_pose_occluded_other.begin(), device_pose_occluded_other.end(), pose_occluded_other.begin());
        }
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
            thrust::copy(device_pose_clutter_points.begin(), device_pose_clutter_points.end(), clutter_cost.begin());
        }
        // thrust::copy(
        //     device_pose_clutter_points.begin(),
        //     device_pose_clutter_points.end(), 
        //     std::ostream_iterator<float>(std::cout, " ")
        // );
        printf("\n");

        result_depth.resize(num_images*real_width*real_height);
        {
            thrust::device_vector<int32_t> v3(depth_image_vec, depth_image_vec + num_images*real_width*real_height);
            thrust::transform(v3.begin(), v3.end(),v3.begin(), max2zero_functor());
            thrust::copy(v3.begin(), v3.end(), result_depth.begin());

        }
        
        std::vector<uint8_t> result_red(num_images*real_width*real_height);
        std::vector<uint8_t> result_green(num_images*real_width*real_height);
        std::vector<uint8_t> result_blue(num_images*real_width*real_height);
        {
            thrust::transform(device_red_int.begin(), device_red_int.end(),
                            device_red_int.begin(), max2zero_functor());
            thrust::copy(device_red_int.begin(), device_red_int.end(), result_red.begin());
            thrust::transform(device_green_int.begin(), device_green_int.end(),
                            device_green_int.begin(), max2zero_functor());
            thrust::copy(device_green_int.begin(), device_green_int.end(), result_green.begin());
            thrust::transform(device_blue_int.begin(), device_blue_int.end(),
                            device_blue_int.begin(), max2zero_functor());
            thrust::copy(device_blue_int.begin(), device_blue_int.end(), result_blue.begin());

        }
        if (result_color.size() > 0) result_color.clear();
        result_color.push_back(result_red);
        result_color.push_back(result_green);
        result_color.push_back(result_blue);


        thrust::transform(device_depth_int.begin_thr(), device_depth_int.end_thr(),
                        device_depth_int.begin_thr(), max2zero_functor());
        return device_depth_int;
    }

    // __global__ void compute_observed_cost_cylinder(
    //     int num_poses,
    //     int observed_cloud_point_num,
    //     uint8_t* cuda_observed_explained,
    //     float* cuda_observed_cloud,
    //     size_t cloud_pitch,
    //     Model::mat4x4* device_poses_ptr,
    //     float* pose_cylinder_radius,
    //     float* observed_total_unexplained)
    // {
    //     /*
    //      * @observed_cloud_point_num - number of points in observed scene
    //      * @cuda_observed_explained (num_poses x observed_point_num) 
    //      *      - binary value indicating whether given point is explained or not based on distance
    //      */
    //     size_t point_index = blockIdx.x*blockDim.x + threadIdx.x;
    //     if(point_index >= num_poses * observed_cloud_point_num) return;

    //     size_t pose_index = point_index/observed_cloud_point_num;
    //     size_t cloud_point_index = point_index % observed_cloud_point_num;

    //     float cylinder_radius = pose_cylinder_radius[pose_index];
    //     Model::mat4x4 pose_entry = device_poses_ptr[pose_index];

    //     float pose_x = pose_entry.a3;
    //     float pose_y = pose_entry.b3;
    //     float pose_z = pose_entry.c3;

    //     float* row_0 = (float *)((char*)cuda_observed_cloud + 0 * cloud_pitch);
    //     float* row_1 = (float *)((char*)cuda_observed_cloud + 1 * cloud_pitch);
    //     float* row_2 = (float *)((char*)cuda_observed_cloud + 2 * cloud_pitch);
    //     float point_x = row_0[cloud_point_index];
    //     float point_y = row_1[cloud_point_index];
    //     float point_z = row_2[cloud_point_index];

    //     // printf("pose_x:%f, pose_y:%f, point_x:%f, point_y:%f, distance:\n", pose_x, pose_y, point_x, point_y);

    //     if (sqrt((point_x - pose_x)*(point_x - pose_x) + (point_y - pose_y)*(point_y - pose_y)) <  cylinder_radius)
    //     {
    //         // lies within the inscribed volume of pose
    //         if (cuda_observed_explained[point_index] == 0)
    //         {
    //             atomicAdd(&observed_total_unexplained[pose_index], 1);
    //         }
    //     }


    // }
    void render_cuda_multi_unified_old(
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
        uint8_t* observed_color,
        int observed_point_num,
        std::vector<float> pose_observed_points_total,
        int* result_observed_cloud_label,
        int cost_type,
        bool calculate_observed_cost,
        float sensor_resolution,
        float color_distance_threshold,
        float occlusion_threshold,
        std::vector<int32_t>& result_depth, 
        std::vector<std::vector<uint8_t>>& result_color,
        float* &result_cloud,
        uint8_t* &result_cloud_color,
        int& result_cloud_point_num,
        int* &result_cloud_pose_map,
        int* &result_dc_index,
        float* &rendered_cost,
        float* &observed_cost,
        float* &points_diff_cost,
        double &peak_memory_usage) {
        /*
         * - @source_mask_label - Label for every pixel in source image, used for segmentation specific occlusion checking
         * - Currently doesnt support pose occlusion or pose occlusion other
         */
        
        // std::string stage = "DEBUG";
        printf("---------------------------------------\n");
        printf("Stage : %s\n", stage.c_str());
        printf("USE_CLUTTER : %d\n", USE_CLUTTER);
        printf("USE_TREE : %d\n", USE_TREE);
        printf("sensor_resolution : %f\n", sensor_resolution);
        printf("color_distance_threshold : %f\n", color_distance_threshold);
        printf("cost_type : %d\n", cost_type);
        printf("point_dim : %d\n", point_dim);
        printf("stride : %d\n", stride);
        printf("depth_factor : %d\n", depth_factor);
        printf("observed_point_num : %d\n", observed_point_num);
        printf("occlusion_threshold : %f\n", occlusion_threshold);
        printf("calculate_observed_cost : %d\n", calculate_observed_cost);

        std::chrono::time_point<std::chrono::system_clock> start, end_1, end_2, end_3, end_3a, end_3b, end_3c, end_4;
        start = std::chrono::system_clock::now();
        // Create device inputs
        int* device_single_result_image;
        cudaMalloc((void**)&device_single_result_image, sizeof(int));
        cudaMemcpy(device_single_result_image, &single_result_image, sizeof(int), cudaMemcpyHostToDevice);
        int num_images;
        if (single_result_image)
        {
            num_images = 1;
        }
        else
        {
            num_images = poses.size();
        }
        const Model::ROI roi= {0, 0, 0, 0};
        const size_t threadsPerBlock = 256;
        // std::cout <<tris[0].color.v1;
        thrust::device_vector<Model::Triangle> device_tris = tris;
        thrust::device_vector<Model::mat4x4> device_poses = poses;
        //// Every index maps a model id to a range of triangles in the triangle vector 
        thrust::device_vector<int> device_tris_model_count_low = tris_model_count;
        thrust::device_vector<int> device_tris_model_count_high = tris_model_count;
        thrust::device_vector<int> device_pose_model_map = pose_model_map;
        thrust::device_vector<int> device_pose_segmentation_label = pose_segmentation_label;

        thrust::device_vector<int32_t> device_source_depth = source_depth;
        thrust::device_vector<uint8_t> device_source_color_red = source_color[0];
        thrust::device_vector<uint8_t> device_source_color_green = source_color[1];
        thrust::device_vector<uint8_t> device_source_color_blue = source_color[2];
        thrust::device_vector<uint8_t> device_source_mask_label = source_mask_label;

        // thrust::copy(
        //     device_tris_model_count.begin(),
        //     device_tris_model_count.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        // printf("\nPose segmentation label : \n");
        // thrust::copy(
        //     device_pose_segmentation_label.begin(),
        //     device_pose_segmentation_label.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        thrust::exclusive_scan(
            device_tris_model_count_low.begin(), device_tris_model_count_low.end(), 
            device_tris_model_count_low.begin(), 0
        ); // in-place scan
        thrust::inclusive_scan(
            device_tris_model_count_high.begin(), device_tris_model_count_high.end(), 
            device_tris_model_count_high.begin()
        ); // in-place scan
        // thrust::copy(
        //     device_tris_model_count_low.begin(),
        //     device_tris_model_count_low.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        // printf("\n");
        // thrust::copy(
        //     device_tris_model_count_high.begin(),
        //     device_tris_model_count_high.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        // printf("\n");
        // thrust::copy(
        //     device_pose_model_map.begin(),
        //     device_pose_model_map.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        printf("\nNumber of triangles : %d\n", tris.size());
        printf("Number of poses : %d\n", num_images);

        size_t real_width = width;
        size_t real_height = height;

        // atomic min only support int32
        
        // Create device outputs
        thrust::device_vector<int> device_pose_occluded(num_images, 0);
        thrust::device_vector<int> device_pose_occluded_other(num_images, 0);
        thrust::device_vector<float> device_pose_clutter_points(num_images, 0);
        thrust::device_vector<float> device_pose_total_points(num_images, 0);

        thrust::device_vector<int32_t> device_depth_int(num_images*real_width*real_height, INT_MAX);
        thrust::device_vector<int32_t> device_lock_int(num_images*real_width*real_height, 0);
        thrust::device_vector<uint8_t> device_red_int(num_images*real_width*real_height, 0);
        thrust::device_vector<uint8_t> device_green_int(num_images*real_width*real_height, 0);
        thrust::device_vector<uint8_t> device_blue_int(num_images*real_width*real_height, 0);

    
        Model::Triangle* device_tris_ptr = thrust::raw_pointer_cast(device_tris.data());
        Model::mat4x4* device_poses_ptr = thrust::raw_pointer_cast(device_poses.data());

        //// Mapping each pose to model
        int* device_pose_model_map_ptr = thrust::raw_pointer_cast(device_pose_model_map.data());

        //// Mapping each model to triangle range
        int* device_tris_model_count_low_ptr = thrust::raw_pointer_cast(device_tris_model_count_low.data());
        int* device_tris_model_count_high_ptr = thrust::raw_pointer_cast(device_tris_model_count_high.data());

        int* device_pose_occluded_vec = thrust::raw_pointer_cast(device_pose_occluded.data());
        int* device_pose_occluded_other_vec = thrust::raw_pointer_cast(device_pose_occluded_other.data());
        float* device_pose_clutter_points_vec = thrust::raw_pointer_cast(device_pose_clutter_points.data());
        float* device_pose_total_points_vec = thrust::raw_pointer_cast(device_pose_total_points.data());
        int* device_pose_segmentation_label_vec = thrust::raw_pointer_cast(device_pose_segmentation_label.data());
        bool use_segmentation_label = false;

        if (device_pose_segmentation_label.size() > 0)
        {
            //// 6-Dof case, segmentation label between pose and source image pixel would be compared for occlusion checking
            use_segmentation_label = true ;
        }
        printf("use_segmentation_label : %d\n", use_segmentation_label);
        int32_t* device_source_depth_vec = thrust::raw_pointer_cast(device_source_depth.data());
        uint8_t* device_source_red_vec = thrust::raw_pointer_cast(device_source_color_red.data());
        uint8_t* device_source_green_vec = thrust::raw_pointer_cast(device_source_color_green.data());
        uint8_t* device_source_blue_vec = thrust::raw_pointer_cast(device_source_color_blue.data());
        uint8_t* device_source_mask_label_vec = thrust::raw_pointer_cast(device_source_mask_label.data());

        int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());
        int32_t* lock_int_vec = thrust::raw_pointer_cast(device_lock_int.data());
        uint8_t* red_image_vec = thrust::raw_pointer_cast(device_red_int.data());
        uint8_t* green_image_vec = thrust::raw_pointer_cast(device_green_int.data());
        uint8_t* blue_image_vec = thrust::raw_pointer_cast(device_blue_int.data());
        
        peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);
        //// Render all poses
        dim3 numBlocks((tris.size() + threadsPerBlock - 1) / threadsPerBlock, poses.size());
        render_triangle_multi<<<numBlocks, threadsPerBlock>>>(device_tris_ptr, tris.size(),
                                                        device_poses_ptr, poses.size(),
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
            thrust::copy(device_pose_clutter_points.begin(), device_pose_clutter_points.end(), clutter_cost.begin());
            // thrust::copy(
            //     device_pose_clutter_points.begin(),
            //     device_pose_clutter_points.end(), 
            //     std::ostream_iterator<float>(std::cout, " ")
            // );
            // printf("\n");
        }
        
        /// Convert INT_MAXs to zeros
        thrust::transform(device_depth_int.begin(), device_depth_int.end(), 
                            device_depth_int.begin(), max2zero_functor());
        thrust::transform(device_red_int.begin(), device_red_int.end(),
                            device_red_int.begin(), max2zero_functor());
        thrust::transform(device_green_int.begin(), device_green_int.end(),
                            device_green_int.begin(), max2zero_functor());
        thrust::transform(device_blue_int.begin(), device_blue_int.end(),
                            device_blue_int.begin(), max2zero_functor());
        
        // Free memory for stuff not needed by cloud construction
        device_tris.clear(); device_tris.shrink_to_fit();
        device_tris_model_count_low.clear(); device_tris_model_count_low.shrink_to_fit();
        device_tris_model_count_high.clear(); device_tris_model_count_high.shrink_to_fit();
        device_pose_model_map.clear(); device_pose_model_map.shrink_to_fit();
        device_poses.clear(); device_poses.shrink_to_fit();
        device_source_depth.clear(); device_source_depth.shrink_to_fit();
        device_source_color_blue.clear(); device_source_color_blue.shrink_to_fit();
        device_source_color_green.clear(); device_source_color_green.shrink_to_fit();
        device_source_color_red.clear(); device_source_color_red.shrink_to_fit();

        end_1 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_1-start;
        printf("*************Rendering Images Done**********\n");
        printf("*************Render time : %f*************\n", elapsed_seconds.count());
        if (stage.compare("DEBUG") == 0 || stage.compare("RENDER") == 0)
        {
            printf("Copying images to CPU\n");
            //// Allocate CPU memory
            std::vector<uint8_t> result_red(num_images*real_width*real_height);
            std::vector<uint8_t> result_green(num_images*real_width*real_height);
            std::vector<uint8_t> result_blue(num_images*real_width*real_height);
            result_depth.resize(num_images*real_width*real_height);
            
            //// Copy from GPU to CPU
            thrust::copy(device_depth_int.begin(), device_depth_int.end(), result_depth.begin());
            thrust::copy(device_red_int.begin(), device_red_int.end(), result_red.begin());
            thrust::copy(device_green_int.begin(), device_green_int.end(), result_green.begin());
            thrust::copy(device_blue_int.begin(), device_blue_int.end(), result_blue.begin());
            result_color.push_back(result_red);
            result_color.push_back(result_green);
            result_color.push_back(result_blue);

            /// Vectors will be free automatically on return

            if (stage.compare("RENDER") == 0) return;
        }
        
        ///////////////////////////////////////////////////////////////

        dim3 threadsPerBlock2D(16, 16);
        assert(real_width % stride == 0);
        dim3 numBlocks2D((real_width/stride * num_images + threadsPerBlock2D.x - 1)/threadsPerBlock2D.x, (real_height/stride + threadsPerBlock2D.y - 1)/threadsPerBlock2D.y);
        thrust::device_vector<int> mask(real_width*real_height*num_images, 0);
        int* mask_ptr = thrust::raw_pointer_cast(mask.data());

        depth_to_mask<<<numBlocks2D, threadsPerBlock2D>>>(depth_image_vec, 
                                                        mask_ptr, 
                                                        real_width, 
                                                        real_height, 
                                                        stride, 
                                                        num_images,
                                                        device_pose_occluded_vec,
                                                        NULL,
                                                        kCameraCX, kCameraCY, kCameraFX, kCameraFY, depth_factor,
                                                        NULL, NULL);
        if (cudaGetLastError() != cudaSuccess) 
        {
            printf("ERROR: Unable to execute kernel depth_to_mask\n");
        }

        //// Create mapping from pixel to corresponding index in point cloud
        int mask_back_temp = mask.back();
        thrust::exclusive_scan(mask.begin(), mask.end(), mask.begin(), 0); // in-place scan
        result_cloud_point_num = mask.back() + mask_back_temp;
        printf("Actual points in all clouds : %d\n", result_cloud_point_num);

        float* cuda_cloud;
        uint8_t* cuda_cloud_color;
        int* cuda_cloud_pose_map;
        size_t query_pitch_in_bytes;

        const unsigned int size_of_float = sizeof(float);
        const unsigned int size_of_int   = sizeof(int);
        const unsigned int size_of_uint   = sizeof(uint8_t);
        int k = 1;

        // cudaMalloc(&cuda_cloud, point_dim * result_cloud_point_num * sizeof(float));
        //// Allocate memory for outputs
        cudaMalloc(&cuda_cloud_color, point_dim * result_cloud_point_num * sizeof(uint8_t));
        cudaMalloc(&cuda_cloud_pose_map, result_cloud_point_num * sizeof(int));
        cudaMallocPitch(&cuda_cloud,   &query_pitch_in_bytes,   result_cloud_point_num * size_of_float, point_dim);

        peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);
        //// Use Mapping to convert images to point clouds
        size_t query_pitch = query_pitch_in_bytes / size_of_float;
        depth_to_2d_cloud<<<numBlocks2D, threadsPerBlock2D>>>(
                            depth_image_vec, red_image_vec, green_image_vec, blue_image_vec,
                            cuda_cloud, query_pitch_in_bytes, cuda_cloud_color, result_cloud_point_num, mask_ptr, width, height, 
                            kCameraCX, kCameraCY, kCameraFX, kCameraFY, depth_factor, stride, num_images, cuda_cloud_pose_map,
                            NULL, NULL, NULL, NULL);
        if (cudaGetLastError() != cudaSuccess) 
        {
            printf("ERROR: Unable to execute kernel depth_to_2d_cloud\n");
        }
        //// Free image memory used during point cloud construction
        device_depth_int.clear(); device_depth_int.shrink_to_fit();
        device_red_int.clear(); device_red_int.shrink_to_fit();
        device_blue_int.clear(); device_blue_int.shrink_to_fit();
        device_green_int.clear(); device_green_int.shrink_to_fit();
        if (stage.compare("DEBUG") == 0 || stage.find("CLOUD") != std::string::npos)
        {
            printf("Copying point clouds to CPU\n");
            //// Allocate CPU memory
            result_cloud = (float*) malloc(point_dim * result_cloud_point_num * sizeof(float));
            result_cloud_color = (uint8_t*) malloc(point_dim * result_cloud_point_num * sizeof(uint8_t));
            result_dc_index = (int*) malloc(num_images * width * height * sizeof(int));
            result_cloud_pose_map = (int*) malloc(result_cloud_point_num * sizeof(int));

            //// Copy to CPU if needed
            cudaMemcpy2D(
                result_cloud,  result_cloud_point_num * size_of_float, cuda_cloud,  query_pitch_in_bytes,  result_cloud_point_num * size_of_float, point_dim, cudaMemcpyDeviceToHost);
            // cudaMemcpy(result_cloud, cuda_cloud, point_dim * result_cloud_point_num * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(result_cloud_color, cuda_cloud_color, point_dim * result_cloud_point_num * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(result_dc_index, mask_ptr, num_images * width * height * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(result_cloud_pose_map, cuda_cloud_pose_map, result_cloud_point_num * sizeof(int), cudaMemcpyDeviceToHost);
            
            /// Exit here if only point clouds are needed - for e.g. before ICP
            /// Free copied stuff
            
            if (stage.compare("CLOUD") == 0) {
                cudaFree(cuda_cloud);
                cudaFree(cuda_cloud_color);
                cudaFree(cuda_cloud_pose_map);
                return;
            }
        }
        // Free any vectors not needed later
        mask.clear(); mask.shrink_to_fit();
        printf("************Point clouds created*************\n");
        end_2 = std::chrono::system_clock::now();
        elapsed_seconds = end_2-end_1;
        printf("************Cloud contruction time : %f************\n", elapsed_seconds.count());


        /////////////////////////////////////////////////////////////////////////////

        // Allocate memory for KNN
        // Query is render and Ref is observed
        float* ref_dev;
        float* dist_dev;
        int* index_dev;
        size_t ref_pitch_in_bytes, dist_pitch_in_bytes, index_pitch_in_bytes;
        // cudaMallocPitch(&cuda_cloud, &query_pitch_in_bytes, result_cloud_point_num * size_of_float, point_dim);
        cudaError_t err0, err1, err2;
        err0 = cudaMallocPitch(&ref_dev, &ref_pitch_in_bytes, observed_point_num * size_of_float, point_dim);
        err1 = cudaMallocPitch(&dist_dev,  &dist_pitch_in_bytes,  result_cloud_point_num * size_of_float, observed_point_num);
        err2 = cudaMallocPitch(&index_dev, &index_pitch_in_bytes, result_cloud_point_num * size_of_int,   k);
        if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
            printf("ERROR: Memory allocation error (cudaMallocPitch)\n");
        }
         // Deduce pitch values
        size_t ref_pitch = ref_pitch_in_bytes / size_of_float;
        size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
        size_t index_pitch = index_pitch_in_bytes / size_of_int;
        if (query_pitch != dist_pitch || query_pitch != index_pitch) {
            printf("ERROR: Invalid pitch value\n");
            return;
        }

        //// Copy observed data
        // cudaMemcpy2D(cuda_cloud, query_pitch_in_bytes, result_cloud, result_cloud_point_num * size_of_float, result_cloud_point_num * size_of_float, point_dim, cudaMemcpyHostToDevice);
        cudaMemcpy2D(ref_dev, ref_pitch_in_bytes, observed_depth, observed_point_num * size_of_float, observed_point_num * size_of_float, point_dim, cudaMemcpyHostToDevice);

        peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);
        // Compute distances and nearest neighbours
        dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
        dim3 grid0(result_cloud_point_num / BLOCK_DIM, result_cloud_point_num / BLOCK_DIM, 1);
        if (result_cloud_point_num % BLOCK_DIM != 0) grid0.x += 1;
        if (result_cloud_point_num   % BLOCK_DIM != 0) grid0.y += 1;
        compute_distances_render<<<grid0, block0>>>(ref_dev, observed_point_num, ref_pitch, cuda_cloud, result_cloud_point_num, query_pitch, point_dim, dist_dev);
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel compute_distances_render\n");
            return;
        }
        printf("compute_distances_render() done\n");
        end_3a = std::chrono::system_clock::now();
        elapsed_seconds = end_3a-end_2;
        printf("*************compute_distances_render time : %f************\n", elapsed_seconds.count());

        dim3 block1(256, 1, 1);
        dim3 grid1(result_cloud_point_num / 256, 1, 1);
        if (result_cloud_point_num % 256 != 0) grid1.x += 1;
        modified_insertion_sort_render<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, result_cloud_point_num, observed_point_num, k);    
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel modified_insertion_sort_render\n");
            return;
        }
        printf("modified_insertion_sort_render() done\n");
        end_3b = std::chrono::system_clock::now();
        elapsed_seconds = end_3b-end_3a;
        printf("*************modified_insertion_sort_render time : %f************\n", elapsed_seconds.count());

        dim3 block2(16, 16, 1);
        dim3 grid2(result_cloud_point_num / 16, k / 16, 1);
        if (result_cloud_point_num % 16 != 0) grid2.x += 1;
        if (k % 16 != 0)        grid2.y += 1;
        compute_sqrt_render<<<grid2, block2>>>(dist_dev, result_cloud_point_num, query_pitch, k);	
        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel compute_sqrt_render\n");
            return;
        }
        printf("compute_sqrt_render() done\n");
        end_3c = std::chrono::system_clock::now();
        elapsed_seconds = end_3c-end_3b;
        printf("*************compute_sqrt_render time : %f************\n", elapsed_seconds.count());

        // float* knn_dist;
        // int* knn_index;
        // cudaMalloc(&knn_dist, result_cloud_point_num * k * size_of_float);
        // cudaMalloc(&knn_index, result_cloud_point_num * k * size_of_int);

        if (stage.compare("DEBUG") == 0)
        {
            float* knn_dist_cpu   = (float*) malloc(result_cloud_point_num * k * sizeof(float));
            int* knn_index_cpu  = (int*)   malloc(result_cloud_point_num * k * sizeof(int));

            cudaMemcpy2D(knn_dist_cpu,  result_cloud_point_num * size_of_float, dist_dev,  dist_pitch_in_bytes,  result_cloud_point_num * size_of_float, k, cudaMemcpyDeviceToHost);
            cudaMemcpy2D(knn_index_cpu, result_cloud_point_num * size_of_int,   index_dev, index_pitch_in_bytes, result_cloud_point_num * size_of_int,   k, cudaMemcpyDeviceToHost);
            
            // cudaMemcpy(knn_dist, knn_dist_cpu, result_cloud_point_num * size_of_float, cudaMemcpyHostToDevice);
            // cudaMemcpy(knn_index, knn_index_cpu, result_cloud_point_num * size_of_int, cudaMemcpyHostToDevice);
            // for(int i = 0; i < result_cloud_point_num; i++){
            //     printf("knn dist:%f\n", knn_dist_cpu[i]);
            // }

            /// Not returning so need to free anything
        }
        //// Free depth point cloud and reference cloud since not needed for cost computation 
        cudaFree(cuda_cloud);
        cudaFree(ref_dev); //TODO
        printf("*************KNN distances computed**********\n");
        end_3 = std::chrono::system_clock::now();
        elapsed_seconds = end_3-end_2;
        printf("*************KNN time : %f************\n", elapsed_seconds.count());

        ///////////////////////////////////////////////////////////////////

        // Allocate outputs
        thrust::device_vector<float> cuda_rendered_cost_vec(num_images, 0);
        float* cuda_rendered_cost = thrust::raw_pointer_cast(cuda_rendered_cost_vec.data());
        thrust::device_vector<float> cuda_pose_point_num_vec(num_images, 0);
        float* cuda_pose_point_num = thrust::raw_pointer_cast(cuda_pose_point_num_vec.data());
        thrust::device_vector<float> cuda_rendered_explained_vec(num_images, 0);

        // Points in observed that get explained by render
        thrust::device_vector<uint8_t> cuda_observed_explained_vec(num_images * observed_point_num, 0);
        uint8_t* cuda_observed_explained = thrust::raw_pointer_cast(cuda_observed_explained_vec.data());
        int* cuda_observed_cloud_label;
        uint8_t* cuda_observed_cloud_color;

        cudaMalloc(&cuda_observed_cloud_color, point_dim * observed_point_num * size_of_uint);
        cudaMemcpy(cuda_observed_cloud_color, observed_color, point_dim * observed_point_num * size_of_uint, cudaMemcpyHostToDevice);

        if (cost_type == 2)
        {
            cudaMalloc(&cuda_observed_cloud_label, observed_point_num * size_of_int);
            cudaMemcpy(cuda_observed_cloud_label, result_observed_cloud_label, observed_point_num * size_of_int, cudaMemcpyHostToDevice);
        }
        peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);

        dim3 numBlocksR((result_cloud_point_num + threadsPerBlock - 1) / threadsPerBlock, 1);
        compute_render_cost<<<numBlocksR, threadsPerBlock>>>(
            dist_dev,
            index_dev,
            cuda_cloud_pose_map,
            device_pose_occluded_vec,
            cuda_rendered_cost,
            sensor_resolution,
            result_cloud_point_num,
            observed_point_num,
            cuda_pose_point_num, // Can be 0 if that pose had no points in it
            cuda_cloud_color,
            cuda_observed_cloud_color,
            cuda_cloud,
            cuda_observed_explained,
            device_pose_segmentation_label_vec,
            cuda_observed_cloud_label,
            cost_type,
            color_distance_threshold);
        
        

        thrust::device_vector<float> percentage_multiplier_val(num_images, 100);
        if (stage.compare("DEBUG") == 0 || stage.find("COST") != std::string::npos)
        {
            printf("Copying rendered cost to CPU\n");
            // Trying to get number of points explained in rendered
            thrust::transform(
                cuda_pose_point_num_vec.begin(), cuda_pose_point_num_vec.end(), 
                cuda_rendered_cost_vec.begin(), cuda_rendered_explained_vec.begin(), 
                thrust::minus<float>()
            );
            thrust::transform(
                cuda_rendered_cost_vec.begin(), cuda_rendered_cost_vec.end(), 
                cuda_pose_point_num_vec.begin(), cuda_rendered_cost_vec.begin(), 
                cost_percentage_functor()
                // thrust::divides<float>()
            );
            thrust::transform(
                cuda_rendered_cost_vec.begin(), cuda_rendered_cost_vec.end(), 
                percentage_multiplier_val.begin(), cuda_rendered_cost_vec.begin(), 
                cost_multiplier_functor()
                // thrust::multiplies<float>()
            );
            // printf("cuda_rendered_cost_vec\n");
            // std::cout << cuda_pose_point_num_vec[49] << std::endl;
            // thrust::copy(
            //     cuda_pose_point_num_vec.begin(),
            //     cuda_pose_point_num_vec.end(), 
            //     std::ostream_iterator<int>(std::cout, " ")
            // );
            // printf("\n");
            rendered_cost = (float*) malloc(num_images * size_of_float);
            cudaMemcpy(rendered_cost, cuda_rendered_cost, num_images * size_of_float, cudaMemcpyDeviceToHost);

            /// Not returning so need to free anything
        }
        printf("*************Render Costs computed**********\n");
        // if (calculate_observed_cost && cost_type == 2)
        if (calculate_observed_cost)
        {
            thrust::device_vector<float> cuda_pose_observed_explained_vec(num_images, 0);
            float* cuda_pose_observed_explained = thrust::raw_pointer_cast(cuda_pose_observed_explained_vec.data());
            thrust::device_vector<float> cuda_pose_points_diff_cost_vec(num_images, 0);

            peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);
        
            dim3 numBlocksO((num_images * observed_point_num + threadsPerBlock - 1) / threadsPerBlock, 1);
            //// Calculate the number of explained points in every pose, by adding
            compute_observed_cost<<<numBlocksO, threadsPerBlock>>>(
                num_images,
                observed_point_num,
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
            thrust::device_vector<float> cuda_pose_observed_points_total_vec = pose_observed_points_total;
            thrust::device_vector<float> cuda_observed_cost_vec(num_images, 0);
            thrust::transform(
                cuda_pose_observed_points_total_vec.begin(), cuda_pose_observed_points_total_vec.end(), 
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
                cuda_pose_observed_points_total_vec.begin(), cuda_observed_cost_vec.begin(), 
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
            if (stage.compare("DEBUG") == 0 || stage.find("COST") != std::string::npos)
            {
                printf("Copying observed cost to CPU\n");
                observed_cost = (float*) malloc(num_images * size_of_float);
                points_diff_cost = (float*) malloc(num_images * size_of_float);

                float* cuda_observed_cost = thrust::raw_pointer_cast(cuda_observed_cost_vec.data());
                float* cuda_pose_points_diff_cost = thrust::raw_pointer_cast(cuda_pose_points_diff_cost_vec.data());
               
                cudaMemcpy(observed_cost, cuda_observed_cost, num_images * size_of_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(points_diff_cost, cuda_pose_points_diff_cost, num_images * size_of_float, cudaMemcpyDeviceToHost);

                /// Not returning so need to free anything
            }
        }
        // else if (calculate_observed_cost && cost_type != 2)
        // {
        //     thrust::device_vector<float> cuda_pose_observed_unexplained_vec(num_images, 0);
        //     float* cuda_pose_observed_unexplained = thrust::raw_pointer_cast(cuda_pose_observed_unexplained_vec.data());
        //     peak_memory_usage = std::max(print_cuda_memory_usage(), peak_memory_usage);
        
        //     thrust::device_vector<float> device_poses_cylinder_radius_vec(num_images, 0.5);
        //     float* device_poses_cylinder_radius_ptr = thrust::raw_pointer_cast(device_poses_cylinder_radius_vec.data());
            
        //     dim3 numBlocksO((num_images * observed_point_num + threadsPerBlock - 1) / threadsPerBlock, 1);
        //     //// Calculate the number of explained points in every pose, by adding
        //     compute_observed_cost_cylinder<<<numBlocksO, threadsPerBlock>>>(
        //         num_images,
        //         observed_point_num,
        //         cuda_observed_explained,
        //         ref_dev,
        //         ref_pitch_in_bytes,
        //         device_poses_ptr,
        //         device_poses_cylinder_radius_ptr,
        //         cuda_pose_observed_unexplained
        //     );
        //     thrust::copy(
        //         cuda_pose_observed_unexplained_vec.begin(),
        //         cuda_pose_observed_unexplained_vec.end(), 
        //         std::ostream_iterator<float>(std::cout, " ")
        //     );
        //     printf("\n");
        //     // if (stage.compare("DEBUG") == 0 || stage.compare("COST") == 0)
        //     // {
        //     //     printf("Copying observed cost to CPU\n");
        //     //     observed_cost = (float*) malloc(num_images * size_of_float);

        //     //     float* cuda_observed_cost = thrust::raw_pointer_cast(cuda_observed_cost_vec.data());
               
        //     //     cudaMemcpy(observed_cost, cuda_observed_cost, num_images * size_of_float, cudaMemcpyDeviceToHost);

        //     //     /// Not returning so need to free anything
        //     // }

        // }

        
        cudaFree(cuda_cloud_color);
        cudaFree(cuda_cloud_pose_map);
        cudaFree(index_dev);
        cudaFree(dist_dev);
        if (cost_type == 2) cudaFree(cuda_observed_cloud_label);
        cudaFree(cuda_observed_cloud_color);

        end_4 = std::chrono::system_clock::now();
        elapsed_seconds = end_4-end_3;
        printf("************Cost Computation time : %f************\n", elapsed_seconds.count());

        printf("---------------------------------------\n");

    }
    struct concatenate_transforms{

        concatenate_transforms(){}

        __host__ __device__
        Model::mat4x4 operator()(Model::mat4x4& transform_a_mat, Eigen::Isometry3f& transform_b) const
        {
            // printf("Before icp : \n");
            // transform_a_mat.print_device();
            Eigen::Matrix4f transform_a = transform_a_mat.to_eigen(100);
            Eigen::Matrix4f transform_new = transform_b.matrix() * transform_a;
            Model::mat4x4 transform_new_mat;
            transform_new_mat.init_from_eigen(transform_new.cast<double>(), 100);
            // printf("After icp : \n");
            // transform_new_mat.print_device();
            return transform_new_mat;
        }
    };

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
        std::vector<float> pose_observed_points_total,
        int* result_observed_cloud_label,
        int cost_type,
        bool calculate_observed_cost,
        float sensor_resolution,
        float color_distance_threshold,
        float occlusion_threshold,
        bool do_icp,
        std::vector<int32_t>& result_depth, 
        std::vector<std::vector<uint8_t>>& result_color,
        float* &result_cloud,
        uint8_t* &result_cloud_color,
        int& result_cloud_point_num,
        int* &result_cloud_pose_map,
        int* &result_dc_index,
        std::vector<Model::mat4x4>& adjusted_poses,
        float* &rendered_cost,
        float* &observed_cost,
        float* &points_diff_cost,
        gpu_stats& stats) {
        /*
         * - @source_mask_label - Label for every pixel in source image, used for segmentation specific occlusion checking
         * - Currently doesnt support pose occlusion or pose occlusion other
         */
        
        // std::string stage = "DEBUG";
        // gpu_stats stats;
        printf("---------------------------------------\n");
        printf("Stage : %s\n", stage.c_str());
        printf("USE_CLUTTER : %d\n", USE_CLUTTER);
        printf("USE_TREE : %d\n", USE_TREE);
        printf("sensor_resolution : %f\n", sensor_resolution);
        printf("color_distance_threshold : %f\n", color_distance_threshold);
        printf("cost_type : %d\n", cost_type);
        printf("point_dim : %d\n", point_dim);
        printf("stride : %d\n", stride);
        printf("depth_factor : %d\n", depth_factor);
        printf("observed_point_num : %d\n", observed_point_num);
        printf("occlusion_threshold : %f\n", occlusion_threshold);
        printf("calculate_observed_cost : %d\n", calculate_observed_cost);

        std::chrono::time_point<std::chrono::system_clock> start, end_1, end_2, end_3, end_3a, end_3b, end_3c, end_4;
        start = std::chrono::system_clock::now();
        // Create device inputs
        // int* device_single_result_image;
        // cudaMalloc((void**)&device_single_result_image, sizeof(int));
        // cudaMemcpy(device_single_result_image, &single_result_image, sizeof(int), cudaMemcpyHostToDevice);
        int num_images;
        if (single_result_image)
        {
            num_images = 1;
        }
        else
        {
            num_images = poses.size();
        }
        // const Model::ROI roi= {0, 0, 0, 0};
        const size_t threadsPerBlock = 256;
        // std::cout <<tris[0].color.v1;
        thrust::device_vector<Model::Triangle> device_tris = tris;
        thrust::device_vector<Model::mat4x4> device_poses = poses;
        //// Every index maps a model id to a range of triangles in the triangle vector 
        thrust::device_vector<int> device_tris_model_count_low = tris_model_count;
        thrust::device_vector<int> device_tris_model_count_high = tris_model_count;
        thrust::device_vector<int> device_pose_model_map = pose_model_map;
        thrust::device_vector<int> device_pose_segmentation_label = pose_segmentation_label;

        thrust::device_vector<int32_t> device_source_depth = source_depth;
        thrust::device_vector<uint8_t> device_source_color_red = source_color[0];
        thrust::device_vector<uint8_t> device_source_color_green = source_color[1];
        thrust::device_vector<uint8_t> device_source_color_blue = source_color[2];
        thrust::device_vector<uint8_t> device_source_mask_label = source_mask_label;

        size_t real_width = width;
        size_t real_height = height;

        // Create device outputs
        thrust::device_vector<int> device_pose_occluded;
        thrust::device_vector<int> device_pose_occluded_other;
        thrust::device_vector<float> device_pose_clutter_points;
        thrust::device_vector<float> device_pose_total_points;

        thrust::device_vector<int32_t> device_depth_int;
        thrust::device_vector<int32_t> device_lock_int;
        thrust::device_vector<uint8_t> device_red_int;
        thrust::device_vector<uint8_t> device_green_int;
        thrust::device_vector<uint8_t> device_blue_int;
        image_render(device_tris,
                    device_poses,
                    device_pose_model_map,
                    device_tris_model_count_low,
                    device_source_depth,
                    device_source_color_red,
                    device_source_color_green,
                    device_source_color_blue,
                    device_source_mask_label,
                    device_pose_segmentation_label,
                    num_images,
                    width,
                    height,
                    proj_mat,
                    occlusion_threshold,
                    single_result_image,
                    device_pose_occluded,
                    device_pose_occluded_other,
                    device_pose_clutter_points,
                    device_pose_total_points,
                    device_depth_int,
                    device_red_int,
                    device_green_int,
                    device_blue_int,
                    stats);
        // int* device_pose_occluded_vec = thrust::raw_pointer_cast(device_pose_occluded.data());
        // int32_t* depth_image_vec = thrust::raw_pointer_cast(device_depth_int.data());
        // uint8_t* red_image_vec = thrust::raw_pointer_cast(device_red_int.data());
        // uint8_t* green_image_vec = thrust::raw_pointer_cast(device_green_int.data());
        // uint8_t* blue_image_vec = thrust::raw_pointer_cast(device_blue_int.data());
        // int* device_pose_segmentation_label_vec = thrust::raw_pointer_cast(device_pose_segmentation_label.data());
        if (USE_CLUTTER) {
            thrust::copy(device_pose_clutter_points.begin(), device_pose_clutter_points.end(), clutter_cost.begin());
        }
        end_1 = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_1-start;
        printf("*************Rendering Images Done**********\n");
        printf("*************Render time : %f*************\n", elapsed_seconds.count());
        if (stage.compare("DEBUG") == 0 || stage.compare("RENDER") == 0)
        {
            printf("Copying images to CPU\n");
            //// Allocate CPU memory
            std::vector<uint8_t> result_red(num_images*real_width*real_height);
            std::vector<uint8_t> result_green(num_images*real_width*real_height);
            std::vector<uint8_t> result_blue(num_images*real_width*real_height);
            result_depth.resize(num_images*real_width*real_height);
            
            //// Copy from GPU to CPU
            thrust::copy(device_depth_int.begin(), device_depth_int.end(), result_depth.begin());
            thrust::copy(device_red_int.begin(), device_red_int.end(), result_red.begin());
            thrust::copy(device_green_int.begin(), device_green_int.end(), result_green.begin());
            thrust::copy(device_blue_int.begin(), device_blue_int.end(), result_blue.begin());
            result_color.push_back(result_red);
            result_color.push_back(result_green);
            result_color.push_back(result_blue);

            /// Vectors will be free automatically on return

            if (stage.compare("RENDER") == 0) return;
        }
        ///////////////////////////////////////////////////////////////
        const unsigned int size_of_float = sizeof(float);
        const unsigned int size_of_int   = sizeof(int);
        const unsigned int size_of_uint   = sizeof(uint8_t);
        float* cuda_cloud;
        size_t query_pitch_in_bytes;
        // int rendered_cloud_point_num;
        thrust::device_vector<float>   rendered_point_cloud;
        thrust::device_vector<uint8_t> rendered_point_cloud_color;
        thrust::device_vector<int>     rendered_dc_index;
        thrust::device_vector<int>     rendered_cloud_pose_map;
        thrust::device_vector<int>     dummy_vec;
        thrust::device_vector<Eigen::Vector3f> result_cloud_eigen;
        compute_point_clouds(
            device_depth_int,
            device_red_int,
            device_green_int,
            device_blue_int,
            num_images,
            width,
            height,
            kCameraCX,
            kCameraCY,
            kCameraFX,
            kCameraFY,
            depth_factor,
            stride,
            device_pose_occluded,
            cuda_cloud,
            query_pitch_in_bytes,
            result_cloud_eigen,
            rendered_point_cloud,
            rendered_point_cloud_color,
            result_cloud_point_num,
            rendered_dc_index,
            rendered_cloud_pose_map,
            dummy_vec,
            stats
        );
        int k = 1;
        printf("************Point clouds created*************\n");
        end_2 = std::chrono::system_clock::now();
        elapsed_seconds = end_2-end_1;
        printf("************Cloud contruction time : %f************\n", elapsed_seconds.count());
        
        // Copy observed stuff from CPU
        thrust::device_vector<int> observed_cloud_label(result_observed_cloud_label, result_observed_cloud_label + observed_point_num);
        thrust::device_vector<Eigen::Vector3f> observed_cloud_eigen(observed_depth_eigen, observed_depth_eigen + observed_point_num);
        printf("observed_cloud_eigen() size : %d\n", observed_cloud_eigen.size());

        // ///////////////////////////////////////////////////////////////
        // // Testing ICP
        end_3a = end_2;
        if (do_icp)
        {
            thrust::device_vector<int> observed_cloud_label_subtracted(observed_cloud_label.size());
            thrust::device_vector<int> device_pose_segmentation_label_subtracted(device_pose_segmentation_label.size());
            if (device_pose_segmentation_label.size() > 0)
            {
                printf("Subtracting 1 from segmentation labels for ICP\n");
                // Make segmentation labes start from 0
                thrust::device_vector<int> minus_vec(observed_cloud_label.size(), 1);
                thrust::transform(
                    observed_cloud_label.begin(), observed_cloud_label.end(), 
                    minus_vec.begin(), observed_cloud_label_subtracted.begin(), 
                    thrust::minus<float>()
                );

                minus_vec.resize(device_pose_segmentation_label.size(), 1);
                thrust::transform(
                    device_pose_segmentation_label.begin(), device_pose_segmentation_label.end(), 
                    minus_vec.begin(), device_pose_segmentation_label_subtracted.begin(), 
                    thrust::minus<float>()
                );
            }


            std::vector<Eigen::Isometry3f> estimated;
            thrust::device_vector<Eigen::Isometry3f> d_estimated;
            std::unique_ptr<fast_gicp::FastGICPCudaCore> vgicp_cuda;
            int max_iterations_ = 150;
            int k_correspondences_ = 10;
            float rotation_epsilon_ = 2e-3;
            float transformation_epsilon_ = 5e-4;
                
            vgicp_cuda.reset(new fast_gicp::FastGICPCudaCore());
            vgicp_cuda->set_max_iterations(max_iterations_);
            vgicp_cuda->set_rotation_epsilon(rotation_epsilon_);
            vgicp_cuda->set_transformation_epsilon(transformation_epsilon_);
            vgicp_cuda->set_correspondence_randomness(k_correspondences_);
            vgicp_cuda->set_input(result_cloud_eigen,
                                observed_cloud_eigen,
                                rendered_cloud_pose_map,
                                observed_cloud_label_subtracted,
                                device_pose_segmentation_label_subtracted,
                                num_images);
            vgicp_cuda->optimize_multi(estimated);
            d_estimated = estimated;
            thrust::device_vector<Model::mat4x4> device_poses_adjusted(device_poses.size());
            thrust::host_vector<Model::mat4x4> host_poses_adjusted(device_poses.size());
            thrust::transform(
                device_poses.begin(), device_poses.end(), d_estimated.begin(), device_poses_adjusted.begin(),
                concatenate_transforms()
            );
            adjusted_poses.resize(num_images);
            thrust::copy(device_poses_adjusted.begin(), device_poses_adjusted.end(), adjusted_poses.begin());
            // host_poses_adjusted = device_poses_adjusted;
            // for (int i = 0; i < host_poses_adjusted.size(); i++)
            // {
            //     host_poses_adjusted[i].print();
            // }
            printf("*************ICPs computed**********\n");
            end_3a = std::chrono::system_clock::now();
            elapsed_seconds = end_3a-end_2;
            printf("*************ICP time : %f************\n", elapsed_seconds.count());
            stats.icp_runtime = std::max(stats.icp_runtime, (float) elapsed_seconds.count());

            image_render(device_tris,
                device_poses_adjusted,
                device_pose_model_map,
                device_tris_model_count_low,
                device_source_depth,
                device_source_color_red,
                device_source_color_green,
                device_source_color_blue,
                device_source_mask_label,
                device_pose_segmentation_label,
                num_images,
                width,
                height,
                proj_mat,
                occlusion_threshold,
                single_result_image,
                device_pose_occluded,
                device_pose_occluded_other,
                device_pose_clutter_points,
                device_pose_total_points,
                device_depth_int,
                device_red_int,
                device_green_int,
                device_blue_int,
                stats);

            compute_point_clouds(
                device_depth_int,
                device_red_int,
                device_green_int,
                device_blue_int,
                num_images,
                width,
                height,
                kCameraCX,
                kCameraCY,
                kCameraFX,
                kCameraFY,
                depth_factor,
                stride,
                device_pose_occluded,
                cuda_cloud,
                query_pitch_in_bytes,
                result_cloud_eigen,
                rendered_point_cloud,
                rendered_point_cloud_color,
                result_cloud_point_num,
                rendered_dc_index,
                rendered_cloud_pose_map,
                dummy_vec,
                stats
            );
        }
        
        ///////////////////////////////////////////////////////////////

        if (stage.compare("DEBUG") == 0 || stage.find("CLOUD") != std::string::npos)
        {
            printf("Copying point clouds to CPU\n");
            //// Allocate CPU memory
            result_cloud = (float*) malloc(point_dim * result_cloud_point_num * sizeof(float));
            result_cloud_color = (uint8_t*) malloc(point_dim * result_cloud_point_num * sizeof(uint8_t));
            result_dc_index = (int*) malloc(num_images * width * height * sizeof(int));
            result_cloud_pose_map = (int*) malloc(result_cloud_point_num * sizeof(int));

            //// Copy to CPU if needed
            cudaMemcpy2D(
                result_cloud,  result_cloud_point_num * size_of_float, cuda_cloud,  query_pitch_in_bytes,  result_cloud_point_num * size_of_float, point_dim, cudaMemcpyDeviceToHost);
            // cudaMemcpy(result_cloud, cuda_cloud, point_dim * result_cloud_point_num * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(result_cloud_color, thrust::raw_pointer_cast(rendered_point_cloud_color.data()), point_dim * result_cloud_point_num * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(result_dc_index, thrust::raw_pointer_cast(rendered_dc_index.data()), num_images * width * height * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(result_cloud_pose_map, thrust::raw_pointer_cast(rendered_cloud_pose_map.data()), result_cloud_point_num * sizeof(int), cudaMemcpyDeviceToHost);
            
            /// Exit here if only point clouds are needed - for e.g. before ICP
            /// Free copied stuff
            
            if (stage.compare("CLOUD") == 0) {
                cudaFree(cuda_cloud);
                // cudaFree(cuda_cloud_color);
                // cudaFree(cuda_cloud_pose_map);
                return;
            }
        }
        rendered_dc_index.clear(); rendered_dc_index.shrink_to_fit();
        ///////////////////////////////////////////////////////////////
        
        thrust::device_vector<thrust::pair<float, int>> k_neighbors;
        fast_gicp::brute_force_knn_search(result_cloud_eigen, 
                                        observed_cloud_eigen, 
                                        1, 
                                        k_neighbors);
                                        //// thrust::device_vector<int>(0), // NN will not be segmentation specific
                                        //// thrust::device_vector<int>(0), // NN will not be segmentation specific
                                        //*source_label_map,
                                        //*target_label_indices,
                                        //*source_pose_map,
                                        //adjusted_x0s,
                                        //mask_pose_icp);
        printf("*************KNN distances computed**********\n");
        end_3 = std::chrono::system_clock::now();
        elapsed_seconds = end_3-end_3a;
        printf("*************KNN time : %f************\n", elapsed_seconds.count());
        sensor_resolution = sensor_resolution * sensor_resolution;
               
        thrust::device_vector<float> k_distances(k_neighbors.size());
        thrust::device_vector<int> k_indices(k_neighbors.size());
        thrust::transform(k_neighbors.begin(), k_neighbors.end(), k_indices.begin(), fast_gicp::untie_pair_second());
        thrust::transform(k_neighbors.begin(), k_neighbors.end(), k_distances.begin(), fast_gicp::untie_pair_first());
        //////////////////////////////////////////////////////////////

        // Testing new cost compute interface
        thrust::device_vector<uint8_t> observed_cloud_color(observed_color, observed_color + point_dim * observed_point_num);

        thrust::device_vector<float> rendered_poses_observed_points_total = pose_observed_points_total;
        thrust::device_vector<float> cuda_rendered_cost_v;
        thrust::device_vector<float> cuda_observed_cost_v;
        thrust::device_vector<float> cuda_pose_points_diff_cost_vec;
        compute_costs(num_images,
            cost_type,
            calculate_observed_cost,
            sensor_resolution,
            color_distance_threshold,
            observed_cloud_color,
            observed_cloud_label,
            observed_point_num,
            rendered_point_cloud_color,
            rendered_cloud_pose_map,
            device_pose_occluded,
            device_pose_segmentation_label,
            rendered_poses_observed_points_total,
            result_cloud_point_num,
            k_distances,
            k_indices,
            cuda_rendered_cost_v,
            cuda_observed_cost_v,
            cuda_pose_points_diff_cost_vec,
            stats                        
        );
        if (stage.compare("DEBUG") == 0 || stage.find("COST") != std::string::npos)
        {
            printf("Copying rendered cost to CPU\n");
            rendered_cost = (float*) malloc(num_images * size_of_float);
            cudaMemcpy(rendered_cost, thrust::raw_pointer_cast(cuda_rendered_cost_v.data()), num_images * size_of_float, cudaMemcpyDeviceToHost);
            if (calculate_observed_cost)
            {
                printf("Copying observed cost to CPU\n");
                observed_cost = (float*) malloc(num_images * size_of_float);
                points_diff_cost = (float*) malloc(num_images * size_of_float);

                cudaMemcpy(observed_cost, thrust::raw_pointer_cast(cuda_observed_cost_v.data()), num_images * size_of_float, cudaMemcpyDeviceToHost);
                cudaMemcpy(points_diff_cost, thrust::raw_pointer_cast(cuda_pose_points_diff_cost_vec.data()), num_images * size_of_float, cudaMemcpyDeviceToHost);
            }
        }
        // cudaFree(index_dev);
        // cudaFree(dist_dev);
        // if (cost_type == 2) cudaFree(cuda_observed_cloud_label);
        // cudaFree(cuda_observed_cloud_color);

        end_4 = std::chrono::system_clock::now();
        elapsed_seconds = end_4-end_3;
        printf("*************Costs computed**********\n");
        printf("************Cost Computation time : %f************\n", elapsed_seconds.count());
        return;
        ///////////////////////////////////////////////////////////////////
    }

    bool depth2cloud_global(const std::vector<int32_t>& depth_data,
                            const std::vector<std::vector<uint8_t>>& color_data,
                            Eigen::Vector3f* &result_cloud_eigen,
                            float* &result_cloud,
                            uint8_t* &result_cloud_color,
                            int* &dc_index,
                            int &rendered_cloud_point_num,
                            int* &cloud_pose_map,
                            int* &result_observed_cloud_label,
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
                            const std::vector<uint8_t>& label_mask_data,
                            const std::vector<double>& observed_cloud_bounds,
                            const Eigen::Matrix4f* camera_transform)
    {
        printf("depth2cloud_global()\n");
        /**
            Convert a given input to point cloud, used to convert observed images to point cloud
            @label_mask_data - Label of every pixel in input 2D image
            @result_observed_cloud_label - Label of every pixel in output 3D cloud, downsampled
            Returns :
                cloud_pose_map - Mapping of every point in the cloud to a pose number
                rendered_cloud_point_num - Total number of points in the rendered pose arrays
        */

        //// Testing new interface
        thrust::device_vector<uint8_t> d_red_in = color_data[0];
        thrust::device_vector<uint8_t> d_green_in = color_data[1];
        thrust::device_vector<uint8_t> d_blue_in = color_data[2];

        thrust::device_vector<int32_t> d_depth_data = depth_data;
        thrust::device_vector<int> d_poses_occluded = pose_occluded;
        thrust::device_vector<uint8_t> d_label_mask_data = label_mask_data;
        thrust::device_vector<double> d_observed_cloud_bounds = observed_cloud_bounds;
        thrust::device_vector<float>   result_point_cloud;
        thrust::device_vector<uint8_t> result_point_cloud_color;
        thrust::device_vector<int>     result_dc_index;
        thrust::device_vector<int>     result_cloud_pose_map;
        thrust::device_vector<int>     result_cloud_label;
        thrust::device_vector<Eigen::Vector3f> d_result_cloud_eigen;

        Eigen::Matrix4f* d_camera_transform;
        if (camera_transform != NULL)
        {
            printf("Using camera transform to transform point cloud to world frame\n");
            std::cout << *camera_transform << std::endl; 
            cudaMalloc(&d_camera_transform, sizeof(Eigen::Matrix4f));
            cudaMemcpy(d_camera_transform, camera_transform, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
        }

        float* result_2d_point_cloud;
        size_t result_2d_pitch_in_bytes;
        gpu_stats stats;
        compute_point_clouds(
            d_depth_data,
            d_red_in,
            d_green_in,
            d_blue_in,
            num_poses,
            width,
            height,
            kCameraCX,
            kCameraCY,
            kCameraFX,
            kCameraFY,
            depth_factor,
            stride,
            d_poses_occluded,
            result_2d_point_cloud,
            result_2d_pitch_in_bytes,
            d_result_cloud_eigen,
            result_point_cloud,
            result_point_cloud_color,
            rendered_cloud_point_num,
            result_dc_index,
            result_cloud_pose_map,
            result_cloud_label,
            stats,
            d_camera_transform,
            d_label_mask_data,
            d_observed_cloud_bounds
        );
        // thrust::copy(
        //     result_cloud_label.begin(),
        //     result_cloud_label.end(), 
        //     std::ostream_iterator<int>(std::cout, " ")
        // );
        // printf("\n");
        // std::vector<float> c_point_cloud(result_point_cloud.size());
        // std::vector<uint8_t> c_point_cloud_color(result_point_cloud_color.size());
        // std::vector<int> c_dc_index(result_dc_index.size());
        // std::vector<int> d_cloud_pose_map(result_cloud_pose_map.size());
        // std::vector<int> d_cloud_label(result_cloud_label.size());
        // thrust::copy(result_point_cloud.begin(), result_point_cloud.end(), c_point_cloud.begin());
        // thrust::copy(result_point_cloud_color.begin(), result_point_cloud_color.end(), c_point_cloud_color.begin());
        // thrust::copy(result_dc_index.begin(), result_dc_index.end(), c_dc_index.begin());
        // thrust::copy(result_cloud_pose_map.begin(), result_cloud_pose_map.end(), d_cloud_pose_map.begin());
        // thrust::copy(result_cloud_label.begin(), result_cloud_label.end(), d_cloud_label.begin());

        // Assign host pointers
        result_cloud = (float*) malloc(point_dim * rendered_cloud_point_num * sizeof(float));
        result_cloud_eigen = (Eigen::Vector3f*) malloc(rendered_cloud_point_num * sizeof(Eigen::Vector3f));
        result_cloud_color = (uint8_t*) malloc(point_dim * rendered_cloud_point_num * sizeof(uint8_t));
        dc_index = (int*) malloc(num_poses * width * height * sizeof(int));
        cloud_pose_map = (int*) malloc(rendered_cloud_point_num * sizeof(int));
        result_observed_cloud_label = (int*) malloc(rendered_cloud_point_num * sizeof(int));

        cudaMemcpy2D(
            result_cloud,  rendered_cloud_point_num * sizeof(float), result_2d_point_cloud,  result_2d_pitch_in_bytes,  rendered_cloud_point_num * sizeof(float), point_dim, cudaMemcpyDeviceToHost);

        // cudaMemcpy(result_cloud, thrust::raw_pointer_cast(result_point_cloud.data()), point_dim * rendered_cloud_point_num * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_cloud_eigen, thrust::raw_pointer_cast(d_result_cloud_eigen.data()), rendered_cloud_point_num * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost);
        cudaMemcpy(result_cloud_color, thrust::raw_pointer_cast(result_point_cloud_color.data()), point_dim * rendered_cloud_point_num * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(dc_index, thrust::raw_pointer_cast(result_dc_index.data()), num_poses * width * height * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cloud_pose_map, thrust::raw_pointer_cast(result_cloud_pose_map.data()), rendered_cloud_point_num * sizeof(int), cudaMemcpyDeviceToHost);
        if (label_mask_data.size() > 0)
        {
            cudaMemcpy(result_observed_cloud_label, thrust::raw_pointer_cast(result_cloud_label.data()), rendered_cloud_point_num * sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (camera_transform != NULL)
        {
            cudaFree(d_camera_transform);
        }
        cudaFree(result_2d_point_cloud);
        return true;
        
        // thrust::device_vector<uint8_t> d_red_in = color_data[0];
        // thrust::device_vector<uint8_t> d_green_in = color_data[1];
        // thrust::device_vector<uint8_t> d_blue_in = color_data[2];
        // int32_t* depth_data_cuda;
        // int* pose_occluded_cuda;
        // uint8_t* label_mask_data_cuda = NULL;
        // double* observed_cloud_bounds_cuda = NULL;
        // Eigen::Matrix4f* camera_transform_cuda = NULL;
        // // int stride = 5;
        // cudaMalloc(&depth_data_cuda, num_poses * width * height * sizeof(int32_t));
        // cudaMemcpy(depth_data_cuda, depth_data.data(), num_poses * width * height * sizeof(int32_t), cudaMemcpyHostToDevice);
        
        // if (label_mask_data.size() > 0)
        // {
        //     printf("Using segementation labels to create point cloud\n");
        //     cudaMalloc(&label_mask_data_cuda, num_poses * width * height * sizeof(uint8_t));
        //     cudaMemcpy(label_mask_data_cuda, label_mask_data.data(), num_poses * width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
        // }
        // if (observed_cloud_bounds.size() > 0)
        // {
        //     // std::cout << observed_cloud_bounds[0] << " " << observed_cloud_bounds[1] << std::endl;
        //     printf("Using filter bounds to create point cloud\n");
        //     printf("x_min : %f, x_max : %f\n", observed_cloud_bounds[1], observed_cloud_bounds[0]);
        //     printf("y_min : %f, y_max : %f\n", observed_cloud_bounds[3], observed_cloud_bounds[2]);
        //     printf("z_min : %f, z_max : %f\n", observed_cloud_bounds[5], observed_cloud_bounds[4]);
        //     cudaMalloc(&observed_cloud_bounds_cuda, 6 * sizeof(double));
        //     cudaMemcpy(observed_cloud_bounds_cuda, observed_cloud_bounds.data(), 6 * sizeof(double), cudaMemcpyHostToDevice);
        // }
        // if (camera_transform != NULL)
        // {
        //     printf("Using camera transform to transform point cloud to world frame\n");
        //     std::cout << *camera_transform << std::endl; 
        //     cudaMalloc(&camera_transform_cuda, sizeof(Eigen::Matrix4f));
        //     cudaMemcpy(camera_transform_cuda, camera_transform, sizeof(Eigen::Matrix4f), cudaMemcpyHostToDevice);
        // }
        // cudaMalloc(&pose_occluded_cuda, num_poses * sizeof(int));
        // cudaMemcpy(pose_occluded_cuda, pose_occluded.data(), num_poses * sizeof(int), cudaMemcpyHostToDevice);

        // dim3 threadsPerBlock(16, 16);
        // assert(width % stride == 0);
        // dim3 numBlocks((width/stride * num_poses + threadsPerBlock.x - 1)/threadsPerBlock.x, (height/stride + threadsPerBlock.y - 1)/threadsPerBlock.y);

        // thrust::device_vector<int> mask(width*height*num_poses, 0);
        // int* mask_ptr = thrust::raw_pointer_cast(mask.data());

        // depth_to_mask<<<numBlocks, threadsPerBlock>>>(depth_data_cuda, 
        //                                               mask_ptr, width, 
        //                                               height, 
        //                                               stride, 
        //                                               num_poses,
        //                                               pose_occluded_cuda,
        //                                               label_mask_data_cuda,
        //                                               kCameraCX, kCameraCY, kCameraFX, kCameraFY, depth_factor,
        //                                               observed_cloud_bounds_cuda, camera_transform_cuda);
            
        // if (cudaGetLastError() != cudaSuccess) 
        // {
        //     printf("ERROR: Unable to execute kernel depth_to_mask\n");
        //     return false;
        // }
        // // cudaDeviceSynchronize();

        // // Create mapping from pixel to corresponding index in point cloud
        // int mask_back_temp = mask.back();
        // thrust::exclusive_scan(mask.begin(), mask.end(), mask.begin(), 0); // in-place scan
        // rendered_cloud_point_num = mask.back() + mask_back_temp;
        // printf("Actual points in all clouds : %d\n", rendered_cloud_point_num);

        // float* cuda_cloud;
        // uint8_t* cuda_cloud_color;
        // int* cuda_cloud_pose_map;
        // int* cuda_cloud_mask_label;
        // size_t query_pitch_in_bytes;

        // // cudaMalloc(&cuda_cloud, point_dim * rendered_cloud_point_num * sizeof(float));
        // cudaMallocPitch(&cuda_cloud,   &query_pitch_in_bytes,   rendered_cloud_point_num * sizeof(float), point_dim);
        // cudaMalloc(&cuda_cloud_color, point_dim * rendered_cloud_point_num * sizeof(uint8_t));
        // cudaMalloc(&cuda_cloud_pose_map, rendered_cloud_point_num * sizeof(int));
        // if (label_mask_data.size() > 0)
        // {
        //     cudaMalloc(&cuda_cloud_mask_label, rendered_cloud_point_num * sizeof(int));
        // }

        // result_cloud = (float*) malloc(point_dim * rendered_cloud_point_num * sizeof(float));
        // result_cloud_color = (uint8_t*) malloc(point_dim * rendered_cloud_point_num * sizeof(uint8_t));
        // dc_index = (int*) malloc(num_poses * width * height * sizeof(int));
        // cloud_pose_map = (int*) malloc(rendered_cloud_point_num * sizeof(int));
        // result_observed_cloud_label = (int*) malloc(rendered_cloud_point_num * sizeof(int));

        

        // uint8_t* red_in = thrust::raw_pointer_cast(d_red_in.data());
        // uint8_t* green_in = thrust::raw_pointer_cast(d_green_in.data());
        // uint8_t* blue_in = thrust::raw_pointer_cast(d_blue_in.data());

        // depth_to_2d_cloud<<<numBlocks, threadsPerBlock>>>(
        //                     depth_data_cuda, red_in, green_in, blue_in,
        //                     cuda_cloud, query_pitch_in_bytes, cuda_cloud_color, rendered_cloud_point_num, mask_ptr, width, height, 
        //                     kCameraCX, kCameraCY, kCameraFX, kCameraFY, depth_factor, stride, num_poses, cuda_cloud_pose_map,
        //                     label_mask_data_cuda, cuda_cloud_mask_label, observed_cloud_bounds_cuda, camera_transform_cuda);
        // // depth_to_cloud<<<numBlocks, threadsPerBlock>>>(
        // //                     depth_data_cuda, red_in, green_in, blue_in,
        // //                     cuda_cloud, cuda_cloud_color, rendered_cloud_point_num, mask_ptr, width, height, 
        // //                     kCameraCX, kCameraCY, kCameraFX, kCameraFY, depth_factor, stride, cuda_cloud_pose_map,
        // //                     label_mask_data_cuda, cuda_cloud_mask_label);
        // if (cudaGetLastError() != cudaSuccess) 
        // {
        //     printf("ERROR: Unable to execute kernel depth_to_2d_cloud\n");
        //     return false;
        // }
        // // cudaDeviceSynchronize();
        // cudaMemcpy2D(
        //         result_cloud,  rendered_cloud_point_num * sizeof(float), cuda_cloud,  query_pitch_in_bytes,  rendered_cloud_point_num * sizeof(float), point_dim, cudaMemcpyDeviceToHost);
        // // cudaMemcpy(result_cloud, cuda_cloud, point_dim * rendered_cloud_point_num * sizeof(float), cudaMemcpyDeviceToHost);
        // cudaMemcpy(result_cloud_color, cuda_cloud_color, point_dim * rendered_cloud_point_num * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        // cudaMemcpy(dc_index, mask_ptr, num_poses * width * height * sizeof(int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(cloud_pose_map, cuda_cloud_pose_map, rendered_cloud_point_num * sizeof(int), cudaMemcpyDeviceToHost);
        // if (label_mask_data.size() > 0)
        // {
        //     cudaMemcpy(result_observed_cloud_label, cuda_cloud_mask_label, rendered_cloud_point_num * sizeof(int), cudaMemcpyDeviceToHost);
        // }
        // // for (int i = 0; i < rendered_cloud_point_num; i++)
        // // {
        // //     printf("%d ", cloud_pose_map[i]);
        // // }
        // // printf("\n");
        // // for(int n = 0; n < num_poses; n ++)
        // // {
        // //     for(int i = 0; i < height; i ++)
        // //     {
        // //         for(int j = 0; j < width; j ++)
        // //         {
        // //             int index = n*width*height + (i*width + j);
        // //             int cloud_index = mask[index];
        // //             // printf("cloud_i:%d\n", cloud_index);
        // //             if (depth_data[index] > 0)
        // //             {
        // //                 // printf("x:%f,y:%f,z:%f\n", 
        // //                 // result_cloud[3*cloud_index], result_cloud[3*cloud_index + 1], result_cloud[3*cloud_index + 2]);
        // //             }
        // //         }
        // //     }
        // // }
        // // if (cudaGetLastError() != cudaSuccess) 
        // // {
        // //     printf("ERROR: Unable to execute kernel depth_to_cloud\n");
        // //     cudaFree(depth_data_cuda);
        // //     cudaFree(pose_occluded_cuda);
        // //     cudaFree(cuda_cloud);
        // //     cudaFree(cuda_cloud_color);
        // //     cudaFree(cuda_cloud_pose_map);
        // //     if (label_mask_data != NULL)
        // //     {
        // //         cudaFree(cuda_cloud_mask_label);
        // //     }
        // //     return false;
        // // }
        // printf("depth2cloud_global() Done\n");
        // cudaFree(depth_data_cuda);
        // cudaFree(pose_occluded_cuda);
        // cudaFree(cuda_cloud);
        // cudaFree(cuda_cloud_color);
        // cudaFree(cuda_cloud_pose_map);
        // if (label_mask_data.size() > 0)
        // {
        //     cudaFree(cuda_cloud_mask_label);
        // }
        // if (observed_cloud_bounds.size() > 0)
        // {
        //     cudaFree(observed_cloud_bounds_cuda);
        // }
        // if (camera_transform != NULL)
        // {
        //     cudaFree(camera_transform_cuda);
        // }
        // return true;
    }
    
    bool compute_rgbd_cost(
        float &sensor_resolution,
        float* knn_dist,
        int* knn_index,
        int* poses_occluded,
        int* cloud_pose_map,
        float* observed_cloud,
        uint8_t* observed_cloud_color,
        float* rendered_cloud,
        uint8_t* rendered_cloud_color,
        int rendered_cloud_point_num,
        int observed_cloud_point_num,
        int num_poses,
        float* &rendered_cost,
        std::vector<float> pose_observed_points_total,
        float* &observed_cost,
        int* pose_segmentation_label,
        int* result_observed_cloud_label,
        int cost_type,
        bool calculate_observed_cost
    )
    {
        /*
         * Function not mainted
         * @pose_observed_points_total - number of total points in observed scene corresponding to given object
         * It is calculated using segmentation label
         * @result_observed_cloud_label - label for every point in the observed cloud, used in calculating render cost,
         * A point is penalized only if it belongs to same label (pose of point and closest observed point)
         */
        // for (int i = 0; i < num_poses; i++)
        // {
        //     printf("%d ", poses_occluded[i]);
        // }
        // printf("\n");
        printf("compute_cost()\n");

        float* cuda_knn_dist;
        int* cuda_knn_index;
        // float* cuda_sensor_resolution;
        int* cuda_poses_occluded;
        int* cuda_cloud_pose_map;
        float* cuda_rendered_cost;
        float* cuda_pose_point_num;
        uint8_t* cuda_observed_cloud_color;
        uint8_t* cuda_rendered_cloud_color;
        float* cuda_rendered_cloud;
        uint8_t* cuda_observed_explained;

        int* cuda_pose_segmentation_label;
        int* cuda_observed_cloud_label;

        const unsigned int size_of_float = sizeof(float);
        const unsigned int size_of_int   = sizeof(int);
        const unsigned int size_of_uint   = sizeof(uint8_t);

        cudaMalloc(&cuda_knn_dist, rendered_cloud_point_num * size_of_float);
        cudaMalloc(&cuda_knn_index, rendered_cloud_point_num * size_of_int);
        cudaMalloc(&cuda_cloud_pose_map, rendered_cloud_point_num * size_of_int);
        cudaMalloc(&cuda_observed_cloud_color, 3 * observed_cloud_point_num * size_of_uint);
        cudaMalloc(&cuda_rendered_cloud, 3 * rendered_cloud_point_num * size_of_float);
        cudaMalloc(&cuda_rendered_cloud_color, 3 * rendered_cloud_point_num * size_of_uint);
        cudaMalloc(&cuda_poses_occluded, num_poses * size_of_int);
        cudaMalloc(&cuda_pose_segmentation_label, num_poses * size_of_int);
        cudaMalloc(&cuda_observed_cloud_label, observed_cloud_point_num * size_of_int);

        thrust::device_vector<float> cuda_rendered_cost_vec(num_poses, 0);
        cuda_rendered_cost = thrust::raw_pointer_cast(cuda_rendered_cost_vec.data());
        thrust::device_vector<float> cuda_pose_point_num_vec(num_poses, 0);
        cuda_pose_point_num = thrust::raw_pointer_cast(cuda_pose_point_num_vec.data());

        thrust::device_vector<uint8_t> cuda_observed_explained_vec(num_poses * observed_cloud_point_num, 0);
        cuda_observed_explained = thrust::raw_pointer_cast(cuda_observed_explained_vec.data());

        cudaMemcpy(cuda_knn_dist, knn_dist, rendered_cloud_point_num * size_of_float, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_knn_index, knn_index, rendered_cloud_point_num * size_of_int, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_cloud_pose_map, cloud_pose_map, rendered_cloud_point_num * size_of_int, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_observed_cloud_color, observed_cloud_color, 3 * observed_cloud_point_num * size_of_uint, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_rendered_cloud, rendered_cloud, 3 * rendered_cloud_point_num * size_of_float, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_rendered_cloud_color, rendered_cloud_color, 3 * rendered_cloud_point_num * size_of_uint, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_poses_occluded, poses_occluded, num_poses * size_of_int, cudaMemcpyHostToDevice);
        
        cudaMemcpy(cuda_pose_segmentation_label, pose_segmentation_label, num_poses * size_of_int, cudaMemcpyHostToDevice);
        cudaMemcpy(cuda_observed_cloud_label, result_observed_cloud_label, observed_cloud_point_num * size_of_int, cudaMemcpyHostToDevice);

        const size_t threadsPerBlock = 256;
        dim3 numBlocksR((rendered_cloud_point_num + threadsPerBlock - 1) / threadsPerBlock, 1);
        compute_render_cost<<<numBlocksR, threadsPerBlock>>>(
            cuda_knn_dist,
            cuda_knn_index,
            cuda_cloud_pose_map,
            cuda_poses_occluded,
            cuda_rendered_cost,
            sensor_resolution,
            rendered_cloud_point_num,
            observed_cloud_point_num,
            cuda_pose_point_num,
            cuda_rendered_cloud_color,
            cuda_observed_cloud_color,
            cuda_rendered_cloud,
            cuda_observed_explained,
            cuda_pose_segmentation_label,
            cuda_observed_cloud_label,
            cost_type,
            15);
        

        

        if (cudaGetLastError() != cudaSuccess) {
            printf("ERROR: Unable to execute kernel\n");
            cudaFree(cuda_knn_dist);
            cudaFree(cuda_knn_index);
            cudaFree(cuda_cloud_pose_map); 
            cudaFree(cuda_observed_cloud_color); 
            cudaFree(cuda_rendered_cloud); 
            cudaFree(cuda_rendered_cloud_color); 
            cudaFree(cuda_poses_occluded); 
            // cudaFree(cuda_rendered_cost); 
            // cudaFree(cuda_observed_explained); 
            cudaFree(cuda_pose_segmentation_label);
            cudaFree(cuda_observed_cloud_label);
            return false;
        }

        thrust::device_vector<float> rendered_multiplier_val(num_poses, 100);
        if (true)
        {
            thrust::transform(
                cuda_rendered_cost_vec.begin(), cuda_rendered_cost_vec.end(), 
                cuda_pose_point_num_vec.begin(), cuda_rendered_cost_vec.begin(), 
                thrust::divides<float>()
            );
            thrust::transform(
                cuda_rendered_cost_vec.begin(), cuda_rendered_cost_vec.end(), 
                rendered_multiplier_val.begin(), cuda_rendered_cost_vec.begin(), 
                thrust::multiplies<float>()
            );
        }
        rendered_cost = (float*) malloc(num_poses * size_of_float);
        cudaMemcpy(rendered_cost, cuda_rendered_cost, num_poses * size_of_float, cudaMemcpyDeviceToHost);


        // Compute observe cost using points marked in render cost kernel
        if (calculate_observed_cost)
        {
            thrust::device_vector<float> cuda_pose_observed_explained_vec(num_poses, 0);
            float* cuda_pose_observed_explained = thrust::raw_pointer_cast(cuda_pose_observed_explained_vec.data());

            dim3 numBlocksO((num_poses * observed_cloud_point_num + threadsPerBlock - 1) / threadsPerBlock, 1);
            compute_observed_cost<<<numBlocksO, threadsPerBlock>>>(
                num_poses,
                observed_cloud_point_num,
                cuda_observed_explained,
                cuda_pose_observed_explained
            );
            
            // Subtract total observed points for each pose with explained points for each pose
            thrust::device_vector<float> cuda_pose_observed_points_total_vec = pose_observed_points_total;
            thrust::device_vector<float> cuda_observed_cost_vec(num_poses, 0);
            thrust::transform(
                cuda_pose_observed_points_total_vec.begin(), cuda_pose_observed_points_total_vec.end(), 
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
                cuda_pose_observed_points_total_vec.begin(), cuda_observed_cost_vec.begin(), 
                thrust::divides<float>()
            );

            // Multiply by 100
            thrust::transform(
                cuda_observed_cost_vec.begin(), cuda_observed_cost_vec.end(), 
                rendered_multiplier_val.begin(), cuda_observed_cost_vec.begin(), 
                thrust::multiplies<float>()
            );

            // printf("Observed cost\n");
            // thrust::copy(
            //     cuda_observed_cost_vec.begin(),
            //     cuda_observed_cost_vec.end(), 
            //     std::ostream_iterator<int>(std::cout, " ")
            // );

            observed_cost = (float*) malloc(num_poses * size_of_float);
            float* cuda_observed_cost = thrust::raw_pointer_cast(cuda_observed_cost_vec.data());
            cudaMemcpy(observed_cost, cuda_observed_cost, num_poses * size_of_float, cudaMemcpyDeviceToHost);
        
        }

        // result_observed_explained = (uint8_t*) malloc(num_poses * observed_cloud_point_num * size_of_uint);
        // cudaMemcpy(result_observed_explained, cuda_observed_explained, num_poses * observed_cloud_point_num * size_of_uint, cudaMemcpyDeviceToHost);

        // for (int i = 0; i < num_poses; i++)
        // {
        //     printf("%f ", rendered_cost[i]);
        // }
        // printf("\n");

        printf("compute_cost() done\n");
        cudaFree(cuda_knn_dist);
        cudaFree(cuda_knn_index);
        cudaFree(cuda_cloud_pose_map); 
        cudaFree(cuda_observed_cloud_color); 
        cudaFree(cuda_rendered_cloud); 
        cudaFree(cuda_rendered_cloud_color); 
        cudaFree(cuda_poses_occluded); 
        // cudaFree(cuda_rendered_cost); 
        // cudaFree(cuda_observed_explained); 
        cudaFree(cuda_pose_segmentation_label);
        cudaFree(cuda_observed_cloud_label);
        return true;
    }
}

