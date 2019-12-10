#pragma once

#ifdef CUDA_ON
// cuda
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

#else
// invalidate cuda macro
#define __device__
#define __host__

#endif
// #include "../../glm/glm.hpp"
// #include "../../glm/ext.hpp"
// load ply
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
namespace hist_prioritize {



#ifdef CUDA_ON
// thrust device vector can't be used in cpp by design
// same codes in cuda renderer,
// because we don't want these two related to each other
template <typename T>
class device_vector_holder{
public:
    T* __gpu_memory;
    size_t __size;
    bool valid = false;
    device_vector_holder(){}
    device_vector_holder(size_t size);
    device_vector_holder(size_t size, T init);
    ~device_vector_holder();

    T* data(){return __gpu_memory;}
    thrust::device_ptr<T> data_thr(){return thrust::device_ptr<T>(__gpu_memory);}
    T* begin(){return __gpu_memory;}
    thrust::device_ptr<T> begin_thr(){return thrust::device_ptr<T>(__gpu_memory);}
    T* end(){return __gpu_memory + __size;}
    thrust::device_ptr<T> end_thr(){return thrust::device_ptr<T>(__gpu_memory + __size);}

    size_t size(){return __size;}

    void __malloc(size_t size);
    void __free();
};
struct pose
{
    float x;
    float y;
    float theta;
    int hist[7];
    bool const operator<(const pose &p) const{
        return x < p.x || (x == p.x && y < p.y) || (x==p.x && y == p.y && theta < p.theta);
    }
};
extern template class device_vector_holder<int>;
#endif

#ifdef CUDA_ON
    using Int_holder = device_vector_holder<int>;
#else
    using Int_holder = std::vector<int>;
#endif


#ifdef CUDA_ON
std::vector<pose> compare_hist(const int width, const int height,const float x_min,const float x_max,
                              const float y_min,const float y_max,
                              const float theta_min,const float theta_max,
                              const float trans_res, const float angle_res,
                              const int32_t ob_pixel_num,
                              const std::vector<std::vector<uint8_t>>& observed,
                              const std::vector<std::vector<float> >& cam_matrix,
                              const std::vector<float>& bounding_boxes,
                              const std::vector<int>& hist_vector
                              );
#endif



__host__ __device__ inline
float std__max(float a, float b){return (a>b)? a: b;};
__host__ __device__ inline
float std__min(float a, float b){return (a<b)? a: b;};
}
