#ifndef QUAD_TREE_TRAVERSOR
#define QUAD_TREE_TRAVERSOR

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <tuple>
#include <cstdio>

#include "quad_tree_builder.cuh"

struct TsneApproxCond{
    inline bool __device__ __host__ operator()(float x_com, float y_com, float x, float y){
        return false;
    }
};

struct TsneNodeHanlder{
    inline float __device__ __host__ operator()(float x_com, float y_com, float x, float y){
        return 0.0f;
    }
};

struct TsneLeafHandler{
    inline float __device__ __host__ operator()(float x_com, float y_com, float x, float y){
        return 0.0f;
    }
};

/*
    F shall be a functor which can be called to compute function for
    single point during tree traversal. Ideally we could allocate a vector
    this way we compute gradient and write to seperate place for each point
    and after that we reduce.
*/
template <typename ApproxCond, typename NodeHanlder, typename LeafHandler>
class QuadTreeTraversor{
public:
    QuadTreeTraversor() = default;

    inline void set_face_lenght(float face_length){
        this->face_length = face_length;
    }

    inline void load_points(
        thrust::device_vector<float> x,
        thrust::device_vector<float> y
    ){
        this->x = std::move(x);
        this->y = std::move(y);
    }

    inline std::tuple<
        thrust::device_vector<float>,
        thrust::device_vector<float>
    > get_points(void){
        return {
            std::move(x),
            std::move(y)
        };
    }

    inline void load_tree(
        thrust::device_vector<uint32_t> f_pos,
        thrust::device_vector<uint32_t> length,
        thrust::device_vector<uint8_t> is_leaf,
        thrust::device_vector<float> x_com,
        thrust::device_vector<float> y_com
    ){
        this->f_pos = std::move(f_pos);
        this->length = std::move(length);
        this->is_leaf = std::move(is_leaf);
        this->x_com = std::move(x_com);
        this->y_com = std::move(y_com);
    }

    inline std::tuple<
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>,
        thrust::device_vector<float>,
        thrust::device_vector<float>
    > get_tree(void){
        return {
            std::move(f_pos),
            std::move(length),
            std::move(is_leaf),
            std::move(x_com),
            std::move(y_com)
        };
    }

    /*
    This function will traverse the tree for loaded points and store
    result for each point. Better than to summarize gradient directly
    to single value due to lock contention. At this point single vector
    should not hugely increase memory usage, after tree construction.
    !!! Sorting input points spacially to lower warp divergence !!!
    */
    inline thrust::device_vector<float> traverse(){
        thrust::device_vector<float> res(x.size());

        uint32_t *f_pos_d = f_pos.data().get(); 
        uint32_t *length_d = length.data().get();
        uint8_t *is_leaf_d = is_leaf.data().get();
        float *x_com_d = x_com.data().get();
        float *y_com_d = y_com.data().get();
        const float *x_d = x.data().get();
        const float *y_d = y.data().get();
        const float face_length_d = face_length;
        thrust::for_each(
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_counting_iterator<uint32_t>(0),
                    x.begin(),
                    y.begin()
                )
            ),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    thrust::make_counting_iterator<uint32_t>(x.size()),
                    x.end(),
                    y.end()
                )
            ),
            [f_pos_d, length_d, is_leaf_d, x_com_d, y_com_d, x_d, y_d, face_length_d]
            __device__ __host__ (thrust::tuple<uint32_t, float, float> t){
                ApproxCond approx_cond;
                NodeHanlder node_handler;
                LeafHandler leaf_handler;
                
                uint32_t idx = thrust::get<0>(t);
                float x = thrust::get<1>(t);
                float y = thrust::get<2>(t);
                float res = 0;
                /* Has to be at least 4 * tree_max_height */
                uint32_t stack[128];
                uint32_t top = 0;

                stack[top++] = 0;

                while (top)
                {
                    uint32_t node_idx = stack[--top];

                    if (is_leaf_d[node_idx])
                    {
                        /* This loop can be unrolled as well */
                        for(uint32_t i = 0; i < length_d[node_idx]; i++)
                        {
                            res += leaf_handler(
                                x_d[f_pos_d[node_idx] + i],
                                y_d[f_pos_d[node_idx] + i],
                                x, y
                            );
                        }
                        continue;
                    }

                    if(approx_cond(
                        x_com_d[node_idx],
                        y_com_d[node_idx],
                        x, y)
                    ){
                        node_handler(
                            x_d[f_pos_d[node_idx]],
                            y_d[f_pos_d[node_idx]],
                            x, y
                        );
                        continue;
                    }

                    #pragma unroll 4
                    for (uint32_t i = 0; i < 4; ++i)
                    {
                        if(i < length_d[node_idx]) stack[top++] = f_pos_d[node_idx] + i;
                    }
                }
            }
        );

        return res;
    }

private:
    constexpr float theta() { return 0.5f; };

    float face_length;

    /* Do not modify points here */
    thrust::device_vector<float> x;
    thrust::device_vector<float> y;

    thrust::device_vector<uint32_t> f_pos;
    thrust::device_vector<uint32_t> length;
    thrust::device_vector<uint8_t> is_leaf;
    thrust::device_vector<float> x_com;
    thrust::device_vector<float> y_com;
};

#endif