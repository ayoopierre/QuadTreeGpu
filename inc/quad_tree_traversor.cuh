#ifndef QUAD_TREE_TRAVERSOR
#define QUAD_TREE_TRAVERSOR

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include <cstdint>
#include <tuple>

#include "quad_tree_builder.cuh"

/*
    F shall be a functor which can be called to compute function for
    single point during tree traversal. Ideally we could allocate a vector
    this way we compute gradient and write to seperate place for each point
    and after that we reduce.
*/
template <typename F>
class QuadTreeTraversor{
public:
    QuadTreeTraversor(F callback) : callback(callback) {};

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
        thrust::for_each(
            thrust::make_zip_iterator(thrust::make_tuple(
                thrust::make_counting_iterator<uint32_t>(0),
                x.begin(),
                y.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                thrust::make_counting_iterator<uint32_t>(x.size())
                x.end(),
                y.end())),
            [f_pos_d, length_d, is_leaf_d, x_com_d, y_com_d] __device__ __host__
            (thrust::tuple<uint32_t, float, float> t){
                uint32_t pnt_idx = thrust::get<0>(t);
                float x = thrust::get<1>(t);
                float y = thrust::get<2>(t);

                uint32_t i = 0;

                uint32_t stack[ParallelQuadtreeBuilder::get_max_height()];
                uint32_t top = 0;

                while(!is_leaf_d[i]){
                    /* Assume tree and points in global coos */ 
                    float dist_sq = (x_com_d[i] - x) * (x_com_d[i] - x) + (y_com_d[i] - y) * (y_com_d[i] - y);
                    float s; // Get side length from builder
                }
            }
        )
    }

private:
    constexpr float theta = 0.5f;

    F callback;

    /* Do not modify points here */
    const thrust::device_vector<float> x;
    const thrust::device_vector<float> y;

    thrust::device_vector<uint32_t> f_pos;
    thrust::device_vector<uint32_t> length;
    thrust::device_vector<uint8_t> is_leaf;
    thrust::device_vector<float> x_com;
    thrust::device_vector<float> y_com;
};

#endif