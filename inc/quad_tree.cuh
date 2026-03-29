#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/unique.h>

#include <cstdlib>
#include <iostream>

class ParallelQuadtree
{
public:
    using Morton = uint32_t;

    ParallelQuadtree(thrust::device_vector<float> &x,
                     thrust::device_vector<float> &y,
                     thrust::device_vector<float> &m)
        : x(x), y(y), m(m)
    {
    };

    void build_tree();
    /* has to stay public for lambda accessibility for thrust */
    void compute_codes();
    void find_leaves();
    void dump_internals();
private:
    /* Maximum of points in a single leaf */
    static constexpr size_t T = 32;
    /* Maximum height of quadtree */
    static constexpr size_t H_max = 32;

    /* Input data*/
    thrust::device_vector<float> &x;
    thrust::device_vector<float> &y;
    thrust::device_vector<float> &m;

    /*  */
    thrust::device_vector<uint64_t> code;
    thrust::device_vector<bool> is_leaf;
    thrust::device_vector<size_t> first_child_idx;
    thrust::device_vector<size_t> children_idx;
};