#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>
#include <thrust/unique.h>
#include <thrust/gather.h>

#include <cstdlib>
#include <iostream>
#include <tuple>

class ParallelQuadtree
{
public:
    using Morton = uint32_t;

    ParallelQuadtree(thrust::device_vector<float> &x,
                     thrust::device_vector<float> &y,
                     thrust::device_vector<float> &m)
        : x(x), y(y), m(m) {
          };

    void build_tree();
    /* Has to stay public for lambda accessibility for thrust */
    // Internal fields filling functions
    void compute_codes();

    // Helpers
    std::tuple<thrust::device_vector<uint32_t>,
               thrust::device_vector<uint32_t>,
               thrust::device_vector<uint64_t>>
    find_leafes();
    std::tuple<thrust::device_vector<uint64_t>,
               thrust::device_vector<uint32_t>,
               thrust::device_vector<uint8_t>>
    generate_quadrants_for_level(const thrust::device_vector<uint64_t> &code,
                                 const thrust::device_vector<uint64_t> &below_code, int level);
    std::tuple<thrust::device_vector<uint64_t>,
               thrust::device_vector<uint32_t>,
               thrust::device_vector<uint8_t>>
    generate_quadrants_for_level2(const thrust::device_vector<uint64_t> &code,
                                  const thrust::device_vector<uint64_t> &below_code, int level);
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