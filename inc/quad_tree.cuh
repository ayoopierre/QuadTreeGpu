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
#include <type_traits>

#include "thrust_vector_concept.cuh"
#include "logging.cuh"

template <QuadTreeBackend Backend>
class QuadTree
{
public:
    // template <typename T>
    // using thrust_vector = typename Backend::template thrust_vector_impl<T>;
    /* Did not know about this mechanism - pure struct tag would suffice */
    // template <typename T>
    // using thrust_vector = std::conditional_t<
    //     std::is_same<Backend, HostTag>::value,
    //     thrust::host_vector<T>,
    //     thrust::device_vector<T>>;
    /* Scales better - for new implementation provide new selector */
    template <typename T>
    using thrust_vector = typename thrust_vector_selector<T, Backend>::type;
    template <typename T>
    using thrust_policy = typename thrust_vector_selector<T, Backend>::policy;

    QuadTree(
        thrust_vector<float> &x,
        thrust_vector<float> &y,
        thrust_vector<float> &m)
        : x(std::move(x)), y(std::move(y)), m(std::move(m)) {
          };

    void build_tree();
    /* has to stay public for lambda accessibility for thrust */
    void compute_codes();
    void find_leafes();
    void dump_internals();

private:
    /* Maximum of points in a single leaf */
    static constexpr size_t T = 32;
    /* Maximum height of quadtree */
    static constexpr size_t H_max = 32;

    /* Input data*/
    thrust_vector<float> x;
    thrust_vector<float> y;
    thrust_vector<float> m;

    /*  */
    thrust_vector<uint64_t> code;
    thrust_vector<bool> is_leaf;
    thrust_vector<size_t> first_child_idx;
    thrust_vector<size_t> children_idx;
};