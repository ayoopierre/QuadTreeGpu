#include "quad_tree.cuh"

#include <thrust/extrema.h>
#include <thrust/pair.h>

__device__ uint64_t expand_bits(uint32_t &v)
{
    uint64_t x = v & 0x00000000FFFFFFFF;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}

void ParallelQuadtree::build_tree()
{
    compute_codes();

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), m.begin()));
    thrust::stable_sort_by_key(code.begin(), code.end(), zip_begin);
}

void ParallelQuadtree::compute_codes()
{
    auto x_min_max = thrust::minmax_element(x.begin(), x.end());
    auto y_min_max = thrust::minmax_element(y.begin(), y.end());

    thrust::device_vector<float> x_norm(x.size());
    thrust::device_vector<float> y_norm(y.size());

    auto trans_x = [x_min_max] __device__(const float &f)
    {
        return (f - *x_min_max.first) / *x_min_max.second;
    };
    auto trans_y = [y_min_max] __device__(const float &f)
    {
        return (f - *y_min_max.first) / *y_min_max.second;
    };
    /* Maybe single-pass zip iterator transform - vectors of same shape? */
    thrust::transform(x.begin(), x.end(), x_norm.begin(), trans_x);
    thrust::transform(y.begin(), y.end(), y_norm.begin(), trans_y);

    code.resize(x.size());

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x_norm.begin(), y_norm.begin()));
    auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(x_norm.end(), y_norm.end()));

    // clang-format off
    thrust::transform(zip_begin, zip_end, code.begin(), 
        [] __device__ (thrust::tuple<float, float> t) {
            float a = thrust::get<0>(t);
            float b = thrust::get<1>(t);

            uint32_t ix = (uint32_t)(fmin(fmax(a, 0.0f), 1.0f) * 4294967295.0f);
            uint32_t iy = (uint32_t)(fmin(fmax(b, 0.0f), 1.0f) * 4294967295.0f);

            return expand_bits(iy) | (expand_bits(ix) << 1);
        }
    );
    // clang-format on
}

void ParallelQuadtree::find_leaves()
{
    auto zip_begin_1 = thrust::make_zip_iterator(thrust::make_tuple(code.begin(), code.begin() + 1));
    auto zip_end_1 = thrust::make_zip_iterator(thrust::make_tuple(code.end() - 1, code.end()));

    /* There is at mose |Points| leafes and this is temporary anyways */
    thrust::device_vector<uint32_t> is_segment_start(x.size());

    // clang-format off
    thrust::transform(zip_begin_1, zip_end_1, is_segment_start.begin(),
        [] __device__ (thrust::tuple<uint64_t, uint64_t> t){
            uint64_t a = thrust::get<0>(t);
            uint64_t b = thrust::get<1>(t);

            /* Get prefix only */
            a = a >> (64 - 2 * H_max);
            b = b >> (64 - 2 * H_max);

            return (a != b) ? 1 : 0;
        }
    );
    // clang-format on

    uint32_t groups = thrust::reduce(is_segment_start.begin(), is_segment_start.end());
    thrust::device_vector<uint32_t> group_offsets(groups);

    // clang-format off
    thrust::copy_if(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(is_segment_start.size()),
        is_segment_start.begin(),
        group_offsets.begin(),
        [] __device__ (const uint32_t a) { return a == 1; }
    );
    // clang-format on

    /*
    Now we can enforce len(leaf) < T. To do so we can check
    in which group we are, and check how far from segment
    start we are. Knowing that we can check if we should
    subdivide to satisfy threshold condition. Subdivisions
    write additional "1" to is_segment_begin, and after that
    we can recompute leaf offsets and and lenghts.
    */

    thrust::device_vector<uint32_t> segment_group_id(is_segment_start.size());
    thrust::exclusive_scan(is_segment_start.begin(), is_segment_start.end(), segment_group_id.begin());

    // clang-format off
    auto zip_begin_2 = thrust::make_zip_iterator(
        thrust::make_tuple(
            segment_group_id.begin(),
            thrust::make_counting_iterator<uint32_t>(0)
        )
    );

    auto zip_end_2 = thrust::make_zip_iterator(
        thrust::make_tuple(
            segment_group_id.end(),
            thrust::make_counting_iterator<uint32_t>(segment_group_id.size())
        )
    );
    
    thrust::transform(zip_begin_2, zip_end_2, is_segment_start.begin(),
        [group_offsets_arr = group_offsets.data()] __device__ (thrust::tuple<uint32_t, uint32_t> t){
            uint32_t group_id = thrust::get<0>(t);
            uint32_t index = thrust::get<1>(t);
            uint32_t group_offset = group_offsets_arr[index];
            uint32_t offset_in_group = index - group_offset;

            return (offset_in_group > T && offset_in_group % T == 0) ? 1 : 0;
        }
    );
    // clang-format on

    /* Now we should have "1" whenever we should create new leaf taking T into account */
    groups = thrust::reduce(is_segment_start.begin(), is_segment_start.end());
    group_offsets.resize(groups);

    // clang-format off
    thrust::copy_if(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(is_segment_start.size()),
        is_segment_start.begin(),
        group_offsets.begin(),
        [] __device__ (const uint32_t a) { return a == 1; }
    );
    // clang-format on

    thrust::device_vector<uint32_t> lengths(groups);

    // clang-format off
    auto zip_begin_3 = thrust::make_zip_iterator(
        thrust::make_tuple(group_offsets.begin(),
        group_offsets.begin() + 1
    ));
    auto zip_end_3 = thrust::make_zip_iterator(
        thrust::make_tuple(group_offsets.end() - 1,
         group_offsets.end()
    ));

    thrust::transform(zip_begin_3, zip_end_3, lengths.begin(),
        [] __device__(thrust::tuple<uint32_t, uint32_t> t) 
        { 
            uint32_t a = thrust::get<0>(t);
            uint32_t b = thrust::get<1>(t);

            return b - a;
        }
    );
    // clang-format on
}

void ParallelQuadtree::dump_internals()
{
    thrust::host_vector<uint64_t> hcode(code);
    for (const auto &lhx : hcode)
    {
        std::cout << lhx << ", ";
    }
    std::cout << std::endl;
}
