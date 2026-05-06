#include "quad_tree.cuh"

#include <list>

#include <thrust/extrema.h>
#include <thrust/pair.h>

template <typename T>
static thrust::device_vector<T> compress_vector(std::list<thrust::device_vector<T>> &vector_list)
{
    size_t total_len = 0;
    for (thrust::device_vector<T> &vector : vector_list)
        total_len += vector.size();

    thrust::device_vector<T> compressed(total_len);

    size_t offset = 0;
    while (!vector_list.empty())
    {
        thrust::device_vector<T> vector = vector_list.front();
        vector_list.pop_front();

        cudaMemcpy(
            compressed.data().get() + offset,
            vector.data().get(),
            sizeof(T) * vector.size(),
            cudaMemcpyKind::cudaMemcpyDeviceToDevice);
        offset += vector.size();
    }

    return std::move(compressed);
}

__device__ __host__ uint64_t expand_bits(uint32_t &v)
{
    uint64_t x = v & 0x00000000FFFFFFFF;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    return x;
}

template <typename T>
static void dump_device_vector(const thrust::device_vector<T> &v, const char *prefix)
{
    std::cout << prefix << " : ";
    for (const T &e : v)
    {
        std::cout << (uint32_t)e << ", ";
    }
    std::cout << std::endl;
}

void ParallelQuadtree::build_tree()
{
    compute_codes();

    auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(x.begin(), y.begin(), m.begin()));
    thrust::stable_sort_by_key(code.begin(), code.end(), zip_begin);

    thrust::device_vector<uint64_t> p_key;
    thrust::device_vector<uint32_t> nlen;
    thrust::device_vector<uint8_t> clen;
    {
        std::list<thrust::device_vector<uint8_t>> node_children_list;
        std::list<thrust::device_vector<uint32_t>> node_points_list;
        std::list<thrust::device_vector<uint64_t>> node_code_list;

        for (int i = H_max; i >= 0; i--)
        {
            thrust::device_vector<uint8_t> node_children;
            thrust::device_vector<uint32_t> node_points;
            thrust::device_vector<uint64_t> node_codes;

            const thrust::device_vector<uint64_t> *prev_codes = node_code_list.empty() ? nullptr : &node_code_list.front();

            std::tie(node_codes, node_points, node_children) =
                generate_quadrants_for_level(code, prev_codes ? *prev_codes : thrust::device_vector<uint64_t>{}, i);

            node_children_list.push_front(std::move(node_children));
            node_points_list.push_front(std::move(node_points));
            node_code_list.push_front(std::move(node_codes));
        }

        //printf("Compress\n");
        p_key = compress_vector<uint64_t>(node_code_list);
        nlen = compress_vector<uint32_t>(node_points_list);
        clen = compress_vector<uint8_t>(node_children_list);
        // dump_device_vector<uint32_t>(nlen, "NUMBER OF POINTS UNDER QUAD: ");
        // //printf("NODES BEFORE TRIM %d\n", (int)p_key.size());
    }

    //printf("Trim\n");
    trim_redundant_nodes(p_key, nlen, clen);
    //printf("Done\n");
    // //printf("NODES AFTER TRIM %d\n", (int)p_key.size());
    cudaDeviceSynchronize();
}

/*
    Could be better:
        1. Find min/max for x and y
        2. In single transform step
           normalize and compute code
*/
void ParallelQuadtree::compute_codes()
{
    auto x_min_max = thrust::minmax_element(x.begin(), x.end());
    auto y_min_max = thrust::minmax_element(y.begin(), y.end());

    thrust::device_vector<float> x_norm(x.size());
    thrust::device_vector<float> y_norm(y.size());

    auto trans_x = [x_min_max] __device__ __host__(const float &f)
    {
        return (f - *x_min_max.first) / *x_min_max.second;
    };
    auto trans_y = [y_min_max] __device__ __host__(const float &f)
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
        [] __device__ __host__ (thrust::tuple<float, float> t) {
            float a = thrust::get<0>(t);
            float b = thrust::get<1>(t);

            uint32_t ix = (uint32_t)(fmin(fmax(a, 0.0f), 1.0f) * 4294967295.0f);
            uint32_t iy = (uint32_t)(fmin(fmax(b, 0.0f), 1.0f) * 4294967295.0f);

            return expand_bits(iy) | (expand_bits(ix) << 1);
        }
    );
    // clang-format on
}

std::tuple<
    thrust::device_vector<uint64_t>,
    thrust::device_vector<uint32_t>,
    thrust::device_vector<uint8_t>> 
ParallelQuadtree::generate_quadrants_for_level(const thrust::device_vector<uint64_t>& code,
    const thrust::device_vector<uint64_t>& below_code, int level)
{
    thrust::device_vector<uint8_t, DeviceArenaAllocator<uint8_t>> quad_change_indicator(
        code.size(),
        DeviceArenaAllocator<uint8_t>(internal_arena)
    );
    {
        /*
        1. Curentlly shift on each code happens twice. We
        could compute shift using transform and than find
        indicator seperatly. 
        2. From codes below we could extract this
        information, and there is less codes below.
        */
        const uint64_t *code_d = code.data().get();
        thrust::transform(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(quad_change_indicator.size()),
            quad_change_indicator.begin(),
            [level, code_d] __device__ __host__ (uint32_t i){
                if(i == 0) return true;

                uint64_t a = code_d[i];
                uint64_t b = code_d[i - 1];

                a = a >> (64 - 2 * level);
                b = b >> (64 - 2 * level);

                return a != b;
        });
    }

    uint32_t num_quadrants = thrust::reduce(quad_change_indicator.begin(), quad_change_indicator.end(), 0);
    /* Codes of all valid quadrants at level k */
    thrust::device_vector<uint64_t> quad_codes(num_quadrants);

    thrust::copy_if(
        code.begin(), code.end(),
        quad_change_indicator.begin(),
        quad_codes.begin(),
        [] __device__ __host__ (uint8_t v) { return 0 != v; }
    );

    /* Number of child points for this quadrant */
    thrust::device_vector<uint32_t> quad_point_count(num_quadrants);
    {
        thrust::device_vector<uint32_t, DeviceArenaAllocator<uint32_t>> quad_end_offset(
            num_quadrants + 1,
            DeviceArenaAllocator<uint32_t>(internal_arena)
        );
        uint8_t *quad_change_indicator_d = quad_change_indicator.data().get();
        uint32_t num_points = code.size();
        thrust::copy_if(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(num_points + 1),
            quad_end_offset.begin(),
            [num_points, quad_change_indicator_d] __device__ __host__ (uint32_t i){ 
                return (i == num_points) ? true : quad_change_indicator_d[i]; 
            }
        );

        uint32_t *quad_end_offset_d = quad_end_offset.data().get();
        thrust::transform(
            thrust::make_counting_iterator<uint32_t>(1),
            thrust::make_counting_iterator<uint32_t>(num_quadrants + 1),
            quad_point_count.begin(),
            [quad_end_offset_d] __device__ __host__ (uint32_t i){
                return quad_end_offset_d[i] - quad_end_offset_d[i - 1];
            }
        );
    }

    /* Can safely restore arena now */
    internal_arena->reset();

    /* Number of child nodes for this quadrant */
    thrust::device_vector<uint8_t> quad_children_count(num_quadrants);
    if(level != H_max)
    {
        thrust::device_vector<uint8_t, DeviceArenaAllocator<uint8_t>> quadrant_change_indicator(
            below_code.size(),
            DeviceArenaAllocator<uint8_t>(internal_arena)
        );

        const uint64_t *below_code_d = below_code.data().get(); 
        thrust::transform(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(below_code.size()),
            quadrant_change_indicator.begin(),
            [below_code_d, level] __device__ __host__ (uint32_t i){
                if(i == 0) return true;

                uint64_t a = below_code_d[i - 1] >> (64 - 2 * level);
                uint64_t b = below_code_d[i] >> (64 - 2 * level);

                return a != b;
            }
        );

        thrust::device_vector<uint32_t, DeviceArenaAllocator<uint32_t>> quad_end_offset(
            num_quadrants + 1,
            DeviceArenaAllocator<uint32_t>(internal_arena)
        );
        uint8_t *quadrant_change_indicator_d = quadrant_change_indicator.data().get();
        uint32_t num_quads_below = below_code.size();
        thrust::copy_if(
            thrust::make_counting_iterator<uint32_t>(0),
            thrust::make_counting_iterator<uint32_t>(num_quads_below + 1),
            quad_end_offset.begin(),
            [num_quads_below, quadrant_change_indicator_d] __device__ __host__ (uint32_t i){
                uint32_t ret = (i == num_quads_below) ? true : quadrant_change_indicator_d[i];
                return ret; 
            }
        );

        uint32_t *quad_end_offset_d = quad_end_offset.data().get();
        thrust::transform(
            thrust::make_counting_iterator<uint32_t>(1),
            thrust::make_counting_iterator<uint32_t>(num_quadrants + 1),
            quad_children_count.begin(),
            [quad_end_offset_d] __device__ __host__ (uint32_t i){
                return quad_end_offset_d[i] - quad_end_offset_d[i - 1];
            }
        );
    }
    else{
        thrust::fill(quad_children_count.begin(), quad_children_count.end(), 0);
    }

    /* Cleanup arena */
    internal_arena->reset();

    return std::make_tuple<
        thrust::device_vector<uint64_t>,
        thrust::device_vector<uint32_t>,
        thrust::device_vector<uint8_t>>
    (
        std::move(quad_codes),
        std::move(quad_point_count),
        std::move(quad_children_count)
    );
}

void ParallelQuadtree::trim_redundant_nodes(thrust::device_vector<uint64_t>& p_key,
    thrust::device_vector<uint32_t>& nlen, thrust::device_vector<uint8_t>& clen)
{
    thrust::device_vector<uint32_t> node_child_start(clen.size()); 
    /* Value initialization is important - by default exclusive_scan happens on uint8_t and it overflows */
    //printf("Excl san\n");
    auto cast = [] __device__ __host__ (uint8_t v) { return static_cast<uint32_t>(v); };
    thrust::exclusive_scan(
        thrust::make_transform_iterator(clen.begin(), cast),
        thrust::make_transform_iterator(clen.end(), cast),
        node_child_start.begin(),
        uint32_t{0}
    );

    //printf("Id gather\n");
    thrust::device_vector<uint32_t> parent_id(clen.size());
    uint32_t *node_child_start_d = node_child_start.data().get();
    uint32_t *parent_id_d = parent_id.data().get();
    thrust::for_each(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(parent_id.size()),
        [node_child_start_d, parent_id_d] __device__ __host__ (uint32_t i){
            /* Uncoalesed - idk how to do diffrent*/
            parent_id_d[node_child_start_d[i]] = i; 
            // //printf("Parent ID - node - %u : parent - %u\n", i, node_child_start_d[i]);
        }
    );

    //printf("Id prop\n");
    thrust::inclusive_scan(parent_id.begin(), parent_id.end(),
        parent_id.begin(), cuda::maximum());

    //printf("Count gather\n");
    thrust::device_vector<uint32_t> parent_point_count(clen.size());
    uint32_t *nlen_d = nlen.data().get();
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(clen.size()),
        parent_point_count.begin(),
        [parent_id_d, nlen_d] __device__ __host__ (uint32_t i){
            // //printf("Parent id %u\n", parent_id_d[i]);
            return nlen_d[parent_id_d[i]];
        }
    );

    internal_arena->reset();

    auto zip_begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_counting_iterator<uint32_t>(0),
            p_key.begin(),
            nlen.begin(),
            clen.begin()
        )
    );

    auto zip_end = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::make_counting_iterator<uint32_t>(p_key.size()),
            p_key.end(),
            nlen.end(),
            clen.end()
        )
    );

    //printf("Redudant node trim\n");
    uint32_t *parent_point_count_d = parent_point_count.data().get();
    uint32_t threshold = T;
    auto end = thrust::remove_if(
        zip_begin, zip_end,
        [parent_point_count_d, threshold] __device__ __host__ (thrust::tuple<uint32_t, uint64_t, uint32_t, uint8_t> t){
            uint32_t i = thrust::get<0>(t);
            // //printf("%u\n", parent_point_count_d[i]);
            return parent_point_count_d[i] <= threshold;
        }
    );

    auto end_tuple = end.get_iterator_tuple();

    p_key.erase(thrust::get<1>(end_tuple), p_key.end());
    nlen.erase(thrust::get<2>(end_tuple), nlen.end());
    clen.erase(thrust::get<3>(end_tuple), clen.end());

    //printf("Shrink\n");
    p_key.shrink_to_fit();
    nlen.shrink_to_fit();
    clen.shrink_to_fit();
}

void ParallelQuadtree::fill_tree(thrust::device_vector<uint64_t> &p_key, 
thrust::device_vector<uint32_t>& nlen, thrust::device_vector<uint8_t>& clen)
{
    thrust::device_vector<uint8_t, DeviceArenaAllocator<uint8_t>>
        s_leaf(p_key.size(), DeviceArenaAllocator<uint8_t>(internal_arena));
    size_t threshold = T;
    uint32_t *nlen_d = nlen.data().get();
    uint8_t *clen_d = clen.data().get();
    thrust::transform(
        thrust::make_counting_iterator<uint32_t>(0),
        thrust::make_counting_iterator<uint32_t>(is_leaf.size()),
        is_leaf.begin(),
        [nlen_d, clen_d, threshold] __device__ __host__ (uint32_t i){
            /* If less than threshold or no children => node is leaf */
            return nlen_d[i] <= threshold || clen_d[i] == 0;
        }
    );
    /* set nlen to 0 if node is not leaf - such that it contributes 0 to prefix sum */
    thrust::replace_if(nlen.begin(), nlen.end(), is_leaf.begin(),
        [] __device__ __host__ (uint8_t mask) { return !mask; }, 0);

    /* set clen for leaf nodes to 0 */
    thrust::replace_if(clen.begin(), clen.end(), is_leaf.begin(),
        [] __device__ __host__ (uint8_t mask) { return mask; }, 0);

    /* do prefix sum, this will figure out offsets in point array where each leaf points to */

    internal_arena->reset();
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
