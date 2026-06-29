#include "quad_tree_builder.cuh"
#include "node.cuh"
#include "quad_tree_traversor.cuh"

#include <random>
#include <memory>
#include <chrono>

std::unique_ptr<float[]> generate_random_floats(size_t N, float min, float max)
{
    std::unique_ptr<float[]> data(new float[N]);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);

    for (size_t i = 0; i < N; ++i)
    {
        data[i] = dist(gen);
    }

    return data;
}

constexpr size_t N = 1000000;

int main(void)
{
    try
    {
        printf("Allocate points\n");
        auto host_buffer_x = generate_random_floats(N, -1.0f, 1.0f);
        auto host_buffer_y = generate_random_floats(N, -1.0f, 1.0f);
        auto host_buffer_z = generate_random_floats(N, 0.0f, 1.0f);
        printf("Create GPU vectors\n");
        thrust::device_vector<float> x(host_buffer_x.get(), host_buffer_x.get() + N);
        thrust::device_vector<float> y(host_buffer_y.get(), host_buffer_y.get() + N);
        thrust::device_vector<float> m(host_buffer_z.get(), host_buffer_z.get() + N);
        thrust::device_vector<uint32_t> nlen, f_pos, length;
        thrust::device_vector<uint8_t> is_leaf;
        thrust::device_vector<float> x_com, y_com;

        printf("Create class\n");
        ParallelQuadtreeBuilder p(std::move(x), std::move(y), std::move(m));

        printf("Build tree\n");
        auto beg = std::chrono::high_resolution_clock().now();

        std::tie(nlen, f_pos, length, is_leaf, x_com, y_com) = p.build_tree();
        std::tie(x, y, m) = p.retrive_arguments();

        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock().now() - beg).count() << "\n";

        QuadTreeTraversor<
            TsneApproxCond,
            TsneNodeHanlder,
            TsneLeafHandler
        > traversor;

        traversor.load_points(std::move(x), std::move(y));
        traversor.load_tree(
            std::move(nlen),
            std::move(f_pos),
            std::move(length),
            std::move(is_leaf),
            std::move(x_com),
            std::move(y_com)
        );
        traversor.set_face_lenght(p.get_face_len());

        beg = std::chrono::high_resolution_clock().now();
        traversor.traverse();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock().now() - beg).count() << "\n";

        cudaDeviceSynchronize();
    }
    catch(...)
    {
        printf("It seems that we throw and arena was to small\n");
    }

    return 0;
}