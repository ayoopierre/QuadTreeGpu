#include "inc/quad_tree.cuh"
#include "inc/node.cuh"

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

constexpr size_t N = 10;

int main(void)
{
    try
    {
        printf("Allocate points\n");
        auto host_buffer_x = generate_random_floats(N, 0.0f, 1.0f);
        auto host_buffer_y = generate_random_floats(N, 0.0f, 1.0f);
        auto host_buffer_z = generate_random_floats(N, 0.0f, 1.0f);
        printf("Create GPU vectors\n");
        thrust::device_vector<float> x(host_buffer_x.get(), host_buffer_x.get() + N);
        thrust::device_vector<float> y(host_buffer_y.get(), host_buffer_y.get() + N);
        thrust::device_vector<float> m(host_buffer_z.get(), host_buffer_z.get() + N);

        printf("Create class\n");
        ParallelQuadtree p(std::move(x), std::move(y), std::move(m));

        printf("Build tree\n");
        auto clock = std::chrono::high_resolution_clock();
        auto beg = clock.now();
        p.build_tree();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(clock.now() - beg).count() << "\n";

        cudaDeviceSynchronize();
    }
    catch(...)
    {
        printf("It seems that we throw and arena was to small\n");
    }

    return 0;
}