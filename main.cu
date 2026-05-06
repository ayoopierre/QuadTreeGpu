#include "inc/quad_tree.cuh"

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
        auto host_buffer_x = generate_random_floats(N, 0.0f, 1.0f);
        auto host_buffer_y = generate_random_floats(N, 0.0f, 1.0f);
        auto host_buffer_z = generate_random_floats(N, 0.0f, 1.0f);
        thrust::device_vector<float> x(host_buffer_x.get(), host_buffer_x.get() + N);
        thrust::device_vector<float> y(host_buffer_y.get(), host_buffer_y.get() + N);
        thrust::device_vector<float> m(host_buffer_z.get(), host_buffer_z.get() + N);

        ParallelQuadtree p(std::move(x), std::move(y), std::move(m));

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

    std::cout << "All done\n";

    return 0;
}