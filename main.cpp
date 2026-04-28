#include "inc/quad_tree.cuh"

#include <random>
#include <memory>

std::unique_ptr<float[]> generate_random_floats(size_t N, float min, float max) {
    std::unique_ptr<float[]> data(new float[N]);

    std::random_device rd;              // seed source
    std::mt19937 gen(rd());             // Mersenne Twister RNG
    std::uniform_real_distribution<float> dist(min, max);

    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(gen);
    }

    return data;
}

constexpr size_t N = 1000;

int main(void)
{
    // thrust::device_vector<float> x = {0.0f, 0.1f, 0.9f, 1.0f, 0.9999f};
    // thrust::device_vector<float> y = {0.0f, 0.1f, 0.9f, 1.0f, 0.9999f};
    // thrust::device_vector<float> m = {1.0f, 1.0f, 1.0f};

    auto host_buffer_x = generate_random_floats(N, 0.0f, 1.0f);
    auto host_buffer_y = generate_random_floats(N, 0.0f, 1.0f);
    auto host_buffer_z = generate_random_floats(N, 0.0f, 1.0f);
    thrust::device_vector<float> x(host_buffer_x.get(), host_buffer_x.get() + N);
    thrust::device_vector<float> y(host_buffer_y.get(), host_buffer_y.get() + N);
    thrust::device_vector<float> m(host_buffer_z.get(), host_buffer_z.get() + N);

    ParallelQuadtree p(x, y, m);
    p.build_tree();
    // p.dump_internals();

    // std::cout << "Done\n";

    return 0;
}