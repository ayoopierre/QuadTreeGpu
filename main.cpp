#include "inc/quad_tree.cuh"

int main(void)
{
    thrust::device_vector<float> x = {0.0f, 0.0000001f, 0.999f, 1.0f, 0.9999f};
    thrust::device_vector<float> y = {0.0f, 0.0000001f, 0.999f, 1.0f, 0.9999f};
    thrust::device_vector<float> m = {1.0f, 1.0f, 1.0f};

    ParallelQuadtree p(x, y, m);
    p.build_tree();
    p.dump_internals();

    std::cout << "Done\n";

    return 0;
}