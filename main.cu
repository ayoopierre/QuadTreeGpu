#include <random>

#include "inc/quad_tree.cuh"

int main(void)
{
    thrust::device_vector<float> x = {0.0f, 5.0f, 3.0f};
    thrust::device_vector<float> y = {1.0f, 2.0f, 2.0f};
    thrust::device_vector<float> m = {1.0f, 1.0f, 1.0f};

    thrust::host_vector<float> hx = {0.0f, 5.0f, 3.0f};
    thrust::host_vector<float> hy = {1.0f, 2.0f, 2.0f};
    thrust::host_vector<float> hm = {1.0f, 1.0f, 1.0f};

    QuadTree<DeviceTag> tree(x, y, m);
    QuadTree<HostTag> htree(hx, hy, hm);

    tree.build_tree();
    // htree.build_tree();

    std::cout << "Done\n";

    return 0;
}