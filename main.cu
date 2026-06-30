#define NCCL_DEBUG INFO

#include "quad_tree_builder.cuh"
#include "node.cuh"
#include "quad_tree_traversor.cuh"
#include "NcclRing.cuh"

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

int main(int argc, char** argv)
{
    printf("Init MPI context\n");
    MPI_Init(&argc, &argv);

    printf("Create MPI Ring topology\n");
    NcclRing ring(MPI_COMM_WORLD);

    constexpr size_t N = 1024;

    float *sendbuf, *recvbuf;

    printf("Prepare tensors\n");
    cudaMalloc(&sendbuf, N * sizeof(float));
    cudaMalloc(&recvbuf, N * sizeof(float));

    printf("Start group\n");
    // Send to the right, receive from the left.
    ncclGroupStart();

    printf("Enqueue send operation\n");
    NCCL_CHECK(ncclSend(
        sendbuf,
        N,
        ncclFloat,
        ring.right(),
        ring.comm(),
        ring.stream()));

    printf("Enqueue recv operation\n");
    NCCL_CHECK(ncclRecv(
        recvbuf,
        N,
        ncclFloat,
        ring.left(),
        ring.comm(),
        ring.stream()));

    printf("End group\n");
    ncclGroupEnd();
    printf("Sent tensors\n");

    printf("Synchronize NCCL stream\n");
    cudaStreamSynchronize(ring.stream());

    cudaFree(sendbuf);
    cudaFree(recvbuf);

    MPI_Finalize();
}