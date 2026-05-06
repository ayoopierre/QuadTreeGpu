Parallel-primitive based quad-tree construction on GPU and CPU using Thrust.

Refrence: https://adms-conf.org/2019-camera-ready/zhang_adms19.pdf

Note: Fix building for stanalone library for linking to pure C++ projects.

# Issues
Using Nsight Compute and Nsight Systems tools following observations an be made:
1. Total kernel computation takes time in tens of ms dominating operation being sort which took ~4ms.
2. Execution is dominated by cudaMalloc, cudaFree and synchronization between streams.

In code there can be seen that there are many allocations/deallocations of temporary vectors which are required in the algorithm.