#ifndef ASYNC_ALLOCATOR
#define ASYNC_ALLOCATOR

#include <cuda.h>

#include <thrust/device_vector.h>
#include <thrust/system/cuda/execution_policy.h>

class GpuArena {
public:
    GpuArena(size_t size) : total_size(size), offset(0) {
        cudaMalloc(&base_ptr, total_size);
    }

    GpuArena(const GpuArena&) = delete;
    GpuArena& operator=(const GpuArena&) = delete;

    ~GpuArena() {
        cudaFree(base_ptr);
    }

    inline void* allocate(size_t bytes) {
        /* Alligment is required for many CUDA operations */
        size_t aligned_bytes = (bytes + 255) & ~255;
        
        if (offset + aligned_bytes > total_size) {
            throw std::runtime_error("Arena to small\n");
        }

        void* ptr = static_cast<char*>(base_ptr) + offset;
        offset += aligned_bytes;
        return ptr;
    }

    inline void deallocate(void* ptr, size_t bytes) {

    }

    inline void reset() {
        offset = 0;
    }

private:
    void* base_ptr = nullptr;
    size_t total_size;
    size_t offset;
};

template <typename T>
struct ArenaAllocator {
    using value_type = T;

    GpuArena* arena;

    ArenaAllocator(GpuArena* a) : arena(a) {}

    template <typename U>
    ArenaAllocator(const ArenaAllocator<U>& other) : arena(other.arena) {}

    T* allocate(size_t n) {
        return static_cast<T*>(arena->allocate(n * sizeof(T)));
    }

    void deallocate(T* p, size_t n) {
        arena->deallocate(p, n * sizeof(T));
    }

    template <typename U>
    bool operator==(const ArenaAllocator<U>& other) const { return arena == other.arena; }

    template <typename U>
    bool operator!=(const ArenaAllocator<U>& other) const { return arena != other.arena; }
};

#endif