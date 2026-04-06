#ifndef THRUST_VECTOR_CONCEPT
#define THRUST_VECTOR_CONCEPT

#include <concepts>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

struct DeviceTag
{
    static constexpr bool is_device = true;
    template <typename T>
    using thrust_vector_impl = thrust::device_vector<T>;
};

struct HostTag
{
    static constexpr bool is_device = false;
    template <typename T>
    using thrust_vector_impl = thrust::host_vector<T>;
};

template <typename B>
concept QuadTreeBackend = requires {
    typename B::template thrust_vector_impl<int>;
    { B::is_device } -> std::convertible_to<bool>;
};

template <typename T, typename BackendType>
struct thrust_vector_selector;

template <typename T>
struct thrust_vector_selector<T, HostTag> {
    using type = thrust::host_vector<T>;
    using policy = thrust::host;
};

template <typename T>
struct thrust_vector_selector<T, DeviceTag> {
    using type = thrust::device_vector<T>;
    using policy = thrust::device;
};

#endif