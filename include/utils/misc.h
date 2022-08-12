//
// Created by lwilkinson on 7/21/22.
//

#pragma once

#include <stdint.h>
#include <utility>
#include <vector>

#include "utils/Vec.h"

//
//  Cache utils
//

template<typename  T>
T* cacheline_align_ptr(T* ptr) {
    return (T*)(((uintptr_t(ptr) + 63) / 64) * 64);
}

//
//  Statistics
//

template<typename Scalar>
double median(std::vector<Scalar> &v) {
    size_t n = v.size() / 2;
    nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}

template<typename Scalar>
double mean(std::vector<Scalar> &v) {
    Scalar sum = 0;
    for (const auto &val: v) { sum += val; }
    return sum / v.size();
}


//
//  Static checking
//

template <int ... Is>
constexpr bool is_in(int i, std::integer_sequence<int, Is...>) { return ((i == Is) || ...); }

#define CHECK_IS_IN(val, ...) \
    static_assert(is_in(val, std::integer_sequence<int, __VA_ARGS__>{}), #val " not in " #__VA_ARGS__);

//
//  Buffer Utils
//

template<typename Vec, bool one_iter_max>
inline int _zero_vectorized_loop(int i, typename Vec::Scalar *array, int numel) {
    using VecType = typename Vec::Type;

    if constexpr(one_iter_max) {
        if (numel - i >= VecType::size()) {
            VecType vec(0); vec.store(&array[i]);
            i += VecType::size();
        }
    } else {
        VecType vec(0);
        for (; i < numel - (VecType::size() + 1); i += VecType::size())
            vec.store(&array[i]);
    }

    return i;
}

template<typename T>
void zero(T * array, int numel) {
    int i = 0;

    // Inlined vectorized loops
    i = _zero_vectorized_loop<Vec<T, 512>, false>(i, array, numel);
    i = _zero_vectorized_loop<Vec<T, 256>, true >(i, array, numel);
    i = _zero_vectorized_loop<Vec<T, 128>, true >(i, array, numel);

    // Scalar loop
    for (; i < numel; i++) { array[i] = 0; }
}
