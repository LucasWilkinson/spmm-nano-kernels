//
// Created by lwilkinson on 7/10/22.
//

#ifndef DNN_SPMM_BENCH_ALGORITHMICUTILS_H
#define DNN_SPMM_BENCH_ALGORITHMICUTILS_H

#include <vector>
#include <algorithm>
#include <numeric>

template <typename T>
std::vector<size_t> argsort(const std::vector<T> &v) {
    std::vector<size_t> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
    return idx;
}

#endif //DNN_SPMM_BENCH_ALGORITHMICUTILS_H
