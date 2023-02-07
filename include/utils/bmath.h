//
// Created by lwilkinson on 10/7/22.
//

#pragma once

template<typename T>
inline __attribute__((__always_inline__))
T ceil_div(T x, T y) {
    return ((x + y - 1) / y);
}

template<typename T>
inline __attribute__((__always_inline__))
T floor_div(T x, T y) {
    return (x / y);
}

template<typename T>
inline __attribute__((__always_inline__))
T next_multiple(T x, T multiple) {
    return std::max(ceil_div(x, multiple) * multiple, multiple);
}

template<typename T>
T prev_multiple(T x, T multiple) {
    return std::max(floor_div(x, multiple) * multiple, multiple);
}
template<typename T>
inline __attribute__((__always_inline__))
T largest_multiple_leq(T x, T multiple) {
  return (x / multiple) * multiple;
}
