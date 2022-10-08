//
// Created by lwilkinson on 10/7/22.
//

#pragma once

template<typename T>
inline __attribute__((__always_inline__))
T next_largest_multiple(T x, T multiple) {
    return ((x + multiple - 1) / multiple) * multiple;
}
