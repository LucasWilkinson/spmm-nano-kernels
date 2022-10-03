//
// Created by lwilkinson on 5/25/22.
//

#ifndef DNN_SPMM_BENCH_VEC_TYPE_UTILS_H
#define DNN_SPMM_BENCH_VEC_TYPE_UTILS_H

#ifdef __AVX512F__
#include "vectorclass.h"
#define VECTORCLASS_ENABLED=1

template<typename _Scalar, int vector_width>
struct Vec { using Type = Vec8f; using Scalar = _Scalar; };

template<> struct Vec<float,  128> { using Type = Vec4f;  using Scalar = float;  static const int vec_width_bits = 128; };
template<> struct Vec<float,  256> { using Type = Vec8f;  using Scalar = float;  static const int vec_width_bits = 256; };
template<> struct Vec<float,  512> { using Type = Vec16f; using Scalar = float;  static const int vec_width_bits = 512; };
template<> struct Vec<double, 128> { using Type = Vec2d;  using Scalar = double; static const int vec_width_bits = 128; };
template<> struct Vec<double, 256> { using Type = Vec4d;  using Scalar = double; static const int vec_width_bits = 256; };
template<> struct Vec<double, 512> { using Type = Vec8d;  using Scalar = double; static const int vec_width_bits = 512; };
#endif

#ifdef __ARM_NEON__
#include "arm_neon.h"

template<typename _Scalar, int vector_width>
struct Vec { using Type = float32x4_t; using Scalar = _Scalar; };

template<> struct Vec<float,  128> { using Type = float32x4_t;  using Scalar = float;  static const int vec_width_bits = 128; };
//template<> struct Vec<double, 128> { using Type = float64x2_t;  using Scalar = double; static const int vec_width_bits = 128; };
#endif

#endif //DNN_SPMM_BENCH_VEC_TYPE_UTILS_H
