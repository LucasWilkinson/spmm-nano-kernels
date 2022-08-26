//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_MICROKERNELDESC_H
#define DNN_SPMM_BENCH_MICROKERNELDESC_H


#include <vector>
#include "utils/Vec.h"
#include "utils/tiling.h"

template <
  typename _MicroKernel
>
struct MicroKernelDesc {
  using Vec = ::Vec<typename _MicroKernel::Scalar, _MicroKernel::vec_width_bits>;
  using Scalar = typename _MicroKernel::Scalar;
  using VecType = typename Vec::Type;
  using MicroKernel = _MicroKernel;

  static const int M_r = _MicroKernel::M_r;
  static const int N_r = _MicroKernel::N_r;
  static const int N_r_reg = _MicroKernel::N_r_reg;
  static const int vec_width_bits = Vec::vec_width_bits;

  using RegTile = RegTiles<M_r, N_r>;
  using TileDims = ::TileDims<M_r, N_r>;
};

#endif // DNN_SPMM_BENCH_MICROKERNELDESC_H
