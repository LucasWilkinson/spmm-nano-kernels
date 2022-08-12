//
// Created by lwilkinson on 8/10/22.
//

#pragma once


#include "Matrix.h"
#include "COO.h"

#include <mkl.h>
#include <vector>

#include "utils/tiling.h"
#include "SOPMicroKernelBase.h"


namespace sop {

//
//  Type Utils
//

template <typename _Ptr, typename _Index>
struct CSRStorageTypes {
  using Ptr = _Ptr;
  using Index = _Index;
};

template <typename _Vec, typename _CSRStorageTypes, typename _RegTile>
struct KernelDesc {
  using Vec = _Vec;
  using Scalar = typename Vec::Scalar;
  using VecType = typename Vec::Type;
  using RegTile = _RegTile;
  using CSRStorageTypes = _CSRStorageTypes;

  static const int M_r = RegTile::M_r;
  static const int N_r = RegTile::N_r * VecType::size();
  static const int N_r_reg = RegTile::N_r;
  static const int vec_width_bits = _Vec::vec_width_bits;

  using TileDims = ::TileDims<M_r, N_r>;
  using Executor = SOPMicroKernelIntrin<Scalar, vec_width_bits, M_r, N_r_reg>;
};

};