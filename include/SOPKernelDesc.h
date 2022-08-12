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

enum PackingStrategy {
  PREPACK,
  PARTIAL_PACKING,
  NO_PACKING
};

template<enum PackingStrategy _C, enum PackingStrategy _B>
struct PackingDesc {
  static const PackingStrategy C_PACKING = _C;
  static const PackingStrategy B_PACKING = _B;
};

template<>
struct PackingDesc<NO_PACKING, NO_PACKING> {
  static const PackingStrategy C_PACKING = NO_PACKING;
  static const PackingStrategy B_PACKING = NO_PACKING;
};

template <typename _Ptr, typename _Index>
struct CSRStorageTypes {
  using Ptr = _Ptr;
  using Index = _Index;
};

template <
  typename _Vec,
  typename _CSRSTypes,
  typename _RegTile,
  typename _PackingDesc
>
struct KernelDesc {
  using Vec = _Vec;
  using Scalar = typename Vec::Scalar;
  using VecType = typename Vec::Type;
  using RegTile = _RegTile;
  using CSRStorageTypes = _CSRSTypes;
  using PackingDesc = _PackingDesc;

  static const int M_r = RegTile::M_r;
  static const int N_r = RegTile::N_r * VecType::size();
  static const int N_r_reg = RegTile::N_r;
  static const int vec_width_bits = _Vec::vec_width_bits;
  static const PackingStrategy C_PACKING = _PackingDesc::C_PACKING;
  static const PackingStrategy B_PACKING = _PackingDesc::B_PACKING;

  using TileDims = ::TileDims<M_r, N_r>;
  using Executor = SOPMicroKernelIntrin<Scalar, vec_width_bits, M_r, N_r_reg>;
};

};