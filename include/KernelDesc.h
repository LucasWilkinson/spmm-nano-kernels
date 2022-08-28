//
// Created by lwilkinson on 8/10/22.
//

#pragma once


#include "Matrix.h"
#include "COO.h"

#include <vector>

#include "utils/tiling.h"
#include "MicroKernelBase.h"

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
  typename _Scalar,
  typename _CSRSTypes,
  typename _PackingDesc,
  enum Schedule _S
>
struct KernelDesc {
  using Scalar = _Scalar;
  using CSRStorageTypes = _CSRSTypes;
  using PackingDesc = _PackingDesc;

  static const PackingStrategy C_PACKING = _PackingDesc::C_PACKING;
  static const PackingStrategy B_PACKING = _PackingDesc::B_PACKING;
  static const Schedule Sched = _S;
};

using KDFloatNoPacking =
    KernelDesc<float, CSRStorageTypes<float*, int>, PackingDesc<NO_PACKING, NO_PACKING>, NKM>;
using KDFloatCPartialPacking =
    KernelDesc<float, CSRStorageTypes<float*, int>, PackingDesc<PARTIAL_PACKING, NO_PACKING>, NKM>;

};