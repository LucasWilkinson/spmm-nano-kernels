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
  PACK,
  NO_PACKING
};


enum UPanelReorderingStrategy {
  NO_REORDERING,
  LOAD_BALANCING
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
  enum UPanelReorderingStrategy _UPanelReorder>
struct KernelDesc {
  using Scalar = _Scalar;
  using CSRStorageTypes = _CSRSTypes;
  using PackingDesc = _PackingDesc;

  static const PackingStrategy C_PACKING = _PackingDesc::C_PACKING;
  static const PackingStrategy B_PACKING = _PackingDesc::B_PACKING;
  static const UPanelReorderingStrategy UPanelOrder = _UPanelReorder;
};

template<typename Scalar>
using KD_PI =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        NO_REORDERING
    >;


template<typename Scalar>
using KD_PILoadBalanced =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        LOAD_BALANCING
    >;


template<typename Scalar>
using KD_Intel=
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        NO_REORDERING
    >;


template<typename Scalar>
using KD_IntelLoadBalanced =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        LOAD_BALANCING
    >;


template<typename Scalar>
using KD_IntelCPacked =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<PACK, NO_PACKING>,
        NO_REORDERING
    >;

template<typename Scalar>
using KD_IntelLoadBalancedCPacked =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<PACK, NO_PACKING>,
        LOAD_BALANCING
    >;

template<typename Scalar>
using KD_IntelPacked =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<PACK, PACK>,
        NO_REORDERING
    >;

template<typename Scalar>
using KD_IntelLoadBalancedPacked =
    KernelDesc<
        Scalar,
        CSRStorageTypes<Scalar*, int>,
        PackingDesc<PACK, PACK>,
        LOAD_BALANCING
    >;
};
