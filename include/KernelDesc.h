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
  enum Schedule _Sched,
  enum UPanelReorderingStrategy _UPanelReorder>
struct KernelDesc {
  using Scalar = _Scalar;
  using CSRStorageTypes = _CSRSTypes;
  using PackingDesc = _PackingDesc;

  static const PackingStrategy C_PACKING = _PackingDesc::C_PACKING;
  static const PackingStrategy B_PACKING = _PackingDesc::B_PACKING;
  static const Schedule Sched = _Sched;
  static const UPanelReorderingStrategy UPanelOrder = _UPanelReorder;
};


using KD_PIFloatSplitN =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C1_MKN,
        NO_REORDERING
    >;

using KD_PIFloatSplitM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C1_NmKM,
        NO_REORDERING
    >;

using KD_PIFloatLoadBalancedSplitM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C1_MKN,
        LOAD_BALANCING
    >;

using KD_IntelFloatNKM=
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C3_nmNKM,
        NO_REORDERING
    >;

using KD_IntelFloatLoadBalancedNKM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C3_nmNKM,
        LOAD_BALANCING
    >;

using KD_IntelFloatKNM=
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C3_nmKNM,
        NO_REORDERING
    >;

using KD_IntelFloatLoadBalancedKNM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<NO_PACKING, NO_PACKING>,
        C3_nmKNM,
        LOAD_BALANCING
    >;

using KD_IntelFloatCPackedKNM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, NO_PACKING>,
        C3_nmKNM,
        NO_REORDERING
    >;

using KD_IntelFloatCPackedNKM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, NO_PACKING>,
        C3_nmNKM,
        NO_REORDERING
    >;

using KD_IntelFloatLoadBalancedCPackedKNM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, NO_PACKING>,
        C3_nmKNM,
        LOAD_BALANCING
    >;

using KD_IntelFloatPackedKNM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, PACK>,
        C3_nmKNM,
        NO_REORDERING
    >;

using KD_IntelFloatPackedNKM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, PACK>,
        C3_nmNKM,
        NO_REORDERING
    >;


using KD_IntelFloatLoadBalancedCPackedNKM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, PACK>,
        C3_nmNKM,
        LOAD_BALANCING
    >;

using KD_IntelFloatLoadBalancedPackedKNM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, PACK>,
        C3_nmKNM,
        LOAD_BALANCING
    >;

using KD_IntelFloatLoadBalancedPackedNKM =
    KernelDesc<
        float,
        CSRStorageTypes<float*, int>,
        PackingDesc<PACK, PACK>,
        C3_nmNKM,
        LOAD_BALANCING
    >;

};
