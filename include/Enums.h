//
// Created by lwilkinson on 8/11/22.
//

#pragma once

namespace sop {

//
//   Enumerations
//

enum Schedule {
  C3_nmNKM,
  C3_nmKNM,
  C1_NmKM,
  C1_MKN
};

enum TileType {
  SPARSE_CSR,
  SPARSE_SOP,
  DENSE,

  // Execution type only, TODO seperate out
  SPARSE_MKL,
};

enum DenseTileMergingStrategy {
  ALL_SPARSE,
  ROW_BASED,
  UNREORDERED_ROW_BASED,
  AGG_ROW_BASED,
  PCT_BASED
};

enum SparseTileMergingStrategy { ALL_CSR, ALL_SOP, MERGE_ROW_SOP, MERGE_SOP };
enum ExecutionStrategy {
  UNTILED_MKL,
  TILED_SPARSE,
  BOTH_TILED
};

};
