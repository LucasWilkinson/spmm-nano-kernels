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
  C1_nmKM,
  C1_nmKN
};

enum RuntimeSchedule: int {
    nmKNM, // 0
    nmNKM, // 1
    nmMNK, // 2
    nmNMK, // 3 incorrect results
    nmKMN, // 4
    nmMKN, // 5
    nmKM, // 6
    nmMK, // 7 incorrect results
    nmKN, // 8
    nmNK, // 9 incorrect results
    nmMN, // 10
    nmNM, // 11
    nmK, // 12 incorrect results
    nmN, // 13
    nmM // 14
};

enum TileType {
  SPARSE_CSR,
  SPARSE_SOP,
  DENSE,
  EMPTY_TILE,

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
