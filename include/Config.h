//
// Created by lwilkinson on 8/11/22.
//
#include "Enums.h"

#pragma once


namespace sop {

enum TilingStrategy: int {
  MANUAL_TILING = 0,
  CAKE_TILING = 1,
  CAKE_TILING_WITH_TLB_COMPENSATION = 2,
};

struct TileConfig {
  // The deemed tile sizes; If we don't tile along one dimension, then one should be set to the full dimension
  int M_c = 16;
  int K_c = 256;
  int N_c = 64;

  int runtimeSchedule = nmM; // 4, 5 sigsegv, 6, 7, 12, 14 incorrect answer

  TilingStrategy tiling_strategy = CAKE_TILING_WITH_TLB_COMPENSATION;
  float beta = 1.0;
  bool sparse_a = true;
  int max_tlb_entries = 64;
  int tlb_page_size = 4096;
};

};