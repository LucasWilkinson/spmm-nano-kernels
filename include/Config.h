//
// Created by lwilkinson on 8/11/22.
//

#pragma once


namespace sop {

enum TilingStrategy: int {
  MANUAL_TILING = 0,
  CAKE_TILING = 1,
  CAKE_TILING_WITH_TLB_COMPENSATION = 2,
};

struct TileConfig {
  int M_c = 16;
  int K_c = 256;
  int N_c = 64;

  TilingStrategy tiling_strategy = CAKE_TILING_WITH_TLB_COMPENSATION;
  float beta = 1.0;
  bool sparse_a = true;
};

};