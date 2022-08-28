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
  int m_tile = 16;
  int k_tile = 256;
  int n_tile = 64;

  TilingStrategy tiling_strategy = CAKE_TILING_WITH_TLB_COMPENSATION;
};

};