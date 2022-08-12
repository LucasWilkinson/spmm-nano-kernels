//
// Created by lwilkinson on 8/11/22.
//

#pragma once


namespace sop {

enum TilingStrategy {
  CAKE_TILING,
  MANUAL_TILING,
};

struct TileConfig {
  int m_tile = 16;
  int k_tile = 256;
  int n_tile = 64;
  TilingStrategy tiling_strategy = CAKE_TILING;
};

};