//
// Created by lwilkinson on 8/2/22.
//

#pragma once

#include <math.h>
#include <vector>
#include "utils/shape.h"
#include "utils/bmath.h"

struct TileLoc      { int ti = 0; int tj = 0; int tid = 0; SubmatrixLoc loc; };

struct TileLocs {

  enum IterationOrder {
    ROW_FIRST,
    COL_FIRST
  };

 private:
  enum IterationOrder m_iteration_order = COL_FIRST;
  int m_num_i_tiles = 0;
  int m_num_j_tiles = 0;
  std::vector<TileLoc> m_locs;

 public:
  TileLocs() = default;
  TileLocs(const TileLocs&) = default;
  TileLocs(Shape tile_shape, Shape matrix_shape, enum IterationOrder iteration_order):
        m_iteration_order(iteration_order) {
    m_num_i_tiles = ceil_div(matrix_shape.rows, (tile_shape.rows));
    m_num_j_tiles = ceil_div(matrix_shape.cols, (tile_shape.cols));
    m_locs.resize(m_num_i_tiles * m_num_j_tiles);

    for (int tid = 0; tid < (m_num_i_tiles * m_num_j_tiles); tid++) {
      int ti, tj;

      if (m_iteration_order == COL_FIRST) {
        tj = tid % m_num_j_tiles;
        ti = tid / m_num_j_tiles;
      } else {
        ti = tid % m_num_i_tiles;
        tj = tid / m_num_i_tiles;
      }

      m_locs[tid] = {
          ti,
          tj,
          tid,
          {{ti * tile_shape.rows,
            std::min((ti + 1) * tile_shape.rows, matrix_shape.rows)},
           {tj * tile_shape.cols,
            std::min((tj + 1) * tile_shape.cols, matrix_shape.cols)}}
      };
    }
  }

  int num_i_tiles() const { return m_num_i_tiles; }
  int num_j_tiles() const { return m_num_j_tiles; }

  TileLoc at_tid(int tid) { return m_locs[tid]; }
  TileLoc at(int ti, int tj) {
    if (m_iteration_order == COL_FIRST) {
      return m_locs[(ti * num_j_tiles()) + tj];
    } else {
      return m_locs[(tj * num_i_tiles()) + ti];
    }
  }


  std::vector<TileLoc> row_panel(int ti) const {
    std::vector<TileLoc> tile_locs;
    tile_locs.reserve(num_j_tiles());

    for (int tj = 0; tj < num_j_tiles(); tj++) {
      if (m_iteration_order == COL_FIRST) {
        tile_locs.push_back(m_locs[(ti * num_j_tiles()) + tj]);
      } else {
        tile_locs.push_back(m_locs[(tj * num_i_tiles()) + ti]);
      }
    }

    return tile_locs;
  }


  std::vector<TileLoc> row_panel(IntRange ti_range) {
    std::vector<TileLoc> tile_locs;
    for (int ti = ti_range.start; ti < ti_range.end; ti++) {
      for (int tj = 0; tj < num_j_tiles(); tj++) {
        if (m_iteration_order == COL_FIRST) {
          tile_locs.push_back(m_locs[(ti * num_j_tiles()) + tj]);
        } else {
          tile_locs.push_back(m_locs[(tj * num_i_tiles()) + ti]);
        }
      }
    }

    return tile_locs;
  }

  std::vector<TileLoc> slice(IntRange ti_range, IntRange tj_range) {
    std::vector<TileLoc> tile_locs;
    for (int ti = ti_range.start; ti < ti_range.end; ti++) {
      for (int tj = tj_range.start; tj < tj_range.end; tj++) {
        if (m_iteration_order == COL_FIRST) {
          tile_locs.push_back(m_locs[(ti * num_j_tiles()) + tj]);
        } else {
          tile_locs.push_back(m_locs[(tj * num_i_tiles()) + ti]);
        }
      }
    }

    return tile_locs;
  }

  std::vector<SubmatrixLoc> submatrix_locs(IntRange ti_range, IntRange tj_range) {
    auto tile_locs = slice(ti_range, tj_range);

    std::vector<SubmatrixLoc> submatrix_locs;
    submatrix_locs.reserve(tile_locs.size());

    for (const auto& tile_loc: tile_locs) submatrix_locs.push_back(tile_loc.loc);
    return submatrix_locs;
  }

  std::vector<TileLoc>::iterator begin() { return m_locs.begin(); }
  std::vector<TileLoc>::iterator end()   { return m_locs.end();   }
};