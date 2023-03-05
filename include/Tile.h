//
// Created by lwilkinson on 7/12/22.
//

#pragma once


#include <assert.h>
#include <vector>
#include <numeric>

#include "utils/error.h"
#include "utils/misc.h"

#include "COO.h"
#include "Storage.h"
#include "MicroKernelBase.h"
#include "MicroKernelPacker.h"

using std::array;
using std::array;
using std::pair;

namespace sop {

template <typename Scalar>
class Tile {
  template <typename T>
  int countbits(T ch) {
    int n = 0;
    if (ch) {
      do
        n++;
      while (0 != (ch = ch & (ch - 1)));
    }
    return n;
  }

  SubmatrixLoc m_tile_loc;
  COO<Scalar>& m_coo_matrix;
  std::shared_ptr<MicroKernelPacker<Scalar>> m_packer;
  int m_num_panels;

 public:
  // Scalar* m_values;

  int num_panels() const { return m_num_panels; }

  Tile(COO<Scalar>& coo_matrix,
       SubmatrixLoc tile_loc,
       std::shared_ptr<MicroKernelPacker<Scalar>> packer)
      : m_coo_matrix(coo_matrix),
        m_tile_loc(tile_loc),
        m_packer(packer) {
    m_num_panels = m_tile_loc.rows.size() / packer->M_r;

    ERROR_AND_EXIT_IF(m_tile_loc.rows.size() % packer->M_r != 0,
                      "Tile size is not a multiple of M_r");
  }

  void pack(MicroKernelPackedData<Scalar>* panel_descs) {
    for (int panel_id = 0; panel_id < m_num_panels; panel_id++) {
      SubmatrixLoc panel_loc = m_tile_loc;
      panel_loc.rows.start += panel_id * m_packer->M_r;
      panel_loc.rows.end = panel_loc.rows.start + m_packer->M_r;

      m_packer->pack(panel_descs[panel_id], panel_loc, m_coo_matrix);
    }
  }

  void pack_coalesced(MicroKernelPackedData<Scalar>* panel_descs, uint8_t* buffer = nullptr) {
    pack(panel_descs);

    for (int panel_id = 0; panel_id < m_num_panels; panel_id++) {
      buffer = m_packer->repack_coalesced(panel_descs[panel_id], buffer);
    }
  }
};

}; // namespace sop
