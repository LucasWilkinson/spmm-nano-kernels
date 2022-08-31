//
// Created by lwilkinson on 8/11/22.
//

#pragma once

#include <math.h>
#include <memory>
#include <numeric>
#include <iostream>
#include <fstream>
#include <sstream>

#include "utils/error.h"
#include "utils/algorithmic.h"
#include "utils/Vec.h"

#include "cake_block_dims.h"
#include "TileLocs.h"
#include "COO.h"

#include "Enums.h"
#include "Storage.h"
#include "Config.h"
#include "Tile.h"
#include "Executor.h"
#include "ExecutorFactory.h"
#include "MicroKernelPackerFactory.h"
#include "mapping_io.h"

//#define PACK_B
using std::vector;

namespace sop {

template <typename KernelDesc>
class MatMul {
  using Scalar = typename KernelDesc::Scalar;

  using CSRPtr = int;
  using CSRIndex = int;

  struct MergedTile {
    enum TileType type;

    IntRange ti;
    IntRange tj;

    bool load_c = true;
    int num_tiles() const {
      return ti.size() * tj.size();
    }
  };

  TileConfig config;
  int total_num_tiles = 0;
  int num_threads = 0;
  int m = 0, k = 0;

  std::string executor_id;
  std::string mapping_id;

  ExecutorFactory<KernelDesc>* executor_factory;
  MicroKernelPackerFactory<Scalar>* packer_factory;
  std::shared_ptr<MicroKernelPacker<Scalar>> packer;

  Shape matrix_tiled_shape;
  Shape tile_shape;
  Shape matrix_shape;
  TileLocs tile_locs;

  DenseMatrix<int> nnz_in_tile;
  DenseMatrix<double> tile_densities;
  DenseMatrix<bool> tile_is_dense;
  DenseMatrix<TileType> tile_type;

  struct Stats {
    int total_tile_count = 0;

    int sop_tiles_count = 0;
    int sop_tiles_nnz_count = 0;
    int sop_tiles_padding = 0;

    int csr_tiles_count = 0;
    int csr_tiles_nnz_count = 0;

    int dense_tiles_count = 0;
    int dense_tiles_padding = 0;
    int dense_tiles_nnz_count = 0;
  } stats;

  using _PackedTile = PackedTile<Scalar>;

  vector<vector<int>> row_panels_per_thread;
  vector<vector<MergedTile>> tiles_to_pack; // Post-scheduling, Pre-Packing
  vector<vector<_PackedTile>> packed_tiles;

  vector<int> panel_swizzle;

  COO<Scalar>* coo = nullptr;
  void* linear_buffer = nullptr;

  void inspect_and_pack() {
    free(linear_buffer);
    linear_buffer = nullptr;
    stats = Stats();

    matrix_shape = { coo->rows(), coo->cols() };
    tile_shape = {config.m_tile, config.k_tile};

    tile_locs = TileLocs(tile_shape, matrix_shape, TileLocs::COL_FIRST);
    matrix_tiled_shape = {tile_locs.num_i_tiles(), tile_locs.num_j_tiles()};

    //schedule_panels();
    pack_tiles();
    pack_linear();
  }

 public:
  MatMul(
      int m, int k,
      int           b_col_predict,
      const Scalar* values,
      const int*    row_offsets,
      const int*    column_indices,
      TileConfig    config_,
      int           num_threads,
      std::string   executor_id,
      std::string   mapping_id)
      : m(m), k(k), config(config_),
        num_threads(num_threads),
        executor_id(executor_id),
        executor_factory(ExecutorFactory<KernelDesc>::get_factory(executor_id)),
        packer_factory(MicroKernelPackerFactory<Scalar>::get_factory(executor_id)),
        mapping_id(mapping_id) {
    coo = new COO<Scalar>(m, k, row_offsets, column_indices, values);

    ERROR_AND_EXIT_IF(!executor_factory, "Executor factory not found");
    ERROR_AND_EXIT_IF(!packer_factory, "Packer factory not found");

    ERROR_AND_EXIT_IF(packer_factory->M_r != executor_factory->M_r,
                      "M_r mismatch between packer and executor");

    std::string filepath(__FILE__);
    auto end_of_path = filepath.find_last_of('/');
    filepath = filepath.substr(0, end_of_path + 1);

    auto nanokernel_mapping = read_pattern_mapping(mapping_id,
       {"mappings/", filepath + "../mappings/"});
    packer = packer_factory->create_specialized_packer(nanokernel_mapping);

    if (config.tiling_strategy == CAKE_TILING ||
        config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {
      cake_cntx_t* cake_cntx = cake_query_cntx();

      cake_cntx->nr = executor_factory->N_r;
      cake_cntx->mr = executor_factory->M_r;
      cake_cntx->ncores = num_threads;

      cache_dims_t* cache_dims = get_cache_dims_3(
          m, b_col_predict, k, num_threads, cake_cntx, KMN,
          nullptr, double(coo->nnz()) / (m * k), false, true);

      ERROR_AND_EXIT_IF(!cache_dims->m_c || !cache_dims->k_c || !cache_dims->n_c,
                        "Invalid cache dimensions");

      config.m_tile = cache_dims->m_c;
      config.k_tile = cache_dims->k_c;
      config.n_tile = cache_dims->n_c;

      if (config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {
        static const int tlb_entries = 64;
        static const int tlb_entries_target = 64;
        static const int page_size = 4096;

        int tlb_entries_used = (b_col_predict * config.k_tile) / page_size;

        //std::cout << "TLB entries: " << config.k_tile << " " << tlb_entries_used << std::endl;

        if (tlb_entries_used > tlb_entries_target) {
          int new_k_tile = (tlb_entries_target * page_size) / b_col_predict;
          int diff = config.k_tile - new_k_tile;
          config.k_tile = new_k_tile;

          // Generous realloc to N-tile
          //const int N_r = KernelDesc::N_r;
          //config.n_tile += ((diff + N_r - 1) / N_r) * N_r;
        }

        tlb_entries_used = (b_col_predict * config.k_tile) / page_size;
        //std::cout << "Updated TLB entries: " << config.k_tile << " " << tlb_entries_used << std::endl;
      }

      free(cake_cntx);
      free(cache_dims);
    }

    inspect_and_pack();
  }

  TileConfig get_config() const {
    //std::cout << "Config: " << config.m_tile << " " << config.k_tile << " " << config.n_tile << std::endl;
    return config;
  }

  ~MatMul() {
    delete coo;
    free(linear_buffer);
  }

  // This operator overloading enables calling
  // operator function () on objects of increment
  void operator()(Scalar* C, const Scalar* B, int b_cols) const {
    struct Executor* executor = create_executor(C, B, b_cols) ;
    (*executor)();
    delete executor;
  }

  // This operator overloading enables calling
  // operator function () on objects of increment
  struct Executor* create_executor(Scalar* C, const Scalar* B, int b_cols) const {
    return executor_factory->create_specialized_executor(
            m, k, b_cols, packed_tiles, B, C, 1, num_threads, config);
  }


private:
  void schedule_panels() {

  }

  void pack_tiles() {
    bool parallel_tiles_packed = false;

    packed_tiles.resize(matrix_tiled_shape.rows);
    for (auto& panel_packed_tiles : packed_tiles){
      panel_packed_tiles.resize(matrix_tiled_shape.cols);
    }

    for (int ti = 0; ti < matrix_tiled_shape.rows; ti++) {
      const auto panel_tile_locs = tile_locs.row_panel(ti);
      for (int tj = 0; tj < panel_tile_locs.size(); tj++) {
        _pack_ukernel(panel_tile_locs[tj].loc,
                      tj != 0,
                      packed_tiles[ti][tj]);
      }
    }
  }

  void pack_linear() {
    int linear_size = 0;
    for (auto& panel : packed_tiles)
      for (auto& tile : panel)
        linear_size += tile.linear_size_in_bytes();

    linear_buffer = aligned_alloc(64, linear_size);
    void* linear_buffer_tmp = linear_buffer;

    for (auto& panel : packed_tiles)
      for (auto& tile : panel)
        tile = std::move(tile.pack_linear(&linear_buffer_tmp));
  }

  void debug_print(const PackedTile<KernelDesc>& tile, int thread = -1) {
    std::cout << "Tile" << std::endl;
    std::cout << "  thread: " << thread << std::endl;
    std::cout << "  type:   " << tile.type << std::endl;
    std::cout << "  loc:    (" << tile.loc.rows.start << ", "
              << tile.loc.rows.end << ")";
    std::cout << " (" << tile.loc.cols.start << ", " << tile.loc.cols.end << ")"
              << std::endl;
    std::cout << "  shape:  (" << tile.shape.rows << ", " << tile.shape.cols
              << ")" << std::endl;
    std::cout << "  load_c: " << tile.load_c << std::endl;
  }

  //
  //  Packing helpers
  //   typename Super::Task& t = this->task;

  SubmatrixLoc _tile_loc(const MergedTile& merged_tile) {
    if (merged_tile.num_tiles() > 1) {
      return merge_locs(
          tile_locs.submatrix_locs(merged_tile.ti, merged_tile.tj));
    } else {
      return tile_locs.at(merged_tile.ti.start, merged_tile.tj.start).loc;
    }
  }

  void _pack_ukernel(const SubmatrixLoc t_loc, bool load_c,
                 _PackedTile& tile_to_pack) {
    tile_to_pack.type = SPARSE_SOP;
    tile_to_pack.loc = t_loc;
    tile_to_pack.shape = t_loc.shape();
    tile_to_pack.load_c = load_c;
    tile_to_pack.free_on_destruction = true;

    Tile<Scalar> sop_tile(*coo, t_loc, packer);

    tile_to_pack.sop.num_panels = sop_tile.num_panels();
    tile_to_pack.sop.panel_descs = new MicroKernelPackedData[sop_tile.num_panels()];
    sop_tile.pack(tile_to_pack.sop.panel_descs);

    stats.sop_tiles_count++;
//    stats.sop_tiles_nnz_count += tile.nnz();
//    stats.sop_tiles_padding += sop_tile.num_padded_nnz();
  }
};
};