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
#include "utils/type_name.h"

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
  static const Schedule schedule = KernelDesc::Sched;

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
  int M_r = 0, N_r = 0;

  int total_num_tiles = 0;
  int num_threads = 0;
  int m = 0, k = 0;

  std::string executor_id;
  std::string mapping_id;

  ExecutorFactory<KernelDesc>* executor_factory;
  Executor<Scalar>* executor = nullptr;
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

  vector<int> upanel_swizzle;

  COO<Scalar>* coo = nullptr;
  void* linear_buffer = nullptr;

  void inspect_and_pack() {
    free(linear_buffer);
    linear_buffer = nullptr;
    stats = Stats();

    matrix_shape = { coo->rows(), coo->cols() };
    tile_shape = {config.M_c, config.K_c};

    tile_locs = TileLocs(tile_shape, matrix_shape, TileLocs::COL_FIRST);
    matrix_tiled_shape = {tile_locs.num_i_tiles(), tile_locs.num_j_tiles()};

    reorder_upanels();
    pack_tiles();
    pack_linear();
  }

 public:
  int require_storage = 0;

  MatMul(
      COO<Scalar>* coo,
      int           b_col_predict,
      TileConfig    config_,
      int           num_threads,
      std::string   executor_id,
      std::string   mapping_id
  ):  coo(coo),
      m(coo->rows()), k(coo->cols()), config(config_),
      num_threads(num_threads),
      executor_id(executor_id),
      mapping_id(mapping_id),
      executor_factory(ExecutorFactory<KernelDesc>::get_factory(executor_id)),
      packer_factory(MicroKernelPackerFactory<Scalar>::get_factory(executor_id))
  {
    ERROR_AND_EXIT_IF(!executor_factory,
      "Executor factory not found: " << executor_id <<
      " for kernel desc: " << type_name<KernelDesc>() <<
      ", Registered factories: " << ExecutorFactory<KernelDesc>::dump_registered_factories());
    ERROR_AND_EXIT_IF(!packer_factory, "Packer factory not found");
    ERROR_AND_EXIT_IF(packer_factory->M_r != executor_factory->M_r,
                      "M_r mismatch between packer and executor");

    std::string filepath(__FILE__);
    auto end_of_path = filepath.find_last_of('/');
    filepath = filepath.substr(0, end_of_path + 1);

    auto nanokernel_mapping = read_pattern_mapping(mapping_id,
       {"mappings/", filepath + "../mappings/"});
    packer = packer_factory->create_specialized_packer(nanokernel_mapping);

    M_r = executor_factory->M_r;
    N_r = executor_factory->N_r;

    if (schedule == C1_NmKM) {
        // Multiple of N_r and larger than or equal to N
        config.N_c = ((b_col_predict + N_r - 1) / N_r) * N_r;
    } else if (schedule == C1_MKN) {
        config.M_c = m;
    } else if (config.tiling_strategy == CAKE_TILING ||
               config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {
      cake_cntx_t* cake_cntx = cake_query_cntx();

      cake_cntx->nr = executor_factory->N_r;
      cake_cntx->mr = executor_factory->M_r;
      cake_cntx->ncores = num_threads;

      cache_dims_t* cache_dims = get_cache_dims_4(
          m, b_col_predict, k, num_threads, cake_cntx, KMN,
          nullptr, true,
          double(coo->nnz()) / (m * k),
          config.beta,
          true, true);

//      cache_dims_t* cache_dims = get_cache_dims_3(
//          m, b_col_predict, k, num_threads, cake_cntx, KMN,
//          nullptr, double(coo->nnz()) / (m * k), true, true);

      ERROR_AND_EXIT_IF(!cache_dims->m_c || !cache_dims->k_c || !cache_dims->n_c,
                        "Invalid cache dimensions");

      config.M_c = cache_dims->m_c;
      config.K_c = cache_dims->k_c;
      config.N_c = cache_dims->n_c;

      if (config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {
        if (config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {

          int BC_size_bytes = (b_col_predict * config.K_c) * sizeof(Scalar);
          int tlb_entries_used =  BC_size_bytes / config.tlb_page_size;

          if (tlb_entries_used > config.max_tlb_entries) {
            int target_size_bytes = (config.max_tlb_entries * config.tlb_page_size) ;
            int new_k_tile = target_size_bytes / (b_col_predict * sizeof(Scalar));
            config.K_c = new_k_tile;
          }
        }
      }

      free(cake_cntx);
      free(cache_dims);
    }

    inspect_and_pack();
    delete coo;
  }

  MatMul(
      int m, int k, int b_col_predict,
      const Scalar* values,
      const int*    row_offsets,
      const int*    column_indices,
      TileConfig    config_,
      int           num_threads,
      std::string   executor_id,
      std::string   mapping_id
  ): MatMul(new COO<Scalar>(m, k, row_offsets, column_indices, values),
            b_col_predict, config_, num_threads, executor_id, mapping_id) {
  }

  TileConfig get_config() const {
    //std::cout << "Config: " << config.m_tile << " " << config.k_tile << " " << config.n_tile << std::endl;
    return config;
  }

  ~MatMul() {
    if (executor) delete executor;
    free(linear_buffer);
  }

  // This operator overloading enables calling
  // operator function () on objects of increment
  void operator()(Scalar* C, const Scalar* B,
                  const Scalar* bias = nullptr,
                  enum Activation activation = NONE,
                  const Scalar min = std::numeric_limits<Scalar>::min(),
                  const Scalar max = std::numeric_limits<Scalar>::max()) const {
    if (!executor) { ERROR_AND_EXIT("Executor not initialized"); }
    (*executor)(C, B, bias, activation, min, max);
  }

  // This operator overloading enables calling
  // operator function () on objects of increment
  void allocate_executor(int b_cols) {
      if (executor)  delete executor;
      executor = executor_factory->create_specialized_executor(
          m, k, b_cols, 1,
          packed_tiles, upanel_swizzle,
          num_threads, config
      );
  }


private:

  double cost(int uti, int utj) {
    IntRange rows = { uti * M_r, std::min((uti + 1) * M_r, m) };
    IntRange cols = { utj * config.K_c, std::min((utj + 1) * config.K_c, k) };
    SubmatrixLoc upanel_loc = { rows, cols };

    // Just use nnz count for now, could be replaced with proper nanokernel
    //   cost model (post packing)
    return coo->submatrix_nnz_count(upanel_loc);
  }

  void reorder_upanels() {
    if (KernelDesc::UPanelOrder == LOAD_BALANCING) {
      int upanels_per_M_c = config.M_c / M_r;
      int num_upanels = upanels_per_M_c * matrix_tiled_shape.rows;
      upanel_swizzle.resize(num_upanels);

      vector<double> panel_costs(num_upanels);
      for (int uti = 0; uti < num_upanels; uti++) {
        for (int utj = 0; utj < matrix_tiled_shape.cols; utj++) {
          panel_costs[uti] += cost(uti, utj);
        }
      }

      auto upanels_sorted_by_cost = argsort(panel_costs);

      // Alternate from scheduling most expensive panels to cheapest panels
      //   in an attempt load balance
      int curr_idx_from_top = 0;
      int curr_idx_from_bot = upanels_sorted_by_cost.size() - 1;
      int curr_Mb_offset = 0;
      int Mb = m / config.M_c;

      while (curr_idx_from_top <= curr_idx_from_bot) {
        // Schedule from top
        for (int curr_Mb_tile = 0; curr_Mb_tile < Mb; curr_Mb_tile++,
             curr_idx_from_top++) {

          if (curr_idx_from_top > curr_idx_from_bot) break;
          int offset = curr_Mb_tile * upanels_per_M_c + curr_Mb_offset;
          upanel_swizzle[offset] = upanels_sorted_by_cost[curr_idx_from_top];
        }
        curr_Mb_offset += 1;

        // Schedule from bottom
        for (int curr_Mb_tile = 0; curr_Mb_tile < Mb; curr_Mb_tile++,
             curr_idx_from_bot--) {

          if (curr_idx_from_top > curr_idx_from_bot) break;
          int offset = curr_Mb_tile * upanels_per_M_c + curr_Mb_offset;
          upanel_swizzle[offset] = upanels_sorted_by_cost[curr_idx_from_bot];
        }
        curr_Mb_offset += 1;
      }
    }
  }

  void pack_tiles() {
    bool parallel_tiles_packed = false;

    packed_tiles.resize(matrix_tiled_shape.rows);
    for (auto& panel_packed_tiles : packed_tiles){
      panel_packed_tiles.resize(matrix_tiled_shape.cols);
    }

    ERROR_AND_EXIT_IF(config.M_c % M_r != 0, "M_tile must be a multiple of M_r");
    const int panels_per_tile = config.M_c / M_r;

    #pragma omp parallel for num_threads(16) schedule(dynamic)
    for (int ti = 0; ti < matrix_tiled_shape.rows; ti++) {
      const auto panel_tile_locs = tile_locs.row_panel(ti);
      for (int tj = 0; tj < panel_tile_locs.size(); tj++) {
        auto t_loc = panel_tile_locs[tj].loc;
        packed_tiles[ti][tj].type = SPARSE_SOP;
        packed_tiles[ti][tj].loc = t_loc;
        packed_tiles[ti][tj].shape = t_loc.shape();
        packed_tiles[ti][tj].load_c = tj != 0;
        packed_tiles[ti][tj].free_on_destruction = true;
        packed_tiles[ti][tj].sop.num_panels = panels_per_tile;
        packed_tiles[ti][tj].sop.panel_descs =
            new MicroKernelPackedData[panels_per_tile];

        auto panel_descs = packed_tiles[ti][tj].sop.panel_descs;
        for (int panel_id = 0; panel_id < panels_per_tile; panel_id++) {
          int global_panel_id = ti * panels_per_tile + panel_id;

          if (KernelDesc::UPanelOrder != NO_REORDERING) {
            global_panel_id = upanel_swizzle[global_panel_id];
          }

          SubmatrixLoc panel_loc = t_loc;
          panel_loc.rows.start = global_panel_id * M_r;
          panel_loc.rows.end = (global_panel_id + 1) * M_r;

          packer->pack(panel_descs[panel_id], panel_loc, *coo);
        }
      }
    }
  }

  void pack_linear() {
    int linear_size = 0;
    for (auto& panel : packed_tiles)
      for (auto& tile : panel)
        linear_size += tile.linear_size_in_bytes();

    require_storage = linear_size;
    linear_buffer = aligned_alloc(4096, linear_size);
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
};
};