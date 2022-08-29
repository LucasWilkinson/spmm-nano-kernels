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

  enum DenseTileMergingStrategy dense_merging_strategy = ROW_BASED;
  enum SparseTileMergingStrategy sparse_merging_strategy = ALL_CSR;
  enum ExecutionStrategy execution_strategy = UNTILED_MKL;

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
  vector<MergedTile> merged_tiles; // Pre-scheduling

  vector<vector<int>> row_panels_per_thread;
  vector<vector<MergedTile>> tiles_to_pack; // Post-scheduling, Pre-Packing
  vector<vector<_PackedTile>> packed_tiles;

  vector<int> row_swizzle;
  vector<int> panel_swizzle;

  const int* row_reordering;

  COO<Scalar>* coo = nullptr;
  void* linear_buffer = nullptr;

  void classify_and_merge_tiles() {
    // Merge tiles and mark which ones should be processed sparse or
    switch (dense_merging_strategy) {
      case ALL_SPARSE:
        dense_merge_all_sparse();
        break;
      default: {
        std::cout << "Unsupported dense merge" << std::endl;
        exit(-1);
      }
    }

    switch (sparse_merging_strategy) {
      case ALL_CSR:
        break;
      case ALL_SOP:
        sparse_merge_all_sop();
        break;
      default: {
        std::cout << "Unsupported sparse merge" << std::endl;
        exit(-1);
      }
    }
  }

  void inspect_and_pack() {
    free(linear_buffer);
    linear_buffer = nullptr;
    stats = Stats();

    merged_tiles.resize(0);

    matrix_shape = { coo->rows(), coo->cols() };
    tile_shape = {config.m_tile, config.k_tile};

    tile_locs = TileLocs(tile_shape, matrix_shape, TileLocs::COL_FIRST);
    matrix_tiled_shape = {tile_locs.num_i_tiles(), tile_locs.num_j_tiles()};

    merged_tiles.reserve(matrix_tiled_shape.area());

    build_tile_density_matrix();
    classify_and_merge_tiles();
    schedule_tiles();
    pack_tiles();
    pack_linear();
  }

 public:
  MatMul(
      int m, int k,
      int b_col_predict,
      const Scalar* values,
      const int* row_offsets,
      const int* column_indices,
      TileConfig _config,
      int num_threads,
      std::string executor_id,
      std::string mapping_id,
      enum DenseTileMergingStrategy dense_merging_strategy = ALL_SPARSE,
      enum SparseTileMergingStrategy sparse_merging_strategy = ALL_SOP,
      enum ExecutionStrategy execution_strategy = TILED_SPARSE,
      const int* row_reordering = nullptr)
      : m(m), k(k), config(_config),
        num_threads(num_threads),
        executor_id(executor_id),
        executor_factory(ExecutorFactory<KernelDesc>::get_factory(executor_id)),
        packer_factory(MicroKernelPackerFactory<Scalar>::get_factory(executor_id)),
        mapping_id(mapping_id),
        dense_merging_strategy(dense_merging_strategy),
        sparse_merging_strategy(sparse_merging_strategy),
        execution_strategy(execution_strategy),
        row_reordering(row_reordering) {
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

    if (config.tiling_strategy == CAKE_TILING
        || config.tiling_strategy == CAKE_TILING_WITH_TLB_COMPENSATION) {
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
  SubmatrixLoc merge_locs(const std::vector<SubmatrixLoc>& tiles) {
    return std::accumulate(
        tiles.begin(),
        tiles.end(),
        tiles[0],
        [](const SubmatrixLoc& loc1, const SubmatrixLoc& loc2) -> SubmatrixLoc {
          return {
              {std::min(loc1.rows.start, loc2.rows.start),
               std::max(loc1.rows.end, loc2.rows.end)},
              {std::min(loc1.cols.start, loc2.cols.start),
               std::max(loc1.cols.end, loc2.cols.end)}};
        });
  }

  void build_tile_density_matrix() {
    nnz_in_tile = DenseMatrix<int>(matrix_tiled_shape);
    tile_densities = DenseMatrix<double>(matrix_tiled_shape);
    tile_type = DenseMatrix<TileType>(matrix_tiled_shape);

    for (const auto& t_loc : tile_locs) {
      assert(
          (t_loc.ti * matrix_tiled_shape.cols) + t_loc.tj <
          matrix_tiled_shape.area());
      int ti = t_loc.ti, tj = t_loc.tj;

      nnz_in_tile.at(ti, tj) = coo->submatrix_nnz_count(t_loc.loc);
      tile_densities.at(ti, tj) =
          double(nnz_in_tile.at(ti, tj)) / double(tile_shape.area());
    }
  }

  void update_tile_type_matrix() {
    for (const auto& merged_tile : merged_tiles) {
      for (int ti = merged_tile.ti.start; ti < merged_tile.ti.end; ti++) {
        for (int tj = merged_tile.tj.start; tj < merged_tile.tj.end; tj++) {
          tile_type.at(ti, tj) = merged_tile.type;
        }
      }
    }
  }

  void dense_merge_all_sparse() {
    for (int ti = 0; ti < matrix_tiled_shape.rows; ti++) {
      for (int tj = 0; tj < matrix_tiled_shape.cols; tj++) {
        merged_tiles.push_back({SPARSE_CSR, {ti, ti + 1}, {tj, tj + 1}});
      }
    }
  }

  void sparse_merge_all_sop() {
    for (auto& merged_tile : merged_tiles) {
      if (merged_tile.type == SPARSE_CSR) {
        merged_tile.type = SPARSE_SOP;
      }
    }
  }

  void schedule_tiles() {
    // Group by row panels
    vector<double> cost_per_panel(matrix_tiled_shape.rows);
    vector<vector<int>> panel_indices(matrix_tiled_shape.rows);

    for (int t = 0; t < merged_tiles.size(); t++) {
      auto& tile = merged_tiles[t];

      if (tile.ti.size() != 1) {
        std::cerr << "For the packed implmentation we currently do not "
                     "support merged row panels";
        exit(-1);
      }

      // If the tile belongs to a single row panel,
      //    i.e. does not overlap multiple
      if (tile.ti.size() != 1) {
        std::cerr << "For the packed implmentation we currently do not "
                     "support tiles merged across row panels";
        exit(-1);
      }

      int row_panel_id = tile.ti.start;
      panel_indices[row_panel_id].push_back(t);

      // TODO: update cost model
      switch (tile.type) {
        case SPARSE_CSR:
          cost_per_panel[row_panel_id] +=
              2.0 * nnz_in_tile.at(tile.ti.start, tile.tj.start);
          break;
        case SPARSE_SOP:
          cost_per_panel[row_panel_id] +=
              1.5 * nnz_in_tile.at(tile.ti.start, tile.tj.start);
          break;
        case DENSE:
          cost_per_panel[row_panel_id] +=
              1.0 * tile.num_tiles() * tile_shape.area();
          break;
        default:
          std::cerr << "No scheduling cost model for tile type " << tile.type
                    << std::endl;
          exit(-1);
      }
    }

    panel_swizzle.resize(num_threads);

    // Sort by cost per panel
    auto sorted_panels = argsort(cost_per_panel);

    // Alternate from scheduling most expensive panels to cheapest panels in an attempt load balance
    int curr_idx_from_top = 0, curr_idx_from_bot = sorted_panels.size() - 1;

    // Roundrobin scheduling, alternating scheduling from top and bottom
    while (curr_idx_from_top <= curr_idx_from_bot) {
      // Schedule from top
      for (int thrd_id = 0; thrd_id < num_threads; thrd_id++, curr_idx_from_top++) {
        if (curr_idx_from_top > curr_idx_from_bot) break;
        panel_swizzle[thrd_id] = curr_idx_from_top;
      }

      // Schedule from bottom
      for (int thrd_id = 0; thrd_id < num_threads; thrd_id++, curr_idx_from_bot--) {
        if (curr_idx_from_top > curr_idx_from_bot) break;
        panel_swizzle[thrd_id] = curr_idx_from_bot;
      }
    }

//    std::cout << "Panel schedule: " << std::endl;
//    for (const auto& cost : cost_per_panel) {
//      std::cout << cost << std::endl;
//    }

    for (auto& _panel_indices : panel_indices) {
      auto& first_tile = merged_tiles[_panel_indices[0]];

      // If this is the first tile in the rowpanel then mark that we should not
      //   load_c (i.e. zero c)
      first_tile.load_c = true;
    }

    tiles_to_pack.resize(panel_indices.size());
    for (int i = 0; i < panel_indices.size(); i++) {
      tiles_to_pack[i].resize(0);
      tiles_to_pack[i].reserve(panel_indices[i].size());

      bool first_tile = true;
      for (auto& idx : panel_indices[i]) {
        tiles_to_pack[i].push_back(merged_tiles[idx]);
        tiles_to_pack[i].back().load_c = !first_tile;
        first_tile = false;
      }
    }

    //        // Alternate from scheduling most expensive panels to cheapest panels in an attempt load balance int curr_idx_from_top = 0, curr_idx_from_bot = sorted_panels.size() - 1;
    //
    //        // Roundrobin scheduling, alternating scheduling from top and bottom while (curr_idx_from_top <= curr_idx_from_bot) {
    //          // Schedule from top
    //          for (int thrd_id = 0; thrd_id < task.nThreads; thrd_id++, curr_idx_from_top++) {
    //            if (curr_idx_from_top > curr_idx_from_bot) break;
    //
    //            for (auto& tile_idx : panel_indices[curr_idx_from_top])
    //              par_tiles_to_pack[thrd_id].push_back(merged_tiles[tile_idx]);
    //          }
    //
    //          // Schedule from bottom
    //          for (int thrd_id = 0; thrd_id < task.nThreads; thrd_id++, curr_idx_from_bot--) {
    //            if (curr_idx_from_top > curr_idx_from_bot) break;
    //
    //            for (auto& tile_idx : panel_indices[curr_idx_from_bot])
    //              par_tiles_to_pack[thrd_id].push_back(merged_tiles[tile_idx]);
    //          }
    //        }
  }

  void pack_tiles() {
    bool parallel_tiles_packed = false;

    // Only used when `execution_strategy == UNTILED_MKL`
    COO<Scalar> coo_sparse_part(coo->rows(), coo->cols());

    auto pack_tile = [this, &coo_sparse_part](
                         const MergedTile& merged_tile,
                         _PackedTile& tile_to_pack) {
      auto t_type = merged_tile.type;
      bool tile_is_sparse =
          (t_type == SPARSE_CSR ||
           t_type == SPARSE_SOP ||
           t_type == SPARSE_MKL);

      // if we use a full decomposition execution strategy don't pack the tile
      //   just merge it into the larger sparse part
      if (execution_strategy == UNTILED_MKL && tile_is_sparse) {
        std::cerr << "UNTILED_MKL unsupported" << std::endl;
        exit(-1);
      }

      switch (t_type) {
        case SPARSE_CSR: {
          _pack_csr(merged_tile, tile_to_pack);
          break;
        }

        case SPARSE_SOP: {
          _pack_sop(merged_tile, tile_to_pack);
          break;
        }

        case DENSE: {
          _pack_dense(merged_tile, tile_to_pack);
          break;
        }

        default:
          std::cerr << "Unsupported tile type" << std::endl;
          exit(-1);
      };
      return true;
    };

    packed_tiles.resize(tiles_to_pack.size());

    // Pack Parallel Tiles
    #pragma parallel for num_threads(16) collapse(2)
    for (int i = 0; i < tiles_to_pack.size(); i++) {
      packed_tiles[i].resize(tiles_to_pack[i].size());
      for (int j = 0; j < tiles_to_pack[i].size(); j++) {
        pack_tile(tiles_to_pack[i][j], packed_tiles[i][j]);
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

  void _pack_csr(const MergedTile& merged_tile, _PackedTile& tile_to_pack) {
    auto t_loc = _tile_loc(merged_tile);

    auto tile = coo->submatrix_extract(t_loc);
    int nnz_in_tile = tile.nnz();

    tile_to_pack.type = SPARSE_CSR;
    tile_to_pack.loc = t_loc;
    tile_to_pack.shape = t_loc.shape();
    tile_to_pack.load_c = merged_tile.load_c;
    tile_to_pack.free_on_destruction = true;

    tile_to_pack.csr.values = new Scalar[nnz_in_tile];
    tile_to_pack.csr.ptrs = new CSRPtr[tile.rows() + 1];
    tile_to_pack.csr.inds = new CSRIndex[nnz_in_tile];

    nnz_in_tile = tile.populate_csr(
        tile_to_pack.csr.values,
        tile_to_pack.csr.ptrs,
        tile_to_pack.csr.inds,
        COUNT);

    if (nnz_in_tile < 0) {
      std::cerr << "Failed to pack CSR" << std::endl;
      exit(-1);
    }

    stats.csr_tiles_count++;
    stats.csr_tiles_nnz_count += nnz_in_tile;
  }

  void _pack_sop(const MergedTile& merged_tile, _PackedTile& tile_to_pack) {
    auto t_loc = _tile_loc(merged_tile);

    tile_to_pack.type = SPARSE_SOP;
    tile_to_pack.loc = t_loc;
    tile_to_pack.shape = t_loc.shape();
    tile_to_pack.load_c = merged_tile.load_c;
    tile_to_pack.free_on_destruction = true;

    Tile<Scalar> sop_tile(*coo, t_loc, packer);

    tile_to_pack.sop.num_panels = sop_tile.num_panels();
    tile_to_pack.sop.panel_descs = new MicroKernelPackedData[sop_tile.num_panels()];
    sop_tile.pack(tile_to_pack.sop.panel_descs);

    stats.sop_tiles_count++;
//    stats.sop_tiles_nnz_count += tile.nnz();
//    stats.sop_tiles_padding += sop_tile.num_padded_nnz();
  }

  void _pack_dense(const MergedTile& merged_tile, _PackedTile& tile_to_pack) {
    auto t_loc = _tile_loc(merged_tile);

    tile_to_pack.type = DENSE;
    tile_to_pack.loc = t_loc;
    tile_to_pack.shape = t_loc.shape();
    tile_to_pack.load_c = merged_tile.load_c;
    tile_to_pack.free_on_destruction = true;

    tile_to_pack.dense.values = new Scalar[t_loc.shape().area()];
    int nnz_in_tile =
        coo->submatrix_extract(t_loc).populate_dense(tile_to_pack.dense.values);

    stats.dense_tiles_count++;
    stats.dense_tiles_nnz_count += nnz_in_tile;
    stats.dense_tiles_padding += t_loc.shape().area() - nnz_in_tile;
  }
};
};