//
// Created by lwilkinson on 5/25/22.
//

#pragma once

#include <assert.h>
#include <omp.h>
#include <chrono>
#include <vectorclass.h>

#include "utils/Vec.h"

#include "boost/preprocessor/repetition/repeat.hpp"

#include "Config.h"
#include "KernelDesc.h"
#include "MicroKernelDesc.h"
#include "packing.h"

namespace sop {

using std::vector;

struct Executor {
  Executor() = default;
  virtual ~Executor() = default;
  virtual void execute_row_panel(int tii) = 0;
  virtual void operator()() = 0;
};

template <typename F, typename ... Ts>
void report_time(bool report, const std::string& name, F&& f, Ts&&...args)
{
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start_time, end_time;
  using dur = std::chrono::duration<double>;

  if (report) {
    start_time = std::chrono::high_resolution_clock::now();
    std::forward<F>(f)(std::forward<Ts>(args)...);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout << std::endl << name << ": "
              << dur(end_time - start_time).count() * 1e6 << "us"
              << std::endl;
  } else {
    std::forward<F>(f)(std::forward<Ts>(args)...);
  }
}

template <typename KernelDesc, typename MicroKernelDesc>
struct ExecutorSpecialized: Executor {
  using Scalar = typename KernelDesc::Scalar;
  using CSRPtr = typename KernelDesc::CSRStorageTypes::Ptr;

  using RegTile = typename MicroKernelDesc::RegTile;
  using TileDims = typename MicroKernelDesc::TileDims;
  using VecType = typename MicroKernelDesc::VecType;
  using MicroKernel = typename MicroKernelDesc::MicroKernel;

  static_assert(std::is_same<Scalar, typename MicroKernel::Scalar>::value,
              "Scalar type mismatch");
  using PackedTile = sop::PackedTile<Scalar>;

  const static PackingStrategy C_PACKING = KernelDesc::PackingDesc::C_PACKING;
  const static PackingStrategy B_PACKING = KernelDesc::PackingDesc::B_PACKING;

  const vector<vector<PackedTile>>& tiles;
  const Scalar* __restrict__ B;
  Scalar* __restrict__ C;

  Scalar* __restrict__ C_packed = nullptr;
  Scalar* __restrict__ C_packed_partial_global = nullptr;
  Scalar* __restrict__ B_packed = nullptr;

  int M, K, N;
  int batch_size;
  int num_threads;

  const TileConfig& config;
  const TileDims td;

  int M_c, K_c, N_c;
  static constexpr int M_r = TileDims::M_r;
  static constexpr int N_r = TileDims::N_r;

  const int c_N_r = N_c / N_r;

  bool partial_N_c_loop = false;
  bool partial_N_r_loop = false;
  int final_N_c_loop_N_r_count = 0;
  int final_N_r_loop_rem = 0;
  typename MicroKernel::Mask final_N_r_rem_mask;

  bool report_packing_time = false;

  ExecutorSpecialized(
    int M, int K, int N,
    const vector<vector<PackedTile>>& tiles,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C,
    int batch_size,
    int num_threads,
    const TileConfig& config
  ): M(M), K(K), N(N), tiles(tiles), B(B), C(C), batch_size(batch_size),
        num_threads(num_threads), config(config),
        td(M, K, N, config.m_tile, config.k_tile,
           std::max(config.n_tile, N_r), num_threads),
        M_c(td.M_c), K_c(td.K_c), N_c(td.N_c)
  {
    int N_c_rem = (N % N_c);
    int N_r_rem = (N % N_r);

    partial_N_c_loop = (N_c_rem != 0);
    partial_N_r_loop = (N_r_rem != 0);

    final_N_c_loop_N_r_count = N_c_rem / N_r;
    final_N_r_loop_rem = N_r_rem;
    final_N_r_rem_mask = MicroKernel::precomp_mask(N_r_rem);

    if (C_PACKING == PREPACK) {
      C_packed =
          new (std::align_val_t(4096)) Scalar[td.M_padded * td.N_padded]();
    }

    if (C_PACKING == PARTIAL_PACKING) {
      C_packed_partial_global =
          new (std::align_val_t(4096)) Scalar[td.M_padded * td.N_padded];
    }

    if (B_PACKING == PREPACK || B_PACKING == PARTIAL_PACKING) {
      B_packed =
          new (std::align_val_t(4096)) Scalar[td.K_padded * td.N_padded];
    }
  }

  ~ExecutorSpecialized() {
    // Clean-up!
    if (C_PACKING == PREPACK) {
      ::operator delete[](C_packed, std::align_val_t(4096));
    }

    if (C_PACKING == PARTIAL_PACKING) {
      ::operator delete[](C_packed_partial_global, std::align_val_t(4096));
    }

    if (B_PACKING == PREPACK || B_PACKING == PARTIAL_PACKING) {
      ::operator delete[](B_packed, std::align_val_t(4096));
    }
  }

  /******************************************
   *    Patial C Packed
   ******************************************/

  void _inner_M_c_loop_partial_packed_c(int iii, int jjj,
                               const PackedTile& pt,
                               Scalar* __restrict__ C_packed,
                               const bool partial_final_loop) {

    int _c_N_r = (partial_final_loop) ? final_N_c_loop_N_r_count : c_N_r;

    // M_r loop
    for (int pi = 0; pi < pt.sop.num_panels; pi++) {
      int tj = 0, _jj = 0;
      const auto& panel_desc = pt.sop.panel_descs[pi];

      uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
      float* __restrict__     values = panel_desc.values;
      int* __restrict__       pattern_counts = panel_desc.nkern_counts;
      int                     num_col_indices = panel_desc.num_col_indices;

      for (; tj < _c_N_r; tj++, _jj += N_r) { // N_r loop
        MicroKernel::_microkernel_packed_C_max_acc(
          M, K, N,
          pattern_counts, col_indices, values, num_col_indices,
          B + jjj + _jj,
          C_packed + jjj * M_c + (pi * M_r) * N_c + _jj * M_r,
          pt.load_c
        );
      }

      if (partial_final_loop && partial_N_r_loop) {
        MicroKernel::_microkernel_masked_packed_C_max_acc(
          final_N_r_loop_rem,
          M, K, N,
          pattern_counts, col_indices, values, num_col_indices,
          B + jjj + _jj,
          C_packed + jjj * M_c + (pi * M_r) * N_c + (tj * N_r) * M_r,
          final_N_r_rem_mask,
          pt.load_c
        );
      }
    }
  }

  void _execute_row_panel_partial_packed_C_KN(int tii) {
    using std::min;

    ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);
    int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;
    const int iii = tii * M_c;

    Scalar* __restrict__ C_packed_partial =
        C_packed_partial_global + iii * N_padded;

     // K_c loop
    int tjj = 0, jjj = 0;
    for (; tjj < Nb_full; tjj++, jjj += N_c) {
      for (int tkk = 0; tkk < Kb; tkk++) {
        _inner_M_c_loop_partial_packed_c(
            iii, jjj, tiles[tii][tkk], C_packed_partial, false);
      }
    }

    if (partial_N_c_loop || partial_N_r_loop) {
      for (int tkk = 0; tkk < Kb; tkk++) {
        _inner_M_c_loop_partial_packed_c(
            iii, jjj, tiles[tii][tkk], C_packed_partial, true);
      }
    }

    //report_time(report_packing_time, "Unpack C",
    unpack_C_partial_M_c<Scalar, TileDims>(
        min(M - iii, M_c), C + iii * N, C_packed_partial, td);
  }


  void _execute_row_panel_partial_packed_C_NK(int tii) {
    using std::min;

    ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);
    int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;
    const int iii = tii * M_c;

    Scalar* __restrict__ C_packed_partial =
        C_packed_partial_global + iii * N_padded;

    // K_c loop
    for (int tkk = 0; tkk < Kb; tkk++) {
      int tjj = 0, jjj = 0;
      for (; tjj < Nb_full; tjj++, jjj += N_c) {
          _inner_M_c_loop_partial_packed_c(
              iii, jjj, tiles[tii][tkk], C_packed_partial, false);
      }

      if (partial_N_c_loop || partial_N_r_loop) {
        _inner_M_c_loop_partial_packed_c(
            iii, jjj, tiles[tii][tkk], C_packed_partial, true);
      }
    }

    //report_time(report_packing_time, "Unpack C",
    unpack_C_partial_M_c<Scalar, TileDims>(
        min(M - iii, M_c), C + iii * N, C_packed_partial, td);
  }

  /******************************************
   *    Not Packed
   ******************************************/

  void _inner_M_c_loop(int iii, int jjj,
                      const PackedTile& pt,
                      const bool partial_final_loop) {

    int _c_N_r = (partial_final_loop) ? final_N_c_loop_N_r_count : c_N_r;

    // M_r loop
    for (int pi = 0; pi < pt.sop.num_panels; pi++) {
      int tj = 0, jj = jjj;
      const auto& panel_desc = pt.sop.panel_descs[pi];

      uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
      float* __restrict__     values = panel_desc.values;
      int* __restrict__       pattern_counts = panel_desc.nkern_counts;
      int                     num_col_indices = panel_desc.num_col_indices;

      for (; tj < _c_N_r; tj++, jj += N_r) { // N_r loop
        MicroKernel::_microkernel_max_acc(
          M, K, N,
          pattern_counts, col_indices, values, num_col_indices,
          B + jj,
          C + jj + (pi * M_r + iii) * N,
          pt.load_c
        );
      }

      if (partial_final_loop && partial_N_r_loop) {
        MicroKernel::_microkernel_masked_max_acc(
          final_N_r_loop_rem,
          M, K, N,
          pattern_counts, col_indices, values, num_col_indices,
          B + jj,
          C + jj + (pi * M_r + iii) * N,
          final_N_r_rem_mask,
          pt.load_c
        );
      }
    }
  }

  void _execute_row_panel_NK(int tii) {
    using std::min;

    ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);
    int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;
    const int iii = tii * M_c;

    // K_c loop
    for (int tkk = 0; tkk < Kb; tkk++) {
      int tjj = 0, jjj = 0;
      for (; tjj < Nb_full; tjj++, jjj += N_c) {
        _inner_M_c_loop(iii, jjj, tiles[tii][tkk], false);
      }

      if (partial_N_c_loop || partial_N_r_loop) {
        _inner_M_c_loop(iii, jjj, tiles[tii][tkk], true);
      }
    }
  }

  void _execute_row_panel_KN(int tii) {
    using std::min;

    ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);
    int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;
    const int iii = tii * M_c;

    // K_c loop
    int tjj = 0, jjj = 0;
    for (; tjj < Nb_full; tjj++, jjj += N_c) {
      for (int tkk = 0; tkk < Kb; tkk++) {
        _inner_M_c_loop(iii, jjj, tiles[tii][tkk], false);
      }
    }

    if (partial_N_c_loop || partial_N_r_loop) {
      for (int tkk = 0; tkk < Kb; tkk++) {
        _inner_M_c_loop(iii, jjj, tiles[tii][tkk], true);
      }
    }
  }

  /******************************************
   *    Outer Loop
   ******************************************/

  void execute_row_panel(int tii) {
    using std::min;

    if (C_PACKING == PARTIAL_PACKING) {
      if (KernelDesc::Sched == KNM) {
        _execute_row_panel_partial_packed_C_KN(tii);
      } else {
        _execute_row_panel_partial_packed_C_NK(tii);
      }
    } else {
      if (KernelDesc::Sched == KNM) {
        _execute_row_panel_KN(tii);
      } else {
        _execute_row_panel_NK(tii);
      }
    }
  }

  void operator()() {
//    if (N % config.n_tile && C_PACKING == NO_PACKING) {
//      std::cerr << "TODO: fix cleanup code " << N;
//      std::cerr << " " << config.n_tile << std::endl;
//      exit(-1);
//    }

    // TODO: Reimplement B packing
    static_assert(B_PACKING == NO_PACKING);

    #pragma omp parallel for schedule(static)
    for (int tii = 0; tii < td.Mb; tii++) {
      execute_row_panel(tii);
    }

    report_packing_time = false;
  }
};

};