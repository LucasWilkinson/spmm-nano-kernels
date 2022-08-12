
//
// Created by lwilkinson on 5/25/22.
//

#include <assert.h>
#include <mkl.h>
#include <omp.h>
#include <chrono>

#include "utils/Vec.h"

#include "boost/preprocessor/repetition/repeat.hpp"
#include "vectorclass/vectorclass.h"

#include "SOPKernelDesc.h"
#include "packing.h"

namespace sop {

using std::vector;
using Config = TileConfig;

enum PackingStrategy {
  PREPACK,
  PARTIAL_PACKING,
  NO_PACKING
};

template<enum PackingStrategy _C, enum PackingStrategy _B>
struct PackingDesc {
  const static PackingStrategy C_PACKING_STRATEGY = _C;
  const static PackingStrategy B_PACKING_STRATEGY = _B;
};

template<>
struct PackingDesc<NO_PACKING, NO_PACKING> {
  const static PackingStrategy C_PACKING_STRATEGY = NO_PACKING;
  const static PackingStrategy B_PACKING_STRATEGY = NO_PACKING;
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


template <typename KernelDesc, typename PackingDesc>
struct SOPExecutor {
  using Scalar = typename KernelDesc::Scalar;
  using RegT = typename KernelDesc::RegTile;
  using TileDims = typename KernelDesc::TileDims;
  using VecType = typename KernelDesc::VecType;
  using Executor = typename KernelDesc::Executor;
  using CSRPtr = typename KernelDesc::CSRStorageTypes::Ptr;
  using T = typename KernelDesc::RegTile;
  using _PackedTile = PackedTile<KernelDesc>;

  const static PackingStrategy C_PACKING_STRATEGY
      = PackingDesc::C_PACKING_STRATEGY;
  const static PackingStrategy B_PACKING_STRATEGY
      = PackingDesc::B_PACKING_STRATEGY;

  const vector<vector<_PackedTile>>& tiles;
  const Scalar* __restrict__ B;
  Scalar* __restrict__ C;

  Scalar* __restrict__ C_packed = nullptr;
  Scalar* __restrict__ C_packed_partial_global = nullptr;
  Scalar* __restrict__ B_packed = nullptr;


  int M, K, N;
  int batch_size;
  int num_threads;

  const Config& config;
  const TileDims td;

  int M_c, K_c, N_c;
  static constexpr int M_r = TileDims::M_r;
  static constexpr int N_r = TileDims::N_r;

  const int c_N_r = N_c / N_r;

  bool partial_N_c_loop = false;
  bool partial_N_r_loop = false;
  int final_N_c_loop_N_r_count = 0;
  int final_N_r_loop_rem = 0;
  Executor::Mask final_N_r_rem_mask;

  bool report_packing_time = false;

  SOPExecutor(
      int M, int K, int N,
      const vector<vector<_PackedTile>>& tiles,
      const Scalar* __restrict__ B,
      Scalar* __restrict__ C,
      int batch_size,
      int num_threads,
      const Config& config
  ): M(M), K(K), N(N), tiles(tiles), B(B), C(C), batch_size(batch_size),
        num_threads(num_threads), config(config),
        td(M, K, N, config.m_tile, config.k_tile, config.n_tile, num_threads),
        M_c(td.M_c), K_c(td.K_c), N_c(td.N_c)
  {
    int N_c_rem = (N % N_c);
    int N_r_rem = (N % N_r);

    partial_N_c_loop = (N_c_rem != 0);
    partial_N_r_loop = (N_r_rem != 0);

    final_N_c_loop_N_r_count = N_c_rem / N_r;
    final_N_r_loop_rem = N_r_rem;
    final_N_r_rem_mask = Executor::create_mask(N_r_rem);

    if (C_PACKING_STRATEGY == PREPACK) {
      C_packed = new (std::align_val_t(4096)) Scalar[td.M_padded * td.N_padded]();
    }

    if (C_PACKING_STRATEGY == PARTIAL_PACKING) {
      C_packed_partial_global =
          new (std::align_val_t(4096)) Scalar[td.M_padded * td.N_padded];
    }

    if (B_PACKING_STRATEGY == PREPACK || B_PACKING_STRATEGY == PARTIAL_PACKING) {
      B_packed = new (std::align_val_t(4096)) Scalar[td.K_padded * td.N_padded];
    }
  }

  ~SOPExecutor() {
    // Clean-up!
    if (C_PACKING_STRATEGY == PREPACK) {
      ::operator delete[](C_packed, std::align_val_t(4096));
    }

    if (C_PACKING_STRATEGY == PARTIAL_PACKING) {
      ::operator delete[](C_packed_partial_global, std::align_val_t(4096));
    }

    if (B_PACKING_STRATEGY == PREPACK || B_PACKING_STRATEGY == PARTIAL_PACKING) {
      ::operator delete[](B_packed, std::align_val_t(4096));
    }
  }

  _ai void inner_N_c_loop_full(int iii, int jjj,
                               const _PackedTile& pt,
                               Scalar* __restrict__ C_packed) {
    // M_r loop
    for (int pi = 0; pi < pt.sop.num_panels; pi++) {
      const int o_M_r = pi * M_r * N_c;

      for (int tj = 0; tj < c_N_r; tj++) { // N_r loop
        const int jj = (tj * N_r) + jjj;
        const int o_N_r = (tj * N_r) * M_r;

        if (C_PACKING_STRATEGY == PARTIAL_PACKING) {
          Executor::panel_executor_packed_C_max_acc(
            N,
            pt.sop.panel_descs[pi],
            B + jj,
            C_packed + jjj * M_c + (pi * M_r) * N_c +
                (tj * N_r) * M_r,
            pt.load_c
          );
        } else {
          Executor::panel_executor_max_acc(
            M, K, N,
            pt.sop.panel_descs[pi],
            B + jj,
            C + jj + (pi * M_r + iii) * N,
            pt.load_c
          );
        }
      }
    }
  }


  _ai void inner_N_c_loop_partial(int iii, int jjj,
                                  const _PackedTile& pt,
                                  Scalar* __restrict__ C_packed) {
    // M_r loop
    for (int pi = 0; pi < pt.sop.num_panels; pi++) {
      const int o_M_r = pi * M_r * N_c;

      int tj = 0;
      for (; tj < final_N_c_loop_N_r_count; tj++) { // N_r loop
        const int jj = (tj * N_r) + jjj;
        const int o_N_r = (tj * N_r) * M_r;

        if (C_PACKING_STRATEGY == PARTIAL_PACKING) {
          Executor::panel_executor_packed_C_max_acc(
            N,
            pt.sop.panel_descs[pi],
            B + jj,
            C_packed + jjj * M_c + (pi * M_r) * N_c + (tj * N_r) * M_r,
            pt.load_c
          );
        } else {
          Executor::panel_executor_max_acc(
            M, K, N,
            pt.sop.panel_descs[pi],
            B + jj,
            C + jj + (pi * M_r + iii) * N,
            pt.load_c
          );
        }
      }

      if (partial_N_r_loop) {
        const int jj = (tj * N_r) + jjj;
        const int o_N_r = (tj * N_r) * M_r;

        if (C_PACKING_STRATEGY == PARTIAL_PACKING) {
          Executor::panel_executor_masked_packed_C_max_acc(
            final_N_r_loop_rem,
            N,
            pt.sop.panel_descs[pi],
            B + jj,
            C_packed + jjj * M_c + (pi * M_r) * N_c + (tj * N_r) * M_r,
            final_N_r_rem_mask,
            pt.load_c);
        } else {
          Executor::panel_executor_masked_max_acc(
            M, K, N,
            pt.sop.panel_descs[pi],
            B + jj,
            C + jj + (pi * M_r + iii) * N,
            final_N_r_rem_mask,
            pt.load_c);
        }
      }
    }
  }

  void operator()() {
    using std::min;
    ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);

    if (N % config.n_tile && C_PACKING_STRATEGY == NO_PACKING) {
      std::cerr << "TODO: fix cleanup code " << N;
      std::cerr << " " << config.n_tile << std::endl;
      exit(-1);
    }

    if (C_PACKING_STRATEGY == PREPACK) {
      report_time(report_packing_time, "Packed C",
        pack_C<Scalar, M_r, N_r>,
          C, C_packed, M_c, N_c, M, K, N);
    }

    if (C_PACKING_STRATEGY == PARTIAL_PACKING) {

    }

    int Nb_full = partial_N_c_loop ? Nb - 1 : Nb;

    // TODO: Reimplement B packing
    static_assert(B_PACKING_STRATEGY == NO_PACKING);

    // Parallel M_c's loop
    #pragma omp parallel for schedule(static)
    for (int tii = 0; tii < Mb; tii++) {
      const int iii = tii * M_c;
      Scalar* __restrict__ C_packed_partial =
          C_packed_partial_global + iii * N_padded;

      // K_c loop
      for (int tkk = 0; tkk < Kb; tkk++) {
        const _PackedTile& pt = tiles[tii][tkk];

        int tjj = 0, jjj = 0;
        for (; tjj < Nb_full; tjj++, jjj += N_c) {
          inner_N_c_loop_full(iii, jjj, pt, C_packed_partial);
        }

        if (partial_N_c_loop) {
          inner_N_c_loop_partial(iii, jjj, pt, C_packed_partial);
        }
      }

      if (C_PACKING_STRATEGY == PARTIAL_PACKING) {
        report_time(report_packing_time, "Unpack C",
            unpack_C_partial_M_c<Scalar, TileDims>,
                std::min(M - iii, M_c), C + iii * N, C_packed_partial, td);
      }
    }

    if (C_PACKING_STRATEGY == PREPACK) {
      report_time(report_packing_time, "Unpack C",
          unpack_C<Scalar, M_r, N_r>,
              C, C_packed, M_c, N_c, M, K, N);
    }

    report_packing_time = false;
  }
};

};