//
// Created by lwilkinson on 5/25/22.
//

#pragma once

#include <assert.h>
#include <omp.h>
#include <chrono>
#include <numeric>
//#include <aligned_new>
#include <atomic>

#include "utils/Vec.h"
#include "utils/error.h"

#include "Config.h"
#include "KernelDesc.h"
#include "MicroKernelDesc.h"
#include "ExecutorPacker.h"
#include "packing.h"

namespace sop {

    struct SpinBarrier {
        std::atomic<int> bar = 0; // Counter of threads, faced barrier.
        std::atomic<int> passed = 0; // Number of barriers, passed by all threads.

        void barrier_wait(int P)
        {
          int passed_old = passed.load(std::memory_order_relaxed);

          if(bar.fetch_add(1) == (P - 1))
          {
            // The last thread, faced barrier.
            bar = 0;
            // Synchronize and store in one operation.
            passed.store(passed_old + 1, std::memory_order_release);
          }
          else
          {
            // Not the last thread. Wait others.
            while(passed.load(std::memory_order_relaxed) == passed_old) {};
            // Need to synchronize cache with other threads, passed barrier.
            std::atomic_thread_fence(std::memory_order_acquire);
          }
        }

        void reset() {
          bar = 0;
          passed = 0;
        }
    };




    using std::vector;

    template<typename Scalar>
    struct Executor {
        Executor() = default;
        virtual ~Executor() = default;
        virtual void operator()(Scalar* __restrict__ _C, const Scalar* __restrict__ _B,
                                const Scalar* _bias,
                                enum Activation activation = NONE,
                                const Scalar min = std::numeric_limits<Scalar>::min(),
                                const Scalar max = std::numeric_limits<Scalar>::max()) = 0;

        virtual void begin_threaded(Scalar* __restrict__ _C, const Scalar* __restrict__ _B,
                                    const Scalar* _bias,
                                    enum Activation activation = NONE,
                                    const Scalar min = std::numeric_limits<Scalar>::min(),
                                    const Scalar max = std::numeric_limits<Scalar>::max()) = 0;
        virtual void execute_thread(int p_tile, int thread_id) = 0;
        virtual int num_parallel_tile() const = 0;

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

    template <typename KernelDesc, typename MicroKernelDesc, bool DataTransform>
    class ExecutorSpecialized: public Executor<typename KernelDesc::Scalar> {
        using Scalar = typename KernelDesc::Scalar;

        using RegTile = typename MicroKernelDesc::RegTile;
        using TileDims = typename MicroKernelDesc::TileDims;
        using VecType = typename MicroKernelDesc::VecType;
        using MicroKernel = typename MicroKernelDesc::MicroKernel;

        static_assert(std::is_same<Scalar, typename MicroKernel::Scalar>::value,
                      "Scalar type mismatch");
        using PackedTile = sop::PackedTile<Scalar>;

        const static PackingStrategy C_PACKING = KernelDesc::PackingDesc::C_PACKING;
        const static PackingStrategy B_PACKING = KernelDesc::PackingDesc::B_PACKING;
        const static UPanelReorderingStrategy UPanelOrder = KernelDesc::UPanelOrder;

        const vector<vector<PackedTile>>& tiles;
        const vector<int>& upanel_swizzle;

        const Scalar* __restrict__ B;
        Scalar* __restrict__ C = nullptr;
        const Scalar* __restrict__ bias = nullptr;

        Packer<KernelDesc, MicroKernelDesc>* packer = nullptr;

        int M, K, N;
        int batch_size;
        int num_threads;

        SpinBarrier barrier;
        bool first_run = true;

        TileConfig& config;
        const TileDims td;

        int M_c, K_c, N_c;
        static constexpr int M_r = TileDims::M_r;
        static constexpr int N_r = TileDims::N_r;

        const int c_N_r = N_c / N_r;
        const int c_M_r = M_c / M_r;

        bool partial_N_c_loop = false;
        bool partial_N_r_loop = false;

        int final_N_c_size = 0;
        int final_N_c_loop_N_r_count = 0;
        int final_N_r_loop_rem = 0;
        typename MicroKernel::Mask final_N_r_rem_mask;

        MicroKernel ukernel;

        bool report_packing_time = false;

    public:
        ExecutorSpecialized(
            int M, int K, int N, int batch_size,
            const vector<vector<PackedTile>>& tiles,
            const vector<int>& upanel_swizzle,
            int num_threads,
            TileConfig& config
        ): M(M), K(K), N(N),
           tiles(tiles),
           upanel_swizzle(upanel_swizzle),
           batch_size(batch_size),
           num_threads(num_threads),
           config(config),
           td(M, K, N, config.M_c, config.K_c,
              std::max(config.N_c, N_r), num_threads),
           M_c(td.M_c), K_c(td.K_c), N_c(td.N_c)
        {
            int N_c_rem = (N % N_c);
            int N_r_rem = (N % N_r);

            partial_N_c_loop = (N_c_rem != 0);
            partial_N_r_loop = (N_r_rem != 0);

            final_N_c_size = N_c_rem;
            final_N_c_loop_N_r_count = N_c_rem / N_r;
            final_N_r_loop_rem = N_r_rem;
            final_N_r_rem_mask = MicroKernel::precomp_mask(N_r_rem);

            if (C_PACKING != NO_PACKING || B_PACKING != NO_PACKING) {
                packer = new Packer<KernelDesc, MicroKernelDesc>(M, K, N, M_c, K_c, N_c, num_threads);
                if (C_PACKING != NO_PACKING) packer->allocate_C_buffers();
                if (B_PACKING != NO_PACKING) packer->allocate_B_buffers();
            }
        }

        ~ExecutorSpecialized() {
            if (packer) delete packer;
        }

        template<bool packed_C, bool packed_B>
        inline __attribute__((__always_inline__)) void _inner_nm_loop(
                Scalar* __restrict__ C_p,
                const Scalar* __restrict__ B_p,
                int tii, int jjj,
                const PackedTile& pt,
                const bool partial_Nc_loop,
                const bool final_store) {
            int _c_N_r = (partial_Nc_loop) ? final_N_c_loop_N_r_count : c_N_r;
            const Scalar* __restrict__ B_p_base = B_p;
            Scalar* __restrict__ C_p_base = C_p;
            Scalar* __restrict__ C_o = nullptr;

            if (pt.type == EMPTY_TILE) {
                return;
            }

            // M_r loop
            for (int pi = 0; pi < pt.sop.num_panels; pi++) {
                int tj = 0, jj = jjj;
                const auto& panel_desc = pt.sop.panel_descs[pi];

                // Just to be safe for now
                if constexpr(packed_C)
                    C_p = packer->advance_to_row(C_p_base, pi);

                if constexpr(packed_B)
                    B_p = B_p_base;

                uint32_t* __restrict__  col_inds = (uint32_t*) panel_desc.col_indices;
                Scalar* __restrict__     values = panel_desc.values;
                int* __restrict__       nkern_counts = panel_desc.nkern_counts;

                int global_upanel_id = tii * c_M_r + pi;
                if constexpr(UPanelOrder != NO_REORDERING && !packed_C)
                    global_upanel_id = upanel_swizzle[global_upanel_id];
                int ii = (global_upanel_id * M_r); // Row start of ukernel

                if (ii + M_r > M) {
                    std::cerr << "bad ii " << ii << " " << tii * c_M_r + pi << std::endl;
                    exit(-1);
                }

                for (; tj < _c_N_r; tj++, jj += N_r) { // N_r loop
                    if constexpr(packed_C && !packed_B) {
                        B_p = B + jj;
                        C_o = (!final_store) ? C_p : C + jj + ii * N;

                        ukernel.vectorized(
                            C_p, N_r,
                            C_o, (!final_store) ? N_r : N,
                            B_p, N,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );

                        C_p = packer->seek_to_next_reg_tile(C_p);
                    } else if constexpr(!packed_C && packed_B) {
                        B_p = B_p_base + tj * N_r;
                        C_p = C + jj + ii * N;

                        ukernel.vectorized(
                            C_p, N,
                            B_p, N_c,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );
                    } else if constexpr(packed_C && packed_B) {
                        B_p = B_p_base + tj * N_r;
                        C_o = (!final_store) ? C_p : C + jj + ii * N;

                        ukernel.vectorized(
                            C_p, N_r,
                            C_o, (!final_store) ? N_r : N,
                            B_p, N_c,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );

                        //B_p = packer->seek_to_next_B_tile(B_p);
                        C_p = packer->seek_to_next_reg_tile(C_p);
                    } else {
                        C_p = C + jj + ii * N;
                        B_p = B + jj;

                        ukernel.vectorized(
                            C_p, N,
                            B_p, N,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );
                    }
                }

                if (partial_Nc_loop && partial_N_r_loop) {
                    if constexpr(packed_C && !packed_B) {
                        B_p = B + jj;
                        C_o = (!final_store) ? C_p : C + jj + ii * N;

                        ukernel.cleanup(
                            final_N_r_loop_rem,
                            C_p, N_r,
                            C_o, (!final_store) ? N_r : N,
                            B_p, N,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );

                        // We don't know if we've traversed the entire row of B,
                        //    reset from the base pointer for the next panel
                        //C_p = packer->advance_to_row(C_p_base, pi + 1);
                    } else if constexpr(!packed_C && packed_B) {
                        B_p = B_p_base + tj * N_r;
                        C_p = C + jj + ii * N;

                        ukernel.cleanup(
                            final_N_r_loop_rem,
                            C_p, N,
                            B_p, N_c,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );
                    } else if constexpr(packed_C && packed_B) {
                        B_p = B_p_base + tj * N_r;
                        C_o = (!final_store) ? C_p : C + jj + ii * N;

                        ukernel.cleanup(
                            final_N_r_loop_rem,
                            C_p, N_r,
                            C_o, (!final_store) ? N_r : N,
                            B_p, N_c,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );

                        // We don't know if we've traversed the entire row of B,
                        //    reset from the base pointer for the next panel
                        //C_p = packer->advance_to_row(C_p_base, pi + 1);
                    } else {
                        C_p = C + jj + ii * N;
                        B_p = B + jj;

                        ukernel.cleanup(
                            final_N_r_loop_rem,
                            C_p, N,
                            B_p, N,
                            K,
                            nkern_counts, col_inds, values,
                            pt.load_c, final_store,
                            bias ? bias + ii : nullptr
                        );
                    }
                }
            }
        }

        inline __attribute__((__always_inline__)) void _inner_nm_loop(int tii, int jjj, const PackedTile& pt,
                                   const bool partial_Nc_loop,
                                   const bool final_store) {
            _inner_nm_loop<false, false>(nullptr, nullptr, tii, jjj, pt, partial_Nc_loop, final_store);
        }

        /******************************************
         *    Outer Loop
         ******************************************/

        int num_parallel_tile() const {
            if(config.runtimeSchedule == nmKNM || config.runtimeSchedule == nmNKM || config.runtimeSchedule == nmKM ||
                config.runtimeSchedule == nmNM || config.runtimeSchedule == nmM)
                return td.Mb;
            else if(config.runtimeSchedule == nmMNK || config.runtimeSchedule == nmNMK ||
                config.runtimeSchedule == nmMK || config.runtimeSchedule == nmNK || config.runtimeSchedule == nmK)
                return td.Kb;
            else
                return td.Nb;
        }

        void execute_thread(int p_tile, int thread_id) {
//            ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);

            static constexpr bool packed_C = C_PACKING == PACK;
            static constexpr bool packed_B = B_PACKING == PACK;

            if constexpr(packed_C || packed_B) {
                switch (config.runtimeSchedule) {
                    case nmN: {
                        Scalar* __restrict__ C_p = nullptr;
                        Scalar* __restrict__ B_p = nullptr;

                        int tjj = p_tile;
                        bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);

                        if constexpr(packed_B) {
                            B_p = packer->get_B_packed_buffer(tjj);
                            packer->pack_B(B_p, B + tjj * N_c, N_c);
                        }

                        bool final_store = true;
                        _inner_nm_loop<packed_C, packed_B>(
                                C_p, B_p, 0, tjj * N_c, tiles[0][0], partial_Nc_loop, final_store);
                        break;
                    }
                    case nmMN: {
                        Scalar* __restrict__ C_p = nullptr;
                        Scalar* __restrict__ B_p = nullptr;

                        int tjj = p_tile;
                        bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);

                        if constexpr(packed_B) {
                            B_p = packer->get_B_packed_buffer(tjj);
                            packer->pack_B(B_p, B + tjj * N_c, N_c);
                        }

                        bool final_store = true;
                        for (int tii = 0; tii < td.Mb; tii++) {
                            _inner_nm_loop<packed_C, packed_B>(
                                C_p, B_p, tii, tjj * N_c, tiles[tii][0], partial_Nc_loop, final_store);
                        }
                        break;
                    }
                    case nmKNM: {
                        int tii = p_tile;
                        int tjj = 0, jjj = 0;
                        int Nb_full = partial_N_c_loop || partial_N_r_loop ? td.Nb - 1 : td.Nb;

                        Scalar* __restrict__ C_p = nullptr;
                        Scalar* __restrict__ B_p = nullptr;

                        if constexpr(packed_C) C_p = packer->get_C_packed_buffer(thread_id);

                        for (; tjj < Nb_full; tjj++, jjj += N_c) {
                            if constexpr(packed_B) {
                                B_p = packer->get_B_packed_buffer(tjj);

                                if (!packer->is_B_packed(tjj)) {
                                    packer->pack_B(B_p, B + jjj, thread_id, N_c);
                                    barrier.barrier_wait(num_threads);
                                    if (thread_id == 0) packer->mark_B_packed(tjj);
                                }
                            }

                            for (int tkk = 0; tkk < td.Kb; tkk++) {
                                bool final_store = (tkk == td.Kb - 1);
                                bool partial_Nc_loop = false;
                                _inner_nm_loop<packed_C, packed_B>(
                                    C_p, B_p, tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                            }
                        }

                        if (partial_N_c_loop || partial_N_r_loop) {
                            if constexpr(packed_B) {
                                B_p = packer->get_B_packed_buffer(tjj);

                                if (!packer->is_B_packed(tjj)) {
                                    packer->pack_B(B_p, B + jjj, thread_id, N_c);
                                    barrier.barrier_wait(num_threads);
                                    if (thread_id == 0) packer->mark_B_packed(tjj);
                                }
                            }

                            for (int tkk = 0; tkk < td.Kb; tkk++) {
                                bool final_store = (tkk == td.Kb - 1);
                                bool partial_Nc_loop = true;
                                _inner_nm_loop<packed_C, packed_B>(
                                    C_p, B_p, tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                            }
                        }
                        break;
                    }
                    default:
                        ERROR_AND_EXIT("Not implemented");
                }
            } else {
                int tii, tjj, tkk, jjj, Nb_full;
                bool final_store, partial_Nc_loop;

                 switch (config.runtimeSchedule) {
                     case nmKNM: {
                         tii = p_tile;
                         Nb_full = partial_N_c_loop || partial_N_r_loop ? td.Nb - 1 : td.Nb;
                         tjj = 0;
                         jjj = 0;

                         for (; tjj < Nb_full; tjj++, jjj += N_c) {
                             for (tkk = 0; tkk < td.Kb; tkk++) {
                                 final_store = (tkk == td.Kb - 1);
                                 partial_Nc_loop = false;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         if (partial_N_c_loop || partial_N_r_loop) {
                             for (tkk = 0; tkk < td.Kb; tkk++) {
                                 final_store = (tkk == td.Kb - 1);
                                 partial_Nc_loop = true;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         break;
                     }
                     case nmNKM:
                          tii = p_tile;
                          Nb_full = partial_N_c_loop || partial_N_r_loop ? td.Nb - 1 : td.Nb;
                         for (tkk = 0; tkk < td.Kb; tkk++) {
                             final_store = (tkk == td.Kb - 1);
                             tjj = 0; jjj = 0;
                             for (; tjj < Nb_full; tjj++, jjj += N_c) {
                                 partial_Nc_loop = false;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                             if (partial_N_c_loop || partial_N_r_loop) {
                                 partial_Nc_loop = true;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         break;
                     case nmMNK:
                         tkk = p_tile;
                         final_store = (tkk == td.Kb - 1);
                         Nb_full = partial_N_c_loop || partial_N_r_loop ? td.Nb - 1 : td.Nb;
                         tjj = 0; jjj = 0;
                         for (; tjj < Nb_full; tjj++, jjj += N_c) {
                             for (tii = 0; tii < td.Mb; tii++) {
                                 partial_Nc_loop = false;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         if (partial_N_c_loop || partial_N_r_loop) {
                             for (tii = 0; tii < td.Mb; tii++) {
                                 partial_Nc_loop = true;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         break;
                     case nmNMK:
                         tkk = p_tile;
                         final_store = (tkk == td.Kb - 1);
                         Nb_full = partial_N_c_loop || partial_N_r_loop ? td.Nb - 1 : td.Nb;
                         for (tii = 0; tii < td.Mb; tii++) {
                             tjj = 0; jjj = 0;
                             for (; tjj < Nb_full; tjj++, jjj += N_c) {
                                 partial_Nc_loop = false;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                             if (partial_N_c_loop || partial_N_r_loop) {
                                 partial_Nc_loop = true;
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         break;
                     case nmKMN: {
                         int tjj = p_tile;
                         int jjj = tjj * N_c;
                         partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                         for (tii = 0; tii < td.Mb; tii++) {
                             for (tkk = 0; tkk < td.Kb; tkk++) {
                                 final_store = (tkk == td.Kb - 1);
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         break;
                     }
                     case nmMKN:
                         tjj = p_tile;
                         jjj = tjj * N_c;
                         partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                         for (tkk = 0; tkk < td.Kb; tkk++) {
                             final_store = (tkk == td.Kb - 1);
                             for (tii = 0; tii < td.Mb; tii++) {
                                 _inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);
                             }
                         }
                         break;
                     case nmKM:
                         tii = p_tile;
                         for (tkk = 0; tkk < td.Kb; tkk++) {
                             final_store = (tkk == td.Kb - 1);
                             partial_Nc_loop = partial_N_c_loop || partial_N_r_loop; // since Nc == N
                             _inner_nm_loop(tii, 0, tiles[tii][tkk], partial_Nc_loop, final_store);
                         }
                         break;
                     case nmMK:
                         tkk = p_tile;
                         final_store = (tkk == td.Kb - 1);
                         for (tii = 0; tii < td.Mb; tii++) {
                             partial_Nc_loop = partial_N_c_loop || partial_N_r_loop; // since Nc == N
                             _inner_nm_loop(tii, 0, tiles[tii][tkk], partial_Nc_loop, final_store);
                         }
                         break;
                     case nmKN:
                         tjj = p_tile;
                         partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                         for (tkk = 0; tkk < td.Kb; tkk++) {
                             final_store = (tkk == td.Kb - 1);
                             _inner_nm_loop(0, tjj * N_c, tiles[0][tkk], partial_Nc_loop, final_store);
                         }
                         break;
                     case nmNK:
                         tkk = p_tile;
                         final_store = (tkk == td.Kb - 1);
                         for (tjj = 0; tjj < td.Nb; tjj++) {
                             partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                             _inner_nm_loop(0, tjj * N_c, tiles[0][tkk], partial_Nc_loop, final_store);
                         }
                         break;
                     case nmMN:
                         tjj = p_tile;
                         partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                         final_store = true;
                         for (tii = 0; tii < td.Mb; tii++) {
                             _inner_nm_loop(tii, tjj * N_c, tiles[tii][0], partial_Nc_loop, final_store);
                         }
                         break;
                     case nmNM:
                         tii = p_tile;
                         final_store = true;
                         for (tjj = 0; tjj < td.Nb; tjj++) {
                             partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                             _inner_nm_loop(tii, tjj * N_c, tiles[tii][0], partial_Nc_loop, final_store);
                         }
                         break;
                     case nmK:
                         tkk = p_tile;
                         final_store = (tkk == td.Kb - 1);
                         partial_Nc_loop = partial_N_c_loop || partial_N_r_loop;
                         _inner_nm_loop(0, 0, tiles[0][tkk], partial_Nc_loop, final_store);
                         break;
                     case nmN:
                         tjj = p_tile;
                         partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == td.Nb - 1);
                         final_store = true;
                         _inner_nm_loop(0, tjj * N_c, tiles[0][0], partial_Nc_loop, final_store);
                         break;
                     case nmM:
                         tii = p_tile;
                         final_store = true;
                         partial_Nc_loop = partial_N_c_loop || partial_N_r_loop;
                         _inner_nm_loop(tii, 0, tiles[tii][0], partial_Nc_loop, final_store);
                         break;
                     default:
                         ERROR_AND_EXIT("Schedule not valid");
                 }
            }
        }


        void begin_threaded(Scalar* __restrict__ _C, const Scalar* __restrict__ _B,
                            const Scalar* _bias,
                            enum Activation activation = NONE,
                            const Scalar min = std::numeric_limits<Scalar>::min(),
                            const Scalar max = std::numeric_limits<Scalar>::max()) {

            C = _C; B = _B; bias = _bias;
            ukernel = MicroKernel(activation, min, max);

            ERROR_AND_EXIT_IF(M_c % M_r, "M_c " << M_c << " must be a multiple of M_r " << M_r
                                                << " schedule " << config.runtimeSchedule);
            ERROR_AND_EXIT_IF(N_c % N_r, "N_c " << N_c << " must be a multiple of N_r " << N_r
                                                << " schedule " << config.runtimeSchedule);

            if constexpr(B_PACKING != NO_PACKING) packer->reset_B_packed_flags();
            barrier.reset();
        }

        void operator()(Scalar* __restrict__ _C, const Scalar* __restrict__ _B,
                        const Scalar* _bias,
                        enum Activation activation = NONE,
                        const Scalar min = std::numeric_limits<Scalar>::min(),
                        const Scalar max = std::numeric_limits<Scalar>::max()) {
            // TODO: go back and follow _ member naming convention
            begin_threaded(_C, _B, _bias, activation, min, max);

            #pragma omp parallel for schedule(static)
            for (int p = 0; p < num_parallel_tile(); p++) {
#if defined(_OPENMP)
                int thread_id = omp_get_thread_num();
#else
                int thread_id = 0;
#endif
                execute_thread(p, thread_id);
            }


            report_packing_time = false;
        }
    };

};