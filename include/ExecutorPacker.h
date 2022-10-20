//
// Created by lwilkinson on 7/26/22.
//

#pragma once

#include <assert.h>
#include <omp.h>
#include <chrono>
//#include <aligned_new>

#include "utils/Vec.h"
#include "utils/error.h"
#include "utils/bmath.h"

#if defined(__AVX512F__) || defined(__AVX2__)

#include "rte_memcpy.h"
#endif

#include "Enums.h"
#include "Config.h"
#include "KernelDesc.h"
#include "MicroKernelDesc.h"

#define USE_CUSTOM_MEMCPY 1

#define _ai inline __attribute__((__always_inline__))

namespace sop {

static _ai void *
pack_memcpy(void *__restrict__ dst, const void *__restrict__ src, size_t n)
{

#ifdef __AVX512VL__
    void* ret = dst;
    ERROR_AND_EXIT_IF((uintptr_t) dst & 0x3F, "dst is not aligned to 64 bytes");

    #pragma GCC unroll 4
    for (; n >= 64; n -= 64, dst = (uint8_t *)dst + 64, src = (const uint8_t *)src + 64) {
        _mm512_store_si512((void *)(dst), _mm512_loadu_si512((const void *)(src)));
    }

    if (n) {
        __mmask16 mask = (uint16_t) ((1 << (n / 4)) - 1);
        _mm512_store_si512((void *) (dst), _mm512_maskz_loadu_epi32(mask, (const void *) (src)));
    }

    return ret;
#else
    ERROR_AND_EXIT("pack_memcpy is not implemented for this architecture");
#endif
}

template<typename KernelDesc, typename MicroKernelDesc>
struct Packer {

    using RegTile = typename MicroKernelDesc::RegTile;
    using TileDims = typename MicroKernelDesc::TileDims;
    using VecType = typename MicroKernelDesc::VecType;
    using MicroKernel = typename MicroKernelDesc::MicroKernel;

    using Scalar = typename KernelDesc::Scalar;
    static constexpr Schedule Sched = KernelDesc::Sched;

    int M_c, K_c, N_c;
    int M, K, N;
    int K_p, N_p;
    int nThreads;
    int c_N_r;
    int c_N_c;
    int rows_of_B_per_thread;

    static constexpr int M_r = TileDims::M_r;
    static constexpr int N_r = TileDims::N_r;

    Scalar *__restrict__ C_packed = nullptr;
    Scalar *__restrict__ B_packed = nullptr;
    bool* B_packed_flags = nullptr;

    Packer(
        int M, int K, int N,
        int M_c, int K_c, int N_c,
        int nThreads
    ) : M(M), K(K), N(N), M_c(M_c), K_c(K_c), N_c(N_c), nThreads(nThreads) {
        N_p = next_multiple(N, N_c);
        K_p = next_multiple(K, 16);
        c_N_r = N_c / N_r;
        c_N_c = ceil_div(N, N_c);
        rows_of_B_per_thread = ceil_div(K, nThreads);
    }

    ~Packer() {
        if (C_packed) operator delete[](C_packed, std::align_val_t(64));
        if (B_packed) operator delete[](B_packed, std::align_val_t(64));
    }

    void allocate_C_buffers() {
        C_packed = new(std::align_val_t(64)) Scalar[M_c * N_c * nThreads];
    }

    void allocate_B_buffers() {
        B_packed = new(std::align_val_t(64)) Scalar[K_p * N_p];
        B_packed_flags = new bool[c_N_c]();
    }

    _ai void reset_B_packed_flags () {
        for (int i = 0; i < c_N_c; i++) B_packed_flags[i] = false;
    }

    _ai bool is_B_packed(int tjj) {
        return B_packed_flags[tjj];
    }

    _ai void mark_B_packed(int tjj) {
        B_packed_flags[tjj] = true;
    }

    _ai void pack_B(Scalar *__restrict__ B_packed, const Scalar *__restrict__ B, int n) {
        std::chrono::time_point<std::chrono::high_resolution_clock>
                start_time, end_time;
        using dur = std::chrono::duration<double>;
#pragma ivdep
#pragma unroll 4
        for (int k = K - 1; k >= 0; k--) {
            const Scalar *__restrict__ B_row = B + k * N;
            Scalar *__restrict__ B_p_row = (Scalar *) __builtin_assume_aligned(B_packed + k * N_c, 16);

#if __AVX2__ || __AVX512F__
#if defined(USE_CUSTOM_MEMCPY) && USE_CUSTOM_MEMCPY
            pack_memcpy(B_p_row, B_row, n * sizeof(Scalar));
#else
            rte_memcpy_generic(B_p_row, B_row, n * sizeof(Scalar));
#endif
#endif
        }
    }

    _ai void pack_B(Scalar *__restrict__ B_packed, const Scalar *__restrict__ B, int thread_id, int n) {
        int start_row = thread_id * rows_of_B_per_thread;
        int end_row = std::min(start_row + rows_of_B_per_thread, K);

#pragma ivdep
#pragma unroll 4
        for (int k = start_row; k < end_row; k++) {
            const Scalar *__restrict__ B_row = B + k * N;
            Scalar *__restrict__ B_p_row = (Scalar *) __builtin_assume_aligned(B_packed + k * N_c, 16);

#if __AVX2__ || __AVX512F__
#if defined(USE_CUSTOM_MEMCPY) && USE_CUSTOM_MEMCPY
            pack_memcpy(B_p_row, B_row, n * sizeof(Scalar));
#else
            rte_memcpy_generic(B_p_row, B_row, n * sizeof(Scalar));
#endif
#endif
        }
    }


    _ai void pack_B_2(Scalar *__restrict__ B_packed, const Scalar *__restrict__ B, int thread_id, int _n) {
        int start_row = thread_id * rows_of_B_per_thread;
        int end_row = std::min(start_row + rows_of_B_per_thread, K);

        std::chrono::time_point<std::chrono::high_resolution_clock>
                start_time, end_time;
        using dur = std::chrono::duration<double>;

//        if (thread_id == 0) start_time = std::chrono::high_resolution_clock::now();


        for (int k = start_row; k < end_row; k += 4) {
            int n = _n;

            void *__restrict__ B_p_row0 = (void *) __builtin_assume_aligned(B_packed + (k + 0) * N_c, 16);
            void *__restrict__ B_p_row1 = (void *) __builtin_assume_aligned(B_packed + (k + 1) * N_c, 16);
            void *__restrict__ B_p_row2 = (void *) __builtin_assume_aligned(B_packed + (k + 2) * N_c, 16);
            void *__restrict__ B_p_row3 = (void *) __builtin_assume_aligned(B_packed + (k + 3) * N_c, 16);

            const void *__restrict__ B_row0 = (B + (k + 0) * N);
            const void *__restrict__ B_row1 = (B + (k + 1) * N);
            const void *__restrict__ B_row2 = (B + (k + 2) * N);
            const void *__restrict__ B_row3 = (B + (k + 3) * N);

            constexpr int inc = 64 / sizeof(Scalar);

            #pragma GCC unroll 4
            for (; n >= 64; n -= 64) {
                _mm512_store_si512((void *) (B_p_row0), _mm512_loadu_si512((const void *) (B_row0)));
                _mm512_store_si512((void *) (B_p_row1), _mm512_loadu_si512((const void *) (B_row1)));
                _mm512_store_si512((void *) (B_p_row2), _mm512_loadu_si512((const void *) (B_row2)));
                _mm512_store_si512((void *) (B_p_row3), _mm512_loadu_si512((const void *) (B_row3)));

                B_p_row0 = (void*)((uint8_t*) B_p_row0 + 64);  B_row0 = (void*)((uint8_t*) B_row0 + 64);
                B_p_row1 = (void*)((uint8_t*) B_p_row1 + 64);  B_row1 = (void*)((uint8_t*) B_row1 + 64);
                B_p_row2 = (void*)((uint8_t*) B_p_row2 + 64);  B_row2 = (void*)((uint8_t*) B_row2 + 64);
                B_p_row3 = (void*)((uint8_t*) B_p_row3 + 64);  B_row3 = (void*)((uint8_t*) B_row3 + 64);
            }

            if (n) {
                __mmask16 mask = ~(uint16_t) ((1 << (n / 4)) - 1);
                _mm512_store_si512((void *) (B_p_row0), _mm512_maskz_loadu_epi32(mask, (const void *) (B_row0)));
                _mm512_store_si512((void *) (B_p_row1), _mm512_maskz_loadu_epi32(mask, (const void *) (B_row1)));
                _mm512_store_si512((void *) (B_p_row2), _mm512_maskz_loadu_epi32(mask, (const void *) (B_row2)));
                _mm512_store_si512((void *) (B_p_row3), _mm512_maskz_loadu_epi32(mask, (const void *) (B_row3)));
            }
        }
    }
    _ai Scalar *__restrict__ get_B_packed_buffer(int tjj) {
      return B_packed + (tjj * K_p * N_c);
    }

    _ai Scalar *__restrict__ seek_to_next_B_tile(Scalar *__restrict__ B_p) {
      return B_p + (K_p * N_c);
    }

    _ai const Scalar *__restrict__ seek_to_next_B_tile(const Scalar *__restrict__ B_p) {
      return B_p + (K_p * N_c);
    }

    _ai Scalar *__restrict__ get_C_packed_buffer(int threadId, int m, int n) {
        return C_packed + (threadId * M_c * N_c) + (m * N_p) + n;
    }

    _ai Scalar *__restrict__ get_C_packed_buffer(int threadId) {
        return C_packed + (threadId * M_c * N_c);
    }

    _ai Scalar *__restrict__ seek_to_next_reg_tile(Scalar *__restrict__ buffer) {
        return buffer + (M_r * N_r);
    }

    _ai Scalar *__restrict__ advance_to_row(Scalar *__restrict__ buffer, int row) {
        return buffer + (M_r * N_r) * c_N_r * row;
    }

/*  NOT NEEDED SINCE WE FUSED C UNPACKING INTO THE MICROKERNEL
    void unpack_C_reg_tile(Scalar __restrict__ *C, Scalar __restrict__ *C_packed_tile) {
        Scalar __restrict__ *C_packed_tile_a = (Scalar*) __builtin_assume_aligned(C_packed_tile, 16);
        for (int i = 0; i < M_r; i++) {
            #pragma ivdep
            for (int j = 0; j < N_r; j++) {
                C[i * N + j] = C_packed_tile_a[i * N_r + j];
            }
        }
    }

    void unpack_C_reg_tile(Scalar __restrict__ *C, Scalar __restrict__ *C_packed_tile, int n) {
        Scalar __restrict__ *C_packed_tile_a = (Scalar*) __builtin_assume_aligned(C_packed_tile, 16);
        for (int i = 0; i < M_r; i++) {
            #pragma ivdep
            for (int j = 0; j < n; j++) {
                C[i * N + j] = C_packed_tile_a[i * N_r + j];
            }
        }
    }

    void unpack_C_cache_tile(Scalar __restrict__ *C, Scalar __restrict__ *_C_packed_tile, int n) {
        if constexpr(Sched == C3_nmNKM || Sched == C3_nmKNM) {
            int N_full_blks = n / N_r;
            int N_rem = n % N_r;

            Scalar __restrict__ *C_packed_tile = (Scalar*) __builtin_assume_aligned(_C_packed_tile, 16);

            #pragma ivdep
            for (int ti = 0; ti < c_M_r; ti++) {
                Scalar __restrict__ *Cij =  (Scalar*) __builtin_assume_aligned(C + ti * M_r * N, 16);

                int tj = 0;
                for (; tj < N_full_blks; tj++, Cij += N_r) {
                    int tile_offset = (ti * c_N_r + tj) * M_r * N_r;

                    for (int i = 0; i < M_r; i++) {
                        int packed_offset = tile_offset + i * N_r;
                        int row_offset = i * N;

                        #pragma ivdep
                        for (int j = 0; j < N_r; j++) {
                            Cij[row_offset + j] = C_packed_tile[packed_offset + j];
                        }
                    }
                }

                int tile_offset = (ti * c_N_r + tj) * M_r * N_r;
                for (int i = 0; i < M_r; i++) {
                    for (int j = 0; j < N_rem; j++) {
                        Cij[i * N + j] = C_packed_tile[tile_offset + i * N_r + j];
                    }
                }
            }
        }
    }
*/

};

};