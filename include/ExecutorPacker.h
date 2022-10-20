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

#ifdef __AVX512F__

#define ALIGNMENT_MASK 0x3F

/**
 * AVX512 implementation below
 */

/**
 * Copy 16 bytes from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov16(uint8_t *dst, const uint8_t *src)
{
  __m128i xmm0;

  xmm0 = _mm_loadu_si128((const __m128i *)src);
  _mm_storeu_si128((__m128i *)dst, xmm0);
}

/**
 * Copy 32 bytes from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov32(uint8_t *dst, const uint8_t *src)
{
  __m256i ymm0;

  ymm0 = _mm256_loadu_si256((const __m256i *)src);
  _mm256_storeu_si256((__m256i *)dst, ymm0);
}

/**
 * Copy 64 bytes from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov64(uint8_t *dst, const uint8_t *src)
{
  __m512i zmm0;

  zmm0 = _mm512_loadu_si512((const void *)src);
  _mm512_storeu_si512((void *)dst, zmm0);
}

/**
 * Copy 128 bytes from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov128(uint8_t *dst, const uint8_t *src)
{
  rte_mov64(dst + 0 * 64, src + 0 * 64);
  rte_mov64(dst + 1 * 64, src + 1 * 64);
}

/**
 * Copy 256 bytes from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov256(uint8_t *dst, const uint8_t *src)
{
  rte_mov64(dst + 0 * 64, src + 0 * 64);
  rte_mov64(dst + 1 * 64, src + 1 * 64);
  rte_mov64(dst + 2 * 64, src + 2 * 64);
  rte_mov64(dst + 3 * 64, src + 3 * 64);
}

/**
 * Copy 128-byte blocks from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov128blocks(uint8_t *dst, const uint8_t *src, size_t n)
{
  __m512i zmm0, zmm1;

  while (n >= 128) {
    zmm0 = _mm512_loadu_si512((const void *)(src + 0 * 64));
    n -= 128;
    zmm1 = _mm512_loadu_si512((const void *)(src + 1 * 64));
    src = src + 128;
    _mm512_storeu_si512((void *)(dst + 0 * 64), zmm0);
    _mm512_storeu_si512((void *)(dst + 1 * 64), zmm1);
    dst = dst + 128;
  }
}

/**
 * Copy 512-byte blocks from one location to another,
 * locations should not overlap.
 */
static _ai void
rte_mov512blocks(uint8_t *dst, const uint8_t *src, size_t n)
{
  __m512i zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7;

  while (n >= 512) {
    zmm0 = _mm512_loadu_si512((const void *)(src + 0 * 64));
    n -= 512;
    zmm1 = _mm512_loadu_si512((const void *)(src + 1 * 64));
    zmm2 = _mm512_loadu_si512((const void *)(src + 2 * 64));
    zmm3 = _mm512_loadu_si512((const void *)(src + 3 * 64));
    zmm4 = _mm512_loadu_si512((const void *)(src + 4 * 64));
    zmm5 = _mm512_loadu_si512((const void *)(src + 5 * 64));
    zmm6 = _mm512_loadu_si512((const void *)(src + 6 * 64));
    zmm7 = _mm512_loadu_si512((const void *)(src + 7 * 64));
    src = src + 512;
    _mm512_storeu_si512((void *)(dst + 0 * 64), zmm0);
    _mm512_storeu_si512((void *)(dst + 1 * 64), zmm1);
    _mm512_storeu_si512((void *)(dst + 2 * 64), zmm2);
    _mm512_storeu_si512((void *)(dst + 3 * 64), zmm3);
    _mm512_storeu_si512((void *)(dst + 4 * 64), zmm4);
    _mm512_storeu_si512((void *)(dst + 5 * 64), zmm5);
    _mm512_storeu_si512((void *)(dst + 6 * 64), zmm6);
    _mm512_storeu_si512((void *)(dst + 7 * 64), zmm7);
    dst = dst + 512;
  }
}

static void *
pack_memcpy(void *dst, const void *src, size_t n)
{
  uintptr_t dstu = (uintptr_t)dst;
  uintptr_t srcu = (uintptr_t)src;
  void *ret = dst;
  size_t dstofss;
  size_t bits;

  /**
	 * Copy less than 16 bytes
   */
  if (n < 16) {
    if (n & 0x01) {
      *(uint8_t *)dstu = *(const uint8_t *)srcu;
      srcu = (uintptr_t)((const uint8_t *)srcu + 1);
      dstu = (uintptr_t)((uint8_t *)dstu + 1);
    }
    if (n & 0x02) {
      *(uint16_t *)dstu = *(const uint16_t *)srcu;
      srcu = (uintptr_t)((const uint16_t *)srcu + 1);
      dstu = (uintptr_t)((uint16_t *)dstu + 1);
    }
    if (n & 0x04) {
      *(uint32_t *)dstu = *(const uint32_t *)srcu;
      srcu = (uintptr_t)((const uint32_t *)srcu + 1);
      dstu = (uintptr_t)((uint32_t *)dstu + 1);
    }
    if (n & 0x08)
      *(uint64_t *)dstu = *(const uint64_t *)srcu;
    return ret;
  }

  /**
	 * Fast way when copy size doesn't exceed 512 bytes
   */
  if (n <= 32) {
    rte_mov16((uint8_t *)dst, (const uint8_t *)src);
    rte_mov16((uint8_t *)dst - 16 + n,
              (const uint8_t *)src - 16 + n);
    return ret;
  }
  if (n <= 64) {
    rte_mov32((uint8_t *)dst, (const uint8_t *)src);
    rte_mov32((uint8_t *)dst - 32 + n,
              (const uint8_t *)src - 32 + n);
    return ret;
  }
  if (n <= 512) {
    if (n >= 256) {
      n -= 256;
      rte_mov256((uint8_t *)dst, (const uint8_t *)src);
      src = (const uint8_t *)src + 256;
      dst = (uint8_t *)dst + 256;
    }
    if (n >= 128) {
      n -= 128;
      rte_mov128((uint8_t *)dst, (const uint8_t *)src);
      src = (const uint8_t *)src + 128;
      dst = (uint8_t *)dst + 128;
    }
  COPY_BLOCK_128_BACK63:
    if (n > 64) {
      rte_mov64((uint8_t *)dst, (const uint8_t *)src);
      rte_mov64((uint8_t *)dst - 64 + n,
                (const uint8_t *)src - 64 + n);
      return ret;
    }
    if (n > 0)
      rte_mov64((uint8_t *)dst - 64 + n,
                (const uint8_t *)src - 64 + n);
    return ret;
  }

  /**
	 * Make store aligned when copy size exceeds 512 bytes
   */
  dstofss = ((uintptr_t)dst & 0x3F);
  if (dstofss > 0) {
    dstofss = 64 - dstofss;
    n -= dstofss;
    rte_mov64((uint8_t *)dst, (const uint8_t *)src);
    src = (const uint8_t *)src + dstofss;
    dst = (uint8_t *)dst + dstofss;
  }

  /**
	 * Copy 512-byte blocks.
	 * Use copy block function for better instruction order control,
	 * which is important when load is unaligned.
   */
  rte_mov512blocks((uint8_t *)dst, (const uint8_t *)src, n);
  bits = n;
  n = n & 511;
  bits -= n;
  src = (const uint8_t *)src + bits;
  dst = (uint8_t *)dst + bits;

  /**
	 * Copy 128-byte blocks.
	 * Use copy block function for better instruction order control,
	 * which is important when load is unaligned.
   */
  if (n >= 128) {
    rte_mov128blocks((uint8_t *)dst, (const uint8_t *)src, n);
    bits = n;
    n = n & 127;
    bits -= n;
    src = (const uint8_t *)src + bits;
    dst = (uint8_t *)dst + bits;
  }

  /**
	 * Copy whatever left
   */

   goto COPY_BLOCK_128_BACK63;
}

#endif

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
        if (C_packed) operator delete[](C_packed, std::align_val_t(4096));
        if (B_packed) operator delete[](B_packed, std::align_val_t(4096));
    }

    void allocate_C_buffers() {
        C_packed = new(std::align_val_t(4096)) Scalar[M_c * N_c * nThreads];
    }

    void allocate_B_buffers() {
        B_packed = new(std::align_val_t(4096)) Scalar[K_p * N_p];
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