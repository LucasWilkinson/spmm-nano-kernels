//
// Created by lwilkinson on 7/26/22.
//

#pragma once

#include "utils/tiling.h"


#define _ai inline __attribute__((__always_inline__))

//
//  Packed B
//

template<typename Scalar, typename TileDims>
_ai void pack_B_N_c_K_c_coop(
    int jjj, int kkk,
    int thread_id, int num_threads,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ B_packed,
    const TileDims& td) {
  using VecType = typename Vec<Scalar, 512>::Type;

  ALIAS_TILE_DIMS(TileDims, td);
  int K_thread = std::ceil(K_c / num_threads);
  int _k = K_thread * thread_id;
  for (; _k < std::min(K - kkk,  K_thread * (thread_id + 1)); _k++) {
    for (int _jj = 0; _jj < N_c; _jj += N_r) {
      for (int _j = 0; _j < N_r; _j += 64 / sizeof(Scalar)) {
        int k = kkk + _k;
        int j = jjj + _jj + _j;

        int offset =
            (kkk * N_c) // K_c Loop
            + (_jj * K_c)   // N_r Loop
            + (_k * N_r)  // K   Loop
            + _j;         // N   Loop

        __m512 zmm = _mm512_load_ps(B + k * N + j);
        _mm512_store_ps(B_packed + offset, zmm);
      }
    }
  }
}


template<typename Scalar, int M_r, int N_r>
inline __attribute__((__always_inline__))
int packed_B_offest(int i, int j, int M, int K, int N)
{
    int tj = j / N_r;
    return (tj * N_r * K) + i * N_r;
}

template<typename Scalar, int M_r, int N_r>
void pack_B_N_c(const Scalar* __restrict__ B, Scalar* __restrict__ B_packed,
            int N_c,
            int M, int K, int N) {
  using VecType = typename Vec<Scalar, 512>::Type;

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < K; i++) {
  for (int jj = 0; jj < N_c; jj += N_r) {

      for (int j = 0; j < N_r; j += VecType::size()) {
        //            VecType vec;
        //            vec.load(B + (i * N) + col_start + j);
        //            vec.store_a(B_packed + (tj * K * N_r) + (i * N_r) + j);

        __m512 zmm = _mm512_load_ps(B + (i * N) + jj + j);
        _mm512_store_ps(B_packed + (jj * K) + (i * N_r) + j, zmm);

      }
    }
  }
}


template<bool pad, typename Scalar, typename TileDims>
_ai void pack_B_K_c(
    int kkk,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ B_packed,
    const TileDims& td) {

  ALIAS_TILE_DIMS(TileDims, td);

  for (int jjj = 0; jjj < N; jjj += N_c) {
    for (int _jj = 0; _jj < N_c; _jj += N_r) {
      for (int _k = 0; _k < std::min(K - kkk, K_c); _k++) {
        for (int _j = 0; _j < N_r; _j += 64/sizeof(Scalar)) {
          int k = kkk + _k;
          int j = jjj + _jj + _j;

          int offset =
                (jjj * K)      // N_c Loop
              + (kkk * N_c)    // K_c Loop
              + (_jj * K_c)    // N_r Loop
              + (_k  * N_r)    // K   Loop
              + _j;            // N   Loop

          __m512 zmm = _mm512_load_ps(B + k * N + j);
          _mm512_store_ps(B_packed + offset, zmm);
        }
      }
    }
  }
}


template<typename Scalar, typename TileDims>
void pack_B_2(const Scalar* __restrict__ B,
            Scalar* __restrict__ B_packed,
            const TileDims& td) {
  using VecType = typename Vec<Scalar, 512>::Type;

  ALIAS_TILE_DIMS(TileDims, td);

  int output_offset = 0;
  int K_rem = K % K_c;

  __m512i zero; zero = _mm512_xor_epi64(zero, zero);

  for (int jjj = 0; jjj < N; jjj += N_c) {
    int output_offset = jjj * K_padded;
    for (int kkk = 0; kkk < K - K_rem; kkk += K_c) {
      for (int jj = jjj; jj < N; jj += N_r) {
        for (int k = kkk; k < kkk + K_c; k++) {
          for (int _j = 0; _j < N_r; _j += 64/sizeof(Scalar)) {
            __m512 zmm = _mm512_loadu_ps(B + k * N + jj + _j);
            _mm512_store_ps(B_packed + output_offset, zmm);
            output_offset += 64/sizeof(Scalar);
          }
        }
      }
    }
  }

  if (K_rem) {
    for (int jjj = 0; jjj < N; jjj += N_c) {
      int kkk = K - K_rem;
      //for (int kkk = K - K_rem; kkk < K; kkk += K_c) {
        for (int jj = jjj; jj < N; jj += N_r) {
          for (int k = kkk; k < kkk + K_c; k++) {
            for (int _j = 0; _j < N_r; _j += 64/sizeof(Scalar)) {
              __m512i zmm;
              if (k < K) {
                zmm = _mm512_load_epi32(B + k * N + jj + _j);
              } else {
                zmm = zero;
              }
              _mm512_store_epi32(B_packed + output_offset, zmm);
              output_offset += 64/sizeof(Scalar);
            }
          }
        }
      //}
    }
  }
}

template<typename Scalar, typename TileDims>
_ai void pack_B_N_c(int jjj, const Scalar* __restrict__ B,
                Scalar* __restrict__ B_packed,
                const TileDims& td) {
  using VecType = typename Vec<Scalar, 512>::Type;

  ALIAS_TILE_DIMS(TileDims, td);

  #pragma omp parallel for schedule(static) collapse(2)
  for (int kkk = 0; kkk < K; kkk += K_c) {
    for (int _jj = 0; _jj < N_c; _jj += N_r) {
      for (int _k = 0; _k < std::min(K - kkk, K_c); _k++) {
        for (int _j = 0; _j < N_r; _j += 64/sizeof(Scalar)) {
          int k = kkk + _k;
          int j = jjj + _jj + _j;

          int offset =
              0 //(jjj * K_padded) // N_c Loop
              + (kkk * N_c)    // K_c Loop
              + (_jj * K_c)    // N_r Loop
              + (_k  * N_r)    // K   Loop
              + _j;            // N   Loop

          __m512 zmm = _mm512_load_ps(B + k * N + j);
          _mm512_store_ps(B_packed + offset, zmm);
        }
      }
    }
  }
}


template<typename Scalar, typename TileDims>
void pack_B(const Scalar* __restrict__ B,
            Scalar* __restrict__ B_packed,
            const TileDims& td) {
    using VecType = typename Vec<Scalar, 512>::Type;

    ALIAS_TILE_DIMS(TileDims, td);

    K_c = 64;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int jjj = 0; jjj < N; jjj += N_c) {
     for (int kkk = 0; kkk < K; kkk += K_c) {
      for (int _k = 0; _k < std::min(K - kkk, K_c); _k++) {
        for (int _jj = 0; _jj < N_c; _jj += N_r) {

            for (int _j = 0; _j < N_r; _j += 64/sizeof(Scalar)) {
              int k = kkk + _k;
              int j = jjj + _jj + _j;

              int offset =
                  (jjj + _jj) * K_padded + k * N_r + _j;
//                  (jjj * K_padded) // N_c Loop
//                  + (kkk * N_c)    // K_c Loop
//                  + (_jj * K_c)    // N_r Loop
//                  + (_k  * N_r)    // K   Loop
//                  + _j;            // N   Loop

              __m512 zmm = _mm512_load_ps(B + k * N + j);
              _mm512_store_ps(B_packed + offset, zmm);
            }
          }
        }
      }
    }
}


template<typename Scalar, int M_r, int N_r>
_ai void pack_C_partial(
    const Scalar* __restrict__ C,
    Scalar* __restrict__ C_packed,
    int i_start, int j_start,
    int i_end, int j_end,
    int M_c, int N_c,
    int M, int K, int N
) {
  using VecType = typename Vec<Scalar, 512>::Type;

  for (int iii = i_start; iii < i_end; iii += M_c) {                    // M_c 3
    for (int ii = iii; ii < std::min(M, iii + M_c); ii += M_r) {        // M_r 2
      for (int _i = 0; _i < M_r; _i ++) {
      for (int jjj = j_start; jjj < j_end; jjj += N_c) {                // N_c 4
        for (int jj = jjj; jj < std::min(N, jjj + N_c); jj += N_r) {    // N_r 1
            const int o_N_c = (jjj - j_start) * (i_end - i_start);
            const int o_M_c = (iii - i_start) * N_c;
            const int o_M_r = (ii - iii) * N_c;
            const int o_N_r = (jj - jjj) * M_r;

            const int offset = o_M_c + o_N_c + o_M_r + o_N_r;
//            std::cout << offset << std::endl;
//            if (offset > (i_end - i_start) * (j_end - j_start)) {
//              std::cerr << "offset too large" << std::endl;
//            }

            #pragma unroll
            for (int _j = 0; _j < N_r; _j += VecType::size()) {
              int i = ii + _i;
              int j = jj + _j;

              VecType vec;
              vec.load(&C[(i * N) + j]);
              vec.store_a(C_packed + offset + _i * N_r + _j);
            }
          }
        }
      }
    }
  }
}

template<typename Scalar, int M_r, int N_r>
void pack_C(
    const Scalar* __restrict__ C,
    Scalar* __restrict__ C_packed,
    int M_c, int N_c,
    int M, int K, int N
) {
  pack_C_partial<Scalar, M_r, N_r>(
      C, C_packed,
      0, 0,
      M, N,
      M_c, N_c,
      M, K, N
  );
}

template<typename Scalar, typename TileDims>
_ai void unpack_C_partial_M_c(
    int rows_to_unpack,
    Scalar* __restrict__ C,
    const Scalar* __restrict__ C_packed,
    const TileDims& td
) {
  ALIAS_TILE_DIMS(TileDims, td);
  using VecType = typename Vec<Scalar, 512>::Type;

  for (int jjj = 0; jjj < N; jjj += N_c) {
    for (int ii = 0; ii < rows_to_unpack; ii += M_r) {
      int ooo = 0
          + jjj * M_c
          + ii  * N_c;

      int N_c_end = jjj + N_c;
      bool partial_tile = false;

      if (N_c_end > N) {
        N_c_end = N - (N_r - 1); // adjust to end of full N_r tiles
        partial_tile = true;
      }

      int jj = jjj, _jj = 0;
      int _i_end = std::min(rows_to_unpack - ii, M_r);

      // Full N_r
      for (; jj < N_c_end; jj += N_r, _jj += N_r) {
        for (int _i = 0, i = ii; _i < _i_end; i++, _i++) {
          #pragma unroll
          for (int _j = 0; _j < N_r; _j += VecType::size()) {
            int offset = ooo + (_jj * M_r) + (_i * N_r) + _j;
            VecType v; v.load_a(C_packed + offset);
            v.store(C + (i * N) + jj + _j);
          }
        }
      }

      if (partial_tile) {
        for (int _i = 0, i = ii; _i < _i_end; i++, _i++) {
          // Full Vecs
          int _j = 0;
          int oo = ooo + (_jj * M_r) + (_i * N_r);

          for (; _j < N - jj - (VecType::size() - 1); _j += VecType::size()) {
            VecType v;
            v.load_a(C_packed + oo + _j);
            v.store(C + (i * N) + jj + _j);
          }

          VecType v;
          v.load_a(C_packed + oo + _j);
          v.store_partial(N - (jj + _j), C + (i * N) + jj + _j);
        }
      }
    }
  }
}


template<typename Scalar, int M_r, int N_r>
_ai void unpack_C_partial_M_c_N_c(
    Scalar* __restrict__ C,
    const Scalar* __restrict__ C_packed,
    int i_start, int j_start,
    int M_c, int N_c,
    int M, int K, int N
) {

  using VecType = typename Vec<Scalar, 512>::Type;
  int iii = i_start;
  int jjj = j_start;


  for (int ii = iii; ii < std::min(M, iii + M_c); ii += M_r) {   // M_r 2
    for (int _i = 0, i = ii; _i < M_r; ++_i, ++i) {

      for (int jj = jjj; jj < std::min(N, jjj + N_c); jj += N_r) { // N_r 1

        const int o_N_c = (jjj - j_start) * M_c;
        const int o_M_c = (iii - i_start) * N_c;
        const int o_M_r = (ii - iii) * N_c;
        const int o_N_r = (jj - jjj) * M_r;

        const int offset = o_M_c + o_N_c + o_M_r + o_N_r;
        #pragma unroll
        for (int _j = 0; _j < N_r; _j += VecType::size()) {
          int j = jj + _j;

          __m512 zmm = _mm512_load_ps(C_packed + offset + _i * N_r + _j);
          _mm512_stream_ps(C + (i * N) + j, zmm);
        }
      }
    }
  }
}

template<typename Scalar, int M_r, int N_r>
_ai void unpack_C_partial(
    Scalar* __restrict__ C,
    const Scalar* __restrict__ C_packed,
    int i_start, int j_start,
    int i_end, int j_end,
    int M_c, int N_c,
    int M, int K, int N
) {

  using VecType = typename Vec<Scalar, 512>::Type;

  for (int jjj = j_start; jjj < j_end; jjj += N_c) {                 // N_c 4
    for (int iii = i_start; iii < i_end; iii += M_c) {               // M_c 3
      for (int ii = iii; ii < std::min(M, iii + M_c); ii += M_r) {   // M_r 2
        for (int _i = 0, i = ii; _i < M_r; ++_i, ++i) {

        for (int jj = jjj; jj < std::min(N, jjj + N_c); jj += N_r) { // N_r 1

          const int o_N_c = (jjj - j_start) * (i_end - i_start);
          const int o_M_c = (iii - i_start) * N_c;
          const int o_M_r = (ii - iii) * N_c;
          const int o_N_r = (jj - jjj) * M_r;

          const int offset = o_M_c + o_N_c + o_M_r + o_N_r;
            #pragma unroll
            for (int _j = 0; _j < N_r; _j += VecType::size()) {
              int j = jj + _j;

              VecType vec;
              vec.load_a(C_packed + offset + _i * N_r + _j);
              vec.store(&C[(i * N) + j]);
            }
          }
        }
      }
    }
  }
}

template<typename Scalar, int M_r, int N_r>
void unpack_C(
    Scalar* __restrict__ C,
    const Scalar* __restrict__ C_packed,
    int M_c, int N_c,
    int M, int K, int N
) {
  unpack_C_partial<Scalar, M_r, N_r>(
      C, C_packed,
      0, 0,
      M, N,
      M_c, N_c,
      M, K, N
  );
}