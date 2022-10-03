#pragma once

#include "utils/error.h"
#include "MicroKernelBase.h"
#include "Storage.h"

#include <immintrin.h>


#include "intrin_compatability.h"

namespace sop {
struct MicroKernel_double_77f9d_AVX2_256_8x1 {

        static const uint16_t* supported_nkern_patterns() {
            static uint16_t patterns[] = {
                0b10000000,
                0b00000001,
                0b00000010,
                0b00000100,
                0b00001000,
                0b00010000,
                0b00100000,
                0b01000000,
                0b00001111,
                0b10101010,
                0b00111100,
                0b11000011,
                0b01010101,
                0b11110000,
                0b00111111,
                0b11001111,
                0b11110011,
                0b11111100,
                0b11111111
            };
        
            return patterns;
        }
        
        static uint16_t encode_nkern_pattern(uint16_t nkern_pat) {
            if (nkern_pat == 0b10000000) return 0;
            if (nkern_pat == 0b00000001) return 1;
            if (nkern_pat == 0b00000010) return 2;
            if (nkern_pat == 0b00000100) return 3;
            if (nkern_pat == 0b00001000) return 4;
            if (nkern_pat == 0b00010000) return 5;
            if (nkern_pat == 0b00100000) return 6;
            if (nkern_pat == 0b01000000) return 7;
            if (nkern_pat == 0b00001111) return 8;
            if (nkern_pat == 0b10101010) return 9;
            if (nkern_pat == 0b00111100) return 10;
            if (nkern_pat == 0b11000011) return 11;
            if (nkern_pat == 0b01010101) return 12;
            if (nkern_pat == 0b11110000) return 13;
            if (nkern_pat == 0b00111111) return 14;
            if (nkern_pat == 0b11001111) return 15;
            if (nkern_pat == 0b11110011) return 16;
            if (nkern_pat == 0b11111100) return 17;
            if (nkern_pat == 0b11111111) return 18;
            if (nkern_pat == 0) return sop::ZERO_PATTERN_ID; 
            ERROR_AND_EXIT("Unable to map unsupported nanokernel pattern " <<  (int) nkern_pat);
            return 0;
        }
        
        static uint16_t decode_nkern_pattern(uint16_t nkern_code) {
            if (nkern_code == 0) return 0b10000000;
            if (nkern_code == 1) return 0b00000001;
            if (nkern_code == 2) return 0b00000010;
            if (nkern_code == 3) return 0b00000100;
            if (nkern_code == 4) return 0b00001000;
            if (nkern_code == 5) return 0b00010000;
            if (nkern_code == 6) return 0b00100000;
            if (nkern_code == 7) return 0b01000000;
            if (nkern_code == 8) return 0b00001111;
            if (nkern_code == 9) return 0b10101010;
            if (nkern_code == 10) return 0b00111100;
            if (nkern_code == 11) return 0b11000011;
            if (nkern_code == 12) return 0b01010101;
            if (nkern_code == 13) return 0b11110000;
            if (nkern_code == 14) return 0b00111111;
            if (nkern_code == 15) return 0b11001111;
            if (nkern_code == 16) return 0b11110011;
            if (nkern_code == 17) return 0b11111100;
            if (nkern_code == 18) return 0b11111111;
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to unmap unsupported nanokernel pattern id " << (int) nkern_code);
            return 0;
        }
        
        static uint16_t nnz_count_for_nkern_code(uint16_t nkern_code) {
            if (nkern_code == 0) return 1;
            if (nkern_code == 1) return 1;
            if (nkern_code == 2) return 1;
            if (nkern_code == 3) return 1;
            if (nkern_code == 4) return 1;
            if (nkern_code == 5) return 1;
            if (nkern_code == 6) return 1;
            if (nkern_code == 7) return 1;
            if (nkern_code == 8) return 4;
            if (nkern_code == 9) return 4;
            if (nkern_code == 10) return 4;
            if (nkern_code == 11) return 4;
            if (nkern_code == 12) return 4;
            if (nkern_code == 13) return 4;
            if (nkern_code == 14) return 6;
            if (nkern_code == 15) return 6;
            if (nkern_code == 16) return 6;
            if (nkern_code == 17) return 6;
            if (nkern_code == 18) return 8;
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
            return 0;
        }
        
    using  Mask = uint32_t;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % 4); }

    using Scalar = double;
    static constexpr int M_r = 8;
    static constexpr int N_r = 1 * 4;
    static constexpr int N_r_reg = 1;
    static constexpr int vec_width_bits = 256;
    static constexpr const char* id = "77f9d_AVX2_256_8x1";
    static int max_acc_width_in_vecs() { return 1; };
    static int max_acc_width_in_eles() { return 1 * 4; };

    static int num_nkern_patterns() { return 19; }

    __ALWAYS_INLINE static void _microkernel_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        double* __restrict__       values,
        int                          num_col_indices,
        const double *__restrict__ B,
        double *__restrict__ C,
        const bool load_c)
    {
      
      double* C0 = C + 0 * N;
      double* C1 = C + 1 * N;
      double* C2 = C + 2 * N;
      double* C3 = C + 3 * N;
      double* C4 = C + 4 * N;
      double* C5 = C + 5 * N;
      double* C6 = C + 6 * N;
      double* C7 = C + 7 * N;
      __m256 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm256d_loadu_pd(C0 + 0 * 4);
        cVec10 = _mm256d_loadu_pd(C1 + 0 * 4);
        cVec20 = _mm256d_loadu_pd(C2 + 0 * 4);
        cVec30 = _mm256d_loadu_pd(C3 + 0 * 4);
        cVec40 = _mm256d_loadu_pd(C4 + 0 * 4);
        cVec50 = _mm256d_loadu_pd(C5 + 0 * 4);
        cVec60 = _mm256d_loadu_pd(C6 + 0 * 4);
        cVec70 = _mm256d_loadu_pd(C7 + 0 * 4);
      } else {
        cVec00 = _mm256d_setzero_pd();
        cVec10 = _mm256d_setzero_pd();
        cVec20 = _mm256d_setzero_pd();
        cVec30 = _mm256d_setzero_pd();
        cVec40 = _mm256d_setzero_pd();
        cVec50 = _mm256d_setzero_pd();
        cVec60 = _mm256d_setzero_pd();
        cVec70 = _mm256d_setzero_pd();
      }
      
      int c_idx = 0;
      double* __restrict__ curr_value_ptr = values;
      const double *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 1
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      _mm256d_storeu_pd(C0 + 0 * 4, cVec00);
      _mm256d_storeu_pd(C1 + 0 * 4, cVec10);
      _mm256d_storeu_pd(C2 + 0 * 4, cVec20);
      _mm256d_storeu_pd(C3 + 0 * 4, cVec30);
      _mm256d_storeu_pd(C4 + 0 * 4, cVec40);
      _mm256d_storeu_pd(C5 + 0 * 4, cVec50);
      _mm256d_storeu_pd(C6 + 0 * 4, cVec60);
      _mm256d_storeu_pd(C7 + 0 * 4, cVec70);
      
      

    }



    __ALWAYS_INLINE static void microkernel_max_acc(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const double *__restrict__ B,
        double *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_max_acc(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_cleanup_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        double* __restrict__       values,
        int                          num_col_indices,
        const double *__restrict__ B,
        double *__restrict__ C,
        const bool load_c,
        int  elements_remaining,
        Mask precomp_mask)
    {
      for(; elements_remaining >= 4; elements_remaining -= 4, C += 4, B += 4) {
          
          double* C0 = C + 0 * N;
          double* C1 = C + 1 * N;
          double* C2 = C + 2 * N;
          double* C3 = C + 3 * N;
          double* C4 = C + 4 * N;
          double* C5 = C + 5 * N;
          double* C6 = C + 6 * N;
          double* C7 = C + 7 * N;
          __m256 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
          if (load_c) {
            cVec00 = _mm256d_loadu_pd(C0 + 0 * 4);
            cVec10 = _mm256d_loadu_pd(C1 + 0 * 4);
            cVec20 = _mm256d_loadu_pd(C2 + 0 * 4);
            cVec30 = _mm256d_loadu_pd(C3 + 0 * 4);
            cVec40 = _mm256d_loadu_pd(C4 + 0 * 4);
            cVec50 = _mm256d_loadu_pd(C5 + 0 * 4);
            cVec60 = _mm256d_loadu_pd(C6 + 0 * 4);
            cVec70 = _mm256d_loadu_pd(C7 + 0 * 4);
          } else {
            cVec00 = _mm256d_setzero_pd();
            cVec10 = _mm256d_setzero_pd();
            cVec20 = _mm256d_setzero_pd();
            cVec30 = _mm256d_setzero_pd();
            cVec40 = _mm256d_setzero_pd();
            cVec50 = _mm256d_setzero_pd();
            cVec60 = _mm256d_setzero_pd();
            cVec70 = _mm256d_setzero_pd();
          }
          
          int c_idx = 0;
          double* __restrict__ curr_value_ptr = values;
          const double *__restrict__ B_curr = col_indices[0] * N + B;
          uint32_t * col_indices_curr = col_indices + 1;
          #pragma unroll 1
          for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          _mm256d_storeu_pd(C0 + 0 * 4, cVec00);
          _mm256d_storeu_pd(C1 + 0 * 4, cVec10);
          _mm256d_storeu_pd(C2 + 0 * 4, cVec20);
          _mm256d_storeu_pd(C3 + 0 * 4, cVec30);
          _mm256d_storeu_pd(C4 + 0 * 4, cVec40);
          _mm256d_storeu_pd(C5 + 0 * 4, cVec50);
          _mm256d_storeu_pd(C6 + 0 * 4, cVec60);
          _mm256d_storeu_pd(C7 + 0 * 4, cVec70);
          
          


      }

    }


    
    __ALWAYS_INLINE static void _microkernel_packed_C_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        double* __restrict__       values,
        int                          num_col_indices,
        const double *__restrict__ B,
        double *__restrict__ C,
        const bool load_c)
    {
      
      double* C0 = C + 0 * 4;
      double* C1 = C + 1 * 4;
      double* C2 = C + 2 * 4;
      double* C3 = C + 3 * 4;
      double* C4 = C + 4 * 4;
      double* C5 = C + 5 * 4;
      double* C6 = C + 6 * 4;
      double* C7 = C + 7 * 4;
      __m256 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm256d_load_pd(C0 + 0 * 4);
        cVec10 = _mm256d_load_pd(C1 + 1 * 4);
        cVec20 = _mm256d_load_pd(C2 + 2 * 4);
        cVec30 = _mm256d_load_pd(C3 + 3 * 4);
        cVec40 = _mm256d_load_pd(C4 + 4 * 4);
        cVec50 = _mm256d_load_pd(C5 + 5 * 4);
        cVec60 = _mm256d_load_pd(C6 + 6 * 4);
        cVec70 = _mm256d_load_pd(C7 + 7 * 4);
      } else {
        cVec00 = _mm256d_setzero_pd();
        cVec10 = _mm256d_setzero_pd();
        cVec20 = _mm256d_setzero_pd();
        cVec30 = _mm256d_setzero_pd();
        cVec40 = _mm256d_setzero_pd();
        cVec50 = _mm256d_setzero_pd();
        cVec60 = _mm256d_setzero_pd();
        cVec70 = _mm256d_setzero_pd();
      }
      
      int c_idx = 0;
      double* __restrict__ curr_value_ptr = values;
      const double *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 1
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
        aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

      }

      
      _mm256d_storeu_pd(C0 + 0 * 4, cVec00);
      _mm256d_storeu_pd(C1 + 0 * 4, cVec10);
      _mm256d_storeu_pd(C2 + 0 * 4, cVec20);
      _mm256d_storeu_pd(C3 + 0 * 4, cVec30);
      _mm256d_storeu_pd(C4 + 0 * 4, cVec40);
      _mm256d_storeu_pd(C5 + 0 * 4, cVec50);
      _mm256d_storeu_pd(C6 + 0 * 4, cVec60);
      _mm256d_storeu_pd(C7 + 0 * 4, cVec70);
      
      

    }



    __ALWAYS_INLINE static void microkernel_packed_C_max_acc(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const double *__restrict__ B,
        double *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_C_max_acc(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_cleanup_packed_C_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        double* __restrict__       values,
        int                          num_col_indices,
        const double *__restrict__ B,
        double *__restrict__ C,
        const bool load_c,
        int  elements_remaining,
        Mask precomp_mask)
    {
      for(; elements_remaining >= 4; elements_remaining -= 4, C += 4, B += 4) {
          
          double* C0 = C + 0 * 4;
          double* C1 = C + 1 * 4;
          double* C2 = C + 2 * 4;
          double* C3 = C + 3 * 4;
          double* C4 = C + 4 * 4;
          double* C5 = C + 5 * 4;
          double* C6 = C + 6 * 4;
          double* C7 = C + 7 * 4;
          __m256 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
          if (load_c) {
            cVec00 = _mm256d_load_pd(C0 + 0 * 4);
            cVec10 = _mm256d_load_pd(C1 + 1 * 4);
            cVec20 = _mm256d_load_pd(C2 + 2 * 4);
            cVec30 = _mm256d_load_pd(C3 + 3 * 4);
            cVec40 = _mm256d_load_pd(C4 + 4 * 4);
            cVec50 = _mm256d_load_pd(C5 + 5 * 4);
            cVec60 = _mm256d_load_pd(C6 + 6 * 4);
            cVec70 = _mm256d_load_pd(C7 + 7 * 4);
          } else {
            cVec00 = _mm256d_setzero_pd();
            cVec10 = _mm256d_setzero_pd();
            cVec20 = _mm256d_setzero_pd();
            cVec30 = _mm256d_setzero_pd();
            cVec40 = _mm256d_setzero_pd();
            cVec50 = _mm256d_setzero_pd();
            cVec60 = _mm256d_setzero_pd();
            cVec70 = _mm256d_setzero_pd();
          }
          
          int c_idx = 0;
          double* __restrict__ curr_value_ptr = values;
          const double *__restrict__ B_curr = col_indices[0] * N + B;
          uint32_t * col_indices_curr = col_indices + 1;
          #pragma unroll 1
          for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
            __m256 aVec;
            __m256 bVec0 = _mm256d_loadu_pd(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm256d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm256d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm256d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm256d_fmadd_pd(aVec, bVec0, cVec30);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec40 = _mm256d_fmadd_pd(aVec, bVec0, cVec40);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec50 = _mm256d_fmadd_pd(aVec, bVec0, cVec50);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec60 = _mm256d_fmadd_pd(aVec, bVec0, cVec60);
            aVec = _mm256d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec70 = _mm256d_fmadd_pd(aVec, bVec0, cVec70);

          }

          
          _mm256d_storeu_pd(C0 + 0 * 4, cVec00);
          _mm256d_storeu_pd(C1 + 0 * 4, cVec10);
          _mm256d_storeu_pd(C2 + 0 * 4, cVec20);
          _mm256d_storeu_pd(C3 + 0 * 4, cVec30);
          _mm256d_storeu_pd(C4 + 0 * 4, cVec40);
          _mm256d_storeu_pd(C5 + 0 * 4, cVec50);
          _mm256d_storeu_pd(C6 + 0 * 4, cVec60);
          _mm256d_storeu_pd(C7 + 0 * 4, cVec70);
          
          


      }

    }


    
};

} // sop
