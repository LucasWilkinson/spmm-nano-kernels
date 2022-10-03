#pragma once

#include "utils/error.h"
#include "MicroKernelBase.h"
#include "Storage.h"

#include <immintrin.h>


#include "intrin_compatability.h"

namespace sop {
struct MicroKernel_double_c22a5_AVX512_512_4x4 {

        static const uint16_t* supported_nkern_patterns() {
            static uint16_t patterns[] = {
                0b00000001,
                0b00000010,
                0b00000100,
                0b00001000,
                0b00000011,
                0b00000101,
                0b00000110,
                0b00001001,
                0b00001010,
                0b00001100,
                0b00000111,
                0b00001011,
                0b00001101,
                0b00001110,
                0b00001111
            };
        
            return patterns;
        }
        
        static uint16_t encode_nkern_pattern(uint16_t nkern_pat) {
            if (nkern_pat == 0b00000001) return 0;
            if (nkern_pat == 0b00000010) return 1;
            if (nkern_pat == 0b00000100) return 2;
            if (nkern_pat == 0b00001000) return 3;
            if (nkern_pat == 0b00000011) return 4;
            if (nkern_pat == 0b00000101) return 5;
            if (nkern_pat == 0b00000110) return 6;
            if (nkern_pat == 0b00001001) return 7;
            if (nkern_pat == 0b00001010) return 8;
            if (nkern_pat == 0b00001100) return 9;
            if (nkern_pat == 0b00000111) return 10;
            if (nkern_pat == 0b00001011) return 11;
            if (nkern_pat == 0b00001101) return 12;
            if (nkern_pat == 0b00001110) return 13;
            if (nkern_pat == 0b00001111) return 14;
            if (nkern_pat == 0) return sop::ZERO_PATTERN_ID; 
            ERROR_AND_EXIT("Unable to map unsupported nanokernel pattern " <<  (int) nkern_pat);
            return 0;
        }
        
        static uint16_t decode_nkern_pattern(uint16_t nkern_code) {
            if (nkern_code == 0) return 0b00000001;
            if (nkern_code == 1) return 0b00000010;
            if (nkern_code == 2) return 0b00000100;
            if (nkern_code == 3) return 0b00001000;
            if (nkern_code == 4) return 0b00000011;
            if (nkern_code == 5) return 0b00000101;
            if (nkern_code == 6) return 0b00000110;
            if (nkern_code == 7) return 0b00001001;
            if (nkern_code == 8) return 0b00001010;
            if (nkern_code == 9) return 0b00001100;
            if (nkern_code == 10) return 0b00000111;
            if (nkern_code == 11) return 0b00001011;
            if (nkern_code == 12) return 0b00001101;
            if (nkern_code == 13) return 0b00001110;
            if (nkern_code == 14) return 0b00001111;
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to unmap unsupported nanokernel pattern id " << (int) nkern_code);
            return 0;
        }
        
        static uint16_t nnz_count_for_nkern_code(uint16_t nkern_code) {
            if (nkern_code == 0) return 1;
            if (nkern_code == 1) return 1;
            if (nkern_code == 2) return 1;
            if (nkern_code == 3) return 1;
            if (nkern_code == 4) return 2;
            if (nkern_code == 5) return 2;
            if (nkern_code == 6) return 2;
            if (nkern_code == 7) return 2;
            if (nkern_code == 8) return 2;
            if (nkern_code == 9) return 2;
            if (nkern_code == 10) return 3;
            if (nkern_code == 11) return 3;
            if (nkern_code == 12) return 3;
            if (nkern_code == 13) return 3;
            if (nkern_code == 14) return 4;
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
            return 0;
        }
        
    using  Mask = __mmask8;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % 8); }

    using Scalar = double;
    static constexpr int M_r = 4;
    static constexpr int N_r = 4 * 8;
    static constexpr int N_r_reg = 4;
    static constexpr int vec_width_bits = 512;
    static constexpr const char* id = "c22a5_AVX512_512_4x4";
    static int max_acc_width_in_vecs() { return 4; };
    static int max_acc_width_in_eles() { return 4 * 8; };

    static int num_nkern_patterns() { return 15; }

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
      __m512 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm512d_loadu_pd(C0 + 0 * 8);
        cVec01 = _mm512d_loadu_pd(C0 + 1 * 8);
        cVec02 = _mm512d_loadu_pd(C0 + 2 * 8);
        cVec03 = _mm512d_loadu_pd(C0 + 3 * 8);
        cVec10 = _mm512d_loadu_pd(C1 + 0 * 8);
        cVec11 = _mm512d_loadu_pd(C1 + 1 * 8);
        cVec12 = _mm512d_loadu_pd(C1 + 2 * 8);
        cVec13 = _mm512d_loadu_pd(C1 + 3 * 8);
        cVec20 = _mm512d_loadu_pd(C2 + 0 * 8);
        cVec21 = _mm512d_loadu_pd(C2 + 1 * 8);
        cVec22 = _mm512d_loadu_pd(C2 + 2 * 8);
        cVec23 = _mm512d_loadu_pd(C2 + 3 * 8);
        cVec30 = _mm512d_loadu_pd(C3 + 0 * 8);
        cVec31 = _mm512d_loadu_pd(C3 + 1 * 8);
        cVec32 = _mm512d_loadu_pd(C3 + 2 * 8);
        cVec33 = _mm512d_loadu_pd(C3 + 3 * 8);
      } else {
        cVec00 = _mm512d_setzero_pd();
        cVec01 = _mm512d_setzero_pd();
        cVec02 = _mm512d_setzero_pd();
        cVec03 = _mm512d_setzero_pd();
        cVec10 = _mm512d_setzero_pd();
        cVec11 = _mm512d_setzero_pd();
        cVec12 = _mm512d_setzero_pd();
        cVec13 = _mm512d_setzero_pd();
        cVec20 = _mm512d_setzero_pd();
        cVec21 = _mm512d_setzero_pd();
        cVec22 = _mm512d_setzero_pd();
        cVec23 = _mm512d_setzero_pd();
        cVec30 = _mm512d_setzero_pd();
        cVec31 = _mm512d_setzero_pd();
        cVec32 = _mm512d_setzero_pd();
        cVec33 = _mm512d_setzero_pd();
      }
      
      int c_idx = 0;
      double* __restrict__ curr_value_ptr = values;
      const double *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 1
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      _mm512d_storeu_pd(C0 + 0 * 8, cVec00);
      _mm512d_storeu_pd(C0 + 1 * 8, cVec01);
      _mm512d_storeu_pd(C0 + 2 * 8, cVec02);
      _mm512d_storeu_pd(C0 + 3 * 8, cVec03);
      _mm512d_storeu_pd(C1 + 0 * 8, cVec10);
      _mm512d_storeu_pd(C1 + 1 * 8, cVec11);
      _mm512d_storeu_pd(C1 + 2 * 8, cVec12);
      _mm512d_storeu_pd(C1 + 3 * 8, cVec13);
      _mm512d_storeu_pd(C2 + 0 * 8, cVec20);
      _mm512d_storeu_pd(C2 + 1 * 8, cVec21);
      _mm512d_storeu_pd(C2 + 2 * 8, cVec22);
      _mm512d_storeu_pd(C2 + 3 * 8, cVec23);
      _mm512d_storeu_pd(C3 + 0 * 8, cVec30);
      _mm512d_storeu_pd(C3 + 1 * 8, cVec31);
      _mm512d_storeu_pd(C3 + 2 * 8, cVec32);
      _mm512d_storeu_pd(C3 + 3 * 8, cVec33);
      
      

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
      for(; elements_remaining >= 8; elements_remaining -= 8, C += 8, B += 8) {
          
          double* C0 = C + 0 * N;
          double* C1 = C + 1 * N;
          double* C2 = C + 2 * N;
          double* C3 = C + 3 * N;
          __m512 cVec00, cVec10, cVec20, cVec30;
          if (load_c) {
            cVec00 = _mm512d_loadu_pd(C0 + 0 * 8);
            cVec10 = _mm512d_loadu_pd(C1 + 0 * 8);
            cVec20 = _mm512d_loadu_pd(C2 + 0 * 8);
            cVec30 = _mm512d_loadu_pd(C3 + 0 * 8);
          } else {
            cVec00 = _mm512d_setzero_pd();
            cVec10 = _mm512d_setzero_pd();
            cVec20 = _mm512d_setzero_pd();
            cVec30 = _mm512d_setzero_pd();
          }
          
          int c_idx = 0;
          double* __restrict__ curr_value_ptr = values;
          const double *__restrict__ B_curr = col_indices[0] * N + B;
          uint32_t * col_indices_curr = col_indices + 1;
          #pragma unroll 1
          for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          _mm512d_storeu_pd(C0 + 0 * 8, cVec00);
          _mm512d_storeu_pd(C1 + 0 * 8, cVec10);
          _mm512d_storeu_pd(C2 + 0 * 8, cVec20);
          _mm512d_storeu_pd(C3 + 0 * 8, cVec30);
          
          


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
      
      double* C0 = C + 0 * 8;
      double* C1 = C + 4 * 8;
      double* C2 = C + 8 * 8;
      double* C3 = C + 12 * 8;
      __m512 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm512d_load_pd(C0 + 0 * 8);
        cVec01 = _mm512d_load_pd(C0 + 1 * 8);
        cVec02 = _mm512d_load_pd(C0 + 2 * 8);
        cVec03 = _mm512d_load_pd(C0 + 3 * 8);
        cVec10 = _mm512d_load_pd(C1 + 4 * 8);
        cVec11 = _mm512d_load_pd(C1 + 5 * 8);
        cVec12 = _mm512d_load_pd(C1 + 6 * 8);
        cVec13 = _mm512d_load_pd(C1 + 7 * 8);
        cVec20 = _mm512d_load_pd(C2 + 8 * 8);
        cVec21 = _mm512d_load_pd(C2 + 9 * 8);
        cVec22 = _mm512d_load_pd(C2 + 10 * 8);
        cVec23 = _mm512d_load_pd(C2 + 11 * 8);
        cVec30 = _mm512d_load_pd(C3 + 12 * 8);
        cVec31 = _mm512d_load_pd(C3 + 13 * 8);
        cVec32 = _mm512d_load_pd(C3 + 14 * 8);
        cVec33 = _mm512d_load_pd(C3 + 15 * 8);
      } else {
        cVec00 = _mm512d_setzero_pd();
        cVec01 = _mm512d_setzero_pd();
        cVec02 = _mm512d_setzero_pd();
        cVec03 = _mm512d_setzero_pd();
        cVec10 = _mm512d_setzero_pd();
        cVec11 = _mm512d_setzero_pd();
        cVec12 = _mm512d_setzero_pd();
        cVec13 = _mm512d_setzero_pd();
        cVec20 = _mm512d_setzero_pd();
        cVec21 = _mm512d_setzero_pd();
        cVec22 = _mm512d_setzero_pd();
        cVec23 = _mm512d_setzero_pd();
        cVec30 = _mm512d_setzero_pd();
        cVec31 = _mm512d_setzero_pd();
        cVec32 = _mm512d_setzero_pd();
        cVec33 = _mm512d_setzero_pd();
      }
      
      int c_idx = 0;
      double* __restrict__ curr_value_ptr = values;
      const double *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 1
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
        __m512 bVec1 = _mm512d_loadu_pd(B_curr + 1 * 8);
        __m512 bVec2 = _mm512d_loadu_pd(B_curr + 2 * 8);
        __m512 bVec3 = _mm512d_loadu_pd(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
        cVec01 = _mm512d_fmadd_pd(aVec, bVec1, cVec01);
        cVec02 = _mm512d_fmadd_pd(aVec, bVec2, cVec02);
        cVec03 = _mm512d_fmadd_pd(aVec, bVec3, cVec03);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
        cVec11 = _mm512d_fmadd_pd(aVec, bVec1, cVec11);
        cVec12 = _mm512d_fmadd_pd(aVec, bVec2, cVec12);
        cVec13 = _mm512d_fmadd_pd(aVec, bVec3, cVec13);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
        cVec21 = _mm512d_fmadd_pd(aVec, bVec1, cVec21);
        cVec22 = _mm512d_fmadd_pd(aVec, bVec2, cVec22);
        cVec23 = _mm512d_fmadd_pd(aVec, bVec3, cVec23);
        aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);
        cVec31 = _mm512d_fmadd_pd(aVec, bVec1, cVec31);
        cVec32 = _mm512d_fmadd_pd(aVec, bVec2, cVec32);
        cVec33 = _mm512d_fmadd_pd(aVec, bVec3, cVec33);

      }

      
      _mm512d_storeu_pd(C0 + 0 * 8, cVec00);
      _mm512d_storeu_pd(C0 + 1 * 8, cVec01);
      _mm512d_storeu_pd(C0 + 2 * 8, cVec02);
      _mm512d_storeu_pd(C0 + 3 * 8, cVec03);
      _mm512d_storeu_pd(C1 + 0 * 8, cVec10);
      _mm512d_storeu_pd(C1 + 1 * 8, cVec11);
      _mm512d_storeu_pd(C1 + 2 * 8, cVec12);
      _mm512d_storeu_pd(C1 + 3 * 8, cVec13);
      _mm512d_storeu_pd(C2 + 0 * 8, cVec20);
      _mm512d_storeu_pd(C2 + 1 * 8, cVec21);
      _mm512d_storeu_pd(C2 + 2 * 8, cVec22);
      _mm512d_storeu_pd(C2 + 3 * 8, cVec23);
      _mm512d_storeu_pd(C3 + 0 * 8, cVec30);
      _mm512d_storeu_pd(C3 + 1 * 8, cVec31);
      _mm512d_storeu_pd(C3 + 2 * 8, cVec32);
      _mm512d_storeu_pd(C3 + 3 * 8, cVec33);
      
      

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
      for(; elements_remaining >= 8; elements_remaining -= 8, C += 8, B += 8) {
          
          double* C0 = C + 0 * 8;
          double* C1 = C + 1 * 8;
          double* C2 = C + 2 * 8;
          double* C3 = C + 3 * 8;
          __m512 cVec00, cVec10, cVec20, cVec30;
          if (load_c) {
            cVec00 = _mm512d_load_pd(C0 + 0 * 8);
            cVec10 = _mm512d_load_pd(C1 + 1 * 8);
            cVec20 = _mm512d_load_pd(C2 + 2 * 8);
            cVec30 = _mm512d_load_pd(C3 + 3 * 8);
          } else {
            cVec00 = _mm512d_setzero_pd();
            cVec10 = _mm512d_setzero_pd();
            cVec20 = _mm512d_setzero_pd();
            cVec30 = _mm512d_setzero_pd();
          }
          
          int c_idx = 0;
          double* __restrict__ curr_value_ptr = values;
          const double *__restrict__ B_curr = col_indices[0] * N + B;
          uint32_t * col_indices_curr = col_indices + 1;
          #pragma unroll 1
          for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
            __m512 aVec;
            __m512 bVec0 = _mm512d_loadu_pd(B_curr + 0 * 8);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec00 = _mm512d_fmadd_pd(aVec, bVec0, cVec00);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec10 = _mm512d_fmadd_pd(aVec, bVec0, cVec10);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec20 = _mm512d_fmadd_pd(aVec, bVec0, cVec20);
            aVec = _mm512d_set1_pd(*curr_value_ptr); curr_value_ptr++;
            cVec30 = _mm512d_fmadd_pd(aVec, bVec0, cVec30);

          }

          
          _mm512d_storeu_pd(C0 + 0 * 8, cVec00);
          _mm512d_storeu_pd(C1 + 0 * 8, cVec10);
          _mm512d_storeu_pd(C2 + 0 * 8, cVec20);
          _mm512d_storeu_pd(C3 + 0 * 8, cVec30);
          
          


      }

    }


    
};

} // sop
