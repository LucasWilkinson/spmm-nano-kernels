#pragma once

#include "utils/error.h"
#include "MicroKernelBase.h"
#include "Storage.h"

#include <immintrin.h>


#include "intrin_compatability.h"

namespace sop {
struct MicroKernel_float_64487_AVX2_256_4x2 {

        static const uint16_t* supported_nkern_patterns() {
            static uint16_t patterns[] = {
                0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
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
        if (nkern_pat == 0b00000111) return 4;
        if (nkern_pat == 0b00001011) return 5;
        if (nkern_pat == 0b00001101) return 6;
        if (nkern_pat == 0b00001110) return 7;
        if (nkern_pat == 0b00001111) return 8;
            if (nkern_pat == 0) return sop::ZERO_PATTERN_ID; 
            ERROR_AND_EXIT("Unable to map unsupported nanokernel pattern " <<  (int) nkern_pat);
            return 0;
        }
        
        static uint16_t decode_nkern_pattern(uint16_t nkern_code) {
            if (nkern_code == 0) return 0b00000001;
        if (nkern_code == 1) return 0b00000010;
        if (nkern_code == 2) return 0b00000100;
        if (nkern_code == 3) return 0b00001000;
        if (nkern_code == 4) return 0b00000111;
        if (nkern_code == 5) return 0b00001011;
        if (nkern_code == 6) return 0b00001101;
        if (nkern_code == 7) return 0b00001110;
        if (nkern_code == 8) return 0b00001111;
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to unmap unsupported nanokernel pattern id " << (int) nkern_code);
            return 0;
        }
        
        static uint16_t nnz_count_for_nkern_code(uint16_t nkern_code) {
            if (nkern_code == 0) return 1;
        if (nkern_code == 1) return 1;
        if (nkern_code == 2) return 1;
        if (nkern_code == 3) return 1;
        if (nkern_code == 4) return 3;
        if (nkern_code == 5) return 3;
        if (nkern_code == 6) return 3;
        if (nkern_code == 7) return 3;
        if (nkern_code == 8) return 4;
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
            return 0;
        }
        
    using  Mask = __mmask8.0;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % 8.0); }

    using Scalar = float;
    static constexpr int M_r = 4;
    static constexpr int N_r = 2 * 8.0;
    static constexpr int N_r_reg = 2;
    static constexpr int vec_width_bits = 256;
    static constexpr const char* id = "64487_AVX2_256_4x2";
    static int max_acc_width_in_vecs() { return 2; };
    static int max_acc_width_in_eles() { return 2 * 8.0; };

    static int num_nkern_patterns() { return 9; }

    __ALWAYS_INLINE static void _microkernel_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec01, cVec10, cVec11, cVec20, cVec21, cVec30, cVec31;
      if (load_c) {
        cVec00 = _mm256_loadu_ps(C + 0 * 8.0);
        cVec01 = _mm256_loadu_ps(C + 1 * 8.0);
        C_temp += N;
        cVec10 = _mm256_loadu_ps(C + 0 * 8.0);
        cVec11 = _mm256_loadu_ps(C + 1 * 8.0);
        C_temp += N;
        cVec20 = _mm256_loadu_ps(C + 0 * 8.0);
        cVec21 = _mm256_loadu_ps(C + 1 * 8.0);
        C_temp += N;
        cVec30 = _mm256_loadu_ps(C + 0 * 8.0);
        cVec31 = _mm256_loadu_ps(C + 1 * 8.0);
        C_temp += N;
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec01 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec11 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec21 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
        cVec31 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8.0);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8.0);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      _mm256_storeu_ps(C + 0 * N + 0 * 8.0, cVec00);
      _mm256_storeu_ps(C + 0 * N + 1 * 8.0, cVec01);
      _mm256_storeu_ps(C + 1 * N + 0 * 8.0, cVec10);
      _mm256_storeu_ps(C + 1 * N + 1 * 8.0, cVec11);
      _mm256_storeu_ps(C + 2 * N + 0 * 8.0, cVec20);
      _mm256_storeu_ps(C + 2 * N + 1 * 8.0, cVec21);
      _mm256_storeu_ps(C + 3 * N + 0 * 8.0, cVec30);
      _mm256_storeu_ps(C + 3 * N + 1 * 8.0, cVec31);
      
      

    }



    __ALWAYS_INLINE static void microkernel_max_acc(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
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
    
};

} // sop
