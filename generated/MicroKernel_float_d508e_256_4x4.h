#pragma once

#include "utils/error.h"
#include "MicroKernelBase.h"
#include "Storage.h"

#include <immintrin.h>


#include "intrin_compatability.h"

namespace sop {
struct MicroKernel_float_d508e_256_4x4 {

    static const uint16_t* supported_nkern_patterns() {
        static uint16_t patterns[] = {
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00000011,
            0b00000101,
            0b00001001,
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
        if (nkern_pat == 0b00001001) return 6;
        if (nkern_pat == 0b00001111) return 7;
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
        if (nkern_code == 6) return 0b00001001;
        if (nkern_code == 7) return 0b00001111;
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
        if (nkern_code == 7) return 4;
        if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
        ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
        return 0;
    }
    
    using Mask = __mmask8;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % 8); }

    using Scalar = float;
    static constexpr int M_r = 4;
    static constexpr int N_r = 4 * 8;
    static constexpr int N_r_reg = 4;
    static constexpr int vec_width_bits = 256;
    static constexpr const char* id = "d508e_256_4x4";
    static int max_acc_width_in_vecs() { return 4; };
    static int max_acc_width_in_eles() { return 4 * 8; };

    static int num_nkern_patterns() { return 8; }

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
      __m256 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm256_loadu_ps(C_temp + 0 * 8);
        cVec01 = _mm256_loadu_ps(C_temp + 1 * 8);
        cVec02 = _mm256_loadu_ps(C_temp + 2 * 8);
        cVec03 = _mm256_loadu_ps(C_temp + 3 * 8);
        C_temp += N;
        cVec10 = _mm256_loadu_ps(C_temp + 0 * 8);
        cVec11 = _mm256_loadu_ps(C_temp + 1 * 8);
        cVec12 = _mm256_loadu_ps(C_temp + 2 * 8);
        cVec13 = _mm256_loadu_ps(C_temp + 3 * 8);
        C_temp += N;
        cVec20 = _mm256_loadu_ps(C_temp + 0 * 8);
        cVec21 = _mm256_loadu_ps(C_temp + 1 * 8);
        cVec22 = _mm256_loadu_ps(C_temp + 2 * 8);
        cVec23 = _mm256_loadu_ps(C_temp + 3 * 8);
        C_temp += N;
        cVec30 = _mm256_loadu_ps(C_temp + 0 * 8);
        cVec31 = _mm256_loadu_ps(C_temp + 1 * 8);
        cVec32 = _mm256_loadu_ps(C_temp + 2 * 8);
        cVec33 = _mm256_loadu_ps(C_temp + 3 * 8);
        C_temp += N;
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec01 = _mm256_setzero_ps();
        cVec02 = _mm256_setzero_ps();
        cVec03 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec11 = _mm256_setzero_ps();
        cVec12 = _mm256_setzero_ps();
        cVec13 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec21 = _mm256_setzero_ps();
        cVec22 = _mm256_setzero_ps();
        cVec23 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
        cVec31 = _mm256_setzero_ps();
        cVec32 = _mm256_setzero_ps();
        cVec33 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      _mm256_storeu_ps(C + 0 * N + 0 * 8, cVec00);
      _mm256_storeu_ps(C + 0 * N + 1 * 8, cVec01);
      _mm256_storeu_ps(C + 0 * N + 2 * 8, cVec02);
      _mm256_storeu_ps(C + 0 * N + 3 * 8, cVec03);
      _mm256_storeu_ps(C + 1 * N + 0 * 8, cVec10);
      _mm256_storeu_ps(C + 1 * N + 1 * 8, cVec11);
      _mm256_storeu_ps(C + 1 * N + 2 * 8, cVec12);
      _mm256_storeu_ps(C + 1 * N + 3 * 8, cVec13);
      _mm256_storeu_ps(C + 2 * N + 0 * 8, cVec20);
      _mm256_storeu_ps(C + 2 * N + 1 * 8, cVec21);
      _mm256_storeu_ps(C + 2 * N + 2 * 8, cVec22);
      _mm256_storeu_ps(C + 2 * N + 3 * 8, cVec23);
      _mm256_storeu_ps(C + 3 * N + 0 * 8, cVec30);
      _mm256_storeu_ps(C + 3 * N + 1 * 8, cVec31);
      _mm256_storeu_ps(C + 3 * N + 2 * 8, cVec32);
      _mm256_storeu_ps(C + 3 * N + 3 * 8, cVec33);
      
      

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
    
    __ALWAYS_INLINE static void _microkernel_packed_max_acc(
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm256_load_ps(C + 0 * 8);
        cVec01 = _mm256_load_ps(C + 1 * 8);
        cVec02 = _mm256_load_ps(C + 2 * 8);
        cVec03 = _mm256_load_ps(C + 3 * 8);
        cVec10 = _mm256_load_ps(C + 4 * 8);
        cVec11 = _mm256_load_ps(C + 5 * 8);
        cVec12 = _mm256_load_ps(C + 6 * 8);
        cVec13 = _mm256_load_ps(C + 7 * 8);
        cVec20 = _mm256_load_ps(C + 8 * 8);
        cVec21 = _mm256_load_ps(C + 9 * 8);
        cVec22 = _mm256_load_ps(C + 10 * 8);
        cVec23 = _mm256_load_ps(C + 11 * 8);
        cVec30 = _mm256_load_ps(C + 12 * 8);
        cVec31 = _mm256_load_ps(C + 13 * 8);
        cVec32 = _mm256_load_ps(C + 14 * 8);
        cVec33 = _mm256_load_ps(C + 15 * 8);
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec01 = _mm256_setzero_ps();
        cVec02 = _mm256_setzero_ps();
        cVec03 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec11 = _mm256_setzero_ps();
        cVec12 = _mm256_setzero_ps();
        cVec13 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec21 = _mm256_setzero_ps();
        cVec22 = _mm256_setzero_ps();
        cVec23 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
        cVec31 = _mm256_setzero_ps();
        cVec32 = _mm256_setzero_ps();
        cVec33 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * (N_r) + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_load_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_load_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_load_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      _mm256_store_ps(C + 0 * 8, cVec00);
      _mm256_store_ps(C + 1 * 8, cVec01);
      _mm256_store_ps(C + 2 * 8, cVec02);
      _mm256_store_ps(C + 3 * 8, cVec03);
      _mm256_store_ps(C + 4 * 8, cVec10);
      _mm256_store_ps(C + 5 * 8, cVec11);
      _mm256_store_ps(C + 6 * 8, cVec12);
      _mm256_store_ps(C + 7 * 8, cVec13);
      _mm256_store_ps(C + 8 * 8, cVec20);
      _mm256_store_ps(C + 9 * 8, cVec21);
      _mm256_store_ps(C + 10 * 8, cVec22);
      _mm256_store_ps(C + 11 * 8, cVec23);
      _mm256_store_ps(C + 12 * 8, cVec30);
      _mm256_store_ps(C + 13 * 8, cVec31);
      _mm256_store_ps(C + 14 * 8, cVec32);
      _mm256_store_ps(C + 15 * 8, cVec33);
      
      

    }



    __ALWAYS_INLINE static void microkernel_packed_max_acc(
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_max_acc(
            nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_packed_C_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm256_load_ps(C + 0 * 8);
        cVec01 = _mm256_load_ps(C + 1 * 8);
        cVec02 = _mm256_load_ps(C + 2 * 8);
        cVec03 = _mm256_load_ps(C + 3 * 8);
        cVec10 = _mm256_load_ps(C + 4 * 8);
        cVec11 = _mm256_load_ps(C + 5 * 8);
        cVec12 = _mm256_load_ps(C + 6 * 8);
        cVec13 = _mm256_load_ps(C + 7 * 8);
        cVec20 = _mm256_load_ps(C + 8 * 8);
        cVec21 = _mm256_load_ps(C + 9 * 8);
        cVec22 = _mm256_load_ps(C + 10 * 8);
        cVec23 = _mm256_load_ps(C + 11 * 8);
        cVec30 = _mm256_load_ps(C + 12 * 8);
        cVec31 = _mm256_load_ps(C + 13 * 8);
        cVec32 = _mm256_load_ps(C + 14 * 8);
        cVec33 = _mm256_load_ps(C + 15 * 8);
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec01 = _mm256_setzero_ps();
        cVec02 = _mm256_setzero_ps();
        cVec03 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec11 = _mm256_setzero_ps();
        cVec12 = _mm256_setzero_ps();
        cVec13 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec21 = _mm256_setzero_ps();
        cVec22 = _mm256_setzero_ps();
        cVec23 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
        cVec31 = _mm256_setzero_ps();
        cVec32 = _mm256_setzero_ps();
        cVec33 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        __m256 bVec1 = _mm256_loadu_ps(B_curr + 1 * 8);
        __m256 bVec2 = _mm256_loadu_ps(B_curr + 2 * 8);
        __m256 bVec3 = _mm256_loadu_ps(B_curr + 3 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm256_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm256_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm256_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm256_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm256_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm256_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm256_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm256_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm256_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm256_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm256_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm256_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      _mm256_store_ps(C + 0 * 8, cVec00);
      _mm256_store_ps(C + 1 * 8, cVec01);
      _mm256_store_ps(C + 2 * 8, cVec02);
      _mm256_store_ps(C + 3 * 8, cVec03);
      _mm256_store_ps(C + 4 * 8, cVec10);
      _mm256_store_ps(C + 5 * 8, cVec11);
      _mm256_store_ps(C + 6 * 8, cVec12);
      _mm256_store_ps(C + 7 * 8, cVec13);
      _mm256_store_ps(C + 8 * 8, cVec20);
      _mm256_store_ps(C + 9 * 8, cVec21);
      _mm256_store_ps(C + 10 * 8, cVec22);
      _mm256_store_ps(C + 11 * 8, cVec23);
      _mm256_store_ps(C + 12 * 8, cVec30);
      _mm256_store_ps(C + 13 * 8, cVec31);
      _mm256_store_ps(C + 14 * 8, cVec32);
      _mm256_store_ps(C + 15 * 8, cVec33);
      
      

    }


    
    static void microkernel_packed_C_max_acc(
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
      
        _microkernel_packed_C_max_acc(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_1(
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
      __m256 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm256_loadu_ps(C_temp + 0 * 8);
        C_temp += N;
        cVec10 = _mm256_loadu_ps(C_temp + 0 * 8);
        C_temp += N;
        cVec20 = _mm256_loadu_ps(C_temp + 0 * 8);
        C_temp += N;
        cVec30 = _mm256_loadu_ps(C_temp + 0 * 8);
        C_temp += N;
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm256_storeu_ps(C + 0 * N + 0 * 8, cVec00);
      _mm256_storeu_ps(C + 1 * N + 0 * 8, cVec10);
      _mm256_storeu_ps(C + 2 * N + 0 * 8, cVec20);
      _mm256_storeu_ps(C + 3 * N + 0 * 8, cVec30);
      
      

    }



    __ALWAYS_INLINE static void microkernel_1(
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
      
        _microkernel_1(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_packed_1(
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm256_load_ps(C + 0 * 8);
        cVec10 = _mm256_load_ps(C + 4 * 8);
        cVec20 = _mm256_load_ps(C + 8 * 8);
        cVec30 = _mm256_load_ps(C + 12 * 8);
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * (N_r) + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_load_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm256_store_ps(C + 0 * 8, cVec00);
      _mm256_store_ps(C + 4 * 8, cVec10);
      _mm256_store_ps(C + 8 * 8, cVec20);
      _mm256_store_ps(C + 12 * 8, cVec30);
      
      

    }



    __ALWAYS_INLINE static void microkernel_packed_1(
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_1(
            nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_packed_C_1(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm256_load_ps(C + 0 * 8);
        cVec10 = _mm256_load_ps(C + 4 * 8);
        cVec20 = _mm256_load_ps(C + 8 * 8);
        cVec30 = _mm256_load_ps(C + 12 * 8);
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_loadu_ps(B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm256_store_ps(C + 0 * 8, cVec00);
      _mm256_store_ps(C + 4 * 8, cVec10);
      _mm256_store_ps(C + 8 * 8, cVec20);
      _mm256_store_ps(C + 12 * 8, cVec30);
      
      

    }


    
    static void microkernel_packed_C_1(
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
      
        _microkernel_packed_C_1(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_masked_1(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm256_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 8);
        C_temp += N;
        cVec10 = _mm256_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 8);
        C_temp += N;
        cVec20 = _mm256_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 8);
        C_temp += N;
        cVec30 = _mm256_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 8);
        C_temp += N;
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm256_mask_storeu_ps(C + 0 * N + 0 * 8, last_reg_mask, cVec00);
      _mm256_mask_storeu_ps(C + 1 * N + 0 * 8, last_reg_mask, cVec10);
      _mm256_mask_storeu_ps(C + 2 * N + 0 * 8, last_reg_mask, cVec20);
      _mm256_mask_storeu_ps(C + 3 * N + 0 * 8, last_reg_mask, cVec30);
      
      

    }


    
    __ALWAYS_INLINE static void _microkernel_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {

        int _j = 0;
        for (; _j < N_rem - 7; _j += 8) {
            _microkernel_1(
                M, K, N,
                nkern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }
        
        _microkernel_masked_1(
            M, K, N,
            nkern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }
    
     static void microkernel_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _microkernel_masked_max_acc(
            N_rem, M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }
    
    __ALWAYS_INLINE static void _microkernel_masked_packed_C_1(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m256 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm256_load_ps(C + 0 * 8);
        cVec10 = _mm256_load_ps(C + 4 * 8);
        cVec20 = _mm256_load_ps(C + 8 * 8);
        cVec30 = _mm256_load_ps(C + 12 * 8);
      } else {
        cVec00 = _mm256_setzero_ps();
        cVec10 = _mm256_setzero_ps();
        cVec20 = _mm256_setzero_ps();
        cVec30 = _mm256_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma unroll
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m256 aVec;
        __m256 bVec0 = _mm256_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 8);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm256_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm256_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm256_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm256_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm256_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm256_store_ps(C + 0 * 8, cVec00);
      _mm256_store_ps(C + 4 * 8, cVec10);
      _mm256_store_ps(C + 8 * 8, cVec20);
      _mm256_store_ps(C + 12 * 8, cVec30);
      
      

    }


    
    static void _microkernel_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {

        int _j = 0;
        for (; _j < N_rem - 7; _j += 8) {
            _microkernel_packed_C_1(
                M, K, N,
                nkern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }
        
        _microkernel_masked_packed_C_1(
            M, K, N,
            nkern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }
    
    static void microkernel_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _microkernel_masked_packed_C_max_acc(
            N_rem, M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }
    
};

} // sop
