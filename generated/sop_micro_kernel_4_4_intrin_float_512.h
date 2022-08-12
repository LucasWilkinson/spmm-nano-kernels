#pragma once

#include "SOPMicroKernelBase.h"
#include "SOPStorage.h"

#include <immintrin.h>


namespace sop {

template<> struct SOPMicroKernelIntrin<float, 512, 4, 4> {

    static const uint16_t* supported_patterns() {
        static uint16_t patterns[] = {
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00001110,
            0b00001101,
            0b00001011,
            0b00000111,
            0b00001111
        };
    
        return patterns;
    }
    
    static uint16_t encode_pattern(uint16_t pattern) {
        if (pattern == 0b00000001) return 0;
        if (pattern == 0b00000010) return 1;
        if (pattern == 0b00000100) return 2;
        if (pattern == 0b00001000) return 3;
        if (pattern == 0b00001110) return 4;
        if (pattern == 0b00001101) return 5;
        if (pattern == 0b00001011) return 6;
        if (pattern == 0b00000111) return 7;
        if (pattern == 0b00001111) return 8;
        if (pattern == 0) return ZERO_PATTERN_ID; 
        std::cerr << "Unable to map unsupported pattern " <<  (int) pattern << std::endl;
        exit(-1);
        return 0;
    }
    
    static uint16_t decode_pattern(uint16_t pattern) {
        if (pattern == 0) return 0b00000001;
        if (pattern == 1) return 0b00000010;
        if (pattern == 2) return 0b00000100;
        if (pattern == 3) return 0b00001000;
        if (pattern == 4) return 0b00001110;
        if (pattern == 5) return 0b00001101;
        if (pattern == 6) return 0b00001011;
        if (pattern == 7) return 0b00000111;
        if (pattern == 8) return 0b00001111;
        if (pattern == ZERO_PATTERN_ID) return 0; 
        std::cerr << "Unable to unmap unsupported pattern id " << (int) pattern << std::endl;
        exit(-1);
        return 0;
    }
    
    static uint16_t nnz_count(uint16_t pattern) {
        if (pattern == 0) return 1;
        if (pattern == 1) return 1;
        if (pattern == 2) return 1;
        if (pattern == 3) return 1;
        if (pattern == 4) return 3;
        if (pattern == 5) return 3;
        if (pattern == 6) return 3;
        if (pattern == 7) return 3;
        if (pattern == 8) return 4;
        if (pattern == ZERO_PATTERN_ID) return 0; 
        std::cerr << "Unable to get pop count for pattern id " << (int) pattern << std::endl;
        exit(-1);
        return 0;
    }
    
    using Mask = __mmask16;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % N_r); }

    static const int M_r = 4;
    static const int N_r = 4 * 16;
    static int max_acc_width_in_vecs() { return 4; };
    static int max_acc_width_in_eles() { return 4 * 16; };

    static int number_of_patterns() { return 9; }
    static int panel_height() { return 4; }


    __ALWAYS_INLINE static void _panel_executor_max_acc(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec01 = _mm512_loadu_ps(C_temp + 1 * 16);
        cVec02 = _mm512_loadu_ps(C_temp + 2 * 16);
        cVec03 = _mm512_loadu_ps(C_temp + 3 * 16);
        C_temp += N;
        cVec10 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec11 = _mm512_loadu_ps(C_temp + 1 * 16);
        cVec12 = _mm512_loadu_ps(C_temp + 2 * 16);
        cVec13 = _mm512_loadu_ps(C_temp + 3 * 16);
        C_temp += N;
        cVec20 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec21 = _mm512_loadu_ps(C_temp + 1 * 16);
        cVec22 = _mm512_loadu_ps(C_temp + 2 * 16);
        cVec23 = _mm512_loadu_ps(C_temp + 3 * 16);
        C_temp += N;
        cVec30 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec31 = _mm512_loadu_ps(C_temp + 1 * 16);
        cVec32 = _mm512_loadu_ps(C_temp + 2 * 16);
        cVec33 = _mm512_loadu_ps(C_temp + 3 * 16);
        C_temp += N;
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec01 = _mm512_setzero_ps();
        cVec02 = _mm512_setzero_ps();
        cVec03 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec11 = _mm512_setzero_ps();
        cVec12 = _mm512_setzero_ps();
        cVec13 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec21 = _mm512_setzero_ps();
        cVec22 = _mm512_setzero_ps();
        cVec23 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec31 = _mm512_setzero_ps();
        cVec32 = _mm512_setzero_ps();
        cVec33 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      _mm512_storeu_ps(C + 0 * N + 0 * 16, cVec00);
      _mm512_storeu_ps(C + 0 * N + 1 * 16, cVec01);
      _mm512_storeu_ps(C + 0 * N + 2 * 16, cVec02);
      _mm512_storeu_ps(C + 0 * N + 3 * 16, cVec03);
      _mm512_storeu_ps(C + 1 * N + 0 * 16, cVec10);
      _mm512_storeu_ps(C + 1 * N + 1 * 16, cVec11);
      _mm512_storeu_ps(C + 1 * N + 2 * 16, cVec12);
      _mm512_storeu_ps(C + 1 * N + 3 * 16, cVec13);
      _mm512_storeu_ps(C + 2 * N + 0 * 16, cVec20);
      _mm512_storeu_ps(C + 2 * N + 1 * 16, cVec21);
      _mm512_storeu_ps(C + 2 * N + 2 * 16, cVec22);
      _mm512_storeu_ps(C + 2 * N + 3 * 16, cVec23);
      _mm512_storeu_ps(C + 3 * N + 0 * 16, cVec30);
      _mm512_storeu_ps(C + 3 * N + 1 * 16, cVec31);
      _mm512_storeu_ps(C + 3 * N + 2 * 16, cVec32);
      _mm512_storeu_ps(C + 3 * N + 3 * 16, cVec33);
      
      

    }



    __ALWAYS_INLINE static void panel_executor_max_acc(
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_max_acc(
            M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    #ifdef ENABLE_PACKED_KERNELS
    __ALWAYS_INLINE static void _panel_executor_packed_max_acc(
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec01 = _mm512_load_ps(C + 1 * 16);
        cVec02 = _mm512_load_ps(C + 2 * 16);
        cVec03 = _mm512_load_ps(C + 3 * 16);
        cVec10 = _mm512_load_ps(C + 4 * 16);
        cVec11 = _mm512_load_ps(C + 5 * 16);
        cVec12 = _mm512_load_ps(C + 6 * 16);
        cVec13 = _mm512_load_ps(C + 7 * 16);
        cVec20 = _mm512_load_ps(C + 8 * 16);
        cVec21 = _mm512_load_ps(C + 9 * 16);
        cVec22 = _mm512_load_ps(C + 10 * 16);
        cVec23 = _mm512_load_ps(C + 11 * 16);
        cVec30 = _mm512_load_ps(C + 12 * 16);
        cVec31 = _mm512_load_ps(C + 13 * 16);
        cVec32 = _mm512_load_ps(C + 14 * 16);
        cVec33 = _mm512_load_ps(C + 15 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec01 = _mm512_setzero_ps();
        cVec02 = _mm512_setzero_ps();
        cVec03 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec11 = _mm512_setzero_ps();
        cVec12 = _mm512_setzero_ps();
        cVec13 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec21 = _mm512_setzero_ps();
        cVec22 = _mm512_setzero_ps();
        cVec23 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec31 = _mm512_setzero_ps();
        cVec32 = _mm512_setzero_ps();
        cVec33 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * (N_r) + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_load_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_load_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 1 * 16, cVec01);
      _mm512_store_ps(C + 2 * 16, cVec02);
      _mm512_store_ps(C + 3 * 16, cVec03);
      _mm512_store_ps(C + 4 * 16, cVec10);
      _mm512_store_ps(C + 5 * 16, cVec11);
      _mm512_store_ps(C + 6 * 16, cVec12);
      _mm512_store_ps(C + 7 * 16, cVec13);
      _mm512_store_ps(C + 8 * 16, cVec20);
      _mm512_store_ps(C + 9 * 16, cVec21);
      _mm512_store_ps(C + 10 * 16, cVec22);
      _mm512_store_ps(C + 11 * 16, cVec23);
      _mm512_store_ps(C + 12 * 16, cVec30);
      _mm512_store_ps(C + 13 * 16, cVec31);
      _mm512_store_ps(C + 14 * 16, cVec32);
      _mm512_store_ps(C + 15 * 16, cVec33);
      
      

    }



    __ALWAYS_INLINE static void panel_executor_packed_max_acc(
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_packed_max_acc(
            pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    #endif
    
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_packed_C_max_acc(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec01, cVec02, cVec03, cVec10, cVec11, cVec12, cVec13, cVec20, cVec21, cVec22, cVec23, cVec30, cVec31, cVec32, cVec33;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec01 = _mm512_load_ps(C + 1 * 16);
        cVec02 = _mm512_load_ps(C + 2 * 16);
        cVec03 = _mm512_load_ps(C + 3 * 16);
        cVec10 = _mm512_load_ps(C + 4 * 16);
        cVec11 = _mm512_load_ps(C + 5 * 16);
        cVec12 = _mm512_load_ps(C + 6 * 16);
        cVec13 = _mm512_load_ps(C + 7 * 16);
        cVec20 = _mm512_load_ps(C + 8 * 16);
        cVec21 = _mm512_load_ps(C + 9 * 16);
        cVec22 = _mm512_load_ps(C + 10 * 16);
        cVec23 = _mm512_load_ps(C + 11 * 16);
        cVec30 = _mm512_load_ps(C + 12 * 16);
        cVec31 = _mm512_load_ps(C + 13 * 16);
        cVec32 = _mm512_load_ps(C + 14 * 16);
        cVec33 = _mm512_load_ps(C + 15 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec01 = _mm512_setzero_ps();
        cVec02 = _mm512_setzero_ps();
        cVec03 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec11 = _mm512_setzero_ps();
        cVec12 = _mm512_setzero_ps();
        cVec13 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec21 = _mm512_setzero_ps();
        cVec22 = _mm512_setzero_ps();
        cVec23 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec31 = _mm512_setzero_ps();
        cVec32 = _mm512_setzero_ps();
        cVec33 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        __m512 bVec2 = _mm512_loadu_ps(B_curr + 2 * 16);
        __m512 bVec3 = _mm512_loadu_ps(B_curr + 3 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        cVec02 = _mm512_fmadd_ps(aVec, bVec2, cVec02);
        cVec03 = _mm512_fmadd_ps(aVec, bVec3, cVec03);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        cVec12 = _mm512_fmadd_ps(aVec, bVec2, cVec12);
        cVec13 = _mm512_fmadd_ps(aVec, bVec3, cVec13);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        cVec22 = _mm512_fmadd_ps(aVec, bVec2, cVec22);
        cVec23 = _mm512_fmadd_ps(aVec, bVec3, cVec23);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        cVec32 = _mm512_fmadd_ps(aVec, bVec2, cVec32);
        cVec33 = _mm512_fmadd_ps(aVec, bVec3, cVec33);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 1 * 16, cVec01);
      _mm512_store_ps(C + 2 * 16, cVec02);
      _mm512_store_ps(C + 3 * 16, cVec03);
      _mm512_store_ps(C + 4 * 16, cVec10);
      _mm512_store_ps(C + 5 * 16, cVec11);
      _mm512_store_ps(C + 6 * 16, cVec12);
      _mm512_store_ps(C + 7 * 16, cVec13);
      _mm512_store_ps(C + 8 * 16, cVec20);
      _mm512_store_ps(C + 9 * 16, cVec21);
      _mm512_store_ps(C + 10 * 16, cVec22);
      _mm512_store_ps(C + 11 * 16, cVec23);
      _mm512_store_ps(C + 12 * 16, cVec30);
      _mm512_store_ps(C + 13 * 16, cVec31);
      _mm512_store_ps(C + 14 * 16, cVec32);
      _mm512_store_ps(C + 15 * 16, cVec33);
      
      

    }


    
    __ALWAYS_INLINE static void panel_executor_packed_C_max_acc(
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_packed_C_max_acc(
            M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    #endif
    
    __ALWAYS_INLINE static void _panel_executor_1(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec10 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec20 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec30 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm512_storeu_ps(C + 0 * N + 0 * 16, cVec00);
      _mm512_storeu_ps(C + 1 * N + 0 * 16, cVec10);
      _mm512_storeu_ps(C + 2 * N + 0 * 16, cVec20);
      _mm512_storeu_ps(C + 3 * N + 0 * 16, cVec30);
      
      

    }



    __ALWAYS_INLINE static void panel_executor_1(
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_1(
            M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    #ifdef ENABLE_PACKED_KERNELS
    __ALWAYS_INLINE static void _panel_executor_packed_1(
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec10 = _mm512_load_ps(C + 4 * 16);
        cVec20 = _mm512_load_ps(C + 8 * 16);
        cVec30 = _mm512_load_ps(C + 12 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * (N_r) + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 4 * 16, cVec10);
      _mm512_store_ps(C + 8 * 16, cVec20);
      _mm512_store_ps(C + 12 * 16, cVec30);
      
      

    }



    __ALWAYS_INLINE static void panel_executor_packed_1(
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_packed_1(
            pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    #endif
    
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_packed_C_1(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec10 = _mm512_load_ps(C + 4 * 16);
        cVec20 = _mm512_load_ps(C + 8 * 16);
        cVec30 = _mm512_load_ps(C + 12 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 4 * 16, cVec10);
      _mm512_store_ps(C + 8 * 16, cVec20);
      _mm512_store_ps(C + 12 * 16, cVec30);
      
      

    }


    
    __ALWAYS_INLINE static void panel_executor_packed_C_1(
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_packed_C_1(
            M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    #endif
    
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_masked_1(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec10 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec20 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec30 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm512_mask_storeu_ps(C + 0 * N + 0 * 16, last_reg_mask, cVec00);
      _mm512_mask_storeu_ps(C + 1 * N + 0 * 16, last_reg_mask, cVec10);
      _mm512_mask_storeu_ps(C + 2 * N + 0 * 16, last_reg_mask, cVec20);
      _mm512_mask_storeu_ps(C + 3 * N + 0 * 16, last_reg_mask, cVec30);
      
      

    }


    
    static void _panel_executor_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {

        int _j = 0;
        for (; _j < N_rem - 15; _j += 16) {
            _panel_executor_1(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }
        
        _panel_executor_masked_1(
            M, K, N,
            pattern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }
    
    static void panel_executor_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _panel_executor_masked_max_acc(
            N_rem, M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }
    
    #endif
    
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_masked_packed_C_1(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec10, cVec20, cVec30;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec10 = _mm512_load_ps(C + 4 * 16);
        cVec20 = _mm512_load_ps(C + 8 * 16);
        cVec30 = _mm512_load_ps(C + 12 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = pattern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 4 * 16, cVec10);
      _mm512_store_ps(C + 8 * 16, cVec20);
      _mm512_store_ps(C + 12 * 16, cVec30);
      
      

    }


    
    static void _panel_executor_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {

        int _j = 0;
        for (; _j < N_rem - 15; _j += 16) {
            _panel_executor_packed_C_1(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }
        
        _panel_executor_masked_packed_C_1(
            M, K, N,
            pattern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }
    
    static void panel_executor_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _panel_executor_masked_packed_C_max_acc(
            N_rem, M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }
    
    #endif
    
    static void panel_executor_max_acc_width_N_c(
        int N_c,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
    
        for (int _j = 0; _j < N_c; _j += N_r) {
            _panel_executor_max_acc(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }
    }


    static void panel_executor_cleanup_N_c(
        int N_c_rem,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        Mask mask, const bool load_c)
    {

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
    
        int _j = 0;
        int end_of_full_blocks = (N_c_rem / N_r) * N_r;
        int end_of_partial_blocks = ((N_c_rem - end_of_full_blocks) / N_r) * N_r;
        
        if (end_of_full_blocks) {
            panel_executor_max_acc_width_N_c(
                end_of_full_blocks,
                M, K, N,
                panel_desc,
                B, C,
                load_c);
        }

        for (_j = end_of_full_blocks; _j < end_of_partial_blocks; _j += 16) {
            _panel_executor_1(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }
        
        _panel_executor_masked_1(
            M, K, N,
            pattern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            mask, load_c);
    }


};

} // namespace sop
