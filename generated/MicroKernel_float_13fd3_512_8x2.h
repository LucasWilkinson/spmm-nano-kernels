#pragma once

#include "MicroKernelBase.h"
#include "Storage.h"

#include <immintrin.h>


namespace sop {
struct MicroKernel_float_13fd3_512_8x2 {

    static const uint16_t* supported_nkern_patterns() {
        static uint16_t patterns[] = {
            0b00000001,
            0b00000010,
            0b00000100,
            0b00001000,
            0b00010000,
            0b00100000,
            0b01000000,
            0b10000000,
            0b01010101,
            0b10101010,
            0b11000011,
            0b00111100,
            0b00001111,
            0b11110000,
            0b11111100,
            0b11110011,
            0b11001111,
            0b00111111,
            0b11111111
        };
    
        return patterns;
    }
    
    static uint16_t encode_nkern_pattern(uint16_t nkern_pat) {
        if (nkern_code == 0b00000001) return 0;
        if (nkern_code == 0b00000010) return 1;
        if (nkern_code == 0b00000100) return 2;
        if (nkern_code == 0b00001000) return 3;
        if (nkern_code == 0b00010000) return 4;
        if (nkern_code == 0b00100000) return 5;
        if (nkern_code == 0b01000000) return 6;
        if (nkern_code == 0b10000000) return 7;
        if (nkern_code == 0b01010101) return 8;
        if (nkern_code == 0b10101010) return 9;
        if (nkern_code == 0b11000011) return 10;
        if (nkern_code == 0b00111100) return 11;
        if (nkern_code == 0b00001111) return 12;
        if (nkern_code == 0b11110000) return 13;
        if (nkern_code == 0b11111100) return 14;
        if (nkern_code == 0b11110011) return 15;
        if (nkern_code == 0b11001111) return 16;
        if (nkern_code == 0b00111111) return 17;
        if (nkern_code == 0b11111111) return 18;
        if (nkern_pat == 0) return sop::ZERO_PATTERN_ID; 
        std::cerr << "Unable to map unsupported nanokernel pattern " <<  (int) nkern_pat << std::endl;
        exit(-1);
        return 0;
    }
    
    static uint16_t decode_nkern_pattern(uint16_t nkern_pat) {
        if (nkern_code == 0) return 0b00000001;
        if (nkern_code == 1) return 0b00000010;
        if (nkern_code == 2) return 0b00000100;
        if (nkern_code == 3) return 0b00001000;
        if (nkern_code == 4) return 0b00010000;
        if (nkern_code == 5) return 0b00100000;
        if (nkern_code == 6) return 0b01000000;
        if (nkern_code == 7) return 0b10000000;
        if (nkern_code == 8) return 0b01010101;
        if (nkern_code == 9) return 0b10101010;
        if (nkern_code == 10) return 0b11000011;
        if (nkern_code == 11) return 0b00111100;
        if (nkern_code == 12) return 0b00001111;
        if (nkern_code == 13) return 0b11110000;
        if (nkern_code == 14) return 0b11111100;
        if (nkern_code == 15) return 0b11110011;
        if (nkern_code == 16) return 0b11001111;
        if (nkern_code == 17) return 0b00111111;
        if (nkern_code == 18) return 0b11111111;
        if (nkern_pat == sop::ZERO_PATTERN_ID) return 0; 
        std::cerr << "Unable to unmap unsupported nanokernel pattern id " << (int) nkern_pat << std::endl;
        exit(-1);
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
        std::cerr << "Unable to get pop count for nanokernel code " << (int) nkern_code << std::endl;
        exit(-1);
        return 0;
    }
    
    using Mask = __mmask16;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % 16); }

    using Scalar = float;
    static constexpr int M_r = 8;
    static constexpr int N_r = 2 * 16;
    static constexpr int N_r_reg = 2;
    static constexpr int vec_width_bits = 512;
    static constexpr const char* id = "13fd3_512_8x2";
    static int max_acc_width_in_vecs() { return 2; };
    static int max_acc_width_in_eles() { return 2 * 16; };

    static int num_nkern_patterns() { return 19; }

    __ALWAYS_INLINE static void _microkernel_2(
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
      __m512 cVec00, cVec01, cVec10, cVec11, cVec20, cVec21, cVec30, cVec31, cVec40, cVec41, cVec50, cVec51, cVec60, cVec61, cVec70, cVec71;
      if (load_c) {
        cVec00 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec01 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec10 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec11 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec20 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec21 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec30 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec31 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec40 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec41 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec50 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec51 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec60 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec61 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
        cVec70 = _mm512_loadu_ps(C_temp + 0 * 16);
        cVec71 = _mm512_loadu_ps(C_temp + 1 * 16);
        C_temp += N;
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec01 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec11 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec21 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec31 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec41 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec51 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec61 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
        cVec71 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      _mm512_storeu_ps(C + 0 * N + 0 * 16, cVec00);
      _mm512_storeu_ps(C + 0 * N + 1 * 16, cVec01);
      _mm512_storeu_ps(C + 1 * N + 0 * 16, cVec10);
      _mm512_storeu_ps(C + 1 * N + 1 * 16, cVec11);
      _mm512_storeu_ps(C + 2 * N + 0 * 16, cVec20);
      _mm512_storeu_ps(C + 2 * N + 1 * 16, cVec21);
      _mm512_storeu_ps(C + 3 * N + 0 * 16, cVec30);
      _mm512_storeu_ps(C + 3 * N + 1 * 16, cVec31);
      _mm512_storeu_ps(C + 4 * N + 0 * 16, cVec40);
      _mm512_storeu_ps(C + 4 * N + 1 * 16, cVec41);
      _mm512_storeu_ps(C + 5 * N + 0 * 16, cVec50);
      _mm512_storeu_ps(C + 5 * N + 1 * 16, cVec51);
      _mm512_storeu_ps(C + 6 * N + 0 * 16, cVec60);
      _mm512_storeu_ps(C + 6 * N + 1 * 16, cVec61);
      _mm512_storeu_ps(C + 7 * N + 0 * 16, cVec70);
      _mm512_storeu_ps(C + 7 * N + 1 * 16, cVec71);
      
      

    }



    __ALWAYS_INLINE static void microkernel_2(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_2(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_packed_2(
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C_temp = C;
      __m512 cVec00, cVec01, cVec10, cVec11, cVec20, cVec21, cVec30, cVec31, cVec40, cVec41, cVec50, cVec51, cVec60, cVec61, cVec70, cVec71;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec01 = _mm512_load_ps(C + 1 * 16);
        cVec10 = _mm512_load_ps(C + 2 * 16);
        cVec11 = _mm512_load_ps(C + 3 * 16);
        cVec20 = _mm512_load_ps(C + 4 * 16);
        cVec21 = _mm512_load_ps(C + 5 * 16);
        cVec30 = _mm512_load_ps(C + 6 * 16);
        cVec31 = _mm512_load_ps(C + 7 * 16);
        cVec40 = _mm512_load_ps(C + 8 * 16);
        cVec41 = _mm512_load_ps(C + 9 * 16);
        cVec50 = _mm512_load_ps(C + 10 * 16);
        cVec51 = _mm512_load_ps(C + 11 * 16);
        cVec60 = _mm512_load_ps(C + 12 * 16);
        cVec61 = _mm512_load_ps(C + 13 * 16);
        cVec70 = _mm512_load_ps(C + 14 * 16);
        cVec71 = _mm512_load_ps(C + 15 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec01 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec11 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec21 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec31 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec41 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec51 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec61 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
        cVec71 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * (N_r) + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_load_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 1 * 16, cVec01);
      _mm512_store_ps(C + 2 * 16, cVec10);
      _mm512_store_ps(C + 3 * 16, cVec11);
      _mm512_store_ps(C + 4 * 16, cVec20);
      _mm512_store_ps(C + 5 * 16, cVec21);
      _mm512_store_ps(C + 6 * 16, cVec30);
      _mm512_store_ps(C + 7 * 16, cVec31);
      _mm512_store_ps(C + 8 * 16, cVec40);
      _mm512_store_ps(C + 9 * 16, cVec41);
      _mm512_store_ps(C + 10 * 16, cVec50);
      _mm512_store_ps(C + 11 * 16, cVec51);
      _mm512_store_ps(C + 12 * 16, cVec60);
      _mm512_store_ps(C + 13 * 16, cVec61);
      _mm512_store_ps(C + 14 * 16, cVec70);
      _mm512_store_ps(C + 15 * 16, cVec71);
      
      

    }



    __ALWAYS_INLINE static void microkernel_packed_2(
        const sop::& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_2(
            nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
    __ALWAYS_INLINE static void _microkernel_packed_C_2(
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
      __m512 cVec00, cVec01, cVec10, cVec11, cVec20, cVec21, cVec30, cVec31, cVec40, cVec41, cVec50, cVec51, cVec60, cVec61, cVec70, cVec71;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec01 = _mm512_load_ps(C + 1 * 16);
        cVec10 = _mm512_load_ps(C + 2 * 16);
        cVec11 = _mm512_load_ps(C + 3 * 16);
        cVec20 = _mm512_load_ps(C + 4 * 16);
        cVec21 = _mm512_load_ps(C + 5 * 16);
        cVec30 = _mm512_load_ps(C + 6 * 16);
        cVec31 = _mm512_load_ps(C + 7 * 16);
        cVec40 = _mm512_load_ps(C + 8 * 16);
        cVec41 = _mm512_load_ps(C + 9 * 16);
        cVec50 = _mm512_load_ps(C + 10 * 16);
        cVec51 = _mm512_load_ps(C + 11 * 16);
        cVec60 = _mm512_load_ps(C + 12 * 16);
        cVec61 = _mm512_load_ps(C + 13 * 16);
        cVec70 = _mm512_load_ps(C + 14 * 16);
        cVec71 = _mm512_load_ps(C + 15 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec01 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec11 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec21 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec31 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec41 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec51 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec61 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
        cVec71 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        __m512 bVec1 = _mm512_loadu_ps(B_curr + 1 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        cVec01 = _mm512_fmadd_ps(aVec, bVec1, cVec01);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        cVec11 = _mm512_fmadd_ps(aVec, bVec1, cVec11);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        cVec21 = _mm512_fmadd_ps(aVec, bVec1, cVec21);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        cVec31 = _mm512_fmadd_ps(aVec, bVec1, cVec31);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        cVec41 = _mm512_fmadd_ps(aVec, bVec1, cVec41);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        cVec51 = _mm512_fmadd_ps(aVec, bVec1, cVec51);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        cVec61 = _mm512_fmadd_ps(aVec, bVec1, cVec61);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);
        cVec71 = _mm512_fmadd_ps(aVec, bVec1, cVec71);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 1 * 16, cVec01);
      _mm512_store_ps(C + 2 * 16, cVec10);
      _mm512_store_ps(C + 3 * 16, cVec11);
      _mm512_store_ps(C + 4 * 16, cVec20);
      _mm512_store_ps(C + 5 * 16, cVec21);
      _mm512_store_ps(C + 6 * 16, cVec30);
      _mm512_store_ps(C + 7 * 16, cVec31);
      _mm512_store_ps(C + 8 * 16, cVec40);
      _mm512_store_ps(C + 9 * 16, cVec41);
      _mm512_store_ps(C + 10 * 16, cVec50);
      _mm512_store_ps(C + 11 * 16, cVec51);
      _mm512_store_ps(C + 12 * 16, cVec60);
      _mm512_store_ps(C + 13 * 16, cVec61);
      _mm512_store_ps(C + 14 * 16, cVec70);
      _mm512_store_ps(C + 15 * 16, cVec71);
      
      

    }


    
    static void microkernel_packed_C_2(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_C_2(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }
    
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
      __m512 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec10 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec20 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec30 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec40 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec50 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec60 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
        cVec70 = _mm512_loadu_ps(C_temp + 0 * 16);
        C_temp += N;
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
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

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      _mm512_storeu_ps(C + 0 * N + 0 * 16, cVec00);
      _mm512_storeu_ps(C + 1 * N + 0 * 16, cVec10);
      _mm512_storeu_ps(C + 2 * N + 0 * 16, cVec20);
      _mm512_storeu_ps(C + 3 * N + 0 * 16, cVec30);
      _mm512_storeu_ps(C + 4 * N + 0 * 16, cVec40);
      _mm512_storeu_ps(C + 5 * N + 0 * 16, cVec50);
      _mm512_storeu_ps(C + 6 * N + 0 * 16, cVec60);
      _mm512_storeu_ps(C + 7 * N + 0 * 16, cVec70);
      
      

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
        int                     num_patterns = panel_desc.num_patterns;
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
      __m512 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec10 = _mm512_load_ps(C + 2 * 16);
        cVec20 = _mm512_load_ps(C + 4 * 16);
        cVec30 = _mm512_load_ps(C + 6 * 16);
        cVec40 = _mm512_load_ps(C + 8 * 16);
        cVec50 = _mm512_load_ps(C + 10 * 16);
        cVec60 = _mm512_load_ps(C + 12 * 16);
        cVec70 = _mm512_load_ps(C + 14 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * (N_r) + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
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

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_load_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 2 * 16, cVec10);
      _mm512_store_ps(C + 4 * 16, cVec20);
      _mm512_store_ps(C + 6 * 16, cVec30);
      _mm512_store_ps(C + 8 * 16, cVec40);
      _mm512_store_ps(C + 10 * 16, cVec50);
      _mm512_store_ps(C + 12 * 16, cVec60);
      _mm512_store_ps(C + 14 * 16, cVec70);
      
      

    }



    __ALWAYS_INLINE static void microkernel_packed_max_acc(
        const sop::& panel_desc,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c) {
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_patterns = panel_desc.num_patterns;
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
      __m512 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec10 = _mm512_load_ps(C + 2 * 16);
        cVec20 = _mm512_load_ps(C + 4 * 16);
        cVec30 = _mm512_load_ps(C + 6 * 16);
        cVec40 = _mm512_load_ps(C + 8 * 16);
        cVec50 = _mm512_load_ps(C + 10 * 16);
        cVec60 = _mm512_load_ps(C + 12 * 16);
        cVec70 = _mm512_load_ps(C + 14 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
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

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_loadu_ps(B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 2 * 16, cVec10);
      _mm512_store_ps(C + 4 * 16, cVec20);
      _mm512_store_ps(C + 6 * 16, cVec30);
      _mm512_store_ps(C + 8 * 16, cVec40);
      _mm512_store_ps(C + 10 * 16, cVec50);
      _mm512_store_ps(C + 12 * 16, cVec60);
      _mm512_store_ps(C + 14 * 16, cVec70);
      
      

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
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_C_max_acc(
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
      __m512 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec10 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec20 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec30 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec40 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec50 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec60 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
        cVec70 = _mm512_maskz_loadu_ps(last_reg_mask, C_temp + 0 * 16);
        C_temp += N;
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
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

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      _mm512_mask_storeu_ps(C + 0 * N + 0 * 16, last_reg_mask, cVec00);
      _mm512_mask_storeu_ps(C + 1 * N + 0 * 16, last_reg_mask, cVec10);
      _mm512_mask_storeu_ps(C + 2 * N + 0 * 16, last_reg_mask, cVec20);
      _mm512_mask_storeu_ps(C + 3 * N + 0 * 16, last_reg_mask, cVec30);
      _mm512_mask_storeu_ps(C + 4 * N + 0 * 16, last_reg_mask, cVec40);
      _mm512_mask_storeu_ps(C + 5 * N + 0 * 16, last_reg_mask, cVec50);
      _mm512_mask_storeu_ps(C + 6 * N + 0 * 16, last_reg_mask, cVec60);
      _mm512_mask_storeu_ps(C + 7 * N + 0 * 16, last_reg_mask, cVec70);
      
      

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
        for (; _j < N_rem - 15; _j += 16) {
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
        int                     num_patterns = panel_desc.num_patterns;
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
      __m512 cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
      if (load_c) {
        cVec00 = _mm512_load_ps(C + 0 * 16);
        cVec10 = _mm512_load_ps(C + 2 * 16);
        cVec20 = _mm512_load_ps(C + 4 * 16);
        cVec30 = _mm512_load_ps(C + 6 * 16);
        cVec40 = _mm512_load_ps(C + 8 * 16);
        cVec50 = _mm512_load_ps(C + 10 * 16);
        cVec60 = _mm512_load_ps(C + 12 * 16);
        cVec70 = _mm512_load_ps(C + 14 * 16);
      } else {
        cVec00 = _mm512_setzero_ps();
        cVec10 = _mm512_setzero_ps();
        cVec20 = _mm512_setzero_ps();
        cVec30 = _mm512_setzero_ps();
        cVec40 = _mm512_setzero_ps();
        cVec50 = _mm512_setzero_ps();
        cVec60 = _mm512_setzero_ps();
        cVec70 = _mm512_setzero_ps();
      }
      
      int c_idx = 0;
      auto curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
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

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec20 = _mm512_fmadd_ps(aVec, bVec0, cVec20);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec30 = _mm512_fmadd_ps(aVec, bVec0, cVec30);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        __m512 aVec;
        __m512 bVec0 = _mm512_maskz_loadu_ps(last_reg_mask, B_curr + 0 * 16);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec00 = _mm512_fmadd_ps(aVec, bVec0, cVec00);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec10 = _mm512_fmadd_ps(aVec, bVec0, cVec10);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);

      }

      
      #pragma GCC unroll 2
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
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
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec40 = _mm512_fmadd_ps(aVec, bVec0, cVec40);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec50 = _mm512_fmadd_ps(aVec, bVec0, cVec50);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec60 = _mm512_fmadd_ps(aVec, bVec0, cVec60);
        aVec = _mm512_set1_ps(*curr_value_ptr); curr_value_ptr++;
        cVec70 = _mm512_fmadd_ps(aVec, bVec0, cVec70);

      }

      
      _mm512_store_ps(C + 0 * 16, cVec00);
      _mm512_store_ps(C + 2 * 16, cVec10);
      _mm512_store_ps(C + 4 * 16, cVec20);
      _mm512_store_ps(C + 6 * 16, cVec30);
      _mm512_store_ps(C + 8 * 16, cVec40);
      _mm512_store_ps(C + 10 * 16, cVec50);
      _mm512_store_ps(C + 12 * 16, cVec60);
      _mm512_store_ps(C + 14 * 16, cVec70);
      
      

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
        for (; _j < N_rem - 15; _j += 16) {
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
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _microkernel_masked_packed_C_max_acc(
            N_rem, M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }
    
};

} // sop
