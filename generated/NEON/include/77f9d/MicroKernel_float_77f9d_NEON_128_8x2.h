#pragma once

#include "utils/error.h"
#include "MicroKernelBase.h"
#include "Storage.h"

#include <arm_neon.h>


#include "intrin_compatability.h"

namespace sop {
struct MicroKernel_float_77f9d_NEON_128_8x2 {

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

    using Scalar = float;
    static constexpr int M_r = 8;
    static constexpr int N_r = 2 * 4;
    static constexpr int N_r_reg = 2;
    static constexpr int vec_width_bits = 128;
    static constexpr const char* id = "77f9d_NEON_128_8x2";
    static int max_acc_width_in_vecs() { return 2; };
    static int max_acc_width_in_eles() { return 2 * 4; };

    static int num_nkern_patterns() { return 19; }

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
      
      float* C0 = C + 0 * N;
      float* C1 = C + 1 * N;
      float* C2 = C + 2 * N;
      float* C3 = C + 3 * N;
      float* C4 = C + 4 * N;
      float* C5 = C + 5 * N;
      float* C6 = C + 6 * N;
      float* C7 = C + 7 * N;
      float32x4_t cVec00, cVec01, cVec10, cVec11, cVec20, cVec21, cVec30, cVec31, cVec40, cVec41, cVec50, cVec51, cVec60, cVec61, cVec70, cVec71;
      if (load_c) {
        cVec00 = vld1q_f32(C0 + 0 * 4);
        cVec01 = vld1q_f32(C0 + 1 * 4);
        cVec10 = vld1q_f32(C1 + 0 * 4);
        cVec11 = vld1q_f32(C1 + 1 * 4);
        cVec20 = vld1q_f32(C2 + 0 * 4);
        cVec21 = vld1q_f32(C2 + 1 * 4);
        cVec30 = vld1q_f32(C3 + 0 * 4);
        cVec31 = vld1q_f32(C3 + 1 * 4);
        cVec40 = vld1q_f32(C4 + 0 * 4);
        cVec41 = vld1q_f32(C4 + 1 * 4);
        cVec50 = vld1q_f32(C5 + 0 * 4);
        cVec51 = vld1q_f32(C5 + 1 * 4);
        cVec60 = vld1q_f32(C6 + 0 * 4);
        cVec61 = vld1q_f32(C6 + 1 * 4);
        cVec70 = vld1q_f32(C7 + 0 * 4);
        cVec71 = vld1q_f32(C7 + 1 * 4);
      } else {
        cVec00 = vmovq_n_f32(0);
        cVec01 = vmovq_n_f32(0);
        cVec10 = vmovq_n_f32(0);
        cVec11 = vmovq_n_f32(0);
        cVec20 = vmovq_n_f32(0);
        cVec21 = vmovq_n_f32(0);
        cVec30 = vmovq_n_f32(0);
        cVec31 = vmovq_n_f32(0);
        cVec40 = vmovq_n_f32(0);
        cVec41 = vmovq_n_f32(0);
        cVec50 = vmovq_n_f32(0);
        cVec51 = vmovq_n_f32(0);
        cVec60 = vmovq_n_f32(0);
        cVec61 = vmovq_n_f32(0);
        cVec70 = vmovq_n_f32(0);
        cVec71 = vmovq_n_f32(0);
      }
      
      int c_idx = 0;
      float* __restrict__ curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 1
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      vst1q_f32(C0 + 0 * 4, cVec00);
      vst1q_f32(C0 + 1 * 4, cVec01);
      vst1q_f32(C1 + 0 * 4, cVec10);
      vst1q_f32(C1 + 1 * 4, cVec11);
      vst1q_f32(C2 + 0 * 4, cVec20);
      vst1q_f32(C2 + 1 * 4, cVec21);
      vst1q_f32(C3 + 0 * 4, cVec30);
      vst1q_f32(C3 + 1 * 4, cVec31);
      vst1q_f32(C4 + 0 * 4, cVec40);
      vst1q_f32(C4 + 1 * 4, cVec41);
      vst1q_f32(C5 + 0 * 4, cVec50);
      vst1q_f32(C5 + 1 * 4, cVec51);
      vst1q_f32(C6 + 0 * 4, cVec60);
      vst1q_f32(C6 + 1 * 4, cVec61);
      vst1q_f32(C7 + 0 * 4, cVec70);
      vst1q_f32(C7 + 1 * 4, cVec71);
      
      

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
    
    __ALWAYS_INLINE static void _microkernel_cleanup_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c,
        int  elements_remaining,
        Mask precomp_mask)
    {
      for(; elements_remaining >= 4; elements_remaining -= 4, C += 4, B += 4) {
          
          float* C0 = C + 0 * N;
          float* C1 = C + 1 * N;
          float* C2 = C + 2 * N;
          float* C3 = C + 3 * N;
          float* C4 = C + 4 * N;
          float* C5 = C + 5 * N;
          float* C6 = C + 6 * N;
          float* C7 = C + 7 * N;
          float32x4_t cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
          if (load_c) {
            cVec00 = vld1q_f32(C0 + 0 * 4);
            cVec10 = vld1q_f32(C1 + 0 * 4);
            cVec20 = vld1q_f32(C2 + 0 * 4);
            cVec30 = vld1q_f32(C3 + 0 * 4);
            cVec40 = vld1q_f32(C4 + 0 * 4);
            cVec50 = vld1q_f32(C5 + 0 * 4);
            cVec60 = vld1q_f32(C6 + 0 * 4);
            cVec70 = vld1q_f32(C7 + 0 * 4);
          } else {
            cVec00 = vmovq_n_f32(0);
            cVec10 = vmovq_n_f32(0);
            cVec20 = vmovq_n_f32(0);
            cVec30 = vmovq_n_f32(0);
            cVec40 = vmovq_n_f32(0);
            cVec50 = vmovq_n_f32(0);
            cVec60 = vmovq_n_f32(0);
            cVec70 = vmovq_n_f32(0);
          }
          
          int c_idx = 0;
          float* __restrict__ curr_value_ptr = values;
          const float *__restrict__ B_curr = col_indices[0] * N + B;
          uint32_t * col_indices_curr = col_indices + 1;
          #pragma unroll 1
          for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          vst1q_f32(C0 + 0 * 4, cVec00);
          vst1q_f32(C1 + 0 * 4, cVec10);
          vst1q_f32(C2 + 0 * 4, cVec20);
          vst1q_f32(C3 + 0 * 4, cVec30);
          vst1q_f32(C4 + 0 * 4, cVec40);
          vst1q_f32(C5 + 0 * 4, cVec50);
          vst1q_f32(C6 + 0 * 4, cVec60);
          vst1q_f32(C7 + 0 * 4, cVec70);
          
          


      }

    }


    
    __ALWAYS_INLINE static void _microkernel_packed_C_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c)
    {
      
      float* C0 = C + 0 * 4;
      float* C1 = C + 2 * 4;
      float* C2 = C + 4 * 4;
      float* C3 = C + 6 * 4;
      float* C4 = C + 8 * 4;
      float* C5 = C + 10 * 4;
      float* C6 = C + 12 * 4;
      float* C7 = C + 14 * 4;
      float32x4_t cVec00, cVec01, cVec10, cVec11, cVec20, cVec21, cVec30, cVec31, cVec40, cVec41, cVec50, cVec51, cVec60, cVec61, cVec70, cVec71;
      if (load_c) {
        cVec00 = vld1q_f32(C0 + 0 * 4);
        cVec01 = vld1q_f32(C0 + 1 * 4);
        cVec10 = vld1q_f32(C1 + 2 * 4);
        cVec11 = vld1q_f32(C1 + 3 * 4);
        cVec20 = vld1q_f32(C2 + 4 * 4);
        cVec21 = vld1q_f32(C2 + 5 * 4);
        cVec30 = vld1q_f32(C3 + 6 * 4);
        cVec31 = vld1q_f32(C3 + 7 * 4);
        cVec40 = vld1q_f32(C4 + 8 * 4);
        cVec41 = vld1q_f32(C4 + 9 * 4);
        cVec50 = vld1q_f32(C5 + 10 * 4);
        cVec51 = vld1q_f32(C5 + 11 * 4);
        cVec60 = vld1q_f32(C6 + 12 * 4);
        cVec61 = vld1q_f32(C6 + 13 * 4);
        cVec70 = vld1q_f32(C7 + 14 * 4);
        cVec71 = vld1q_f32(C7 + 15 * 4);
      } else {
        cVec00 = vmovq_n_f32(0);
        cVec01 = vmovq_n_f32(0);
        cVec10 = vmovq_n_f32(0);
        cVec11 = vmovq_n_f32(0);
        cVec20 = vmovq_n_f32(0);
        cVec21 = vmovq_n_f32(0);
        cVec30 = vmovq_n_f32(0);
        cVec31 = vmovq_n_f32(0);
        cVec40 = vmovq_n_f32(0);
        cVec41 = vmovq_n_f32(0);
        cVec50 = vmovq_n_f32(0);
        cVec51 = vmovq_n_f32(0);
        cVec60 = vmovq_n_f32(0);
        cVec61 = vmovq_n_f32(0);
        cVec70 = vmovq_n_f32(0);
        cVec71 = vmovq_n_f32(0);
      }
      
      int c_idx = 0;
      float* __restrict__ curr_value_ptr = values;
      const float *__restrict__ B_curr = col_indices[0] * N + B;
      uint32_t * col_indices_curr = col_indices + 1;
      #pragma unroll 1
      for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
        cVec41 = vfmaq_f32(cVec41, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
        cVec51 = vfmaq_f32(cVec51, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
        cVec61 = vfmaq_f32(cVec61, aVec, bVec1);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec70 = vfmaq_f32(cVec70, aVec, bVec0);
        cVec71 = vfmaq_f32(cVec71, aVec, bVec1);

      }

      
      vst1q_f32(C0 + 0 * 4, cVec00);
      vst1q_f32(C0 + 1 * 4, cVec01);
      vst1q_f32(C1 + 0 * 4, cVec10);
      vst1q_f32(C1 + 1 * 4, cVec11);
      vst1q_f32(C2 + 0 * 4, cVec20);
      vst1q_f32(C2 + 1 * 4, cVec21);
      vst1q_f32(C3 + 0 * 4, cVec30);
      vst1q_f32(C3 + 1 * 4, cVec31);
      vst1q_f32(C4 + 0 * 4, cVec40);
      vst1q_f32(C4 + 1 * 4, cVec41);
      vst1q_f32(C5 + 0 * 4, cVec50);
      vst1q_f32(C5 + 1 * 4, cVec51);
      vst1q_f32(C6 + 0 * 4, cVec60);
      vst1q_f32(C6 + 1 * 4, cVec61);
      vst1q_f32(C7 + 0 * 4, cVec70);
      vst1q_f32(C7 + 1 * 4, cVec71);
      
      

    }



    __ALWAYS_INLINE static void microkernel_packed_C_max_acc(
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
    
    __ALWAYS_INLINE static void _microkernel_cleanup_packed_C_max_acc(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        float* __restrict__       values,
        int                          num_col_indices,
        const float *__restrict__ B,
        float *__restrict__ C,
        const bool load_c,
        int  elements_remaining,
        Mask precomp_mask)
    {
      for(; elements_remaining >= 4; elements_remaining -= 4, C += 4, B += 4) {
          
          float* C0 = C + 0 * 4;
          float* C1 = C + 1 * 4;
          float* C2 = C + 2 * 4;
          float* C3 = C + 3 * 4;
          float* C4 = C + 4 * 4;
          float* C5 = C + 5 * 4;
          float* C6 = C + 6 * 4;
          float* C7 = C + 7 * 4;
          float32x4_t cVec00, cVec10, cVec20, cVec30, cVec40, cVec50, cVec60, cVec70;
          if (load_c) {
            cVec00 = vld1q_f32(C0 + 0 * 4);
            cVec10 = vld1q_f32(C1 + 1 * 4);
            cVec20 = vld1q_f32(C2 + 2 * 4);
            cVec30 = vld1q_f32(C3 + 3 * 4);
            cVec40 = vld1q_f32(C4 + 4 * 4);
            cVec50 = vld1q_f32(C5 + 5 * 4);
            cVec60 = vld1q_f32(C6 + 6 * 4);
            cVec70 = vld1q_f32(C7 + 7 * 4);
          } else {
            cVec00 = vmovq_n_f32(0);
            cVec10 = vmovq_n_f32(0);
            cVec20 = vmovq_n_f32(0);
            cVec30 = vmovq_n_f32(0);
            cVec40 = vmovq_n_f32(0);
            cVec50 = vmovq_n_f32(0);
            cVec60 = vmovq_n_f32(0);
            cVec70 = vmovq_n_f32(0);
          }
          
          int c_idx = 0;
          float* __restrict__ curr_value_ptr = values;
          const float *__restrict__ B_curr = col_indices[0] * N + B;
          uint32_t * col_indices_curr = col_indices + 1;
          #pragma unroll 1
          for(int pat_count = nkern_counts[0]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[9]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[10]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[11]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[12]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[13]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[14]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[15]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[16]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[17]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[18]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec40 = vfmaq_f32(cVec40, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec50 = vfmaq_f32(cVec50, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec60 = vfmaq_f32(cVec60, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec70 = vfmaq_f32(cVec70, aVec, bVec0);

          }

          
          vst1q_f32(C0 + 0 * 4, cVec00);
          vst1q_f32(C1 + 0 * 4, cVec10);
          vst1q_f32(C2 + 0 * 4, cVec20);
          vst1q_f32(C3 + 0 * 4, cVec30);
          vst1q_f32(C4 + 0 * 4, cVec40);
          vst1q_f32(C5 + 0 * 4, cVec50);
          vst1q_f32(C6 + 0 * 4, cVec60);
          vst1q_f32(C7 + 0 * 4, cVec70);
          
          


      }

    }


    
};

} // sop
