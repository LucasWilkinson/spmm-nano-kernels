#pragma once

#include "utils/error.h"
#include "MicroKernelBase.h"
#include "Storage.h"

#include <arm_neon.h>


#include "intrin_compatability.h"

namespace sop {
struct MicroKernel_float_64487_NEON_128_4x3 {

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
        
    using  Mask = uint32_t;
    static Mask create_mask(int n) { return ((1 << n) - 1); }
    static Mask precomp_mask(int N) { return create_mask(N % 4); }

    using Scalar = float;
    static constexpr int M_r = 4;
    static constexpr int N_r = 3 * 4;
    static constexpr int N_r_reg = 3;
    static constexpr int vec_width_bits = 128;
    static constexpr const char* id = "64487_NEON_128_4x3";
    static int max_acc_width_in_vecs() { return 3; };
    static int max_acc_width_in_eles() { return 3 * 4; };

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
      
      float* C0 = C + 0 * N;
      float* C1 = C + 1 * N;
      float* C2 = C + 2 * N;
      float* C3 = C + 3 * N;
      float32x4_t cVec00, cVec01, cVec02, cVec10, cVec11, cVec12, cVec20, cVec21, cVec22, cVec30, cVec31, cVec32;
      if (load_c) {
        cVec00 = vld1q_f32(C0 + 0 * 4);
        cVec01 = vld1q_f32(C0 + 1 * 4);
        cVec02 = vld1q_f32(C0 + 2 * 4);
        cVec10 = vld1q_f32(C1 + 0 * 4);
        cVec11 = vld1q_f32(C1 + 1 * 4);
        cVec12 = vld1q_f32(C1 + 2 * 4);
        cVec20 = vld1q_f32(C2 + 0 * 4);
        cVec21 = vld1q_f32(C2 + 1 * 4);
        cVec22 = vld1q_f32(C2 + 2 * 4);
        cVec30 = vld1q_f32(C3 + 0 * 4);
        cVec31 = vld1q_f32(C3 + 1 * 4);
        cVec32 = vld1q_f32(C3 + 2 * 4);
      } else {
        cVec00 = vmovq_n_f32(0);
        cVec01 = vmovq_n_f32(0);
        cVec02 = vmovq_n_f32(0);
        cVec10 = vmovq_n_f32(0);
        cVec11 = vmovq_n_f32(0);
        cVec12 = vmovq_n_f32(0);
        cVec20 = vmovq_n_f32(0);
        cVec21 = vmovq_n_f32(0);
        cVec22 = vmovq_n_f32(0);
        cVec30 = vmovq_n_f32(0);
        cVec31 = vmovq_n_f32(0);
        cVec32 = vmovq_n_f32(0);
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
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      vst1q_f32(C0 + 0 * 4, cVec00);
      vst1q_f32(C0 + 1 * 4, cVec01);
      vst1q_f32(C0 + 2 * 4, cVec02);
      vst1q_f32(C1 + 0 * 4, cVec10);
      vst1q_f32(C1 + 1 * 4, cVec11);
      vst1q_f32(C1 + 2 * 4, cVec12);
      vst1q_f32(C2 + 0 * 4, cVec20);
      vst1q_f32(C2 + 1 * 4, cVec21);
      vst1q_f32(C2 + 2 * 4, cVec22);
      vst1q_f32(C3 + 0 * 4, cVec30);
      vst1q_f32(C3 + 1 * 4, cVec31);
      vst1q_f32(C3 + 2 * 4, cVec32);
      
      

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
          float32x4_t cVec00, cVec10, cVec20, cVec30;
          if (load_c) {
            cVec00 = vld1q_f32(C0 + 0 * 4);
            cVec10 = vld1q_f32(C1 + 0 * 4);
            cVec20 = vld1q_f32(C2 + 0 * 4);
            cVec30 = vld1q_f32(C3 + 0 * 4);
          } else {
            cVec00 = vmovq_n_f32(0);
            cVec10 = vmovq_n_f32(0);
            cVec20 = vmovq_n_f32(0);
            cVec30 = vmovq_n_f32(0);
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
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

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

          
          vst1q_f32(C0 + 0 * 4, cVec00);
          vst1q_f32(C1 + 0 * 4, cVec10);
          vst1q_f32(C2 + 0 * 4, cVec20);
          vst1q_f32(C3 + 0 * 4, cVec30);
          
          


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
      float* C1 = C + 3 * 4;
      float* C2 = C + 6 * 4;
      float* C3 = C + 9 * 4;
      float32x4_t cVec00, cVec01, cVec02, cVec10, cVec11, cVec12, cVec20, cVec21, cVec22, cVec30, cVec31, cVec32;
      if (load_c) {
        cVec00 = vld1q_f32(C0 + 0 * 4);
        cVec01 = vld1q_f32(C0 + 1 * 4);
        cVec02 = vld1q_f32(C0 + 2 * 4);
        cVec10 = vld1q_f32(C1 + 3 * 4);
        cVec11 = vld1q_f32(C1 + 4 * 4);
        cVec12 = vld1q_f32(C1 + 5 * 4);
        cVec20 = vld1q_f32(C2 + 6 * 4);
        cVec21 = vld1q_f32(C2 + 7 * 4);
        cVec22 = vld1q_f32(C2 + 8 * 4);
        cVec30 = vld1q_f32(C3 + 9 * 4);
        cVec31 = vld1q_f32(C3 + 10 * 4);
        cVec32 = vld1q_f32(C3 + 11 * 4);
      } else {
        cVec00 = vmovq_n_f32(0);
        cVec01 = vmovq_n_f32(0);
        cVec02 = vmovq_n_f32(0);
        cVec10 = vmovq_n_f32(0);
        cVec11 = vmovq_n_f32(0);
        cVec12 = vmovq_n_f32(0);
        cVec20 = vmovq_n_f32(0);
        cVec21 = vmovq_n_f32(0);
        cVec22 = vmovq_n_f32(0);
        cVec30 = vmovq_n_f32(0);
        cVec31 = vmovq_n_f32(0);
        cVec32 = vmovq_n_f32(0);
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
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      #pragma unroll 1
      for(int pat_count = nkern_counts[8]; pat_count > 0; pat_count--) {
        float32x4_t aVec;
        float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
        float32x4_t bVec1 = vld1q_f32(B_curr + 1 * 4);
        float32x4_t bVec2 = vld1q_f32(B_curr + 2 * 4);
        B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
        cVec01 = vfmaq_f32(cVec01, aVec, bVec1);
        cVec02 = vfmaq_f32(cVec02, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
        cVec11 = vfmaq_f32(cVec11, aVec, bVec1);
        cVec12 = vfmaq_f32(cVec12, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
        cVec21 = vfmaq_f32(cVec21, aVec, bVec1);
        cVec22 = vfmaq_f32(cVec22, aVec, bVec2);
        aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
        cVec30 = vfmaq_f32(cVec30, aVec, bVec0);
        cVec31 = vfmaq_f32(cVec31, aVec, bVec1);
        cVec32 = vfmaq_f32(cVec32, aVec, bVec2);

      }

      
      vst1q_f32(C0 + 0 * 4, cVec00);
      vst1q_f32(C0 + 1 * 4, cVec01);
      vst1q_f32(C0 + 2 * 4, cVec02);
      vst1q_f32(C1 + 0 * 4, cVec10);
      vst1q_f32(C1 + 1 * 4, cVec11);
      vst1q_f32(C1 + 2 * 4, cVec12);
      vst1q_f32(C2 + 0 * 4, cVec20);
      vst1q_f32(C2 + 1 * 4, cVec21);
      vst1q_f32(C2 + 2 * 4, cVec22);
      vst1q_f32(C3 + 0 * 4, cVec30);
      vst1q_f32(C3 + 1 * 4, cVec31);
      vst1q_f32(C3 + 2 * 4, cVec32);
      
      

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
          float32x4_t cVec00, cVec10, cVec20, cVec30;
          if (load_c) {
            cVec00 = vld1q_f32(C0 + 0 * 4);
            cVec10 = vld1q_f32(C1 + 1 * 4);
            cVec20 = vld1q_f32(C2 + 2 * 4);
            cVec30 = vld1q_f32(C3 + 3 * 4);
          } else {
            cVec00 = vmovq_n_f32(0);
            cVec10 = vmovq_n_f32(0);
            cVec20 = vmovq_n_f32(0);
            cVec30 = vmovq_n_f32(0);
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
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[1]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[2]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[3]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[4]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[5]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[6]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec00 = vfmaq_f32(cVec00, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

          }

          
          #pragma unroll 1
          for(int pat_count = nkern_counts[7]; pat_count > 0; pat_count--) {
            float32x4_t aVec;
            float32x4_t bVec0 = vld1q_f32(B_curr + 0 * 4);
            B_curr = (*col_indices_curr) * N + B; col_indices_curr++;
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec10 = vfmaq_f32(cVec10, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec20 = vfmaq_f32(cVec20, aVec, bVec0);
            aVec = vld1q_dup_f32(curr_value_ptr); curr_value_ptr++;
            cVec30 = vfmaq_f32(cVec30, aVec, bVec0);

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

          
          vst1q_f32(C0 + 0 * 4, cVec00);
          vst1q_f32(C1 + 0 * 4, cVec10);
          vst1q_f32(C2 + 0 * 4, cVec20);
          vst1q_f32(C3 + 0 * 4, cVec30);
          
          


      }

    }


    
};

} // sop
