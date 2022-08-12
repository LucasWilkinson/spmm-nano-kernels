//
// Created by lwilkinson on 8/11/22.
//

#pragma once

#include "SOPStorage.h"

#define __ALWAYS_INLINE __inline __attribute__((__always_inline__))

namespace sop {

static const uint16_t ZERO_PATTERN_ID = std::numeric_limits<uint16_t>::max();

template <
    typename Scalar,
    int _vec_width,
    int _panel_height,
    int _max_acc_width>
struct SOPMicroKernelIntrin {
  static uint16_t encode_pattern(uint16_t pattern);
  static uint16_t decode_pattern(uint16_t pat_code);
  static uint16_t nnz_count(uint16_t pat_code);

  static const int M_r;
  static const int N_r;

  static int max_acc_width_in_vecs();
  static int max_acc_width_in_eles();

  static int number_of_patterns();
  static const uint16_t* supported_patterns();

  static int panel_height();

  // ... + various with micro kernels, see generated files
};

} // namespace sop