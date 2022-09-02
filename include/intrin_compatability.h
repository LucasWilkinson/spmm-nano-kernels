//
// Created by lwilkinson on 9/2/22.
//

#ifndef DNN_SPMM_BENCH_INTRIN_COMPATABILITY_H
#define DNN_SPMM_BENCH_INTRIN_COMPATABILITY_H


#ifndef __AVX512VL__

#define _mm256_mask_storeu_ps(x, y, z) ;
#define _mm256_maskz_loadu_ps(x, y)    _mm256_setzero_ps();

#endif

#endif // DNN_SPMM_BENCH_INTRIN_COMPATABILITY_H
