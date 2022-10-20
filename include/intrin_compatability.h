//
// Created by lwilkinson on 9/2/22.
//

#ifndef DNN_SPMM_BENCH_INTRIN_COMPATABILITY_H
#define DNN_SPMM_BENCH_INTRIN_COMPATABILITY_H


#ifndef __AVX512VL__

#define _mm256_mask_storeu_ps(x, y, z) ;
#define _mm256_maskz_loadu_ps(x, y)    _mm256_setzero_ps();

#endif

#define _mm128_loadu_ps(x)       _mm_loadu_ps(x);
#define _mm128_storeu_ps(x, y)   _mm_storeu_ps(x, y);
#define _mm128_set1_ps(x)        _mm_set1_ps(x);
#define _mm128_fmadd_ps(x, y, z) _mm_fmadd_ps(x, y, z);
#define _mm128_setzero_ps()      _mm_setzero_ps();
#define _mm128_max_ps(x, y)      _mm_max_ps(x, y);
#define _mm128_min_ps(x, y)      _mm_min_ps(x, y);


#endif // DNN_SPMM_BENCH_INTRIN_COMPATABILITY_H
