#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "c22a5/MicroKernel_float_c22a5_AVX2_256_4x2.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX2_256_4x2", "func": "executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x2", "kernel_desc": "KDFloatCPartialPacking", "M_r": 4, "N_r": 2, "arch": "AVX2", "reg_width_bits": 256}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x2() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_c22a5_AVX2_256_4x2>(4, 16);
}

} // namespace sop
#endif
