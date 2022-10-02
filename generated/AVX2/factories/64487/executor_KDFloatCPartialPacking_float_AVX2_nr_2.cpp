#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_64487_AVX2_256_4x2.h"

namespace sop {

// factory_desc | {"id": "64487_AVX2_256_4x2", "func": "executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x2", "kernel_desc": "KDFloatCPartialPacking", "M_r": 4, "N_r": 2, "arch": "AVX2", "reg_width_bits": 256}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x2() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_64487_AVX2_256_4x2>(4, 16);
}

} // namespace sop
#endif // __AVX2__
