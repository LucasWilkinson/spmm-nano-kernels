#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "77f9d/MicroKernel_float_77f9d_AVX2_8x1.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX2_8x1", "func": "executor_factory_KDFloatCPartialPacking_77f9d_AVX2_8x1", "kernel_desc": "KDFloatCPartialPacking", "M_r": 8, "N_r": 1, "arch": "AVX2", "reg_width_bits": 256}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX2_8x1() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_77f9d_AVX2_8x1>(8, 8);
}

} // namespace sop
#endif // __AVX2__