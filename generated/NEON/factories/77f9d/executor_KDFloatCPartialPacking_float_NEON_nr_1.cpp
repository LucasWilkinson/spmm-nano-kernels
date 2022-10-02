#ifdef __ARM_NEON__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "77f9d/MicroKernel_float_77f9d_NEON_128_8x1.h"

namespace sop {

// factory_desc | {"id": "77f9d_NEON_128_8x1", "func": "executor_factory_KDFloatCPartialPacking_77f9d_NEON_128_8x1", "kernel_desc": "KDFloatCPartialPacking", "M_r": 8, "N_r": 1, "arch": "NEON", "reg_width_bits": 128}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_NEON_128_8x1() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_77f9d_NEON_128_8x1>(8, 4);
}

} // namespace sop
#endif
