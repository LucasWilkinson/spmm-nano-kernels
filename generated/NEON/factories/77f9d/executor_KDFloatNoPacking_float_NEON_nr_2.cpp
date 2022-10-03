#ifdef __ARM_NEON__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "77f9d/MicroKernel_float_77f9d_NEON_128_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_NEON_128_8x2", "func": "executor_factory_KDFloatNoPacking_77f9d_NEON_128_8x2", "kernel_desc": "KDFloatNoPacking", "M_r": 8, "N_r": 2, "arch": "NEON", "reg_width_bits": 128}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_NEON_128_8x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_77f9d_NEON_128_8x2>(8, 8);
}

} // namespace sop
#endif
