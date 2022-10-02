#ifdef __ARM_NEON__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "64487/MicroKernel_float_64487_NEON_128_4x3.h"

namespace sop {

// factory_desc | {"id": "64487_NEON_128_4x3", "func": "executor_factory_KDFloatNoPacking_64487_NEON_128_4x3", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 3, "arch": "NEON", "reg_width_bits": 128}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_NEON_128_4x3() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_64487_NEON_128_4x3>(4, 12);
}

} // namespace sop
#endif
