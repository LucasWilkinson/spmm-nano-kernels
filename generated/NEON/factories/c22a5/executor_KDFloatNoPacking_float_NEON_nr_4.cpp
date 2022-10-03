#ifdef __ARM_NEON__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "c22a5/MicroKernel_float_c22a5_NEON_128_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_NEON_128_4x4", "func": "executor_factory_KDFloatNoPacking_c22a5_NEON_128_4x4", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 4, "arch": "NEON", "reg_width_bits": 128}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_NEON_128_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_c22a5_NEON_128_4x4>(4, 16);
}

} // namespace sop
#endif
