#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_77f9d_AVX2_256_8x1.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX2_256_8x1", "func": "executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1", "kernel_desc": "KDFloatNoPacking", "M_r": 8, "N_r": 1, "arch": "AVX2", "reg_width_bits": 256}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_77f9d_AVX2_256_8x1>(8, 8);
}

} // namespace sop
#endif // __AVX2__
