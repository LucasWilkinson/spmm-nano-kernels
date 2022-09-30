#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "64487/MicroKernel_float_64487_AVX2_4x2.h"

namespace sop {

// factory_desc | {"id": "64487_AVX2_4x2", "func": "executor_factory_KDFloatNoPacking_64487_AVX2_4x2", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 2, "arch": "AVX2", "reg_width_bits": 256}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX2_4x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_64487_AVX2_4x2>(4, 16);
}

} // namespace sop
#endif // __AVX2__
