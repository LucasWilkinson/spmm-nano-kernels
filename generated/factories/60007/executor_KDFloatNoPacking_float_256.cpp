#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_60007_256_4x4.h"

namespace sop {

// factory_desc | {"id": "60007_256_4x4", "func": "executor_factory_KDFloatNoPacking_60007_256_4x4", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 4, "vec_width": 256}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_60007_256_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_60007_256_4x4>(4, 32);
}

} // namespace sop
#endif // __AVX2__
