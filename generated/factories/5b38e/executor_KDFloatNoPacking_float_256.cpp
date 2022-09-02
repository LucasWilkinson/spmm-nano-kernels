#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_5b38e_256_8x2.h"

namespace sop {

// factory_desc | {"id": "5b38e_256_8x2", "func": "executor_factory_KDFloatNoPacking_5b38e_256_8x2", "kernel_desc": "KDFloatNoPacking", "M_r": 8, "N_r": 2, "vec_width": 256}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5b38e_256_8x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_5b38e_256_8x2>(8, 16);
}

} // namespace sop
#endif // __AVX2__
