#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_5b38e_256_8x2.h"

namespace sop {

// factory_desc | {"id": "5b38e_256_8x2", "func": "executor_factory_KDFloatCPartialPacking_5b38e_256_8x2", "kernel_desc": "KDFloatCPartialPacking", "M_r": 8, "N_r": 2}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_5b38e_256_8x2() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_5b38e_256_8x2>(8, 16);
}

} // namespace sop
