#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_6a59f_512_4x4.h"

namespace sop {

// factory_desc | {"id": "6a59f_512_4x4", "func": "executor_factory_KDFloatCPartialPacking_6a59f_512_4x4", "kernel_desc": "KDFloatCPartialPacking", "M_r": 4, "N_r": 4}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_6a59f_512_4x4() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_6a59f_512_4x4>(4, 64);
}

} // namespace sop