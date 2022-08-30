#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_5eab3_256_4x4.h"

namespace sop {

// factory_desc | {"id": "5eab3_256_4x4", "func": "executor_factory_KDFloatNoPacking_5eab3_256_4x4", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 4}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5eab3_256_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_5eab3_256_4x4>(4, 32);
}

} // namespace sop
