#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_520b4_256_4x4.h"

namespace sop {

// factory_desc | {"id": "520b4_256_4x4", "func": "executor_factory_KDFloatNoPackingLoadBalanced_520b4_256_4x4", "kernel_desc": "KDFloatNoPackingLoadBalanced", "M_r": 4, "N_r": 4}
ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_520b4_256_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPackingLoadBalanced, MicroKernel_float_520b4_256_4x4>(4, 32);
}

} // namespace sop