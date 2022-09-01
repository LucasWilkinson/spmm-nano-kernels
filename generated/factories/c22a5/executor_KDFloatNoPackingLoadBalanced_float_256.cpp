#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_c22a5_256_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_256_4x4", "func": "executor_factory_KDFloatNoPackingLoadBalanced_c22a5_256_4x4", "kernel_desc": "KDFloatNoPackingLoadBalanced", "M_r": 4, "N_r": 4}
ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_256_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPackingLoadBalanced, MicroKernel_float_c22a5_256_4x4>(4, 32);
}

} // namespace sop