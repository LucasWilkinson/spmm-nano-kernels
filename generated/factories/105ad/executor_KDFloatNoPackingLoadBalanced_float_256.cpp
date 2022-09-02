#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_105ad_256_4x4.h"

namespace sop {

// factory_desc | {"id": "105ad_256_4x4", "func": "executor_factory_KDFloatNoPackingLoadBalanced_105ad_256_4x4", "kernel_desc": "KDFloatNoPackingLoadBalanced", "M_r": 4, "N_r": 4, "vec_width": 256}
ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_105ad_256_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPackingLoadBalanced, MicroKernel_float_105ad_256_4x4>(4, 32);
}

} // namespace sop
#endif // __AVX2__
