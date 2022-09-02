#ifdef __AVX2__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_77f9d_256_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_256_8x2", "func": "executor_factory_KDFloatNoPackingLoadBalanced_77f9d_256_8x2", "kernel_desc": "KDFloatNoPackingLoadBalanced", "M_r": 8, "N_r": 2, "vec_width": 256}
ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_256_8x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPackingLoadBalanced, MicroKernel_float_77f9d_256_8x2>(8, 16);
}

} // namespace sop
#endif // __AVX2__
