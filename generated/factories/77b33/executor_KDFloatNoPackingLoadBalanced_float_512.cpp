#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_77b33_512_8x2.h"

namespace sop {

// factory_desc | {"id": "77b33_512_8x2", "func": "executor_factory_KDFloatNoPackingLoadBalanced_77b33_512_8x2", "kernel_desc": "KDFloatNoPackingLoadBalanced", "M_r": 8, "N_r": 2, "vec_width": 512}
ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77b33_512_8x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPackingLoadBalanced, MicroKernel_float_77b33_512_8x2>(8, 32);
}

} // namespace sop
#endif // __AVX512VL__
