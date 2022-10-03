#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "c22a5/MicroKernel_double_c22a5_AVX2_256_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX2_256_4x4", "func": "executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x4", "kernel_desc": "KDFloatNoPackingLoadBalanced", "M_r": 4, "N_r": 4, "arch": "AVX2", "reg_width_bits": 256}
ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPackingLoadBalanced, MicroKernel_double_c22a5_AVX2_256_4x4>(4, 16);
}

} // namespace sop
#endif
