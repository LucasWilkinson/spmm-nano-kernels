#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "77f9d/MicroKernel_double_77f9d_AVX512_512_8x4.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX512_512_8x4", "func": "executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x4", "kernel_desc": "KDFloatCPartialPacking", "M_r": 8, "N_r": 4, "arch": "AVX512", "reg_width_bits": 512}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x4() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_double_77f9d_AVX512_512_8x4>(8, 32);
}

} // namespace sop
#endif
