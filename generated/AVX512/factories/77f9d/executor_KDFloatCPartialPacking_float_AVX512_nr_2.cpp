#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "77f9d/MicroKernel_float_77f9d_AVX512_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX512_8x2", "func": "executor_factory_KDFloatCPartialPacking_77f9d_AVX512_8x2", "kernel_desc": "KDFloatCPartialPacking", "M_r": 8, "N_r": 2, "arch": "AVX512", "reg_width_bits": 512}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_8x2() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_77f9d_AVX512_8x2>(8, 32);
}

} // namespace sop
#endif // __AVX512VL__
