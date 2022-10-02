#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_c22a5_AVX512_512_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX512_512_4x4", "func": "executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4", "kernel_desc": "KDFloatCPartialPacking", "M_r": 4, "N_r": 4, "arch": "AVX512", "reg_width_bits": 512}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_c22a5_AVX512_512_4x4>(4, 64);
}

} // namespace sop
#endif // __AVX512VL__
