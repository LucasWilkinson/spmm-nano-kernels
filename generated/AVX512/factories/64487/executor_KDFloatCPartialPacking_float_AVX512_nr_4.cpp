#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "64487/MicroKernel_float_64487_AVX512_4x4.h"

namespace sop {

// factory_desc | {"id": "64487_AVX512_4x4", "func": "executor_factory_KDFloatCPartialPacking_64487_AVX512_4x4", "kernel_desc": "KDFloatCPartialPacking", "M_r": 4, "N_r": 4, "arch": "AVX512", "reg_width_bits": 512}
ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX512_4x4() {
    return new ExecutorFactorySpecialized<KDFloatCPartialPacking, MicroKernel_float_64487_AVX512_4x4>(4, 64);
}

} // namespace sop
#endif // __AVX512VL__
