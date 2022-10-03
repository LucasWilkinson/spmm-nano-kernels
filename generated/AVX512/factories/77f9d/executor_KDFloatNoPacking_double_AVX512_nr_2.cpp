#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "77f9d/MicroKernel_double_77f9d_AVX512_512_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX512_512_8x2", "func": "executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2", "kernel_desc": "KDFloatNoPacking", "M_r": 8, "N_r": 2, "arch": "AVX512", "reg_width_bits": 512}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_double_77f9d_AVX512_512_8x2>(8, 16);
}

} // namespace sop
#endif
