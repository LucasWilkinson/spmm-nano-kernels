#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "c22a5/MicroKernel_float_c22a5_AVX512_512_4x2.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX512_512_4x2", "func": "executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x2", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 2, "arch": "AVX512", "reg_width_bits": 512}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_c22a5_AVX512_512_4x2>(4, 32);
}

} // namespace sop
#endif
