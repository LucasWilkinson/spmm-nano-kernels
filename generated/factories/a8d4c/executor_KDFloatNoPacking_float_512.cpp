#ifdef __AVX512VL__
#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_a8d4c_512_4x4.h"

namespace sop {

// factory_desc | {"id": "a8d4c_512_4x4", "func": "executor_factory_KDFloatNoPacking_a8d4c_512_4x4", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 4, "vec_width": 512}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_a8d4c_512_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_a8d4c_512_4x4>(4, 64);
}

} // namespace sop
#endif // __AVX512VL__
