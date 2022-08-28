#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_80977_512_8x2.h"

namespace sop {

// factory_desc | {"id": "80977_512_8x2", "func": "executor_factory_KDFloatNoPacking_80977_512_8x2", "kernel_desc": "KDFloatNoPacking", "M_r": 8, "N_r": 2}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_80977_512_8x2() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_80977_512_8x2>(8, 32);
}

} // namespace sop
