#include "ExecutorFactory.h"
#include "KernelDesc.h"
#include "MicroKernel_float_7bf97_512_4x4.h"

namespace sop {

// factory_desc | {"id": "7bf97_512_4x4", "func": "executor_factory_KDFloatNoPacking_7bf97_512_4x4", "kernel_desc": "KDFloatNoPacking", "M_r": 4, "N_r": 4}
ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_7bf97_512_4x4() {
    return new ExecutorFactorySpecialized<KDFloatNoPacking, MicroKernel_float_7bf97_512_4x4>(4, 64);
}

} // namespace sop
