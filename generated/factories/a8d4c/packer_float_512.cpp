#include "MicroKernelPackerFactory.h"
#include "MicroKernel_float_a8d4c_512_4x4.h"

namespace sop {

// factory_desc | {"id": "a8d4c_512_4x4", "func": "packer_factory_a8d4c_512_4x4", "scalar": "float", "M_r": 4}
MicroKernelPackerFactory<float>* packer_factory_a8d4c_512_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_a8d4c_512_4x4>(4);
}

} // namespace sop