#include "MicroKernelPackerFactory.h"
#include "MicroKernel_float_c22a5_512_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_512_4x4", "func": "packer_factory_c22a5_512_4x4", "scalar": "float", "M_r": 4}
MicroKernelPackerFactory<float>* packer_factory_c22a5_512_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_c22a5_512_4x4>(4);
}

} // namespace sop
