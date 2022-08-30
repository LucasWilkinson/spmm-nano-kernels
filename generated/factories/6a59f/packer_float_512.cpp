#include "MicroKernelPackerFactory.h"
#include "MicroKernel_float_6a59f_512_4x4.h"

namespace sop {

// factory_desc | {"id": "6a59f_512_4x4", "func": "packer_factory_6a59f_512_4x4", "scalar": "float", "M_r": 4}
MicroKernelPackerFactory<float>* packer_factory_6a59f_512_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_6a59f_512_4x4>(4);
}

} // namespace sop
