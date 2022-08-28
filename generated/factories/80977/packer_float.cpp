#include "MicroKernelPackerFactory.h"
#include "MicroKernel_float_80977_512_8x2.h"

namespace sop {

// factory_desc | {"id": "80977_512_8x2", "func": "packer_factory_80977_512_8x2", "scalar": "float", "M_r": 8}
MicroKernelPackerFactory<float>* packer_factory_80977_512_8x2() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_80977_512_8x2>(8);
}

} // namespace sop
