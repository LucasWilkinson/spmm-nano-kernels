#ifdef __ARM_NEON__
#include "77f9d/MicroKernel_float_77f9d_NEON_128_8x1.h"
#include "MicroKernelPackerFactory.h"
#include "77f9d/MicroKernel_float_77f9d_NEON_128_8x1.h"

namespace sop {

// factory_desc | {"id": "77f9d_NEON_128_8x1", "func": "packer_factory_77f9d_NEON_128_8x1_float", "scalar": "float", "M_r": 8, "N_r": 1, "arch": "NEON", "reg_width_bits": 128}
MicroKernelPackerFactory<float>* packer_factory_77f9d_NEON_128_8x1_float() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_77f9d_NEON_128_8x1>(8);
}

} // namespace sop
#endif
