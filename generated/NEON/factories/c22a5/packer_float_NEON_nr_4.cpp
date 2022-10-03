#ifdef __ARM_NEON__
#include "c22a5/MicroKernel_float_c22a5_NEON_128_4x4.h"
#include "MicroKernelPackerFactory.h"
#include "c22a5/MicroKernel_float_c22a5_NEON_128_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_NEON_128_4x4", "func": "packer_factory_c22a5_NEON_128_4x4", "scalar": "float", "M_r": 4, "N_r": 4, "arch": "NEON", "reg_width_bits": 128}
MicroKernelPackerFactory<float>* packer_factory_c22a5_NEON_128_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_c22a5_NEON_128_4x4>(4);
}

} // namespace sop
#endif
