#ifdef __AVX512VL__
#include "64487/MicroKernel_float_64487_AVX2_256_4x4.h"
#include "MicroKernelPackerFactory.h"
#include "64487/MicroKernel_float_64487_AVX2_256_4x4.h"

namespace sop {

// factory_desc | {"id": "64487_AVX2_256_4x4", "func": "packer_factory_64487_AVX2_256_4x4_float", "scalar": "float", "M_r": 4, "N_r": 4, "arch": "AVX2", "reg_width_bits": 256}
MicroKernelPackerFactory<float>* packer_factory_64487_AVX2_256_4x4_float() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_64487_AVX2_256_4x4>(4);
}

} // namespace sop
#endif
