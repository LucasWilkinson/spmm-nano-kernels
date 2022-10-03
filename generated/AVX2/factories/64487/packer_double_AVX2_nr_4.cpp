#ifdef __AVX512VL__
#include "64487/MicroKernel_double_64487_AVX2_256_4x4.h"
#include "MicroKernelPackerFactory.h"
#include "64487/MicroKernel_double_64487_AVX2_256_4x4.h"

namespace sop {

// factory_desc | {"id": "64487_AVX2_256_4x4", "func": "packer_factory_64487_AVX2_256_4x4", "scalar": "double", "M_r": 4, "N_r": 4, "arch": "AVX2", "reg_width_bits": 256}
MicroKernelPackerFactory<double>* packer_factory_64487_AVX2_256_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_double_64487_AVX2_256_4x4>(4);
}

} // namespace sop
#endif
