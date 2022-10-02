#ifdef __AVX2__
#include "MicroKernelPackerFactory.h"
#include "64487/MicroKernel_float_64487_AVX2_256_4x2.h"

namespace sop {

// factory_desc | {"id": "64487_AVX2_256_4x2", "func": "packer_factory_64487_AVX2_256_4x2", "scalar": "float", "M_r": 4, "N_r": 2, "arch": "AVX2", "reg_width_bits": 256}
MicroKernelPackerFactory<float>* packer_factory_64487_AVX2_256_4x2() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_64487_AVX2_256_4x2>(4);
}

} // namespace sop
#endif // __AVX2__
