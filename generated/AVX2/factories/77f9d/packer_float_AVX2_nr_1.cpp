#ifdef __AVX2__
#include "MicroKernelPackerFactory.h"
#include "77f9d/MicroKernel_float_77f9d_AVX2_256_8x1.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX2_256_8x1", "func": "packer_factory_77f9d_AVX2_256_8x1", "scalar": "float", "M_r": 8, "N_r": 1, "arch": "AVX2", "reg_width_bits": 256}
MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX2_256_8x1() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_77f9d_AVX2_256_8x1>(8);
}

} // namespace sop
#endif // __AVX2__
