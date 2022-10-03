#ifdef __AVX512VL__
#include "77f9d/MicroKernel_float_77f9d_AVX2_256_8x2.h"
#include "MicroKernelPackerFactory.h"
#include "77f9d/MicroKernel_float_77f9d_AVX2_256_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX2_256_8x2", "func": "packer_factory_77f9d_AVX2_256_8x2", "scalar": "float", "M_r": 8, "N_r": 2, "arch": "AVX2", "reg_width_bits": 256}
MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX2_256_8x2() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_77f9d_AVX2_256_8x2>(8);
}

} // namespace sop
#endif
