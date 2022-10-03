#ifdef __AVX512VL__
#include "c22a5/MicroKernel_float_c22a5_AVX2_256_4x4.h"
#include "MicroKernelPackerFactory.h"
#include "c22a5/MicroKernel_float_c22a5_AVX2_256_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX2_256_4x4", "func": "packer_factory_c22a5_AVX2_256_4x4", "scalar": "float", "M_r": 4, "N_r": 4, "arch": "AVX2", "reg_width_bits": 256}
MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX2_256_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_c22a5_AVX2_256_4x4>(4);
}

} // namespace sop
#endif
