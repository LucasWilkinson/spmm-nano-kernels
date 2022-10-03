#ifdef __AVX512VL__
#include "77f9d/MicroKernel_float_77f9d_AVX512_512_8x4.h"
#include "MicroKernelPackerFactory.h"
#include "77f9d/MicroKernel_float_77f9d_AVX512_512_8x4.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX512_512_8x4", "func": "packer_factory_77f9d_AVX512_512_8x4_float", "scalar": "float", "M_r": 8, "N_r": 4, "arch": "AVX512", "reg_width_bits": 512}
MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX512_512_8x4_float() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_77f9d_AVX512_512_8x4>(8);
}

} // namespace sop
#endif
