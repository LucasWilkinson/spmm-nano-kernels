#ifdef __AVX512VL__
#include "MicroKernelPackerFactory.h"
#include "77f9d/MicroKernel_float_77f9d_AVX512_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX512_8x2", "func": "packer_factory_77f9d_AVX512_8x2", "scalar": "float", "M_r": 8, "N_r": 2, "arch": "AVX512", "reg_width_bits": 512}
MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX512_8x2() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_77f9d_AVX512_8x2>(8);
}

} // namespace sop
#endif // __AVX512VL__
