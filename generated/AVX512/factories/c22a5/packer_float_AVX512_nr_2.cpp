#ifdef __AVX512VL__
#include "MicroKernelPackerFactory.h"
#include "c22a5/MicroKernel_float_c22a5_AVX512_4x2.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX512_4x2", "func": "packer_factory_c22a5_AVX512_4x2", "scalar": "float", "M_r": 4, "N_r": 2, "arch": "AVX512", "reg_width_bits": 512}
MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX512_4x2() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_c22a5_AVX512_4x2>(4);
}

} // namespace sop
#endif // __AVX512VL__