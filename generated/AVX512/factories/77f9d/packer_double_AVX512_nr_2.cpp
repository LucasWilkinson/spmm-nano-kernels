#ifdef __AVX512VL__
#include "77f9d/MicroKernel_double_77f9d_AVX512_512_8x2.h"
#include "MicroKernelPackerFactory.h"
#include "77f9d/MicroKernel_double_77f9d_AVX512_512_8x2.h"

namespace sop {

// factory_desc | {"id": "77f9d_AVX512_512_8x2", "func": "packer_factory_77f9d_AVX512_512_8x2", "scalar": "double", "M_r": 8, "N_r": 2, "arch": "AVX512", "reg_width_bits": 512}
MicroKernelPackerFactory<double>* packer_factory_77f9d_AVX512_512_8x2() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_double_77f9d_AVX512_512_8x2>(8);
}

} // namespace sop
#endif
