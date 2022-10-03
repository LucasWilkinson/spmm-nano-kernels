#ifdef __AVX512VL__
#include "c22a5/MicroKernel_double_c22a5_AVX512_512_4x4.h"
#include "MicroKernelPackerFactory.h"
#include "c22a5/MicroKernel_double_c22a5_AVX512_512_4x4.h"

namespace sop {

// factory_desc | {"id": "c22a5_AVX512_512_4x4", "func": "packer_factory_c22a5_AVX512_512_4x4", "scalar": "double", "M_r": 4, "N_r": 4, "arch": "AVX512", "reg_width_bits": 512}
MicroKernelPackerFactory<double>* packer_factory_c22a5_AVX512_512_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_double_c22a5_AVX512_512_4x4>(4);
}

} // namespace sop
#endif
