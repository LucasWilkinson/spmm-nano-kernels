#ifdef __AVX512VL__
#include "MicroKernelPackerFactory.h"
#include "MicroKernel_float_520b4_512_4x4.h"

namespace sop {

// factory_desc | {"id": "520b4_512_4x4", "func": "packer_factory_520b4_512_4x4", "scalar": "float", "M_r": 4, "vec_width": 512}
MicroKernelPackerFactory<float>* packer_factory_520b4_512_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_520b4_512_4x4>(4);
}

} // namespace sop
#endif // __AVX512VL__
