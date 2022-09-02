#ifdef __AVX2__
#include "MicroKernelPackerFactory.h"
#include "MicroKernel_float_105ad_256_4x4.h"

namespace sop {

// factory_desc | {"id": "105ad_256_4x4", "func": "packer_factory_105ad_256_4x4", "scalar": "float", "M_r": 4, "vec_width": 256}
MicroKernelPackerFactory<float>* packer_factory_105ad_256_4x4() {
    return new MicroKernelPackerFactorySpecialized<MicroKernel_float_105ad_256_4x4>(4);
}

} // namespace sop
#endif // __AVX2__
