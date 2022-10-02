#include "MicroKernelPackerFactory.h"

namespace sop {

#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_64487_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX2_256_8x1();
#endif // __AVX2__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_64487_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_64487_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX512_512_8x2();
#endif // __AVX512VL__

struct MicroKernelPackerFactoryFloat: public MicroKernelPackerFactory<float> {
MicroKernelPackerFactoryFloat(): MicroKernelPackerFactory<float>(8) {
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x2", packer_factory_64487_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x2", packer_factory_c22a5_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", packer_factory_77f9d_AVX2_256_8x1());
#endif // __AVX2__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x2", packer_factory_64487_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", packer_factory_64487_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x2", packer_factory_c22a5_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", packer_factory_c22a5_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", packer_factory_77f9d_AVX512_512_8x2());
#endif // __AVX512VL__
}
};

MicroKernelPackerFactoryFloat trip_registration_for_Float;

}
