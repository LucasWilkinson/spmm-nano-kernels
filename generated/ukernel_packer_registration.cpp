#include "MicroKernelPackerFactory.h"

namespace sop {

#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_NEON_128_4x3_float();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_NEON_128_4x4_float();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_NEON_128_8x1_float();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_NEON_128_8x2_float();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern MicroKernelPackerFactory<float>* packer_factory_64487_NEON_128_4x3_float();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern MicroKernelPackerFactory<float>* packer_factory_64487_NEON_128_4x4_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX512_512_4x4_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX512_512_8x2_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX512_512_8x4_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern MicroKernelPackerFactory<float>* packer_factory_64487_AVX512_512_4x4_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_AVX2_256_4x4_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX2_256_8x2_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_AVX2_256_8x1_float();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern MicroKernelPackerFactory<float>* packer_factory_64487_AVX2_256_4x4_float();
#endif
#endif

struct MicroKernelPackerFactoryFloat: public MicroKernelPackerFactory<float> {
MicroKernelPackerFactoryFloat(): MicroKernelPackerFactory<float>(4) {
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("c22a5_NEON_128_4x3", packer_factory_c22a5_NEON_128_4x3_float());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("c22a5_NEON_128_4x4", packer_factory_c22a5_NEON_128_4x4_float());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x1", packer_factory_77f9d_NEON_128_8x1_float());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x2", packer_factory_77f9d_NEON_128_8x2_float());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("64487_NEON_128_4x3", packer_factory_64487_NEON_128_4x3_float());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("64487_NEON_128_4x4", packer_factory_64487_NEON_128_4x4_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", packer_factory_c22a5_AVX512_512_4x4_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", packer_factory_77f9d_AVX512_512_8x2_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", packer_factory_77f9d_AVX512_512_8x4_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", packer_factory_64487_AVX512_512_4x4_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", packer_factory_c22a5_AVX2_256_4x4_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", packer_factory_77f9d_AVX2_256_8x2_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", packer_factory_77f9d_AVX2_256_8x1_float());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", packer_factory_64487_AVX2_256_4x4_float());
#endif
#endif
}
};

MicroKernelPackerFactoryFloat trip_registration_for_Float;

}
