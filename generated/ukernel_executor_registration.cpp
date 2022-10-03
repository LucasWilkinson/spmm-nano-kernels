#include "KernelDesc.h"
#include "ExecutorFactory.h"

namespace sop {

#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_NEON_128_4x4();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_NEON_128_4x4();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_NEON_128_4x4();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_NEON_128_8x1();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_NEON_128_8x1();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_NEON_128_8x1();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_NEON_128_8x2();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_NEON_128_8x2();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_NEON_128_8x2();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_NEON_128_4x4();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_NEON_128_4x4();
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_NEON_128_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x1();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x1();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x1();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x1();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x2();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x4();
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX2_256_4x4();
#endif
#endif

struct ExecutorFactoryKDFloatNoPacking : public ExecutorFactory<KDFloatNoPacking> {
ExecutorFactoryKDFloatNoPacking(){
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("c22a5_NEON_128_4x4", executor_factory_KDFloatNoPacking_c22a5_NEON_128_4x4());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x1", executor_factory_KDFloatNoPacking_77f9d_NEON_128_8x1());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x2", executor_factory_KDFloatNoPacking_77f9d_NEON_128_8x2());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("64487_NEON_128_4x4", executor_factory_KDFloatNoPacking_64487_NEON_128_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatNoPacking_64487_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatNoPacking_64487_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", executor_factory_KDFloatNoPacking_c22a5_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", executor_factory_KDFloatNoPacking_c22a5_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", executor_factory_KDFloatNoPacking_64487_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", executor_factory_KDFloatNoPacking_64487_AVX2_256_4x4());
#endif
#endif
}
};

ExecutorFactoryKDFloatNoPacking trip_registration_for_KDFloatNoPacking;

struct ExecutorFactoryKDFloatNoPackingLoadBalanced : public ExecutorFactory<KDFloatNoPackingLoadBalanced> {
ExecutorFactoryKDFloatNoPackingLoadBalanced(){
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("c22a5_NEON_128_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_NEON_128_4x4());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x1", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_NEON_128_8x1());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_NEON_128_8x2());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("64487_NEON_128_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_NEON_128_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x1());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x1());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX2_256_4x4());
#endif
#endif
}
};

ExecutorFactoryKDFloatNoPackingLoadBalanced trip_registration_for_KDFloatNoPackingLoadBalanced;

struct ExecutorFactoryKDFloatCPartialPacking : public ExecutorFactory<KDFloatCPartialPacking> {
ExecutorFactoryKDFloatCPartialPacking(){
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("c22a5_NEON_128_4x4", executor_factory_KDFloatCPartialPacking_c22a5_NEON_128_4x4());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x1", executor_factory_KDFloatCPartialPacking_77f9d_NEON_128_8x1());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("77f9d_NEON_128_8x2", executor_factory_KDFloatCPartialPacking_77f9d_NEON_128_8x2());
#endif
#endif
#ifdef __ARM_NEON__
#if defined(ENABLE_NEON)
  register_factory("64487_NEON_128_4x4", executor_factory_KDFloatCPartialPacking_64487_NEON_128_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x4", executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x4", executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x1());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x1());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x2", executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x2());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x4());
#endif
#endif
#ifdef __AVX512VL__
#if defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x4", executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x4());
#endif
#endif
}
};

ExecutorFactoryKDFloatCPartialPacking trip_registration_for_KDFloatCPartialPacking;

}
