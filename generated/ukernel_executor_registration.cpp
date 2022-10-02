#include "KernelDesc.h"
#include "ExecutorFactory.h"

namespace sop {

#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x2();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x1();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1();
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x1();
#endif // __AVX2__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x4();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2();
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x2();
#endif // __AVX512VL__

struct ExecutorFactoryKDFloatNoPacking : public ExecutorFactory<KDFloatNoPacking> {
ExecutorFactoryKDFloatNoPacking(){
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x2", executor_factory_KDFloatNoPacking_64487_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x2", executor_factory_KDFloatNoPacking_c22a5_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatNoPacking_77f9d_AVX2_256_8x1());
#endif // __AVX2__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x2", executor_factory_KDFloatNoPacking_64487_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatNoPacking_64487_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x2", executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatNoPacking_c22a5_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatNoPacking_77f9d_AVX512_512_8x2());
#endif // __AVX512VL__
}
};

ExecutorFactoryKDFloatNoPacking trip_registration_for_KDFloatNoPacking;

struct ExecutorFactoryKDFloatNoPackingLoadBalanced : public ExecutorFactory<KDFloatNoPackingLoadBalanced> {
ExecutorFactoryKDFloatNoPackingLoadBalanced(){
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x2", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x2", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX2_256_8x1());
#endif // __AVX2__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x2", executor_factory_KDFloatNoPackingLoadBalanced_64487_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x2", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_AVX512_512_8x2());
#endif // __AVX512VL__
}
};

ExecutorFactoryKDFloatNoPackingLoadBalanced trip_registration_for_KDFloatNoPackingLoadBalanced;

struct ExecutorFactoryKDFloatCPartialPacking : public ExecutorFactory<KDFloatCPartialPacking> {
ExecutorFactoryKDFloatCPartialPacking(){
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("64487_AVX2_256_4x2", executor_factory_KDFloatCPartialPacking_64487_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("c22a5_AVX2_256_4x2", executor_factory_KDFloatCPartialPacking_c22a5_AVX2_256_4x2());
#endif // __AVX2__
#if defined(__AVX2__) && defined(ENABLE_AVX2)
  register_factory("77f9d_AVX2_256_8x1", executor_factory_KDFloatCPartialPacking_77f9d_AVX2_256_8x1());
#endif // __AVX2__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x2", executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("64487_AVX512_512_4x4", executor_factory_KDFloatCPartialPacking_64487_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x2", executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x2());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("c22a5_AVX512_512_4x4", executor_factory_KDFloatCPartialPacking_c22a5_AVX512_512_4x4());
#endif // __AVX512VL__
#if defined(__AVX512VL__) && defined(ENABLE_AVX512)
  register_factory("77f9d_AVX512_512_8x2", executor_factory_KDFloatCPartialPacking_77f9d_AVX512_512_8x2());
#endif // __AVX512VL__
}
};

ExecutorFactoryKDFloatCPartialPacking trip_registration_for_KDFloatCPartialPacking;

}
