#include "KernelDesc.h"
#include "ExecutorFactory.h"

namespace sop {

extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_512_4x4();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_8c90f_512_8x2();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_8c90f_512_8x2();

struct ExecutorFactoryKDFloatNoPacking : public ExecutorFactory<KDFloatNoPacking> {
ExecutorFactoryKDFloatNoPacking(){
  register_factory("c22a5_512_4x4", executor_factory_KDFloatNoPacking_c22a5_512_4x4());
  register_factory("8c90f_512_8x2", executor_factory_KDFloatNoPacking_8c90f_512_8x2());
}
};

ExecutorFactoryKDFloatNoPacking trip_registration_for_KDFloatNoPacking;

struct ExecutorFactoryKDFloatCPartialPacking : public ExecutorFactory<KDFloatCPartialPacking> {
ExecutorFactoryKDFloatCPartialPacking(){
  register_factory("c22a5_512_4x4", executor_factory_KDFloatCPartialPacking_c22a5_512_4x4());
  register_factory("8c90f_512_8x2", executor_factory_KDFloatCPartialPacking_8c90f_512_8x2());
}
};

ExecutorFactoryKDFloatCPartialPacking trip_registration_for_KDFloatCPartialPacking;

}
