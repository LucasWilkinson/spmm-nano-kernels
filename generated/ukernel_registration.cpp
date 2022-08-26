#include "KernelDesc.h"
#include "ExecutorFactory.h"

extern ExecutorFactory<KDFloatNoPacking> executor_factory_13fd3_512_8x2();
extern ExecutorFactory<KDFloatCPartialPacking> executor_factory_13fd3_512_8x2();
extern ExecutorFactory<KDFloatNoPacking> executor_factory_924ca_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking> executor_factory_924ca_512_4x4();

struct ExecutorFactoryKDFloatNoPacking : public ExecutorFactory<KDFloatNoPacking> {
ExecutorFactoryKDFloatNoPacking(){
  register_factory("13fd3_512_8x2", executor_factory_13fd3_512_8x2());
  register_factory("924ca_512_4x4", executor_factory_924ca_512_4x4());
}
};

ExecutorFactoryKDFloatNoPacking trip_registration_for_KDFloatNoPacking;

struct ExecutorFactoryKDFloatCPartialPacking : public ExecutorFactory<KDFloatCPartialPacking> {
ExecutorFactoryKDFloatCPartialPacking(){
  register_factory("13fd3_512_8x2", executor_factory_13fd3_512_8x2());
  register_factory("924ca_512_4x4", executor_factory_924ca_512_4x4());
}
};

ExecutorFactoryKDFloatCPartialPacking trip_registration_for_KDFloatCPartialPacking;

