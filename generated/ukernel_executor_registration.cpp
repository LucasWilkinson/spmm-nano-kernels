#include "KernelDesc.h"
#include "ExecutorFactory.h"

namespace sop {

extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_2a088_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_2a088_512_4x4();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_80977_512_8x2();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_80977_512_8x2();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_ad3b1_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_ad3b1_512_4x4();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_512_4x4();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_512_8x2();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_512_8x2();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_512_4x4();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_520b4_512_4x4();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_520b4_512_4x4();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5b38e_512_8x2();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_5b38e_512_8x2();
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_eb154_512_8x2();
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_eb154_512_8x2();

struct ExecutorFactoryKDFloatNoPacking : public ExecutorFactory<KDFloatNoPacking> {
ExecutorFactoryKDFloatNoPacking(){
  register_factory("2a088_512_4x4", executor_factory_KDFloatNoPacking_2a088_512_4x4());
  register_factory("80977_512_8x2", executor_factory_KDFloatNoPacking_80977_512_8x2());
  register_factory("ad3b1_512_4x4", executor_factory_KDFloatNoPacking_ad3b1_512_4x4());
  register_factory("c22a5_512_4x4", executor_factory_KDFloatNoPacking_c22a5_512_4x4());
  register_factory("77f9d_512_8x2", executor_factory_KDFloatNoPacking_77f9d_512_8x2());
  register_factory("64487_512_4x4", executor_factory_KDFloatNoPacking_64487_512_4x4());
  register_factory("520b4_512_4x4", executor_factory_KDFloatNoPacking_520b4_512_4x4());
  register_factory("5b38e_512_8x2", executor_factory_KDFloatNoPacking_5b38e_512_8x2());
  register_factory("eb154_512_8x2", executor_factory_KDFloatNoPacking_eb154_512_8x2());
}
};

ExecutorFactoryKDFloatNoPacking trip_registration_for_KDFloatNoPacking;

struct ExecutorFactoryKDFloatCPartialPacking : public ExecutorFactory<KDFloatCPartialPacking> {
ExecutorFactoryKDFloatCPartialPacking(){
  register_factory("2a088_512_4x4", executor_factory_KDFloatCPartialPacking_2a088_512_4x4());
  register_factory("80977_512_8x2", executor_factory_KDFloatCPartialPacking_80977_512_8x2());
  register_factory("ad3b1_512_4x4", executor_factory_KDFloatCPartialPacking_ad3b1_512_4x4());
  register_factory("c22a5_512_4x4", executor_factory_KDFloatCPartialPacking_c22a5_512_4x4());
  register_factory("77f9d_512_8x2", executor_factory_KDFloatCPartialPacking_77f9d_512_8x2());
  register_factory("64487_512_4x4", executor_factory_KDFloatCPartialPacking_64487_512_4x4());
  register_factory("520b4_512_4x4", executor_factory_KDFloatCPartialPacking_520b4_512_4x4());
  register_factory("5b38e_512_8x2", executor_factory_KDFloatCPartialPacking_5b38e_512_8x2());
  register_factory("eb154_512_8x2", executor_factory_KDFloatCPartialPacking_eb154_512_8x2());
}
};

ExecutorFactoryKDFloatCPartialPacking trip_registration_for_KDFloatCPartialPacking;

}
