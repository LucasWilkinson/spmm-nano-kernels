#include "MicroKernelPackerFactory.h"

namespace sop {

extern MicroKernelPackerFactory<float>* packer_factory_c22a5_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_8c90f_512_8x2();

struct MicroKernelPackerFactoryFloat: public MicroKernelPackerFactory<float> {
MicroKernelPackerFactoryFloat(): MicroKernelPackerFactory<float>(8) {
  register_factory("c22a5_512_4x4", packer_factory_c22a5_512_4x4());
  register_factory("8c90f_512_8x2", packer_factory_8c90f_512_8x2());
}
};

MicroKernelPackerFactoryFloat trip_registration_for_Float;

}
