#include "MicroKernelPackerFactory.h"

namespace sop {

extern MicroKernelPackerFactory<float>* packer_factory_ad3b1_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_ad3b1_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_77b33_512_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_77b33_256_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_512_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_256_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_64487_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_64487_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_520b4_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_520b4_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_5b38e_512_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_5b38e_256_8x2();

struct MicroKernelPackerFactoryFloat: public MicroKernelPackerFactory<float> {
MicroKernelPackerFactoryFloat(): MicroKernelPackerFactory<float>(8) {
  register_factory("ad3b1_512_4x4", packer_factory_ad3b1_512_4x4());
  register_factory("ad3b1_256_4x4", packer_factory_ad3b1_256_4x4());
  register_factory("c22a5_512_4x4", packer_factory_c22a5_512_4x4());
  register_factory("c22a5_256_4x4", packer_factory_c22a5_256_4x4());
  register_factory("77b33_512_8x2", packer_factory_77b33_512_8x2());
  register_factory("77b33_256_8x2", packer_factory_77b33_256_8x2());
  register_factory("77f9d_512_8x2", packer_factory_77f9d_512_8x2());
  register_factory("77f9d_256_8x2", packer_factory_77f9d_256_8x2());
  register_factory("64487_512_4x4", packer_factory_64487_512_4x4());
  register_factory("64487_256_4x4", packer_factory_64487_256_4x4());
  register_factory("520b4_512_4x4", packer_factory_520b4_512_4x4());
  register_factory("520b4_256_4x4", packer_factory_520b4_256_4x4());
  register_factory("5b38e_512_8x2", packer_factory_5b38e_512_8x2());
  register_factory("5b38e_256_8x2", packer_factory_5b38e_256_8x2());
}
};

MicroKernelPackerFactoryFloat trip_registration_for_Float;

}
