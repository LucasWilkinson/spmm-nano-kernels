#include "MicroKernelPackerFactory.h"

namespace sop {

extern MicroKernelPackerFactory<float>* packer_factory_60007_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_60007_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_0dfe3_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_0dfe3_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_e8c1f_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_e8c1f_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_ad3b1_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_ad3b1_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_f1006_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_f1006_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_5eab3_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_5eab3_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_c22a5_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_77b33_512_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_77b33_256_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_f0bdc_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_f0bdc_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_dad5c_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_dad5c_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_45bec_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_45bec_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_105ad_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_105ad_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_6a59f_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_6a59f_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_d508e_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_d508e_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_91aaa_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_91aaa_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_acd31_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_acd31_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_a8d4c_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_a8d4c_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_512_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_77f9d_256_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_7bf97_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_7bf97_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_64487_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_64487_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_520b4_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_520b4_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_470b8_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_470b8_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_5b38e_512_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_5b38e_256_8x2();
extern MicroKernelPackerFactory<float>* packer_factory_28600_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_28600_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_3e5d4_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_3e5d4_256_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_0e71b_512_4x4();
extern MicroKernelPackerFactory<float>* packer_factory_0e71b_256_4x4();

struct MicroKernelPackerFactoryFloat: public MicroKernelPackerFactory<float> {
MicroKernelPackerFactoryFloat(): MicroKernelPackerFactory<float>(4) {
  register_factory("60007_512_4x4", packer_factory_60007_512_4x4());
  register_factory("60007_256_4x4", packer_factory_60007_256_4x4());
  register_factory("0dfe3_512_4x4", packer_factory_0dfe3_512_4x4());
  register_factory("0dfe3_256_4x4", packer_factory_0dfe3_256_4x4());
  register_factory("e8c1f_512_4x4", packer_factory_e8c1f_512_4x4());
  register_factory("e8c1f_256_4x4", packer_factory_e8c1f_256_4x4());
  register_factory("ad3b1_512_4x4", packer_factory_ad3b1_512_4x4());
  register_factory("ad3b1_256_4x4", packer_factory_ad3b1_256_4x4());
  register_factory("f1006_512_4x4", packer_factory_f1006_512_4x4());
  register_factory("f1006_256_4x4", packer_factory_f1006_256_4x4());
  register_factory("5eab3_512_4x4", packer_factory_5eab3_512_4x4());
  register_factory("5eab3_256_4x4", packer_factory_5eab3_256_4x4());
  register_factory("c22a5_512_4x4", packer_factory_c22a5_512_4x4());
  register_factory("c22a5_256_4x4", packer_factory_c22a5_256_4x4());
  register_factory("77b33_512_8x2", packer_factory_77b33_512_8x2());
  register_factory("77b33_256_8x2", packer_factory_77b33_256_8x2());
  register_factory("f0bdc_512_4x4", packer_factory_f0bdc_512_4x4());
  register_factory("f0bdc_256_4x4", packer_factory_f0bdc_256_4x4());
  register_factory("dad5c_512_4x4", packer_factory_dad5c_512_4x4());
  register_factory("dad5c_256_4x4", packer_factory_dad5c_256_4x4());
  register_factory("45bec_512_4x4", packer_factory_45bec_512_4x4());
  register_factory("45bec_256_4x4", packer_factory_45bec_256_4x4());
  register_factory("105ad_512_4x4", packer_factory_105ad_512_4x4());
  register_factory("105ad_256_4x4", packer_factory_105ad_256_4x4());
  register_factory("6a59f_512_4x4", packer_factory_6a59f_512_4x4());
  register_factory("6a59f_256_4x4", packer_factory_6a59f_256_4x4());
  register_factory("d508e_512_4x4", packer_factory_d508e_512_4x4());
  register_factory("d508e_256_4x4", packer_factory_d508e_256_4x4());
  register_factory("91aaa_512_4x4", packer_factory_91aaa_512_4x4());
  register_factory("91aaa_256_4x4", packer_factory_91aaa_256_4x4());
  register_factory("acd31_512_4x4", packer_factory_acd31_512_4x4());
  register_factory("acd31_256_4x4", packer_factory_acd31_256_4x4());
  register_factory("a8d4c_512_4x4", packer_factory_a8d4c_512_4x4());
  register_factory("a8d4c_256_4x4", packer_factory_a8d4c_256_4x4());
  register_factory("77f9d_512_8x2", packer_factory_77f9d_512_8x2());
  register_factory("77f9d_256_8x2", packer_factory_77f9d_256_8x2());
  register_factory("7bf97_512_4x4", packer_factory_7bf97_512_4x4());
  register_factory("7bf97_256_4x4", packer_factory_7bf97_256_4x4());
  register_factory("64487_512_4x4", packer_factory_64487_512_4x4());
  register_factory("64487_256_4x4", packer_factory_64487_256_4x4());
  register_factory("520b4_512_4x4", packer_factory_520b4_512_4x4());
  register_factory("520b4_256_4x4", packer_factory_520b4_256_4x4());
  register_factory("470b8_512_4x4", packer_factory_470b8_512_4x4());
  register_factory("470b8_256_4x4", packer_factory_470b8_256_4x4());
  register_factory("5b38e_512_8x2", packer_factory_5b38e_512_8x2());
  register_factory("5b38e_256_8x2", packer_factory_5b38e_256_8x2());
  register_factory("28600_512_4x4", packer_factory_28600_512_4x4());
  register_factory("28600_256_4x4", packer_factory_28600_256_4x4());
  register_factory("3e5d4_512_4x4", packer_factory_3e5d4_512_4x4());
  register_factory("3e5d4_256_4x4", packer_factory_3e5d4_256_4x4());
  register_factory("0e71b_512_4x4", packer_factory_0e71b_512_4x4());
  register_factory("0e71b_256_4x4", packer_factory_0e71b_256_4x4());
}
};

MicroKernelPackerFactoryFloat trip_registration_for_Float;

}
