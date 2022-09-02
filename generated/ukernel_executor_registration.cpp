#include "KernelDesc.h"
#include "ExecutorFactory.h"

namespace sop {

#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_60007_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_60007_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_60007_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_60007_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_60007_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_60007_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_0dfe3_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_0dfe3_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_0dfe3_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_0dfe3_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_0dfe3_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_0dfe3_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_e8c1f_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_e8c1f_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_e8c1f_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_e8c1f_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_e8c1f_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_e8c1f_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_ad3b1_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_ad3b1_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_ad3b1_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_ad3b1_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_ad3b1_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_ad3b1_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_f1006_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_f1006_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_f1006_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_f1006_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_f1006_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_f1006_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_5eab3_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5eab3_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5eab3_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_5eab3_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_5eab3_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_5eab3_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_c22a5_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_c22a5_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_c22a5_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77b33_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77b33_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77b33_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77b33_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77b33_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77b33_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_f0bdc_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_f0bdc_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_f0bdc_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_f0bdc_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_f0bdc_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_f0bdc_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_dad5c_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_dad5c_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_dad5c_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_dad5c_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_dad5c_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_dad5c_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_45bec_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_45bec_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_45bec_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_45bec_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_45bec_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_45bec_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_105ad_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_105ad_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_105ad_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_105ad_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_105ad_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_105ad_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_6a59f_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_6a59f_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_6a59f_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_6a59f_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_6a59f_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_6a59f_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_d508e_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_d508e_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_d508e_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_d508e_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_d508e_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_d508e_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_91aaa_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_91aaa_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_91aaa_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_91aaa_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_91aaa_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_91aaa_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_acd31_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_acd31_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_acd31_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_acd31_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_acd31_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_acd31_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_a8d4c_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_a8d4c_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_a8d4c_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_a8d4c_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_a8d4c_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_a8d4c_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_77f9d_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_77f9d_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_77f9d_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_7bf97_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_7bf97_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_7bf97_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_7bf97_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_7bf97_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_7bf97_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_64487_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_64487_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_64487_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_520b4_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_520b4_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_520b4_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_520b4_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_520b4_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_520b4_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_470b8_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_470b8_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_470b8_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_470b8_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_470b8_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_470b8_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_5b38e_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5b38e_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_5b38e_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_5b38e_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_5b38e_512_8x2();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_5b38e_256_8x2();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_28600_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_28600_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_28600_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_28600_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_28600_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_28600_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_3e5d4_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_3e5d4_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_3e5d4_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_3e5d4_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_3e5d4_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_3e5d4_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_0e71b_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_0e71b_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatNoPacking>* executor_factory_KDFloatNoPacking_0e71b_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_0e71b_256_4x4();
#endif // __AVX2__
#ifdef __AVX512VL__
extern ExecutorFactory<KDFloatCPartialPacking>* executor_factory_KDFloatCPartialPacking_0e71b_512_4x4();
#endif // __AVX512VL__
#ifdef __AVX2__
extern ExecutorFactory<KDFloatNoPackingLoadBalanced>* executor_factory_KDFloatNoPackingLoadBalanced_0e71b_256_4x4();
#endif // __AVX2__

struct ExecutorFactoryKDFloatNoPackingLoadBalanced : public ExecutorFactory<KDFloatNoPackingLoadBalanced> {
ExecutorFactoryKDFloatNoPackingLoadBalanced(){
#ifdef __AVX512VL__
  register_factory("60007_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_60007_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("60007_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_60007_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("0dfe3_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_0dfe3_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("0dfe3_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_0dfe3_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("e8c1f_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_e8c1f_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("e8c1f_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_e8c1f_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("ad3b1_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_ad3b1_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("ad3b1_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_ad3b1_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("f1006_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_f1006_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("f1006_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_f1006_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("5eab3_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_5eab3_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("5eab3_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_5eab3_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("c22a5_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("c22a5_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_c22a5_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("77b33_512_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77b33_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("77b33_256_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77b33_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("f0bdc_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_f0bdc_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("f0bdc_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_f0bdc_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("dad5c_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_dad5c_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("dad5c_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_dad5c_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("45bec_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_45bec_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("45bec_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_45bec_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("105ad_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_105ad_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("105ad_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_105ad_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("6a59f_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_6a59f_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("6a59f_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_6a59f_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("d508e_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_d508e_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("d508e_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_d508e_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("91aaa_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_91aaa_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("91aaa_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_91aaa_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("acd31_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_acd31_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("acd31_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_acd31_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("a8d4c_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_a8d4c_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("a8d4c_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_a8d4c_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("77f9d_512_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("77f9d_256_8x2", executor_factory_KDFloatNoPackingLoadBalanced_77f9d_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("7bf97_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_7bf97_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("7bf97_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_7bf97_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("64487_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("64487_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_64487_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("520b4_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_520b4_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("520b4_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_520b4_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("470b8_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_470b8_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("470b8_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_470b8_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("5b38e_512_8x2", executor_factory_KDFloatNoPackingLoadBalanced_5b38e_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("5b38e_256_8x2", executor_factory_KDFloatNoPackingLoadBalanced_5b38e_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("28600_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_28600_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("28600_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_28600_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("3e5d4_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_3e5d4_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("3e5d4_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_3e5d4_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("0e71b_512_4x4", executor_factory_KDFloatNoPackingLoadBalanced_0e71b_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("0e71b_256_4x4", executor_factory_KDFloatNoPackingLoadBalanced_0e71b_256_4x4());
#endif // __AVX2__
}
};

ExecutorFactoryKDFloatNoPackingLoadBalanced trip_registration_for_KDFloatNoPackingLoadBalanced;

struct ExecutorFactoryKDFloatCPartialPacking : public ExecutorFactory<KDFloatCPartialPacking> {
ExecutorFactoryKDFloatCPartialPacking(){
#ifdef __AVX2__
  register_factory("60007_256_4x4", executor_factory_KDFloatCPartialPacking_60007_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("60007_512_4x4", executor_factory_KDFloatCPartialPacking_60007_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("0dfe3_256_4x4", executor_factory_KDFloatCPartialPacking_0dfe3_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("0dfe3_512_4x4", executor_factory_KDFloatCPartialPacking_0dfe3_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("e8c1f_256_4x4", executor_factory_KDFloatCPartialPacking_e8c1f_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("e8c1f_512_4x4", executor_factory_KDFloatCPartialPacking_e8c1f_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("ad3b1_256_4x4", executor_factory_KDFloatCPartialPacking_ad3b1_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("ad3b1_512_4x4", executor_factory_KDFloatCPartialPacking_ad3b1_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("f1006_256_4x4", executor_factory_KDFloatCPartialPacking_f1006_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("f1006_512_4x4", executor_factory_KDFloatCPartialPacking_f1006_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("5eab3_256_4x4", executor_factory_KDFloatCPartialPacking_5eab3_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("5eab3_512_4x4", executor_factory_KDFloatCPartialPacking_5eab3_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("c22a5_256_4x4", executor_factory_KDFloatCPartialPacking_c22a5_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("c22a5_512_4x4", executor_factory_KDFloatCPartialPacking_c22a5_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("77b33_256_8x2", executor_factory_KDFloatCPartialPacking_77b33_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("77b33_512_8x2", executor_factory_KDFloatCPartialPacking_77b33_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("f0bdc_256_4x4", executor_factory_KDFloatCPartialPacking_f0bdc_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("f0bdc_512_4x4", executor_factory_KDFloatCPartialPacking_f0bdc_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("dad5c_256_4x4", executor_factory_KDFloatCPartialPacking_dad5c_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("dad5c_512_4x4", executor_factory_KDFloatCPartialPacking_dad5c_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("45bec_256_4x4", executor_factory_KDFloatCPartialPacking_45bec_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("45bec_512_4x4", executor_factory_KDFloatCPartialPacking_45bec_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("105ad_256_4x4", executor_factory_KDFloatCPartialPacking_105ad_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("105ad_512_4x4", executor_factory_KDFloatCPartialPacking_105ad_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("6a59f_256_4x4", executor_factory_KDFloatCPartialPacking_6a59f_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("6a59f_512_4x4", executor_factory_KDFloatCPartialPacking_6a59f_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("d508e_256_4x4", executor_factory_KDFloatCPartialPacking_d508e_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("d508e_512_4x4", executor_factory_KDFloatCPartialPacking_d508e_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("91aaa_256_4x4", executor_factory_KDFloatCPartialPacking_91aaa_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("91aaa_512_4x4", executor_factory_KDFloatCPartialPacking_91aaa_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("acd31_256_4x4", executor_factory_KDFloatCPartialPacking_acd31_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("acd31_512_4x4", executor_factory_KDFloatCPartialPacking_acd31_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("a8d4c_256_4x4", executor_factory_KDFloatCPartialPacking_a8d4c_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("a8d4c_512_4x4", executor_factory_KDFloatCPartialPacking_a8d4c_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("77f9d_256_8x2", executor_factory_KDFloatCPartialPacking_77f9d_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("77f9d_512_8x2", executor_factory_KDFloatCPartialPacking_77f9d_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("7bf97_256_4x4", executor_factory_KDFloatCPartialPacking_7bf97_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("7bf97_512_4x4", executor_factory_KDFloatCPartialPacking_7bf97_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("64487_256_4x4", executor_factory_KDFloatCPartialPacking_64487_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("64487_512_4x4", executor_factory_KDFloatCPartialPacking_64487_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("520b4_256_4x4", executor_factory_KDFloatCPartialPacking_520b4_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("520b4_512_4x4", executor_factory_KDFloatCPartialPacking_520b4_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("470b8_256_4x4", executor_factory_KDFloatCPartialPacking_470b8_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("470b8_512_4x4", executor_factory_KDFloatCPartialPacking_470b8_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("5b38e_256_8x2", executor_factory_KDFloatCPartialPacking_5b38e_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("5b38e_512_8x2", executor_factory_KDFloatCPartialPacking_5b38e_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("28600_256_4x4", executor_factory_KDFloatCPartialPacking_28600_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("28600_512_4x4", executor_factory_KDFloatCPartialPacking_28600_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("3e5d4_256_4x4", executor_factory_KDFloatCPartialPacking_3e5d4_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("3e5d4_512_4x4", executor_factory_KDFloatCPartialPacking_3e5d4_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("0e71b_256_4x4", executor_factory_KDFloatCPartialPacking_0e71b_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("0e71b_512_4x4", executor_factory_KDFloatCPartialPacking_0e71b_512_4x4());
#endif // __AVX512VL__
}
};

ExecutorFactoryKDFloatCPartialPacking trip_registration_for_KDFloatCPartialPacking;

struct ExecutorFactoryKDFloatNoPacking : public ExecutorFactory<KDFloatNoPacking> {
ExecutorFactoryKDFloatNoPacking(){
#ifdef __AVX2__
  register_factory("60007_256_4x4", executor_factory_KDFloatNoPacking_60007_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("60007_512_4x4", executor_factory_KDFloatNoPacking_60007_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("0dfe3_256_4x4", executor_factory_KDFloatNoPacking_0dfe3_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("0dfe3_512_4x4", executor_factory_KDFloatNoPacking_0dfe3_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("e8c1f_256_4x4", executor_factory_KDFloatNoPacking_e8c1f_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("e8c1f_512_4x4", executor_factory_KDFloatNoPacking_e8c1f_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("ad3b1_256_4x4", executor_factory_KDFloatNoPacking_ad3b1_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("ad3b1_512_4x4", executor_factory_KDFloatNoPacking_ad3b1_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("f1006_256_4x4", executor_factory_KDFloatNoPacking_f1006_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("f1006_512_4x4", executor_factory_KDFloatNoPacking_f1006_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("5eab3_256_4x4", executor_factory_KDFloatNoPacking_5eab3_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("5eab3_512_4x4", executor_factory_KDFloatNoPacking_5eab3_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("c22a5_256_4x4", executor_factory_KDFloatNoPacking_c22a5_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("c22a5_512_4x4", executor_factory_KDFloatNoPacking_c22a5_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("77b33_256_8x2", executor_factory_KDFloatNoPacking_77b33_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("77b33_512_8x2", executor_factory_KDFloatNoPacking_77b33_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("f0bdc_256_4x4", executor_factory_KDFloatNoPacking_f0bdc_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("f0bdc_512_4x4", executor_factory_KDFloatNoPacking_f0bdc_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("dad5c_256_4x4", executor_factory_KDFloatNoPacking_dad5c_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("dad5c_512_4x4", executor_factory_KDFloatNoPacking_dad5c_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("45bec_256_4x4", executor_factory_KDFloatNoPacking_45bec_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("45bec_512_4x4", executor_factory_KDFloatNoPacking_45bec_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("105ad_256_4x4", executor_factory_KDFloatNoPacking_105ad_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("105ad_512_4x4", executor_factory_KDFloatNoPacking_105ad_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("6a59f_256_4x4", executor_factory_KDFloatNoPacking_6a59f_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("6a59f_512_4x4", executor_factory_KDFloatNoPacking_6a59f_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("d508e_256_4x4", executor_factory_KDFloatNoPacking_d508e_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("d508e_512_4x4", executor_factory_KDFloatNoPacking_d508e_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("91aaa_256_4x4", executor_factory_KDFloatNoPacking_91aaa_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("91aaa_512_4x4", executor_factory_KDFloatNoPacking_91aaa_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("acd31_256_4x4", executor_factory_KDFloatNoPacking_acd31_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("acd31_512_4x4", executor_factory_KDFloatNoPacking_acd31_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("a8d4c_256_4x4", executor_factory_KDFloatNoPacking_a8d4c_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("a8d4c_512_4x4", executor_factory_KDFloatNoPacking_a8d4c_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("77f9d_256_8x2", executor_factory_KDFloatNoPacking_77f9d_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("77f9d_512_8x2", executor_factory_KDFloatNoPacking_77f9d_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("7bf97_256_4x4", executor_factory_KDFloatNoPacking_7bf97_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("7bf97_512_4x4", executor_factory_KDFloatNoPacking_7bf97_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("64487_256_4x4", executor_factory_KDFloatNoPacking_64487_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("64487_512_4x4", executor_factory_KDFloatNoPacking_64487_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("520b4_256_4x4", executor_factory_KDFloatNoPacking_520b4_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("520b4_512_4x4", executor_factory_KDFloatNoPacking_520b4_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("470b8_256_4x4", executor_factory_KDFloatNoPacking_470b8_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("470b8_512_4x4", executor_factory_KDFloatNoPacking_470b8_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("5b38e_256_8x2", executor_factory_KDFloatNoPacking_5b38e_256_8x2());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("5b38e_512_8x2", executor_factory_KDFloatNoPacking_5b38e_512_8x2());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("28600_256_4x4", executor_factory_KDFloatNoPacking_28600_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("28600_512_4x4", executor_factory_KDFloatNoPacking_28600_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("3e5d4_256_4x4", executor_factory_KDFloatNoPacking_3e5d4_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("3e5d4_512_4x4", executor_factory_KDFloatNoPacking_3e5d4_512_4x4());
#endif // __AVX512VL__
#ifdef __AVX2__
  register_factory("0e71b_256_4x4", executor_factory_KDFloatNoPacking_0e71b_256_4x4());
#endif // __AVX2__
#ifdef __AVX512VL__
  register_factory("0e71b_512_4x4", executor_factory_KDFloatNoPacking_0e71b_512_4x4());
#endif // __AVX512VL__
}
};

ExecutorFactoryKDFloatNoPacking trip_registration_for_KDFloatNoPacking;

}
