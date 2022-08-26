//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_PANELPACKERFACTORY_H
#define DNN_SPMM_BENCH_PANELPACKERFACTORY_H

#include "MicroKernelPacker.h"
#include <memory>

namespace sop {

template <typename Scalar>
class MicroKernelPackerFactory {

 public:
  const int M_r;
  const int N_r;

  MicroKernelPackerFactory(int M_r = 0, int N_r = 0): M_r(M_r), N_r(N_r) {}
  virtual ~MicroKernelPackerFactory() = default;

  virtual MicroKernelPacker<Scalar>* create_specialized_packer(
      int M_r, std::shared_ptr<NanoKernelMapping> pattern_mapping) {
    return nullptr;
  };

  // Hack for now to enforce initialization order, only works within
  //   one translation unit.
  static std::unordered_map<std::string, MicroKernelPackerFactory*>& get_factories() {
    static std::unordered_map<std::string, MicroKernelPackerFactory*> factories;
    return factories;
  }

  static MicroKernelPackerFactory* get_factory(const std::string& name) {
    return get_factories()[name];
  }

  static void register_factory(const std::string& name, MicroKernelPackerFactory* factory) {
    get_factories()[name] = factory;
  }
};

template <typename _MircoKernel>
class MicroKernelPackerFactorySpeacilized:
    public MicroKernelPackerFactory<typename _MircoKernel::Scalar> {

  using Scalar = typename _MircoKernel::Scalar;
  std::string id;

 public:
  MicroKernelPackerFactory<Scalar>* create_specialized_executor(
      int M_r, std::shared_ptr<NanoKernelMapping> pattern_mapping) override {
    return new MicroKernelPackerSpeaclized<_MircoKernel>(M_r,  pattern_mapping);
  }
};

} // namespace sop


#endif // DNN_SPMM_BENCH_PANELPACKERFACTORY_H
