//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_PANELPACKERFACTORY_H
#define DNN_SPMM_BENCH_PANELPACKERFACTORY_H

#include "MicroKernelPacker.h"
#include "MicroKernelDesc.h"
#include <memory>
#include <unordered_map>

namespace sop {

template <typename Scalar>
class MicroKernelPackerFactory {

 public:
  const int M_r;

  MicroKernelPackerFactory(int M_r = 0): M_r(M_r) {}
  virtual ~MicroKernelPackerFactory() = default;

  virtual std::shared_ptr<MicroKernelPacker<Scalar>> create_specialized_packer(
      std::shared_ptr<NanoKernelMapping> pattern_mapping) {
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
class MicroKernelPackerFactorySpecialized:
    public MicroKernelPackerFactory<typename _MircoKernel::Scalar> {

  using Super = MicroKernelPackerFactory<typename _MircoKernel::Scalar>;
  using Scalar = typename _MircoKernel::Scalar;
  using _MicroKernelDesc = MicroKernelDesc<_MircoKernel>;
  std::string id;

 public:
  MicroKernelPackerFactorySpecialized(int M_r = 0): Super(M_r) {}

  std::shared_ptr<MicroKernelPacker<Scalar>> create_specialized_packer(
      std::shared_ptr<NanoKernelMapping> pattern_mapping) override {
    return std::make_shared<MicroKernelPackerSpeaclized<_MicroKernelDesc>>
        (this->M_r,  pattern_mapping);
  }
};

} // namespace sop


#endif // DNN_SPMM_BENCH_PANELPACKERFACTORY_H
