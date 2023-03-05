//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_EXECUTORFACTORY_H
#define DNN_SPMM_BENCH_EXECUTORFACTORY_H

#include <unordered_map>
#include <sstream>

#include "MicroKernelDesc.h"
#include "Executor.h"
#include "Config.h"

namespace sop {

template <typename _KernelDesc, bool DataTransform>
class ExecutorFactory {
  using Scalar = typename _KernelDesc::Scalar;

public:
  const int M_r;
  const int N_r;

  ExecutorFactory(int M_r = 0, int N_r = 0): M_r(M_r), N_r(N_r) {}
  virtual ~ExecutorFactory() = default;

  virtual Executor<Scalar>* create_specialized_executor(
      int M, int K, int N, int batch_size,
      const vector<vector<PackedTile<Scalar>>>& tiles,
      const vector<int>& upanel_swizzle,
      int num_threads,
      TileConfig& config
  ) { return nullptr; };

  // Hack for now to enforce initialization order, only works within
  //   one translation unit.
  static std::unordered_map<std::string, ExecutorFactory*>& get_factories() {
    static std::unordered_map<std::string, ExecutorFactory*> factories;
    return factories;
  }

  static ExecutorFactory* get_factory(const std::string& name) {
    return get_factories()[name];
  }

  static std::string dump_registered_factories() {
    std::stringstream ss;
    for (auto& [name, factory] : get_factories()) {
        if (factory != nullptr) {
            ss << name << std::endl;
        }
    }
    return ss.str();
  }

  static void register_factory(const std::string& name, ExecutorFactory* factory) {
    get_factories()[name] = factory;
  }
};

template <typename _KernelDesc, typename _MircoKernel, bool DataTransform>
// template <typename ExecutorWithSchedule, typename _KernelDesc, typename _MircoKernel> // TODO: For Schedule
class ExecutorFactorySpecialized: public ExecutorFactory<_KernelDesc, DataTransform> {

  using Super = ExecutorFactory<_KernelDesc, DataTransform>;
  using Scalar = typename _MircoKernel::Scalar;
  std::string id;

 public:
  ExecutorFactorySpecialized(int M_r, int N_r): Super(M_r, N_r) {}

  Executor<Scalar>* create_specialized_executor(
    int M, int K, int N, int batch_size,
    const vector<vector<PackedTile<Scalar>>>& tiles,
    const vector<int>& upanel_swizzle,
    int num_threads,
    TileConfig& config
  ) override {
    // return new ExecutorWithSchedule<_KernelDesc, MicroKernelDesc<_MircoKernel>>( // TODO: For Schedule
    return new ExecutorSpecialized<_KernelDesc, MicroKernelDesc<_MircoKernel>, DataTransform>(
      M, K, N, batch_size, tiles, upanel_swizzle, num_threads, config);
  }
};

};

#endif // DNN_SPMM_BENCH_EXECUTORFACTORY_H
