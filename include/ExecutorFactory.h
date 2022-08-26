//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_EXECUTORFACTORY_H
#define DNN_SPMM_BENCH_EXECUTORFACTORY_H

#include <unordered_map>

#include "MicroKernelDesc.h"
#include "Executor.h"
#include "Config.h"

namespace sop {

template <typename _KernelDesc>
class ExecutorFactory {
  using Scalar = typename _KernelDesc::Scalar;

public:
  const int M_r;
  const int N_r;

  ExecutorFactory(int M_r = 0, int N_r = 0): M_r(M_r), N_r(N_r) {}
  virtual ~ExecutorFactory() = default;

  virtual Executor* create_specialized_executor(
      int M, int K, int N,
      const vector<vector<PackedTile<Scalar>>>& tiles,
      const Scalar* __restrict__ B,
      Scalar* __restrict__ C,
      int batch_size,
      int num_threads,
      const TileConfig& config
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

  static void register_factory(const std::string& name, ExecutorFactory* factory) {
    get_factories()[name] = factory;
  }
};

template <typename _KernelDesc, typename _MircoKernel>
class ExecutorFactorySpeacilized: public ExecutorFactory<_KernelDesc> {

  using Scalar = typename _MircoKernel::Scalar;
  std::string id;

 public:
  Executor* create_specialized_executor(
    int M, int K, int N,
    const vector<vector<PackedTile<Scalar>>>& tiles,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C,
    int batch_size,
    int num_threads,
    const TileConfig& config
  ) override {
    return new ExecutorSpecialized<_KernelDesc, MicroKernelDesc<_MircoKernel>>(
      M, K, N, tiles, B, C, batch_size, num_threads, config);
  }
};

};

#endif // DNN_SPMM_BENCH_EXECUTORFACTORY_H
