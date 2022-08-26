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
  static inline std::unordered_map<std::string, ExecutorFactory*> factories;

  ExecutorFactory() = default;
  virtual ~ExecutorFactory() = default;

  virtual Executor* create_specialized_executor(
      int M, int K, int N,
      const vector<vector<PackedTile<Scalar>>>& tiles,
      const Scalar* __restrict__ B,
      Scalar* __restrict__ C,
      int batch_size,
      int num_threads,
      const TileConfig& config
  ) = 0;
};

template <typename _KernelDesc, typename _MircoKernel>
class ExecutorFactorySpeacilized:
    public ExecutorFactory<_KernelDesc> {

  using Scalar = typename _MircoKernel::Scalar;
  std::string id;

 public:
  ExecutorFactorySpeacilized(): id(_MircoKernel::id) {
    std::cout << "Registering executor factory for " << id << " " << typeid(_KernelDesc).name() << std::endl;
    ExecutorFactory<_KernelDesc>::factories[id] = this;
    std::cout << "Done Registering executor factory for " << id << " " << typeid(_KernelDesc).name() << std::endl;
  };
  virtual ~ExecutorFactorySpeacilized() {
    std::cout << "UnRegistering executor factory for " << id << " " << typeid(_KernelDesc).name() << std::endl;
    ExecutorFactory<_KernelDesc>::factories[id] = nullptr;
  };

  Executor* create_specialized_executor(
    int M, int K, int N,
    const vector<vector<PackedTile<Scalar>>>& tiles,
    const Scalar* __restrict__ B,
    Scalar* __restrict__ C,
    int batch_size,
    int num_threads,
    const TileConfig& config
  ) {
    return new ExecutorSpecialized<_KernelDesc, MicroKernelDesc<_MircoKernel>>(
      M, K, N, tiles, B, C, batch_size, num_threads, config);
  }
};

};

#endif // DNN_SPMM_BENCH_EXECUTORFACTORY_H
