//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_MAPPINGIO_H
#define DNN_SPMM_BENCH_MAPPINGIO_H


#include <string>
#include <vector>

#include "Storage.h"

#include <memory>

namespace sop {

std::shared_ptr<NanoKernelMapping> read_pattern_mapping(
    const std::string& id,
    const std::vector<std::string>& search_dirs
);

} // namespace sop

#endif // DNN_SPMM_BENCH_MAPPINGIO_H
