#include "mapping_to_executor.h"

#include <algorithm>

#include "utils/error.h"


std::unordered_map<
    std::string,
    std::tuple<
        std::string,
        int,
        std::vector<std::pair<std::string, int>>
    >
> mapping_to_executor = {
  {"61fee", {"c22a5", 4, {

#if defined(ENABLE_AVX512)
    {"AVX512", 4},    {"AVX512", 4},
#endif

#if defined(ENABLE_AVX2)
    {"AVX2", 4},
#endif
{"", -1}  }}},
  {"da01e", {"64487", 4, {

#if defined(ENABLE_AVX512)
    {"AVX512", 4},    {"AVX512", 4},
#endif

#if defined(ENABLE_AVX2)
    {"AVX2", 4},
#endif
{"", -1}  }}},
  {"400fa", {"77f9d", 8, {

#if defined(ENABLE_AVX512)
    {"AVX512", 2},
#endif

#if defined(ENABLE_AVX2)
    {"AVX2", 2},
#endif
{"", -1}  }}},
  {"747f9", {"77f9d", 8, {

#if defined(ENABLE_AVX512)
    {"AVX512", 2},
#endif

#if defined(ENABLE_AVX2)
    {"AVX2", 2},
#endif
{"", -1}  }}}
};

std::string get_executor_id(
    const std::string& mapping_id,
    std::string arch,
    int N_r
) {
  ERROR_AND_EXIT_IF(
    mapping_to_executor.find(mapping_id) == mapping_to_executor.end(),
    "Mapping ID not found: " << mapping_id);
  auto [executor_id, M_r, supprted_N_rs] = mapping_to_executor[mapping_id];

  if (N_r >= 1) {
    for (const auto& [arch_supported, N_r_supported] : supprted_N_rs) {
      if (arch_supported == arch && N_r == N_r_supported) break;
    }
    ERROR_AND_EXIT("N_r not supported by mapping");  } else {
    for (const auto& [arch_supported, N_r_supported] : supprted_N_rs) {
      if (arch_supported == arch) {
        N_r = N_r_supported;
        break;
      }
    }
  }

  return executor_id +
      "_" + arch +
      "_" + std::to_string(M_r) + "x" + std::to_string(N_r);
}
