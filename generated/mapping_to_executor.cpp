#include "mapping_to_executor.h"

#include "utils/error.h"


std::unordered_map<
    std::string,
    std::tuple<
        std::string,
        int,
        std::vector<int>
    >
> mapping_to_executor = {
  {"400fa", {"77f9d", 8, { 2 }}},
  {"bf0ff", {"8c90f", 8, { 2 }}},
  {"61fee", {"c22a5", 4, { 4 }}},
  {"da01e", {"64487", 4, { 4 }}}
};

std::string get_executor_id(
    const std::string& mapping_id,
    int vec_width_bits,
    int N_r
) {
  auto [executor_id, M_r, supprted_N_rs] = mapping_to_executor[mapping_id];

  if (N_r >= 1) {
    ERROR_AND_EXIT_IF(std::find(supprted_N_rs.begin(), supprted_N_rs.end(), N_r)
                          == supprted_N_rs.end(),
                      "N_r not supported by mapping");
  } else {
    N_r = supprted_N_rs[0];
  }

  return executor_id +
      "_" + std::to_string(vec_width_bits) +
      "_" + std::to_string(M_r) + "x" + std::to_string(N_r);
}
