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
  {"8ee60", {"a8d4c", 4, { 4 }}},
  {"b48fd", {"60007", 4, { 4 }}},
  {"86ca2", {"f0bdc", 4, { 4 }}},
  {"400fa", {"77f9d", 8, { 2 }}},
  {"4892b", {"6a59f", 4, { 4 }}},
  {"8e68c", {"ad3b1", 4, { 4 }}},
  {"92be7", {"5b38e", 8, { 2 }}},
  {"747f9", {"77f9d", 8, { 2 }}},
  {"5535a", {"28600", 4, { 4 }}},
  {"29556", {"0e71b", 4, { 4 }}},
  {"5280a", {"3e5d4", 4, { 4 }}},
  {"61fee", {"c22a5", 4, { 4 }}},
  {"2da33", {"dad5c", 4, { 4 }}},
  {"dc018", {"105ad", 4, { 4 }}},
  {"da01e", {"64487", 4, { 4 }}},
  {"cf3f2", {"91aaa", 4, { 4 }}},
  {"b792a", {"520b4", 4, { 4 }}},
  {"9bec8", {"f1006", 4, { 4 }}},
  {"2cbb1", {"5eab3", 4, { 4 }}},
  {"30842", {"77b33", 8, { 2 }}}
};

std::string get_executor_id(
    const std::string& mapping_id,
    int vec_width_bits,
    int N_r
) {
  ERROR_AND_EXIT_IF(
    mapping_to_executor.find(mapping_id) == mapping_to_executor.end(),
    "Mapping ID not found: " << mapping_id);
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
