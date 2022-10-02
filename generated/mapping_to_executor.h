#pragma once

#include <unordered_map>
#include <string>
#include <vector>

std::string get_executor_id(
    const std::string& mapping_id,
    std::string arch,
    int vec_width_bits,
    int N_r = -1
);
