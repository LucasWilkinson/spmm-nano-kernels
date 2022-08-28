#pragma once

#include <unordered_map>
#include <vector>

std::string get_executor_id(
    const std::string& mapping_id,
    int vec_width_bits,
    int N_r = -1
);
