#pragma once

#include <unordered_map>
#include <vector>

std::string get_executor_id(
    const std::string& mapping_id,
    std::string arch,
    int N_r = -1
);
