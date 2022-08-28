
def generate_mapping_to_executor(output_root, mapping):
    mappings_map_init = ""
    for key, value in mapping.items():
        excutor_id, M_r, N_r = value
        mappings_map_init += f'  {{"{key}", {{"{excutor_id}", {M_r}, {{ {N_r} }}}}}},\n'

    with open(f'{output_root}/mapping_to_executor.h', 'w') as f:
        f.write(
        f'#pragma once\n\n'
        f'#include <unordered_map>\n'
        f'#include <vector>\n\n'
        f'std::string get_executor_id(\n'
        f'    const std::string& mapping_id,\n'
        f'    int vec_width_bits,\n'
        f'    int N_r = -1\n'
        f');\n'
        )
    with open(f'{output_root}/mapping_to_executor.cpp', 'w') as f:
        f.write(
        f'#include "mapping_to_executor.h"\n'
        f'\n'
        f'#include "utils/error.h"\n'
        f'\n'
        f'\n'
        f'std::unordered_map<\n'
        f'    std::string,\n'
        f'    std::tuple<\n'
        f'        std::string,\n'
        f'        int,\n'
        f'        std::vector<int>\n'
        f'    >\n'
        f'> mapping_to_executor = {{\n'
        f'{mappings_map_init.rstrip().rstrip(",")}\n'
        f'}};\n'
        f'\n'
        f'std::string get_executor_id(\n'
        f'    const std::string& mapping_id,\n'
        f'    int vec_width_bits,\n'
        f'    int N_r\n'
        f') {{\n'
        f'  ERROR_AND_EXIT_IF(\n'
        f'    mapping_to_executor.find(mapping_id) == mapping_to_executor.end(),\n'
        f'    "Mapping ID not found: " << mapping_id);\n'
        f'  auto [executor_id, M_r, supprted_N_rs] = mapping_to_executor[mapping_id];\n'
        f'\n'
        f'  if (N_r >= 1) {{\n'
        f'    ERROR_AND_EXIT_IF(std::find(supprted_N_rs.begin(), supprted_N_rs.end(), N_r)\n'
        f'                          == supprted_N_rs.end(),\n'
        f'                      "N_r not supported by mapping");\n'
        f'  }} else {{\n'
        f'    N_r = supprted_N_rs[0];\n'
        f'  }}\n'
        f'\n'
        f'  return executor_id +\n'
        f'      "_" + std::to_string(vec_width_bits) +\n'
        f'      "_" + std::to_string(M_r) + "x" + std::to_string(N_r);\n'
        f'}}\n'
        )