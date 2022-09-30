
def generate_mapping_to_executor(output_root, mapping):
    mappings_map_init = ""
    for mapping_id, value in mapping.items():
        for executor_id, value2 in value.items():
            mr_set = False
            mappings_map_init += f'  {{"{mapping_id}", {{"{executor_id}",'

            for arch, accs in value2.items():
                if not mr_set:
                    mappings_map_init += f' {accs[0][0]}, {{\n'
                    mr_set = True

                mappings_map_init += f'\n#if defined(ENABLE_{arch})\n'
                for acc in accs:
                    mappings_map_init += f'    {{"{arch}", {acc[1]}}},'
                mappings_map_init += f'\n#endif\n'

            mappings_map_init += f'{{"", -1}}  }}}}}},\n'

    with open(f'{output_root}/mapping_to_executor.h', 'w') as f:
        f.write(
        f'#pragma once\n\n'
        f'#include <unordered_map>\n'
        f'#include <vector>\n\n'
        f'std::string get_executor_id(\n'
        f'    const std::string& mapping_id,\n'
        f'    std::string arch,\n'
        f'    int N_r = -1\n'
        f');\n'
        )
    with open(f'{output_root}/mapping_to_executor.cpp', 'w') as f:
        f.write(
        f'#include "mapping_to_executor.h"\n\n'
        f'#include <algorithm>\n\n'
        f'#include "utils/error.h"\n'
        f'\n'
        f'\n'
        f'std::unordered_map<\n'
        f'    std::string,\n'
        f'    std::tuple<\n'
        f'        std::string,\n'
        f'        int,\n'
        f'        std::vector<std::pair<std::string, int>>\n'
        f'    >\n'
        f'> mapping_to_executor = {{\n'
        f'{mappings_map_init.rstrip().rstrip(",")}\n'
        f'}};\n'
        f'\n'
        f'std::string get_executor_id(\n'
        f'    const std::string& mapping_id,\n'
        f'    std::string arch,\n'
        f'    int N_r\n'
        f') {{\n'
        f'  ERROR_AND_EXIT_IF(\n'
        f'    mapping_to_executor.find(mapping_id) == mapping_to_executor.end(),\n'
        f'    "Mapping ID not found: " << mapping_id);\n'
        f'  auto [executor_id, M_r, supprted_N_rs] = mapping_to_executor[mapping_id];\n'
        f'\n'
        f'  if (N_r >= 1) {{\n'
        f'    for (const auto& [arch_supported, N_r_supported] : supprted_N_rs) {{\n'
        f'      if (arch_supported == arch && N_r == N_r_supported) break;\n'
        f'    }}\n'
        f'    ERROR_AND_EXIT("N_r not supported by mapping");'
        f'  }} else {{\n'
        f'    for (const auto& [arch_supported, N_r_supported] : supprted_N_rs) {{\n'
        f'      if (arch_supported == arch) {{\n'
        f'        N_r = N_r_supported;\n'
        f'        break;\n'
        f'      }}\n'
        f'    }}\n'
        f'  }}\n'
        f'\n'
        f'  return executor_id +\n'
        f'      "_" + arch +\n'
        f'      "_" + std::to_string(M_r) + "x" + std::to_string(N_r);\n'
        f'}}\n'
        )