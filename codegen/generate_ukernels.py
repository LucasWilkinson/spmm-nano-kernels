import math
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import gmpy
import glob
import json
import shutil

from collections import defaultdict
from .codegen_utils import *
from .base_ukernel_codegen import UKernelCodegenBase
from .generate_registration import generate_ukernel_registration
from .generate_mapping import generate_mapping_to_executor

output_root = os.path.abspath(f"{SCRIPT_DIR}/../generated/")
kernel_descs = [
    'KDFloatNoPacking',
    'KDFloatCPartialPacking',
    'KDFloatNoPackingLoadBalanced'
]

mappings_to_generate = ["61fee", "da01e", "400fa", "747f9"]

shutil.rmtree(output_root, ignore_errors=True)
os.makedirs(output_root)

mapping_to_executor = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

for mapping_file in [f'{SCRIPT_DIR}/../mappings/mapping_{mapping_id}.txt' for mapping_id in mappings_to_generate]:
    patterns = set()
    max_mapped = 0
    with open(mapping_file, 'r') as f:
        f.readline()
        for line in f.readlines():
            line = line.split(':')
            max_mapped = max(max_mapped, int(line[0]))
            for pat in json.loads(line[1]):
                patterns.add(pat)

    M_r = int(round(math.log2(max_mapped)))
    N_r_avx512 = 16 // M_r
    N_r_avx2 = 8 // M_r

    mapping_key = mapping_file.split("/")[-1].replace(".txt", "").split('_')[-1]
    common_args = dict(output_root=output_root, build_factories_for=kernel_descs)

    codegen = UKernelCodegenBase(Mr=M_r, nanokernels=list(patterns), output_root=output_root)
    nkern_hash = codegen.nanokernel_hash

    common_args = dict(Nr=N_r_avx512, arch="AVX512", vec_width_bits=512, scalar="float")
    ukern_id = codegen.gen_header(**common_args)
    codegen.gen_factories(**common_args, build_factories_for=kernel_descs)

    mapping_to_executor[mapping_key][f'{nkern_hash}']["AVX512"].append((M_r, N_r_avx512))
    print(f'{mapping_file.split("/")[-1]} -> {ukern_id}')

    if N_r_avx512 == 4:
        common_args = dict(Nr=2, arch="AVX512", vec_width_bits=512, scalar="float")
        ukern_id = codegen.gen_header(**common_args)
        codegen.gen_factories(**common_args, build_factories_for=kernel_descs)

        mapping_to_executor[mapping_key][f'{nkern_hash}']["AVX512"].append((M_r, N_r_avx512))
        print(f'{mapping_file.split("/")[-1]} -> {ukern_id}')

    common_args = dict(Nr=N_r_avx2, arch="AVX2", vec_width_bits=256, scalar="float")
    ukern_id = codegen.gen_header(**common_args)
    codegen.gen_factories(**common_args, build_factories_for=kernel_descs)

    mapping_to_executor[mapping_key][f'{nkern_hash}']["AVX2"].append((M_r, N_r_avx512))
    print(f'{mapping_file.split("/")[-1]} -> {ukern_id}')

with open(f'{output_root}/mapping_to_executor.cpp', 'w') as f:
    f.write('#include <unordered_map>\n')

generate_ukernel_registration(output_root)
generate_mapping_to_executor(output_root, mapping_to_executor)


