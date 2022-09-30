import math
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import gmpy
import glob
import json
import shutil

from collections import defaultdict
from tools.codegen.codegen_utils import *
from SOP.codegen.avx_ukernel_codegen import ukernel_codegen
from SOP.codegen.generate_registration import generate_ukernel_registration
from SOP.codegen.generate_mapping import generate_mapping_to_executor

output_root = os.path.abspath(f'{SCRIPT_DIR}/../generated/')
kernel_descs = ['KDFloatNoPacking', 'KDFloatCPartialPacking', 'KDFloatNoPackingLoadBalanced']

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

    nkern_hash = ukernel_codegen([M_r, N_r_avx512], list(patterns), vec_configs=[('float', "AVX512")], **common_args)
    mapping_to_executor[mapping_key][f'{nkern_hash}']["AVX512"].append((M_r, N_r_avx512))
    print(f'{mapping_file.split("/")[-1]} -> {nkern_hash}_512_{M_r}x{N_r_avx512}')

    if N_r_avx512 == 4:
        nkern_hash = ukernel_codegen([M_r, 2], list(patterns), vec_configs=[('float', "AVX512")], **common_args)
        mapping_to_executor[mapping_key][f'{nkern_hash}']["AVX512"].append((M_r, 2))
        print(f'{mapping_file.split("/")[-1]} -> {nkern_hash}_512_{M_r}x2')

    nkern_hash = ukernel_codegen([M_r, N_r_avx2], list(patterns), vec_configs=[('float', "AVX2")], **common_args)
    mapping_to_executor[mapping_key][f'{nkern_hash}']["AVX2"].append((M_r, N_r_avx2))
    print(f'{mapping_file.split("/")[-1]} -> {nkern_hash}_256_{M_r}x{N_r_avx2}')


with open(f'{output_root}/mapping_to_executor.cpp', 'w') as f:
    f.write('#include <unordered_map>\n')

generate_ukernel_registration(output_root)
generate_mapping_to_executor(output_root, mapping_to_executor)


