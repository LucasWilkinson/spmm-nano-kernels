import math
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import gmpy
import glob
import json

from tools.codegen.codegen_utils import *
from SOP.codegen.ukernel_codegen import ukernel_codegen
from SOP.codegen.generate_registration import generate_ukernel_registration
from SOP.codegen.generate_mapping import generate_mapping_to_executor

output_root = f'{SCRIPT_DIR}/../generated/'
kernel_descs = ['KDFloatNoPacking', 'KDFloatCPartialPacking', 'KDFloatNoPackingLoadBalanced']

mapping_to_executor = {}

for mapping_file in glob.glob(f'{SCRIPT_DIR}/../mappings/*.txt'):
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
    N_r = 16 // M_r

    nkern_hash = ukernel_codegen([M_r, N_r], list(patterns), output_root=output_root, build_factories_for=kernel_descs)
    print(f'{mapping_file.split("/")[-1]} -> {nkern_hash}_x_{M_r}x{N_r}')

    mapping_to_executor[mapping_file.split("/")[-1].replace(".txt", "").split('_')[-1]] \
        = (f'{nkern_hash}', M_r, N_r)


with open(f'{output_root}/mapping_to_executor.cpp', 'w') as f:
    f.write('#include <unordered_map>\n')

generate_ukernel_registration(output_root)
generate_mapping_to_executor(output_root, mapping_to_executor)


