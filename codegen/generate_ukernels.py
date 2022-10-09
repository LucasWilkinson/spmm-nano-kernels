import math
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# import gmpy
import glob
import json
import shutil

from collections import defaultdict
from .codegen_utils import *
from .base_ukernel_codegen import UKernelCodegenBase
from .generate_registration import generate_ukernel_registration
from .generate_mapping import generate_mapping_to_executor

output_root = os.path.abspath(f"{SCRIPT_DIR}/../generated/")
kernel_descs = {
    "AVX512": [
        'KD_IntelFloatNKM',
        'KD_IntelFloatLoadBalancedNKM',
        'KD_IntelFloatCPackedNKM',
        'KD_IntelFloatLoadBalancedCPackedNKM',
        'KD_IntelFloatKNM',
        'KD_IntelFloatLoadBalancedKNM',
        'KD_IntelFloatCPackedKNM',
        'KD_IntelFloatLoadBalancedCPackedKNM'
    ],
    "AVX2": [
        'KD_IntelFloatNKM',
        'KD_IntelFloatLoadBalancedNKM',
        'KD_IntelFloatCPackedNKM',
        'KD_IntelFloatLoadBalancedCPackedNKM',
        'KD_IntelFloatKNM',
        'KD_IntelFloatLoadBalancedKNM',
        'KD_IntelFloatCPackedKNM',
        'KD_IntelFloatLoadBalancedCPackedKNM'
    ],
    "NEON": [
        'KD_PIFloatSplitN',
        'KD_PIFloatSplitM',
        'KD_PIFloatLoadBalancedSplitM',
        #    'KDFloatLoadBalanced'
    ]
}

mappings_to_generate = ["61fee", "da01e", "400fa", "747f9"]

shutil.rmtree(output_root, ignore_errors=True)
os.makedirs(output_root, exist_ok=True)

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

    mapping_key = mapping_file.split("/")[-1].replace(".txt", "").split('_')[-1]
    common_args = dict(output_root=output_root, build_factories_for=kernel_descs)

    codegen = UKernelCodegenBase(Mr=M_r, nanokernels=list(patterns), output_root=output_root)
    nkern_hash = codegen.nanokernel_hash

    scalars_to_generate = {
        "AVX512": ['float'],
        "AVX2": ['float'],
        "NEON": ['float']
    }

    Nrs_to_generate = {
        "AVX512": {4: [6, 4], 8: [3, 2]},
        "AVX2":   {4: [6, 4], 8: [1, 2]},
        "NEON":   {4: [4, 3, 2], 8: [4, 3, 2, 1]},
    }

    vecwidths_to_generate = {
        "AVX512": [512],
        "AVX2":   [256],
        "NEON":   [128],
    }

    kernel_descs

    def gen(arch):
        for scalar in scalars_to_generate[arch]:
            for Nr in Nrs_to_generate[arch][M_r]:
                for vecwidth in vecwidths_to_generate[arch]:
                    common_args = dict(Nr=Nr, arch=arch, vec_width_bits=vecwidth, scalar=scalar)
                    ukern_id = codegen.gen_header(**common_args)
                    codegen.gen_factories(**common_args, build_factories_for=kernel_descs[arch])

                    mapping_to_executor[mapping_key][f'{nkern_hash}'][arch].append((M_r, Nr))
                    print(f'{mapping_file.split("/")[-1]} -> {ukern_id}')

    gen("AVX512")
    gen("AVX2")
    gen("NEON")

with open(f'{output_root}/mapping_to_executor.cpp', 'w') as f:
    f.write('#include <unordered_map>\n')

generate_ukernel_registration(output_root)
generate_mapping_to_executor(output_root, mapping_to_executor)


