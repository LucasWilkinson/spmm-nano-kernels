import math
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import gmpy
import glob
import json

from tools.codegen.codegen_utils import *
from SOP.codegen.ukernel_codegen import ukernel_codegen
from SOP.codegen.generate_registration import generate_ukernel_registration

output_root=f'{SCRIPT_DIR}/../generated/'
kernel_descs = ['KDFloatNoPacking', 'KDFloatCPartialPacking' ]

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

generate_ukernel_registration(output_root)


