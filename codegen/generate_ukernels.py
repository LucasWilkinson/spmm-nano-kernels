import gmpy

from tools.codegen.codegen_utils import *
from SOP.codegen.ukernel_codegen import ukernel_codegen
from SOP.codegen.generate_registration import generate_ukernel_registration

NANO_KERNELS_8 = [
    1 << i for i in range(8) #3 << i for i in range(0, 8, 2)
] + [
    0b01010101, 0b10101010,
    0b11000011, 0b00111100, 0b00001111, 0b11110000,
    0b11111100, 0b11110011, 0b11001111, 0b00111111,
    0b11111111
]

NANO_KERNELS_4 = [
    0b00000001, 0b00000010, 0b00000100, 0b00001000,
    0b00001110, 0b00001101, 0b00001011, 0b00000111,
    0b00001111,
]

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

kernel_descs = ['KDFloatNoPacking', 'KDFloatCPartialPacking' ]

output_root=f'{SCRIPT_DIR}/../generated/'
ukernel_codegen([8, 2], NANO_KERNELS_8, output_root=output_root, build_factories_for=kernel_descs)
ukernel_codegen([4, 4], NANO_KERNELS_4, output_root=output_root, build_factories_for=kernel_descs)

generate_ukernel_registration(output_root)


