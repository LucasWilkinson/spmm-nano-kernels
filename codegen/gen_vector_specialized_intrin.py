import gmpy

from tools.codegen.codegen_utils import *
from SOP.codegen.sop_codegen_intrin import gen_for_vec_height

SUPPORTED_PATTERNS_16 = [
    1 << i for i in range(16)
] + [

   0b0100010001000100, 0b1010101010101010,
   0b11000011, 0b00111100, 0b00001111, 0b11110000,
   0b11111100, 0b11110011, 0b11001111, 0b00111111,
   0b11111111
]


SUPPORTED_PATTERNS_8 = [
    1 << i for i in range(8) #3 << i for i in range(0, 8, 2)
] + [
    0b01010101, 0b10101010,
    0b11000011, 0b00111100, 0b00001111, 0b11110000,
    0b11111100, 0b11110011, 0b11001111, 0b00111111,
    0b11111111
]

SUPPORTED_PATTERNS_4 = [
    0b00000001, 0b00000010, 0b00000100, 0b00001000,
    0b00001110, 0b00001101, 0b00001011, 0b00000111,
    0b00001111,
]

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
gen_for_vec_height('', [8, 2], SUPPORTED_PATTERNS_8,
                   output_path=f'{SCRIPT_DIR}/../generated/sop_micro_kernel_8_2_intrin.h')
gen_for_vec_height('', [4, 4], SUPPORTED_PATTERNS_4,
                   output_path=f'{SCRIPT_DIR}/../generated/sop_micro_kernel_4_4_intrin.h')


