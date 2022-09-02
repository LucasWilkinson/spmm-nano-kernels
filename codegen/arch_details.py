vec_type_info = {
    ("float", 512):  (16, '', 's'),
    ("float", 256):  (8,  '', 's'),
    ("double", 512): (8, 'd', 'd'),
    ("double", 256): (4, 'd', 'd'),
}

min_instruction_sets = {
    512: "__AVX512VL__",
    256: "__AVX2__",
}
