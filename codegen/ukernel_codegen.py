import gmpy
import hashlib
import os
import json

from tools.codegen.codegen_utils import *
from functools import partial


vec_type_info = {
    ("float", 512):  (16, '', 's'),
    ("float", 256):  (8,  '', 's'),
    ("double", 512): (8, 'd', 'd'),
    ("double", 256): (4, 'd', 'd'),
}


def executor_factory_name(kernel_desc, microkernel_id):
    return f'executor_factory_{kernel_desc}_{microkernel_id}'

def packer_factory_name(microkernel_id):
    return f'packer_factory_{microkernel_id}'

def sort_nanokernels(nanokernels):
    return sorted(nanokernels, key=lambda x: gmpy.popcount(x))

def nanokernel_hash(nanokernels):
    nanokernels = sort_nanokernels(nanokernels)
    _hash = hashlib.md5(" ".join([str(p) for p in nanokernels]).encode("utf8")).hexdigest()
    return _hash[-5:]

def microkernel_id(vec_width, acc_dims, nanokernels):
    _hash = hashlib.md5(" ".join([str(p) for p in nanokernels]).encode("utf8")).hexdigest()
    return f'{nanokernel_hash(nanokernels)}_{vec_width}_{acc_dims[0]}x{acc_dims[1]}'


def microkernel_typename(scalar, vec_width, acc_dims, nanokernels):
    return f'MicroKernel_{scalar}_{microkernel_id(vec_width, acc_dims, nanokernels)}'


def unroll_mapping(nnzs):
    if nnzs == 1: return 2
    if nnzs <= 2: return 2
    return 2

def ukernel_codegen(acc_dims, nanokernels,
                    vec_configs = [('float', 512)],
                    build_factories_for=None,
                    output_path=None,
                    output_root=None,
                    namespace=None):
    vec_height = acc_dims[0]
    namespace = namespace if namespace is not None else "sop"
    nanokernels = sort_nanokernels(nanokernels)

    assert output_path is None or output_root is None
    assert build_factories_for is None or output_path is None

    nanokernel_hash_ = nanokernel_hash(nanokernels)

    micro_kernel_typename_names = [
        microkernel_typename(scalar, vec_width, acc_dims, nanokernels)
        for scalar, vec_width in vec_configs
    ]


    supported_patterns = sorted(nanokernels, key=lambda x: gmpy.popcount(x))

    # Utility functions
    supported_patterns_list = ",\n            ".join([f'0b{x:08b}' for x in supported_patterns])

    supported_pattern_getter = f'''
    static const uint16_t* supported_nkern_patterns() {{
        static uint16_t patterns[] = {{
            {supported_patterns_list}
        }};
    
        return patterns;
    }}
    '''

    supported_patterns_encode_cases = "\n        ".join([
        f'if (nkern_pat == 0b{x:08b}) return {i};' for i, x in enumerate(supported_patterns)])

    supported_pattern_encode = f'''
    static uint16_t encode_nkern_pattern(uint16_t nkern_pat) {{
        {supported_patterns_encode_cases}
        if (nkern_pat == 0) return sop::ZERO_PATTERN_ID; 
        ERROR_AND_EXIT("Unable to map unsupported nanokernel pattern " <<  (int) nkern_pat);
        return 0;
    }}
    '''

    supported_patterns_decode_cases = "\n        ".join([
        f'if (nkern_code == {i}) return 0b{x:08b};' for i, x in enumerate(supported_patterns)])

    supported_pattern_decode = f'''
    static uint16_t decode_nkern_pattern(uint16_t nkern_code) {{
        {supported_patterns_decode_cases}
        if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
        ERROR_AND_EXIT("Unable to unmap unsupported nanokernel pattern id " << (int) nkern_code);
        return 0;
    }}
    '''

    pattern_nnz_count_cases = "\n        ".join([
        f'if (nkern_code == {i}) return {gmpy.popcount(x)};' for i, x in enumerate(supported_patterns)])

    pattern_nnz_count = f'''
    static uint16_t nnz_count_for_nkern_code(uint16_t nkern_code) {{
        {pattern_nnz_count_cases}
        if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
        ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
        return 0;
    }}
    '''

    import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    output_root = f'{SCRIPT_DIR}/_C/generated' if output_root is None else output_root

    def ukern_header_path(microkernel_typename):
        return f'{output_root}/{microkernel_typename}.h'

    if build_factories_for is not None:
        factory_output_root = f'{output_root}/factories/{nanokernel_hash_}/'
        os.makedirs(factory_output_root, exist_ok=True)

        print(f'Created output directory {output_root}')

        for (scalar, vec_width), micro_kernel_typename in zip(vec_configs, micro_kernel_typename_names):
            microkernel_id_ = microkernel_id(vec_width, acc_dims, nanokernels)

            header = ukern_header_path(micro_kernel_typename).split('/')[-1]
            factory_desc_json = json.dumps({
                "id": microkernel_id_,
                "func": packer_factory_name(microkernel_id_),
                "scalar": scalar,
                "M_r": acc_dims[0]
            })

            with open(factory_output_root + f'packer_{scalar}.cpp', 'w+') as f:
                f.write(f'#include "MicroKernelPackerFactory.h"\n')
                f.write(f'#include "{header}"\n')
                f.write(f'\n')
                f.write(f'namespace {namespace} {{\n')
                f.write(f'\n')
                f.write(f'// factory_desc | {factory_desc_json}\n')
                f.write(f'MicroKernelPackerFactory<{scalar}>* {packer_factory_name(microkernel_id_)}() {{\n')
                f.write(f'    return new MicroKernelPackerFactorySpecialized<{micro_kernel_typename}>('
                        f'{acc_dims[0]});\n')
                f.write(f'}}\n')
                f.write('\n')
                f.write(f'}} // namespace {namespace}\n')

            for kernel_desc in build_factories_for:
                factory_name = executor_factory_name(kernel_desc, microkernel_id_)
                header = ukern_header_path(micro_kernel_typename).split('/')[-1]
                factory_desc_json = json.dumps({
                    "id": microkernel_id_,
                    "func": factory_name,
                    "kernel_desc": kernel_desc,
                    "M_r": acc_dims[0],
                    "N_r": acc_dims[1],
                })

                with open(factory_output_root + f'executor_{kernel_desc}_{scalar}.cpp', 'w+') as f:
                    f.write(f'#include "ExecutorFactory.h"\n')
                    f.write(f'#include "KernelDesc.h"\n')
                    f.write(f'#include "{header}"\n')
                    f.write(f'\n')
                    f.write(f'namespace {namespace} {{\n')
                    f.write(f'\n')
                    f.write(f'// factory_desc | {factory_desc_json}\n')
                    f.write(f'ExecutorFactory<{kernel_desc}>* {factory_name}() {{\n')
                    f.write(f'    return new ExecutorFactorySpecialized<{kernel_desc}, {micro_kernel_typename}>('
                            f'{acc_dims[0]}, {acc_dims[1]});\n')
                    f.write(f'}}\n')
                    f.write('\n')
                    f.write(f'}} // namespace {namespace}\n')


    for (scalar, vec_width), micro_kernel_typename in zip(vec_configs, micro_kernel_typename_names):
        reg_width_ele, _, _ = vec_type_info[(scalar, vec_width)]
        microkernel_id_ = microkernel_id(vec_width, acc_dims, nanokernels)

        with open(ukern_header_path(micro_kernel_typename), 'w+') as f:
            f.write(f'#pragma once\n\n')
            f.write(f'#include "utils/error.h"\n')
            f.write(f'#include "MicroKernelBase.h"\n')
            f.write(f'#include "Storage.h"\n')
            f.write(f'\n')
            f.write(f'#include <immintrin.h>\n\n')
            f.write(f'\n')
            f.write(f'namespace {namespace} {{')
            f.write(f'\n')
            f.write(f'struct {micro_kernel_typename} {{\n')
            f.write(supported_pattern_getter)
            f.write(supported_pattern_encode)
            f.write(supported_pattern_decode)
            f.write(pattern_nnz_count)
            f.write('\n')
            f.write(f'    using Mask = __mmask{reg_width_ele};\n')
            f.write(f'    static Mask create_mask(int n) {{ return ((1 << n) - 1); }}\n')
            f.write(f'    static Mask precomp_mask(int N) {{ return create_mask(N % {reg_width_ele}); }}\n')
            f.write('\n')
            f.write(f'    using Scalar = {scalar};\n')
            f.write(f'    static constexpr int M_r = {acc_dims[0]};\n')
            f.write(f'    static constexpr int N_r = {acc_dims[1]} * {reg_width_ele};\n')
            f.write(f'    static constexpr int N_r_reg = {acc_dims[1]};\n')
            f.write(f'    static constexpr int vec_width_bits = {vec_width};\n')
            f.write(f'    static constexpr const char* id = "{microkernel_id_}";\n')
            f.write(f'    static int max_acc_width_in_vecs() {{ return {acc_dims[1]}; }};\n')
            f.write(f'    static int max_acc_width_in_eles() {{ return {acc_dims[1]} * {reg_width_ele}; }};\n\n')
            f.write(f'    static int num_nkern_patterns() {{ return {len(nanokernels)}; }}\n')
            f.write(f'')

            acc_widths = [acc_dims[1], 1]
            max_acc_width = max(acc_widths)

            for i, acc_width in enumerate(acc_widths):
                acc_width_str = str(acc_width) if i != 0 else "max_acc"

                sop_panel_executor = f'''
    __ALWAYS_INLINE static void _microkernel_{acc_width_str}(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], acc_width], max_acc_width, nanokernels, 
                           packed_C=False, packed_B=False).emit(6)}
    }}\n\n

    __ALWAYS_INLINE static void microkernel_{acc_width_str}(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c) {{
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_{acc_width_str}(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }}
    '''

                sop_panel_executor_packed = f'''
    __ALWAYS_INLINE static void _microkernel_packed_{acc_width_str}(
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], acc_width], max_acc_width, nanokernels, 
                           packed_C=True, packed_B=True).emit(6)}
    }}\n\n

    __ALWAYS_INLINE static void microkernel_packed_{acc_width_str}(
        const sop::MicroKernelPackedData& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c) {{
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_{acc_width_str}(
            nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }}
    '''

                sop_panel_executor_packed_C = f'''
    __ALWAYS_INLINE static void _microkernel_packed_C_{acc_width_str}(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], acc_width], max_acc_width, nanokernels, 
                           packed_C=True, packed_B=False).emit(6)}
    }}\n\n
    
    static void microkernel_packed_C_{acc_width_str}(
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c) {{
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _microkernel_packed_C_{acc_width_str}(
            M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }}
    '''
                f.write(sop_panel_executor)
                f.write(sop_panel_executor_packed)
                f.write(sop_panel_executor_packed_C)

            sop_panel_executor_packed_masked_C = f'''
    __ALWAYS_INLINE static void _microkernel_masked_1(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], 1], max_acc_width, nanokernels,
                           packed_C=False, packed_B=False, masked=True).emit(6)}
    }}\n\n
    
    __ALWAYS_INLINE static void _microkernel_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {{

        int _j = 0;
        for (; _j < N_rem - {reg_width_ele-1}; _j += {reg_width_ele}) {{
            _microkernel_1(
                M, K, N,
                nkern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }}
        
        _microkernel_masked_1(
            M, K, N,
            nkern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }}
    
     static void microkernel_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {{

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _microkernel_masked_max_acc(
            N_rem, M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }}
    '''
            f.write(sop_panel_executor_packed_masked_C)

            sop_panel_executor_packed_masked_C = f'''
    __ALWAYS_INLINE static void _microkernel_masked_packed_C_1(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], 1], max_acc_width, nanokernels,
                           packed_C=True, packed_B=False, masked=True).emit(6)}
    }}\n\n
    
    static void _microkernel_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {{

        int _j = 0;
        for (; _j < N_rem - {reg_width_ele-1}; _j += {reg_width_ele}) {{
            _microkernel_packed_C_1(
                M, K, N,
                nkern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }}
        
        _microkernel_masked_packed_C_1(
            M, K, N,
            nkern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }}
    
    static void microkernel_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        const sop::MicroKernelPackedData& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {{

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       nkern_counts = panel_desc.nkern_counts;
        int                     num_nkern = panel_desc.num_nkern;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _microkernel_masked_packed_C_max_acc(
            N_rem, M, K, N, nkern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }}
    '''
            f.write(sop_panel_executor_packed_masked_C)

            f.write(f'\n}};\n\n')
            f.write(f'}} // {namespace}\n')

    return nanokernel_hash_

def gen_executor_body(scalar, reg_width_bits, acc_dims, max_acc_width, supported_patterns,
                      packed_C=False, packed_B=False, masked=False):
    PREFETCH_B = False
    reg_width_ele, m_reg_char, mm_func_char = vec_type_info[(scalar, reg_width_bits)]

    m_reg = f'__m{reg_width_bits}'
    m_mask = f'__mmask{reg_width_ele}'
    mm = f'_mm{reg_width_bits}{m_reg_char}'

    mask = 'last_reg_mask' if masked else None

    def intrin(func, *args, mask=None):
        masked = ""
        if mask is not None:
            if "load" in func:
                masked = "maskz_"
            else:
                masked = "mask_"

            if "store" in func:
                args = (args[0],) + (mask,) + args[1:]
            else:
                args = (mask,) + args[0:]

        return f'{mm}_{masked}{func}_p{mm_func_char}({", ".join(args)})'

    def setup_accumulator(acc_dims, load=False):
        if load: assert True
        lines = [f'{scalar}* C_temp = C;']

        reg_def = f'{m_reg} '
        for i in range(acc_dims[0]):
            for k in range(acc_dims[1]):
                reg_def += f'cVec{i}{k}, '

        lines += [reg_def.strip(', ') + ";"]
        lines += ['if (load_c) {']

        if packed_C:
            load_intrin = partial(intrin, "load")
            c_load = lambda i, k: load_intrin(f'C + {i * max_acc_width + k} * {reg_width_ele}')
        else:
            load_intrin = partial(intrin, "loadu")
            c_load = lambda i, k: load_intrin(f'C_temp + {k} * {reg_width_ele}',
                                              mask=None if k < acc_dims[1]-1 else mask)
        for i in range(acc_dims[0]):
            for k in range(acc_dims[1]):
                lines += [f'  cVec{i}{k} = {c_load(i, k)};']
            if not packed_C:
                lines += [f'  C_temp += N;']

        lines += ['} else {']
        for i in range(acc_dims[0]):
            for k in range(acc_dims[1]):
                intrinsic = intrin("setzero", f'')
                lines += [f'  cVec{i}{k} = {intrinsic};']

        lines += ['}']
        return lines

    def store_accumulator(acc_dims):
        if packed_C:
            store_intrin = partial(intrin, "store")
            c_store = lambda i, k: store_intrin(f'C + {i * max_acc_width + k} * {reg_width_ele}', f'cVec{i}{k}')
        else:
            store_intrin = partial(intrin, "storeu")
            c_store = lambda i, k: store_intrin(f'C + {i} * N + {k} * {reg_width_ele}', f'cVec{i}{k}',
                                                mask=None if k < acc_dims[1]-1 else mask)

        return [f'{c_store(i, k)};'
                for i in range(acc_dims[0])
                for k in range(acc_dims[1])]

    def gen_pattern_case(acc_dims, id, pat):
        case_body = Block()

        count_loop = ForLoop(f'int pat_count = nkern_counts[{id}]', 'pat_count > 0', 'pat_count--',
                             unroll=unroll_mapping(gmpy.popcount(pat)))
        count_loop += f'{m_reg} aVec;'

        if packed_B:
            assert masked == False
            load_intrin = partial(intrin, "load")
        else:
            load_intrin = partial(intrin, "loadu")

        b_load = lambda k: load_intrin(f'B_curr + {k} * {reg_width_ele}', mask=None if k < acc_dims[1]-1 else mask)

        for k in range(acc_dims[1]):
            count_loop += f'{m_reg} bVec{k} = {b_load(k)};'

        if packed_B:
            count_loop += f'B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;'
            # for k in range(acc_dims[1]):
            #     count_loop += f'__builtin_prefetch(((uint8_t*) B_curr) + {k * 64}, 0, 3);'
        else:
            count_loop += f'B_curr = (*col_indices_curr) * N + B; col_indices_curr++;'

        pat_tmp = pat
        idx = 0
        while pat_tmp:
            if pat_tmp & 1:
                intrinsic = intrin("set1", '*curr_value_ptr')
                count_loop += f'aVec = {intrinsic}; curr_value_ptr++;'
                for k in range(acc_dims[1]):
                    intrinsic = intrin("fmadd", 'aVec', f'bVec{k}', f'cVec{idx}{k}')
                    count_loop += f'cVec{idx}{k} = {intrinsic};'

            pat_tmp >>= 1
            idx += 1

        case_body += count_loop

        return case_body.sub_elements

    sop_panel_executor = Block()
    sop_panel_executor += ''

    if True or (packed_B or packed_C):
        body = Block()
    else:
        body = ForLoop('int _j = 0', '_j < N_c', f'_j += {acc_dims[1]} * {reg_width_ele}')

    body += setup_accumulator(acc_dims)
    body += 'int c_idx = 0;'
    body += 'auto curr_value_ptr = values;'

    if packed_B:
        body += f'const {scalar} *__restrict__ B_curr = col_indices[0] * (N_r) + B;'
    else:
        body += f'const {scalar} *__restrict__ B_curr = col_indices[0] * N + B;'

    body += 'uint32_t * col_indices_curr = col_indices + 1;'

    for id, pat in enumerate(supported_patterns):
        body += gen_pattern_case(acc_dims, id, pat)

    body += store_accumulator(acc_dims)
    sop_panel_executor += body.sub_elements

    return sop_panel_executor