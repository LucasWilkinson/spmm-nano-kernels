import gmpy

from tools.codegen.codegen_utils import *
from functools import partial


vec_type_info = {
    ("float", 512):  (16, '', 's'),
    ("float", 256):  (8,  '', 's'),
    ("double", 512): (8, 'd', 'd'),
    ("double", 256): (4, 'd', 'd'),
}


def unroll_mapping(nnzs):
    if nnzs == 1: return 2
    if nnzs <= 2: return 2
    return 2


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

        count_loop = ForLoop(f'int pat_count = pattern_counts[{id}]', 'pat_count > 0', 'pat_count--',
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
    # sop_panel_executor += f'uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;'
    # sop_panel_executor += f'float* __restrict__     values = panel_desc.values;'
    # sop_panel_executor += f'int* __restrict__       pattern_counts = panel_desc.pattern_counts;'
    # sop_panel_executor += f'int                     num_patterns = panel_desc.num_patterns;'
    # sop_panel_executor += f'int                     num_col_indices = panel_desc.num_col_indices;'
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


def gen_for_vec_height(kernel_id, acc_dims, supported_patterns, output_path=None, output_root=None):
    vec_height = acc_dims[0]
    reg_num_ele = 16

    supported_patterns = sorted(supported_patterns, key=lambda x: gmpy.popcount(x))

    # Utility functions
    supported_patterns_list = ",\n            ".join([f'0b{x:08b}' for x in supported_patterns])

    supported_pattern_getter = f'''
    static const uint16_t* supported_patterns() {{
        static uint16_t patterns[] = {{
            {supported_patterns_list}
        }};
    
        return patterns;
    }}
    '''

    supported_patterns_encode_cases = "\n        ".join([
        f'if (pattern == 0b{x:08b}) return {i};' for i, x in enumerate(supported_patterns)])

    supported_pattern_encode = f'''
    static uint16_t encode_pattern(uint16_t pattern) {{
        {supported_patterns_encode_cases}
        if (pattern == 0) return ZERO_PATTERN_ID; 
        std::cerr << "Unable to map unsupported pattern " <<  (int) pattern << std::endl;
        exit(-1);
        return 0;
    }}
    '''

    supported_patterns_decode_cases = "\n        ".join([
        f'if (pattern == {i}) return 0b{x:08b};' for i, x in enumerate(supported_patterns)])

    supported_pattern_decode = f'''
    static uint16_t decode_pattern(uint16_t pattern) {{
        {supported_patterns_decode_cases}
        if (pattern == ZERO_PATTERN_ID) return 0; 
        std::cerr << "Unable to unmap unsupported pattern id " << (int) pattern << std::endl;
        exit(-1);
        return 0;
    }}
    '''

    pattern_nnz_count_cases = "\n        ".join([
        f'if (pattern == {i}) return {gmpy.popcount(x)};' for i, x in enumerate(supported_patterns)])

    pattern_nnz_count = f'''
    static uint16_t nnz_count(uint16_t pattern) {{
        {pattern_nnz_count_cases}
        if (pattern == ZERO_PATTERN_ID) return 0; 
        std::cerr << "Unable to get pop count for pattern id " << (int) pattern << std::endl;
        exit(-1);
        return 0;
    }}
    '''

    import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    output_root = f'{SCRIPT_DIR}/_C/generated' if output_root is None else output_root

    output_path = output_path if output_path is not None else f'{output_root}/sop_executor_{kernel_id}.h'


    for scalar, vec_width in [('float', 512)]:
        reg_width_ele, _, _ = vec_type_info[(scalar, vec_width)]

        with open(output_path.replace('.h', f'_{scalar}_{vec_width}.h'), 'w+') as f:
            f.write(f'#pragma once\n\n')
            f.write(f'#include "SOPMicroKernelBase.h"\n')
            f.write(f'#include "SOPStorage.h"\n')
            f.write(f'\n')
            f.write(f'#include <immintrin.h>\n\n')
            f.write(f'\n')
            f.write(f'namespace sop {{\n')
            f.write(f'\n')
            f.write(f'template<> struct SOPMicroKernelIntrin<{scalar}, {vec_width}, {vec_height}, {acc_dims[1]}> {{\n')
            f.write(supported_pattern_getter)
            f.write(supported_pattern_encode)
            f.write(supported_pattern_decode)
            f.write(pattern_nnz_count)
            f.write('\n')
            f.write(f'    using Mask = __mmask{reg_width_ele};\n')
            f.write(f'    static Mask create_mask(int n) {{ return ((1 << n) - 1); }}\n')
            f.write(f'    static Mask precomp_mask(int N) {{ return create_mask(N % N_r); }}\n')
            f.write('\n')
            f.write(f'    static const int M_r = {acc_dims[0]};\n')
            f.write(f'    static const int N_r = {acc_dims[1]} * {reg_num_ele};\n')
            f.write(f'    static int max_acc_width_in_vecs() {{ return {acc_dims[1]}; }};\n')
            f.write(f'    static int max_acc_width_in_eles() {{ return {acc_dims[1]} * {reg_width_ele}; }};\n\n')
            f.write(f'    static int number_of_patterns() {{ return {len(supported_patterns)}; }}\n')
            f.write(f'    static int panel_height() {{ return {vec_height}; }}\n\n')
            f.write(f'')

            if acc_dims[1] == 1:
                acc_widths = [1]
            else:
                acc_widths = [acc_dims[1], 1]

            max_acc_width = max(acc_widths)

            for acc_width in acc_widths:
                acc_width_str = str(acc_width) if acc_width != max_acc_width else "max_acc"

                sop_panel_executor = f'''
    __ALWAYS_INLINE static void _panel_executor_{acc_width_str}(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], acc_width], max_acc_width, supported_patterns, 
                           packed_C=False, packed_B=False).emit(6)}
    }}\n\n

    __ALWAYS_INLINE static void panel_executor_{acc_width_str}(
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c) {{
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_{acc_width_str}(
            M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }}
    '''

                sop_panel_executor_packed = f'''
    #ifdef ENABLE_PACKED_KERNELS
    __ALWAYS_INLINE static void _panel_executor_packed_{acc_width_str}(
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], acc_width], max_acc_width, supported_patterns, 
                           packed_C=True, packed_B=True).emit(6)}
    }}\n\n

    __ALWAYS_INLINE static void panel_executor_packed_{acc_width_str}(
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c) {{
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_packed_{acc_width_str}(
            pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }}
    #endif
    '''

                sop_panel_executor_packed_C = f'''
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_packed_C_{acc_width_str}(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], acc_width], max_acc_width, supported_patterns, 
                           packed_C=True, packed_B=False).emit(6)}
    }}\n\n
    
    __ALWAYS_INLINE static void panel_executor_packed_C_{acc_width_str}(
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c) {{
    
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
      
        _panel_executor_packed_C_{acc_width_str}(
            M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, load_c
        );
    }}
    #endif
    '''
                f.write(sop_panel_executor)
                f.write(sop_panel_executor_packed)
                f.write(sop_panel_executor_packed_C)

            sop_panel_executor_packed_masked_C = f'''
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_masked_1(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], 1], max_acc_width, supported_patterns,
                           packed_C=False, packed_B=False, masked=True).emit(6)}
    }}\n\n
    
    static void _panel_executor_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {{

        int _j = 0;
        for (; _j < N_rem - {reg_width_ele-1}; _j += {reg_width_ele}) {{
            _panel_executor_1(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }}
        
        _panel_executor_masked_1(
            M, K, N,
            pattern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }}
    
    static void panel_executor_masked_max_acc(
        int N_rem,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {{

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _panel_executor_masked_max_acc(
            N_rem, M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }}
    
    #endif
    '''
            f.write(sop_panel_executor_packed_masked_C)

            sop_panel_executor_packed_masked_C = f'''
    #ifdef ENABLE_PACKED_C_KERNELS
    __ALWAYS_INLINE static void _panel_executor_masked_packed_C_1(
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c)
    {{\n{gen_executor_body(scalar, vec_width, [acc_dims[0], 1], max_acc_width, supported_patterns,
                           packed_C=True, packed_B=False, masked=True).emit(6)}
    }}\n\n
    
    static void _panel_executor_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        int* __restrict__            pattern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__       C,
        Mask last_reg_mask,
        const bool load_c) {{

        int _j = 0;
        for (; _j < N_rem - {reg_width_ele-1}; _j += {reg_width_ele}) {{
            _panel_executor_packed_C_1(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }}
        
        _panel_executor_masked_packed_C_1(
            M, K, N,
            pattern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            last_reg_mask, load_c);
    }}
    
    static void panel_executor_masked_packed_C_max_acc(
        int N_rem,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        Mask last_reg_mask,
        const bool load_c) {{

        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
     
        _panel_executor_masked_packed_C_max_acc(
            N_rem, M, K, N, pattern_counts, col_indices, values, num_col_indices, B, C, last_reg_mask, load_c);
    }}
    
    #endif
    '''
            f.write(sop_panel_executor_packed_masked_C)

            sop_panel_executor = f'''
    static void panel_executor_max_acc_width_N_c(
        int N_c,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c)
    {{\n
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
    
        for (int _j = 0; _j < N_c; _j += N_r) {{
            _panel_executor_max_acc(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }}
    }}\n\n'''
            f.write(sop_panel_executor)

            sop_panel_executor = f'''
    static void panel_executor_cleanup_N_c(
        int N_c_rem,
        int M, int K, int N,
        const PanelUsingCounts& panel_desc,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        Mask mask, const bool load_c)
    {{\n
        uint32_t* __restrict__  col_indices = (uint32_t*) panel_desc.col_indices;
        float* __restrict__     values = panel_desc.values;
        int* __restrict__       pattern_counts = panel_desc.pattern_counts;
        int                     num_patterns = panel_desc.num_patterns;
        int                     num_col_indices = panel_desc.num_col_indices;
    
        int _j = 0;
        int end_of_full_blocks = (N_c_rem / N_r) * N_r;
        int end_of_partial_blocks = ((N_c_rem - end_of_full_blocks) / N_r) * N_r;
        
        if (end_of_full_blocks) {{
            panel_executor_max_acc_width_N_c(
                end_of_full_blocks,
                M, K, N,
                panel_desc,
                B, C,
                load_c);
        }}

        for (_j = end_of_full_blocks; _j < end_of_partial_blocks; _j += {reg_width_ele}) {{
            _panel_executor_1(
                M, K, N,
                pattern_counts, col_indices, values, num_col_indices,
                B + _j,
                C + _j,
                load_c);
        }}
        
        _panel_executor_masked_1(
            M, K, N,
            pattern_counts, col_indices, values, num_col_indices,
            B + _j,
            C + _j,
            mask, load_c);
    }}\n\n'''
            f.write(sop_panel_executor)
            f.write(f'\n}};\n\n')

            f.write(f'}} // namespace sop\n')


