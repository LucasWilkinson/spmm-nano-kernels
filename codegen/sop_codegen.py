import gmpy

from tools.codegen.codegen_utils import *


def gen_executor_body(acc_dims, supported_patterns, packed_C=False, packed_B=False):
    PRECOMPUTE_B = False
    PREFETCH_B = False

    def setup_accumulator(acc_dims, load=False):
        if load: assert True
        lines = [f'Scalar* C_temp = C;']

        for i in range(acc_dims[0]):
            for k in range(acc_dims[1]):
                lines += [f'VecType cVec{i}{k};']

        lines += ['if (load_c) {']

        if packed_C:
            print("packed")
            for i in range(acc_dims[0]):
                for k in range(acc_dims[1]):
                    lines += [f'  cVec{i}{k}.load(C + {i * acc_dims[1] + k} * VecType::size());']
        else:
            for i in range(acc_dims[0]):
                for k in range(acc_dims[1]):
                    lines += [f'  cVec{i}{k}.load(C_temp + {k} * VecType::size());']
                lines += [f'  C_temp += n;']

        lines += ['} else {']
        for i in range(acc_dims[0]):
            for k in range(acc_dims[1]):
                lines += [f'  cVec{i}{k} ^= cVec{i}{k}; // zero c']

        lines += ['}']
        return lines

    def store_accumulator(acc_dims):
        if packed_C:
            return [f'cVec{i}{k}.store(C + {i * acc_dims[1] + k} * VecType::size());'
                    for i in range(acc_dims[0])
                    for k in range(acc_dims[1])]
        else:
            return [f'cVec{i}{k}.store(&C[{i} * n + {k} * VecType::size()]);'
                    for i in range(acc_dims[0])
                    for k in range(acc_dims[1])]

    def gen_pattern_case(acc_dims, id, pat):
        case_body = Block()

        count_loop = ForLoop(f'int pat_count = pattern_counts[{id}]', 'pat_count > 0', 'pat_count--', unroll=2)
        count_loop += f'VecType aVec;'

        if PRECOMPUTE_B:
            if packed_B:
                assert True
            else:
                for k in range(acc_dims[1]):
                    count_loop += f'VecType bVec{k}; bVec{k}.load(B_ptrs[c_idx] + {k} * VecType::size());'
        else:
            if packed_B:
                for k in range(acc_dims[1]):
                    count_loop += f'VecType bVec{k}; bVec{k}.load_a(B_curr + {k} * VecType::size());'
            else:
                for k in range(acc_dims[1]):
                    count_loop += f'VecType bVec{k}; bVec{k}.load(B_curr + {k} * VecType::size());'

        if PREFETCH_B:
            if packed_B:
                assert True
            else:
                count_loop += f'B_curr = B_next;'
                count_loop += f'B_next = B_next_next;'
                count_loop += f'B_next_next = col_indices[(++c_idx) + 2] * n + B;'
                count_loop += [
                    f'#pragma unroll',
                    f'for (int i = 0; i < ({k} * VecType::size() * sizeof(float)) / 64; i ++) {{',
                    f'    __builtin_prefetch(((uint8_t*) B_next_next) + 64 * i, 0, 3);',
                    f'}}',
                ]
        else:
            if packed_B:
                count_loop += f'B_curr = (*col_indices_curr) * (N_r) + B; col_indices_curr++;'
                # for k in range(acc_dims[1]):
                #     count_loop += f'__builtin_prefetch(((uint8_t*) B_curr) + {k * 64}, 0, 3);'
            else:
                count_loop += f'B_curr = col_indices[++c_idx] * n + B;'

        pat_tmp = pat
        idx = 0
        while pat_tmp:
            if pat_tmp & 1:
                count_loop += f'aVec = VecType(*curr_value_ptr); curr_value_ptr++;'
                for k in range(acc_dims[1]):
                    count_loop += f'cVec{idx}{k} = mul_add(aVec, bVec{k}, cVec{idx}{k});'

            pat_tmp >>= 1
            idx += 1

        case_body += count_loop

        return case_body.sub_elements

    sop_panel_executor = Block()
    sop_panel_executor += f'if (panel_desc.num_patterns <= 0) return;'
    sop_panel_executor += f'float* values = panel_desc.values;'
    sop_panel_executor += f'uint32_t * col_indices = (uint32_t*) panel_desc.col_indices;'
    sop_panel_executor += f'int* pattern_counts = panel_desc.pattern_counts;'
    sop_panel_executor += f'int num_patterns = panel_desc.num_patterns;'
    sop_panel_executor += f'int num_col_indices = panel_desc.num_col_indices;'
    sop_panel_executor += ''

    if PRECOMPUTE_B:
        sop_panel_executor += [
            f'const float* B_ptrs[num_col_indices];',
            f'',
            f'#pragma ivdep',
            f'#pragma vector nontemporal (col_indices)',
            f'#pragma prefetch col_indices:_MM_HINT_T1',
            f'#pragma temporal (B_ptrs)',
            f'#pragma unroll 16',
            f'for (int i = 0; i < num_col_indices; i ++) {{',
            f'    B_ptrs[i] = (const typename StorageTypes::Scalar *) uintptr_t(B) + uintptr_t(col_indices[i]) * n;',
            f'}}',
        ]

    if packed_B or packed_C:
        nvecs_loops = Block()
    else:
        nvecs_loops = ForLoop('int n_vec = 0', 'n_vec < n_vecs', f'n_vec += {acc_dims[1]}')

    nvecs_loops += setup_accumulator(acc_dims)
    nvecs_loops += 'int c_idx = 0;'
    nvecs_loops += 'auto curr_value_ptr = values;'

    if packed_B:
        nvecs_loops += 'const Scalar *__restrict__ B_curr = col_indices[0] * (N_r) + B;'
    else:
        nvecs_loops += 'const Scalar *__restrict__ B_curr = col_indices[0] * n + B;'

    nvecs_loops += 'uint32_t * col_indices_curr = col_indices + 1;'

    if PREFETCH_B:
        nvecs_loops += 'const Scalar *__restrict__ B_next = col_indices[1] * n + B;'
        nvecs_loops += 'const Scalar *__restrict__ B_next_next = col_indices[2] * n + B;'
        nvecs_loops += [
            f'#pragma unroll',
            f'for (int i = 0; i < ({acc_dims[1]} * VecType::size() * sizeof(float)) / 64; i ++) {{',
            f'    __builtin_prefetch(((uint8_t*) B_curr) + 64 * i, 0, 3);',
            f'}}',
        ]
        nvecs_loops += [
            f'#pragma unroll',
            f'for (int i = 0; i < ({acc_dims[1]} * VecType::size() * sizeof(float)) / 64; i ++) {{',
            f'    __builtin_prefetch(((uint8_t*) B_next) + 64 * i, 0, 3);',
            f'}}',
        ]
        nvecs_loops += [
            f'#pragma unroll',
            f'for (int i = 0; i < ({acc_dims[1]} * VecType::size() * sizeof(float)) / 64; i ++) {{',
            f'    __builtin_prefetch(((uint8_t*) B_next_next) + 64 * i, 0, 3);',
            f'}}',
        ]

    for id, pat in enumerate(supported_patterns):
        nvecs_loops += gen_pattern_case(acc_dims, id, pat)

    nvecs_loops += store_accumulator(acc_dims)

    if not packed_B:
        if PRECOMPUTE_B:
            nvecs_loops += [
                f'#pragma ivdep',
                f'#pragma unroll 16',
                f'for (int i = 0; i < num_col_indices; i ++) {{',
                f'    B_ptrs[i] += {acc_dims[1]}* VecType::size();',
                f'}}',
            ]
        else:
            nvecs_loops += f'B += {acc_dims[1]} * VecType::size();'

        nvecs_loops += f'C += {acc_dims[1]} * VecType::size();'
    sop_panel_executor += nvecs_loops.sub_elements

    return sop_panel_executor


def gen_for_vec_height(kernel_id, acc_dims, supported_patterns, output_path=None, output_root=None):
    vec_height = acc_dims[0]

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

    sop_multiple_panel_executor = '''
    __ALWAYS_INLINE static void sop_multiple_panel_executor(
            int n_tile,
            uint32_t m, uint32_t k, uint32_t n,
            const PanelUsingCounts* panel_descs, uint32_t num_panels,
            const Scalar *__restrict__ B,
            Scalar *__restrict__ C,
            const bool load_c
    ) {
        for (int t = 0; t < num_panels; t++) {
            sop_panel_executor(
                n_tile,
                m, k, n,
                panel_descs[t],
                B, &C[t * panel_height() * n], 
                load_c
            );
        }
    }
    '''

    sop_panel_executor = f'''
    __ALWAYS_INLINE static void sop_panel_executor (
        int n_vecs,
        int m, int k, int n,
        const PanelUsingCounts panel_desc,
        const Scalar *__restrict__ B,
        Scalar *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(acc_dims, supported_patterns, packed_C=False, packed_B=False).emit(6)}
    }}'''

    sop_panel_executor_packed = f'''
    __ALWAYS_INLINE static void sop_panel_executor_packed(
        const PanelUsingCounts panel_desc,
        const Scalar *__restrict__ B,
        Scalar *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(acc_dims, supported_patterns, packed_C=True, packed_B=True).emit(6)}
    }}'''

    sop_panel_executor_packed_C = f'''
    __ALWAYS_INLINE static void sop_panel_executor_packed_C(
        uint32_t n,
        const PanelUsingCounts panel_desc,
        const Scalar *__restrict__ B,
        Scalar *__restrict__ C,
        const bool load_c)
    {{\n{gen_executor_body(acc_dims, supported_patterns, packed_C=True, packed_B=False).emit(6)}
    }}'''

    sop_multiple_panel_executor_packed = '''
    __ALWAYS_INLINE static void sop_multiple_panel_executor_packed(
            const int N_c,
            uint32_t m, uint32_t k, uint32_t n,
            const PanelUsingCounts* panel_descs, uint32_t num_panels,
            const Scalar *__restrict__ B,
            Scalar *__restrict__ C,
            const bool load_c
    ) {
        const int Nr_tiles = N_c / N_r;
        const int c_K_r = k / M_r;
        const int c_N_r = n / N_r;
        assert(N_c % N_r == 0);
        for (int ti = 0; ti < num_panels; ti++) {
            for (int Nr_tile = 0; Nr_tile < Nr_tiles; Nr_tile++) {
                sop_panel_executor_packed(
                    N_c,
                    m, k, n,
                    panel_descs[ti],
                    &B[(Nr_tile) * (k * N_r)], 
                    &C[(ti * c_N_r + Nr_tile) * (M_r * N_r)], 
                    load_c
                );
            }
        }
    }
    '''

    with open(output_path, 'w+') as f:
        f.write(f'#pragma once\n\n')
        f.write(f'#include "SOP_utils.h"\n\n')
        f.write(f'')
        f.write(f'template<typename _Vec>\n')
        f.write(f'struct SOPExecutor<_Vec, {vec_height}, {acc_dims[1]}> {{\n')
        f.write(f'    using Vec = _Vec;\n')
        f.write(f'    using Scalar = typename Vec::Scalar;\n')
        f.write(f'    using VecType = typename Vec::Type;\n\n')
        f.write(supported_pattern_getter)
        f.write(supported_pattern_encode)
        f.write(supported_pattern_decode)
        f.write(pattern_nnz_count)
        f.write('\n')
        f.write(f'    static const int M_r = {acc_dims[0]};\n')
        f.write(f'    static const int N_r = {acc_dims[1]} * VecType::size();\n')
        f.write(f'    static int sop_accumulator_width_in_vecs() {{ return {acc_dims[1]}; }};\n\n')
        f.write(f'    static int number_of_patterns() {{ return {len(supported_patterns)}; }}\n')
        f.write(f'    static int panel_height() {{ return {vec_height}; }}\n\n')
        f.write(sop_panel_executor)
        f.write(sop_panel_executor_packed)
        f.write(sop_panel_executor_packed_C)
        f.write(sop_multiple_panel_executor)
        f.write(sop_multiple_panel_executor_packed)
        f.write(f'\n}};\n\n')


