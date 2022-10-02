import gmpy
import hashlib
import os
import json

from .codegen_utils import *
from .arch_details import *
from functools import partial

##
#   Utils
##


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


def microkernel_id(arch, vec_width_bits, acc_dims, nanokernels):
    _hash = hashlib.md5(" ".join([str(p) for p in nanokernels]).encode("utf8")).hexdigest()
    return f'{nanokernel_hash(nanokernels)}_{arch}_{vec_width_bits}_{acc_dims[0]}x{acc_dims[1]}'


def microkernel_typename(scalar, arch, vec_width_bits, acc_dims, nanokernels):
    return f'MicroKernel_{scalar}_{microkernel_id(arch, vec_width_bits, acc_dims, nanokernels)}'


def unroll_mapping(nnzs):
    if nnzs == 1: return 2
    if nnzs <= 2: return 2
    return 2


class UKernelCodegenBase:
    def __init__(self, Mr, nanokernels, output_root=None, namespace=None):
        import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
        output_root = f'{SCRIPT_DIR}/_C/generated' if output_root is None else output_root

        self.Mr = Mr
        self.nanokernels = sort_nanokernels(nanokernels)
        self.nanokernel_hash = nanokernel_hash(self.nanokernels)
        self.output_root = output_root
        self.namespace = namespace if namespace is not None else "sop"

    def gen_factories(self, Nr, arch, vec_width_bits, scalar, build_factories_for=None):
        typename = microkernel_typename(scalar, arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)

        reg_width_bits = instruction_set_reg_width[arch]
        reg_width_ele, _, _ = vec_type_info[(scalar, reg_width_bits)]
        microkernel_id_ = microkernel_id(arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)

        header = "/".join(self._header_path(arch, typename).split('/')[-2:])
        factory_desc_json = json.dumps({
            "id": microkernel_id_,
            "func": packer_factory_name(microkernel_id_),
            "scalar": scalar,
            "M_r": self.Mr,
            "N_r": Nr,
            "arch": arch,
            "reg_width_bits": reg_width_bits,
        })

        with open(self._factories_dir(arch) + f'packer_{scalar}_{arch}_nr_{Nr}.cpp', 'w+') as f:
            f.write(f'#ifdef {min_instruction_sets[reg_width_bits]}\n')
            f.write(f'#include "MicroKernelPackerFactory.h"\n')
            f.write(f'#include "{header}"\n')
            f.write(f'\n')
            f.write(f'namespace {self.namespace} {{\n')
            f.write(f'\n')
            f.write(f'// factory_desc | {factory_desc_json}\n')
            f.write(f'MicroKernelPackerFactory<{scalar}>* {packer_factory_name(microkernel_id_)}() {{\n')
            f.write(f'    return new MicroKernelPackerFactorySpecialized<{typename}>('
                    f'{self.Mr});\n')
            f.write(f'}}\n')
            f.write('\n')
            f.write(f'}} // namespace {self.namespace}\n')
            f.write(f'#endif // {min_instruction_sets[reg_width_bits]}\n')

        for kernel_desc in build_factories_for:
            factory_name = executor_factory_name(kernel_desc, microkernel_id_)
            factory_desc_json = json.dumps({
                "id": microkernel_id_,
                "func": factory_name,
                "kernel_desc": kernel_desc,
                "M_r": self.Mr,
                "N_r": Nr,
                "arch": arch,
                "reg_width_bits": reg_width_bits,
            })

            with open(self._factories_dir(arch) + f'executor_{kernel_desc}_{scalar}_{arch}_nr_{Nr}.cpp', 'w+') as f:
                f.write(f'#ifdef {min_instruction_sets[reg_width_bits]}\n')
                f.write(f'#include "ExecutorFactory.h"\n')
                f.write(f'#include "KernelDesc.h"\n')
                f.write(f'#include "{self._header_filename(typename)}"\n')
                f.write(f'\n')
                f.write(f'namespace {self.namespace} {{\n')
                f.write(f'\n')
                f.write(f'// factory_desc | {factory_desc_json}\n')
                f.write(f'ExecutorFactory<{kernel_desc}>* {factory_name}() {{\n')
                f.write(f'    return new ExecutorFactorySpecialized<{kernel_desc}, {typename}>('
                        f'{self.Mr}, {Nr*reg_width_ele});\n')
                f.write(f'}}\n')
                f.write('\n')
                f.write(f'}} // namespace {self.namespace}\n')
                f.write(f'#endif // {min_instruction_sets[reg_width_bits]}\n')

    def gen_header(self, Nr, arch, vec_width_bits, scalar='float'):
        typename = microkernel_typename(scalar, arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)
        ukern_id = microkernel_id(arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)
        vec_width_ele = vec_width_bits / SCALAR_SIZE_BITS[scalar]

        with open(self._header_path(arch, typename), 'w+') as f:
            f.write(f'#pragma once\n\n')
            f.write(f'#include "utils/error.h"\n')
            f.write(f'#include "MicroKernelBase.h"\n')
            f.write(f'#include "Storage.h"\n')
            f.write(f'\n')
            f.write(f'#include <immintrin.h>\n\n')
            f.write(f'\n')
            f.write(f'#include "intrin_compatability.h"\n')
            f.write(f'\n')
            f.write(f'namespace {self.namespace} {{')
            f.write(f'\n')
            f.write(f'struct {typename} {{\n')
            f.write(self._emit_supported_patterns())
            f.write(self._emit_nkern_encoder())
            f.write(self._emit_nkern_decoder())
            f.write(self._emit_nkern_nnz_count())
            f.write('\n')
            f.write(f'    using  Mask = __mmask{vec_width_ele};\n')
            f.write(f'    static Mask create_mask(int n) {{ return ((1 << n) - 1); }}\n')
            f.write(f'    static Mask precomp_mask(int N) {{ return create_mask(N % {vec_width_ele}); }}\n')
            f.write('\n')
            f.write(f'    using Scalar = {scalar};\n')
            f.write(f'    static constexpr int M_r = {self.Mr};\n')
            f.write(f'    static constexpr int N_r = {Nr} * {vec_width_ele};\n')
            f.write(f'    static constexpr int N_r_reg = {Nr};\n')
            f.write(f'    static constexpr int vec_width_bits = {vec_width_bits};\n')
            f.write(f'    static constexpr const char* id = "{ukern_id}";\n')
            f.write(f'    static int max_acc_width_in_vecs() {{ return {Nr}; }};\n')
            f.write(f'    static int max_acc_width_in_eles() {{ return {Nr} * {vec_width_ele}; }};\n\n')
            f.write(f'    static int num_nkern_patterns() {{ return {len(self.nanokernels)}; }}\n')
            f.write(f'')

            body = partial(self._emit_executor_body, arch=arch, vec_width_bits=vec_width_bits)
            f.write(self._emit_microkernels(body, [Nr, 1], scalar))

            #f.write(self._emit_microkernels(self, typename, Nr, scalar))
            f.write(f'\n}};\n\n')
            f.write(f'}} // {self.namespace}\n')

        return ukern_id

    def _include_dir(self, arch):
        path = f'{self.output_root}/{arch}/include/{self.nanokernel_hash}/'
        os.makedirs(path, exist_ok=True)
        return path

    def _factories_dir(self, arch):
        path = f'{self.output_root}/{arch}/factories/{self.nanokernel_hash}/'
        os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _header_filename(typename):
        return typename + '.h'

    def _header_path(self, arch, typename):
        return self._include_dir(arch) + self._header_filename(typename)

    def _emit_supported_patterns(self):
        supported_patterns_list = ",\n            ".join([f'0b{x:08b}' for x in self.nanokernels])
        return f'''
        static const uint16_t* supported_nkern_patterns() {{
            static uint16_t patterns[] = {{
                {supported_patterns_list}
            }};
        
            return patterns;
        }}
        '''

    def _emit_nkern_encoder(self):
        supported_patterns_encode_cases = "\n        ".join([
            f'if (nkern_pat == 0b{x:08b}) return {i};' for i, x in enumerate(self.nanokernels)])
        return f'''
        static uint16_t encode_nkern_pattern(uint16_t nkern_pat) {{
            {supported_patterns_encode_cases}
            if (nkern_pat == 0) return sop::ZERO_PATTERN_ID; 
            ERROR_AND_EXIT("Unable to map unsupported nanokernel pattern " <<  (int) nkern_pat);
            return 0;
        }}
        '''

    def _emit_nkern_decoder(self):
        supported_patterns_decode_cases = "\n        ".join([
            f'if (nkern_code == {i}) return 0b{x:08b};' for i, x in enumerate(self.nanokernels)])
        return f'''
        static uint16_t decode_nkern_pattern(uint16_t nkern_code) {{
            {supported_patterns_decode_cases}
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to unmap unsupported nanokernel pattern id " << (int) nkern_code);
            return 0;
        }}
        '''

    def _emit_nkern_nnz_count(self):
        pattern_nnz_count_cases = "\n        ".join([
            f'if (nkern_code == {i}) return {gmpy.popcount(x)};' for i, x in enumerate(self.nanokernels)])
        return f'''
        static uint16_t nnz_count_for_nkern_code(uint16_t nkern_code) {{
            {pattern_nnz_count_cases}
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
            return 0;
        }}
        '''

    def _emit_executor_body(self, Nr, arch, vec_width_bits, scalar, packed_C=False, packed_B=False, mask=None):
        arch_details = {
            'AVX512': AVX512(),
            'AVX2': AVX2(),
        }[arch]

        arch_intrin_gen = ArchIntrinGenerator(arch_details, vec_width_bits, scalar)
        vec_width_ele = vec_width_bits / SCALAR_SIZE_BITS[scalar]
        m_reg = arch_intrin_gen.vec_type()

        def setup_accumulator():
            lines = [f'{scalar}* C_temp = C;']

            reg_def = f'{arch_intrin_gen.vec_type()} '
            for i in range(self.Mr):
                for k in range(Nr):
                    reg_def += f'cVec{i}{k}, '

            lines += [reg_def.strip(', ') + ";"]
            lines += ['if (load_c) {']

            if packed_C:
                c_load = lambda i, k: arch_intrin_gen.load_intrin(f'C + {i * Nr + k} * {vec_width_ele}', aligned=True)
            else:
                if mask is None:
                    c_load = lambda i, k: arch_intrin_gen.load_intrin(f'C + {k} * {vec_width_ele}')
                else:
                    c_load = lambda i, k: arch_intrin_gen.load_intrin(f'C + {k} * {vec_width_ele}', mask=mask)

            for i in range(self.Mr):
                for k in range(Nr):
                    lines += [f'  cVec{i}{k} = {c_load(i, k)};']
                if not packed_C:
                    lines += [f'  C_temp += N;']

            lines += ['} else {']
            for i in range(self.Mr):
                for k in range(Nr):
                    lines += [f'  cVec{i}{k} = {arch_intrin_gen.setzero_intrin()};']

            lines += ['}']
            return lines

        def store_accumulator(acc_dims):
            if packed_C:
                c_store = lambda i, k: \
                    arch_intrin_gen.store_intrin(f'C + {i * Nr + k} * {vec_width_ele}', f'cVec{i}{k}')
            else:
                c_store = lambda i, k: \
                    arch_intrin_gen.store_intrin(f'C + {i} * N + {k} * {vec_width_ele}', f'cVec{i}{k}', mask=mask)

            return [f'{c_store(i, k)};'
                    for i in range(acc_dims[0])
                    for k in range(acc_dims[1])]

        def gen_pattern_case(acc_dims, id, pat):
            case_body = Block()

            count_loop = ForLoop(f'int pat_count = nkern_counts[{id}]', 'pat_count > 0', 'pat_count--',
                                 unroll=unroll_mapping(gmpy.popcount(pat)))
            count_loop += f'{m_reg} aVec;'

            if packed_B:
                assert mask is None
                load_intrin = partial(arch_intrin_gen.load_intrin, aligned=True)
            else:
                load_intrin = partial(arch_intrin_gen.load_intrin, aligned=False)

            b_load = lambda k: load_intrin(f'B_curr + {k} * {vec_width_ele}', mask=mask)

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
                    count_loop += f'aVec = {arch_intrin_gen.broadcast_intrin("*curr_value_ptr")}; curr_value_ptr++;'
                    for k in range(acc_dims[1]):
                        intrinsic = arch_intrin_gen.fma_intrin('aVec', f'bVec{k}', f'cVec{idx}{k}')
                        count_loop += f'cVec{idx}{k} = {intrinsic};'

                pat_tmp >>= 1
                idx += 1

            case_body += count_loop

            return case_body.sub_elements

        sop_panel_executor = Block()
        sop_panel_executor += ''

        body = Block()
        body += setup_accumulator()
        body += 'int c_idx = 0;'
        body += 'auto curr_value_ptr = values;'

        if packed_B:
            body += f'const {scalar} *__restrict__ B_curr = col_indices[0] * (N_r) + B;'
        else:
            body += f'const {scalar} *__restrict__ B_curr = col_indices[0] * N + B;'

        body += 'uint32_t * col_indices_curr = col_indices + 1;'

        for id, pat in enumerate(self.nanokernels):
            body += gen_pattern_case([self.Mr, Nr], id, pat)

        body += store_accumulator([self.Mr, Nr])
        sop_panel_executor += body.sub_elements

        return sop_panel_executor

    @staticmethod
    def _emit_microkernels(body: callable, Nrs, scalar, name=""):
        if len(name) > 0 and name[0] != '_':
            name = '_' + name

        for i, Nr in enumerate(sorted(Nrs, reverse=True)):
            acc_width_str = str(Nr) if i != 0 else "max_acc"

            return f'''
    __ALWAYS_INLINE static void _microkernel{name}_{acc_width_str}(
        int M, int K, int N,
        int* __restrict__            nkern_counts,
        uint32_t* __restrict__       col_indices,
        {scalar}* __restrict__       values,
        int                          num_col_indices,
        const {scalar} *__restrict__ B,
        {scalar} *__restrict__ C,
        const bool load_c)
    {{\n{body(Nr=Nr, scalar=scalar).emit(6)}
    }}\n\n

    __ALWAYS_INLINE static void microkernel{name}_{acc_width_str}(
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
