# import gmpy
import hashlib
import os
import json

from dataclasses import dataclass
from .codegen_utils import *
from .arch_details import *
from functools import partial

##
#   Utils
##


def popcount(x):
    return bin(x).count("1")


def executor_factory_name(kernel_desc, microkernel_id):
    return f'executor_factory_{kernel_desc}_{microkernel_id}'


def packer_factory_name(scalar, microkernel_id):
    return f'packer_factory_{microkernel_id}_{scalar}'


def sort_nanokernels(nanokernels):
    return sorted(nanokernels, key=lambda x: popcount(x))


def nanokernel_hash(nanokernels):
    nanokernels = sort_nanokernels(nanokernels)
    _hash = hashlib.md5(" ".join([str(p) for p in nanokernels]).encode("utf8")).hexdigest()
    return _hash[-5:]


def microkernel_id(arch, vec_width_bits, acc_dims, nanokernels):
    _hash = hashlib.md5(" ".join([str(p) for p in nanokernels]).encode("utf8")).hexdigest()
    return f'{nanokernel_hash(nanokernels)}_{arch}_{vec_width_bits}_{acc_dims[0]}x{acc_dims[1]}'


def microkernel_typename(scalar, arch, vec_width_bits, acc_dims, nanokernels):
    return f'MicroKernel_{scalar}_{microkernel_id(arch, vec_width_bits, acc_dims, nanokernels)}'


def unroll_mapping(arch_str, nnzs):
    if nnzs == 1: return 2
    return 1


class UKernelCodegenBase:
    supported_archs = {
        'AVX512': AVX512(),
        'AVX2': AVX2(),
        'NEON': NEON(),
    }

    size_of_scalar = {
        "float": 4,
        "double": 8
    }

    def __init__(self, Mr, nanokernels, output_root=None, namespace=None, fuse_bias=False, fuse_minmax=False):
        import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
        output_root = f'{SCRIPT_DIR}/_C/generated' if output_root is None else output_root

        self.Mr = Mr
        self.nanokernels = sort_nanokernels(nanokernels)
        self.nanokernel_hash = nanokernel_hash(self.nanokernels)
        self.output_root = output_root
        self.namespace = namespace if namespace is not None else "sop"

    def gen_factories(self, Nr, arch, vec_width_bits, scalar, build_factories_for=None):
        typename = microkernel_typename(scalar, arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)
        arch_details = self.supported_archs[arch]

        reg_width_bits = vec_width_bits
        reg_width_ele = int(vec_width_bits / SCALAR_SIZE_BITS[scalar])
        microkernel_id_ = microkernel_id(arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)

        header = "/".join(self._header_path(arch, typename).split('/')[-2:])
        factory_desc_json = json.dumps({
            "id": microkernel_id_,
            "func": packer_factory_name(scalar, microkernel_id_),
            "scalar": scalar,
            "M_r": self.Mr,
            "N_r": Nr,
            "arch": arch,
            "reg_width_bits": reg_width_bits,
        })

        with open(self._factories_dir(arch) + f'packer_{scalar}_{arch}_nr_{Nr}.cpp', 'w+') as f:
            f.write(f'{arch_details.preprocessor_guard()}\n')
            f.write(f'#include "{header}"\n')
            f.write(f'#include "MicroKernelPackerFactory.h"\n')
            f.write(f'#include "{header}"\n')
            f.write(f'\n')
            f.write(f'namespace {self.namespace} {{\n')
            f.write(f'\n')
            f.write(f'// factory_desc | {factory_desc_json}\n')
            f.write(f'MicroKernelPackerFactory<{scalar}>* {packer_factory_name(scalar, microkernel_id_)}() {{\n')
            f.write(f'    return new MicroKernelPackerFactorySpecialized<{typename}>('
                    f'{self.Mr});\n')
            f.write(f'}}\n')
            f.write('\n')
            f.write(f'}} // namespace {self.namespace}\n')
            f.write(f'#endif\n')

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
                f.write(f'{arch_details.preprocessor_guard()}\n')
                f.write(f'#include "ExecutorFactory.h"\n')
                f.write(f'#include "KernelDesc.h"\n')
                f.write(f'#include "{self.nanokernel_hash}/{self._header_filename(typename)}"\n')
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
                f.write(f'#endif\n')

    def gen_header(self, Nr, arch, vec_width_bits, scalar='float'):
        typename = microkernel_typename(scalar, arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)
        ukern_id = microkernel_id(arch, vec_width_bits, [self.Mr, Nr], self.nanokernels)
        vec_width_ele = int(vec_width_bits / SCALAR_SIZE_BITS[scalar])
        arch_details = self.supported_archs[arch]

        with open(self._header_path(arch, typename), 'w+') as f:
            f.write(f'#pragma once\n\n')
            f.write(f'#include "utils/error.h"\n')
            f.write(f'#include "MicroKernelBase.h"\n')
            f.write(f'#include "Storage.h"\n')
            f.write(f'\n')
            f.write(f'{arch_details.intrin_include()}\n\n')
            f.write(f'\n')
            f.write(f'#include "intrin_compatability.h"\n')
            f.write(f'\n')
            f.write(f'namespace {self.namespace} {{')
            f.write(f'\n')
            f.write(f'struct {typename} {{\n')

            f.write(f'    enum Activation activation = NONE;\n')
            f.write(f'    {scalar} min = std::numeric_limits<{scalar}>::min();\n')
            f.write(f'    {scalar} max = std::numeric_limits<{scalar}>::max();\n\n')
            f.write(f'    {typename}(enum Activation activation = NONE, \n'
                    f'                {scalar} min = std::numeric_limits<{scalar}>::min(),\n'
                    f'                {scalar} max = std::numeric_limits<{scalar}>::max()):\n'
                    f'                  activation(activation),\n'
                    f'                  min(min), max(max) {{}}\n')
            f.write(self._emit_assembly(arch, Nr))
            f.write(self._emit_supported_patterns())
            f.write(self._emit_nkern_encoder())
            f.write(self._emit_nkern_decoder())
            f.write(self._emit_nkern_nnz_count())
            f.write('\n')
            f.write(f'    using  Mask = {arch_details.mask_type(scalar, vec_width_bits)};\n')
            f.write(f'    static constexpr Mask create_mask(int n) {{ return ((uint64_t) (1 << n) - 1); }}\n')
            f.write(f'    static constexpr Mask precomp_mask(int N) {{ return create_mask(N % {vec_width_ele}); }}\n')
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

            common_args = dict(arch=arch, vec_width_bits=vec_width_bits)

            main_body = partial(self._emit_executor_body_vectorized, **common_args)
            cleanup_body = partial(self._emit_executor_body_cleanup, **common_args)
            f.write(self._emit_microkernels(main_body, cleanup_body, Nr, scalar, name=""))

            # main_body = partial(self._emit_executor_body_vectorized, **common_args, packed_C=True)
            # cleanup_body = partial(self._emit_executor_body_cleanup, **common_args, packed_C=True)
            # f.write(self._emit_microkernels(main_body, cleanup_body, Nr, scalar, name="packed_C"))

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
        supported_patterns_list = ",\n                ".join([f'0b{x:08b}' for x in self.nanokernels])
        return f'''
        static const uint16_t* supported_nkern_patterns() {{
            static uint16_t patterns[] = {{
                {supported_patterns_list}
            }};
        
            return patterns;
        }}
        '''

    def _emit_nkern_encoder(self):
        supported_patterns_encode_cases = "\n            ".join([
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
        supported_patterns_decode_cases = "\n            ".join([
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
        pattern_nnz_count_cases = "\n            ".join([
            f'if (nkern_code == {i}) return {popcount(x)};' for i, x in enumerate(self.nanokernels)])
        return f'''
        static uint16_t nnz_count_for_nkern_code(uint16_t nkern_code) {{
            {pattern_nnz_count_cases}
            if (nkern_code == sop::ZERO_PATTERN_ID) return 0; 
            ERROR_AND_EXIT("Unable to get pop count for nanokernel code " << (int) nkern_code);
            return 0;
        }}
        '''

    def _emit_assembly(self, arch, Nr):
        if arch == "NEON+no":
            from codegen.asm.arm_gen_nano_kernels import gen_nano_kernel_asm
            return "\n\n" + \
                "inline __attribute__((__always_inline__)) " + \
                gen_nano_kernel_asm(self.Mr, Nr, self.nanokernels) + "\n" + \
                "inline __attribute__((__always_inline__)) " + \
                gen_nano_kernel_asm(self.Mr, 1, self.nanokernels)
        else:
            return ""

    @dataclass
    class Intrinsics:
        vec_type: str
        vec_width_ele: str
        load: callable
        load_aligned: callable
        store: callable
        store_aligned: callable
        fma: callable
        zero_reg: callable
        broadcast_from_ptr: callable
        broadcast: callable
        min: callable
        max: callable
        alignr: callable

    @staticmethod
    def nnz_iterator(pat):
        id, loc = 0, 0
        while pat:
            if pat & 1:
                yield id, loc
                id += 1
            pat >>= 1
            loc += 1

    def _gen_nano_kernel_preload_B(self, scalar, Nr, intrinsics: Intrinsics, pat, alignB=False):
        reg_t = intrinsics.vec_type
        vec_width_ele = intrinsics.vec_width_ele
        case_body = Block()

        b_load = lambda k: intrinsics.load(f'B_curr + {k} * {vec_width_ele}')

        b_load_a = lambda k, mask: intrinsics.load(f'B_curr_aligned + {k} * {vec_width_ele}', mask=mask) if vec_width_ele > 1 and alignB else intrinsics.load(f'B_curr + {k} * {vec_width_ele}')
        # b_load_a = lambda k, mask: "B_LOAD_A"

        # case_body += f'if (B_curr & ({vec_width_ele} - 1) && B_curr & (4-1)) {{'
        # case_body += f'    // B is not aligned to 16 bytes, but is aligned to 4 bytes'
        # case_body += f'    switch(B_curr & ~(16-1)) {{'
        # case_body += f'        case 4: {{'
        # case_body += f'            auto B_curr_aligned = B_curr - 4;'
        # case_body += f'            {reg_t} ba0 = {b_load_a(0, f"{4-1} << 12") };'
        # case_body += f'}}'
        # case_body += f'}}'
        for k in range(Nr):
            case_body += f'{reg_t} b{k};'

        if alignB and False:
            case_body += f'if ((uintptr_t)B_curr & ({vec_width_ele * 4} - 1) && !( (uintptr_t)B_curr & (16-1))) {{'
            case_body += f'    // B is not aligned to 16 bytes, but is aligned to 4 bytes'
            case_body += f'    switch((uintptr_t)B_curr & (64-1)) {{'
            case_body += f'        case 48: {{'
            case_body += f'            auto B_curr_aligned = B_curr - 12;'
            case_body += f'            {reg_t} ba0 = {b_load_a(0, f"0b1111000000000000") };'
            for k in range(Nr - 1):
                case_body += f'            {reg_t} ba{k + 1} = {b_load_a(k + 1, None)};'
                if vec_width_ele > 1 and alignB:
                    case_body += f'            b{k} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{k+1}", f"(__m512i)ba{k}", f"12")};'
            case_body += f'            {reg_t} ba{Nr} = {b_load_a(Nr, f"0b0000111111111111") };'
            if vec_width_ele > 1 and alignB:
                    case_body += f'            b{Nr-1} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{Nr}", f"(__m512i)ba{Nr-1}", f"12")};'
            # if vec_width_ele > 1 and alignB:
            #     for k in range(Nr):
            #         case_body += f'            b{k} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{k}", f"(__m512i)ba{k + 1}", f"12")};'
            case_body += f'            break;'
            case_body += f'        }}'
            case_body += f'        case 32: {{'
            case_body += f'            auto B_curr_aligned = B_curr - 8;'
            case_body += f'            {reg_t} ba0 = {b_load_a(0, f"0b1111111100000000") };'
            for k in range(Nr - 1):
                case_body += f'            {reg_t} ba{k + 1} = {b_load_a(k + 1, None)};'
                if vec_width_ele > 1 and alignB:
                    case_body += f'            b{k} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{k+1}", f"(__m512i)ba{k}", f"8")};'
            case_body += f'            {reg_t} ba{Nr} = {b_load_a(Nr, f"0b0000000011111111") };'
            if vec_width_ele > 1 and alignB:
                    case_body += f'            b{Nr-1} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{Nr}", f"(__m512i)ba{Nr-1}", f"8")};'
            # if vec_width_ele > 1 and alignB:
            #     for k in range(Nr):
            #         case_body += f'            b{k} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{k}", f"(__m512i)ba{k + 1}", f"8")};'
            case_body += f'            break;'
            case_body += f'        }}'
            case_body += f'        case 16: {{'
            case_body += f'            auto B_curr_aligned = B_curr - 4;'
            case_body += f'            {reg_t} ba0 = {b_load_a(0, f"0b1111111111110000") };'
            for k in range(Nr - 1):
                case_body += f'            {reg_t} ba{k + 1} = {b_load_a(k + 1, None)};'
                if vec_width_ele > 1 and alignB:
                    case_body += f'            b{k} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{k+1}", f"(__m512i)ba{k}", f"4")};'
            case_body += f'            {reg_t} ba{Nr} = {b_load_a(Nr, f"0b0000000000001111") };'
            if vec_width_ele > 1 and alignB:
                    case_body += f'            b{Nr-1} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{Nr}", f"(__m512i)ba{Nr-1}", f"4")};'
            # if vec_width_ele > 1 and alignB:
            #     for k in range(Nr):
            #         case_body += f'            b{k} = ({reg_t}) {intrinsics.alignr(f"(__m512i)ba{k}", f"(__m512i)ba{k + 1}", f"4")};'
            case_body += f'            break;'
            case_body += f'        }}'
            case_body += f'     }}'
            case_body += f'}}'

            case_body += f'else {{'
            for k in range(Nr):
                case_body += f'b{k} = {intrinsics.load(f"B_curr + {k} * {vec_width_ele}")};'
            case_body += f'}}'
        else:
            for k in range(Nr):
                case_body += f'b{k} = {intrinsics.load(f"B_curr + {k} * {vec_width_ele}")};'

        case_body += f'B_curr = (*col_indices_curr) * B_stride + B; col_indices_curr++;'

        for _, loc in self.nnz_iterator(pat):
            case_body += f'{reg_t} a{loc} = {intrinsics.broadcast_from_ptr(f"curr_value_ptr")};'
            case_body += f'curr_value_ptr++;'
            case_body += [f'c{loc}{k} = {intrinsics.fma(f"a{loc}", f"b{k}", f"c{loc}{k}")};' for k in range(Nr)]

        return case_body.sub_elements

    def _gen_nano_kernel_preload_A(self, scalar, Nr, intrinsics: Intrinsics, pat):
        reg_t = intrinsics.vec_type
        vec_width_ele = intrinsics.vec_width_ele
        size_of_scalar = self.size_of_scalar[scalar]

        if vec_width_ele > 1 and popcount(pat) > 2:
            nnz = popcount(pat)

            case_body = Block()
            for i in range(0, nnz, vec_width_ele):
                case_body += f'{reg_t} a{i} = {intrinsics.load(f"curr_value_ptr + {i}")};'
            case_body += f'curr_value_ptr = ({scalar}*)((uintptr_t) curr_value_ptr + {nnz * size_of_scalar});'

            for k in range(Nr):
                case_body += f'{reg_t} b{k} = {intrinsics.load(f"B_curr + {k} * {vec_width_ele}")};'
                for offset, loc in self.nnz_iterator(pat):
                    lane = offset % vec_width_ele
                    case_body += f'c{loc}{k} = {intrinsics.fma( f"b{k}", f"a{offset - lane}", f"c{loc}{k}", lane=lane)};'

            case_body += f'B_curr = ({scalar}*)((*col_indices_curr++) * scaled_B_stride + (uintptr_t) B);'
        else:
            case_body = Block()
            case_body += [f'{reg_t} a{loc} = {intrinsics.broadcast_from_ptr("curr_value_ptr")}; curr_value_ptr++;'
                          for _, loc in self.nnz_iterator(pat)]

            for k in range(Nr):
                b_loc = f"({scalar}*)((uintptr_t) B_curr + {k} * {vec_width_ele * size_of_scalar})"
                case_body += f'{reg_t} b{k} = {intrinsics.load(b_loc)};'
                for _, loc in self.nnz_iterator(pat):
                    case_body += f'c{loc}{k} = {intrinsics.fma(f"a{loc}", f"b{k}", f"c{loc}{k}")};'

            case_body += f'B_curr = ({scalar}*)((*col_indices_curr++) * scaled_B_stride + (uintptr_t) B);'

        return case_body.sub_elements

    def _emit_executor_body(self,
                            arch_str,
                            Nr,
                            scalar,
                            intrinsics: Intrinsics,
                            alignB=False):
        reg_t = intrinsics.vec_type
        vec_width_ele = intrinsics.vec_width_ele
        size_of_scalar = 4


        sop_panel_executor = Block()
        sop_panel_executor += ''
        body = Block()

        ##
        #   Setup Accumulator
        ##
        body += f'{reg_t} ' + ", ".join([f'c{i}{j}' for i in range(self.Mr) for j in range(Nr)]) + ';'

        c_row = lambda i: f' C + {i} * C_stride'
        c_load = lambda i, k: intrinsics.load(f'C{i} + {k} * {vec_width_ele}')

        body += f'uint64_t scaled_B_stride = B_stride * {size_of_scalar};'
        body += f'uint64_t scaled_C_stride = C_stride * {size_of_scalar};'

        body += 'if (load_c) {'
        body += [f'  {scalar}* __restrict__ C{i} = {c_row(i)};' for i in range(self.Mr)]
        body += [f'  c{i}{k} = {c_load(i, k)};' for i in range(self.Mr) for k in range(Nr)]
        body += '} else if (bias) {'
        body += [f'  c{i}{k} = {intrinsics.broadcast_from_ptr(f"bias + {i}")};'
                 for i in range(self.Mr) for k in range(Nr)]
        body += '} else {'
        body += [f'  c{i}{k} = {intrinsics.zero_reg()};' for i in range(self.Mr) for k in range(Nr)]
        body += '}'

        body += 'int c_idx = 0;'
        body += f'{scalar} *__restrict__ curr_value_ptr = values;'

        body += f'const {scalar} *__restrict__ B_curr ' \
                f'= ({scalar}*) (col_indices[0] * scaled_B_stride + (uintptr_t)B);'
        body += 'uint32_t * col_indices_curr = col_indices + 1;'

        ##
        #   Gen Nanokernels
        ##
        for id, pat in enumerate(self.nanokernels):
            nkern_loop = ForLoop(f'int pat_count = nkern_counts[{id}]', 'pat_count > 0', 'pat_count--',
                                 unroll=unroll_mapping(arch_str, nnzs=popcount(pat)))

            if arch_str == "NEON":
                nkern_loop += self._gen_nano_kernel_preload_A(scalar, Nr, intrinsics, pat)
            else:
                nkern_loop += self._gen_nano_kernel_preload_B(scalar, Nr, intrinsics, pat, alignB=alignB)
            body += nkern_loop

        ##
        #   Store accumulator
        ##
        c_store = lambda i, k: intrinsics.store(f'C_out + {i} * C_out_stride + {k * vec_width_ele}', f'c{i}{k}')

        body += 'if (activation == MINMAX && apply_activation) {'
        body += f'  {reg_t} min_vec = {intrinsics.broadcast("min")};'
        body += f'  {reg_t} max_vec = {intrinsics.broadcast("max")};'
        for i in range(self.Mr):
            for k in range(Nr):
                body += f'   c{i}{k} = {intrinsics.min(f"c{i}{k}", "max_vec")};'
                body += f'   c{i}{k} = {intrinsics.max(f"c{i}{k}", "min_vec")};'
                body += f'   {c_store(i, k)};'
        body += '} else {'
        body += [f'  {c_store(i, k)};' for i in range(self.Mr) for k in range(Nr)]
        body += '}'
        sop_panel_executor += body.sub_elements

        return sop_panel_executor

    def _emit_executor_body_vectorized(self, Nr, arch, vec_width_bits, scalar,
                                       packed_C=False, packed_B=False,
                                       mask=None):
        arch_str = arch
        arch = self.supported_archs[arch]
        assert arch.supports_scalar(scalar)

        # TODO: should really be aarch64
        if arch_str == "NEON+no":
            return f'''
            asm_{self.Mr}x{Nr}(
                C, C_stride,
                C_out, C_out_stride,
                B, B_stride,
                nkern_counts, col_indices, values,
                load_c, apply_activation,
                bias,
                activation == MINMAX,
                &min, &max
            );
            '''
        else:
            arch_intrinsics = ArchIntrinGenerator(arch, vec_width_bits, scalar) #
            vec_width_ele = int(vec_width_bits / SCALAR_SIZE_BITS[scalar])
            vector_intrinsics = self.Intrinsics(
                vec_type=arch_intrinsics.vec_type(),
                vec_width_ele=vec_width_ele,
                load=arch_intrinsics.load_intrin,
                load_aligned=partial(arch_intrinsics.load_intrin, aligned=True),
                store=arch_intrinsics.store_intrin,
                store_aligned=partial(arch_intrinsics.store_intrin, aligned=True),
                broadcast_from_ptr=arch_intrinsics.broadcast_from_ptr_intrin,
                broadcast=arch_intrinsics.broadcast_intrin,
                fma=arch_intrinsics.fma_intrin,
                zero_reg=arch_intrinsics.setzero_intrin,
                min=arch_intrinsics.min_intrin,
                max=arch_intrinsics.max_intrin,
                alignr=arch_intrinsics.alignr_intrin
            )

            return self._emit_executor_body(arch_str, Nr, scalar, vector_intrinsics, \
                alignB=True if arch_str == 'AVX512' or arch_str == 'AVX2' else False).emit(6)

    def _emit_executor_body_cleanup(self, Nr, arch, vec_width_bits, scalar,
                                    packed_C=False, packed_B=False,
                                    mask=None):
        arch_str = arch
        arch = self.supported_archs[arch]
        assert arch.supports_scalar(scalar)

        arch_intrinsics = ArchIntrinGenerator(arch, vec_width_bits, scalar)
        vec_width_ele = int(vec_width_bits / SCALAR_SIZE_BITS[scalar])
        vector_intrinsics = self.Intrinsics(
            vec_type=arch_intrinsics.vec_type(),
            vec_width_ele=vec_width_ele,
            load=arch_intrinsics.load_intrin,
            load_aligned=partial(arch_intrinsics.load_intrin, aligned=True),
            store=arch_intrinsics.store_intrin,
            store_aligned=partial(arch_intrinsics.store_intrin, aligned=True),
            broadcast_from_ptr=arch_intrinsics.broadcast_from_ptr_intrin,
            broadcast=arch_intrinsics.broadcast_intrin,
            fma=arch_intrinsics.fma_intrin,
            zero_reg=arch_intrinsics.setzero_intrin,
            min=arch_intrinsics.min_intrin,
            max=arch_intrinsics.max_intrin,
            alignr=arch_intrinsics.alignr_intrin
        )

        scalar_intrinsics = self.Intrinsics(
            vec_type=scalar,
            vec_width_ele=1,
            load=lambda x: f'*({x})',
            load_aligned=lambda x: f'*({x})',
            store=lambda x, y: f'*({x}) = {y}',
            store_aligned=lambda x, y: f'*({x}) = {y}',
            broadcast_from_ptr=lambda x: f'*({x})',
            broadcast=lambda x: f'{x}',
            fma=lambda a, b, c, lane=0: f'{a} * {b} + {c}',
            zero_reg=lambda: '0',
            min=lambda x, y: f'(({x} > {y}) ? {y} : {x})',
            max=lambda x, y: f'(({x} > {y}) ? {x} : {y})',
            alignr=None
        )

        func_body = Block()
        vec_cleanup_loop = ForLoop(f'', f'elements_remaining >= {vec_width_ele}',
                                   f'elements_remaining -= {vec_width_ele}, '
                                   f'C += {vec_width_ele}, C_out += {vec_width_ele}, B += {vec_width_ele}')
        vec_cleanup_loop += self._emit_executor_body(arch_str, 1, scalar, vector_intrinsics, alignB=False)

        scalar_cleanup_loop = ForLoop(f'', f'elements_remaining', f'elements_remaining--, '
                                      f'C += 1, C_out += 1, B += 1')
        scalar_cleanup_loop += self._emit_executor_body(arch_str, 1, scalar, scalar_intrinsics, alignB=False)

        func_body += vec_cleanup_loop
        func_body += scalar_cleanup_loop
        return func_body.emit(6)

    @staticmethod
    def _emit_microkernels(main_body: callable, cleanup_body: callable, Nr, scalar, name="", masked=False):
        mask_str = ""
        mask_arg_string =""
        mask = None

        if masked:
            mask = 'mask'
            mask_str = f' {mask},'
            mask_arg_string = f'Mask {mask},'

        if len(name) > 0 and name[0] != '_':
            name = '_' + name

        return f'''
        
    inline __attribute__((__always_inline__)) void vectorized(
        {scalar} *__restrict__ C, const int C_stride,
        {scalar} *__restrict__ C_out, const int C_out_stride, 
        const {scalar} *__restrict__ B, const int B_stride,
        int* __restrict__ nkern_counts,
        uint32_t* __restrict__ col_indices,
        {scalar}* __restrict__ values,
        const bool load_c,
        const bool apply_activation,
        const {scalar} *__restrict__ bias = nullptr)
    {{\n{main_body(Nr=Nr, scalar=scalar)}
    }}\n\n
        
    inline __attribute__((__always_inline__)) void vectorized(
        {scalar} *__restrict__ C, const int C_stride, 
        const {scalar} *__restrict__ B, const int B_stride,
        int* __restrict__ nkern_counts,
        uint32_t* __restrict__ col_indices,
        {scalar}* __restrict__ values,
        const bool load_c,
        const bool apply_activation,
        const {scalar} *__restrict__ bias = nullptr)
    {{\n
        vectorized(C, C_stride, C, C_stride, B, B_stride,
                   nkern_counts, col_indices, values, 
                   load_c, apply_activation, bias);
    }}\n\n
    
    inline __attribute__((__always_inline__)) void cleanup(
        int elements_remaining,
        {scalar} *__restrict__ C, const int C_stride,
        {scalar} *__restrict__ C_out, const int C_out_stride, 
        const {scalar} *__restrict__ B, const int B_stride,
        int* __restrict__ nkern_counts,
        uint32_t* __restrict__ col_indices,
        {scalar}* __restrict__ values,
        const bool load_c,
        const bool apply_activation,
        const {scalar} *__restrict__ bias = nullptr)
    {{\n{cleanup_body(Nr=Nr, scalar=scalar)}
    }}\n\n
    
    inline __attribute__((__always_inline__)) void cleanup(
        int elements_remaining,
        {scalar} *__restrict__ C, const int C_stride, 
        const {scalar} *__restrict__ B, const int B_stride,
        int* __restrict__ nkern_counts,
        uint32_t* __restrict__ col_indices,
        {scalar}* __restrict__ values,
        const bool load_c,
        const bool apply_activation,
        const {scalar} *__restrict__ bias = nullptr)
    {{\n
        cleanup(elements_remaining,C, C_stride, C, C_stride, B, B_stride,
                nkern_counts, col_indices, values,
                load_c, apply_activation, bias);
    }}\n\n
    '''
