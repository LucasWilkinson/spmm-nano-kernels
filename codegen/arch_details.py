from abc import ABC, abstractmethod
from functools import partial
from re import T

SCALAR_SIZE_BITS = {
    'float16': 16,
    'float': 32,
    'double': 64,
}


class Arch:
    def __init__(self):
        super().__init__()

    @abstractmethod
    def supports_masks(self) -> bool:
        pass

    @abstractmethod
    def supports_vec_width_bits(self, vec_width_bits) -> bool:
        pass

    @abstractmethod
    def supports_scalar(self, scalar) -> bool:
        pass

    @abstractmethod
    def preprocessor_guard(self):
        pass

    @abstractmethod
    def intrin_include(self):
        pass

    @abstractmethod
    def mask_type(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def vec_type(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def load_intrin(self, scalar, vec_width_bits, aligned=False):
        pass

    def masked_load_intrin(self, scalar, vec_width_bits, aligned=False):
        raise NotImplementedError()

    @abstractmethod
    def store_intrin(self, scalar, vec_width_bits, aligned=False):
        pass

    def masked_store_intrin(self, scalar, vec_width_bits, aligned=False):
        raise NotImplementedError()

    @abstractmethod
    def fma_intrin(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def setzero_intrin(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def broadcast_intrin(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def broadcast_from_ptr_intrin(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def min_intrin(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def max_intrin(self, scalar, vec_width_bits):
        pass

    @abstractmethod
    def alignr_intrin(self, scalar, vec_width_bits):
        pass


class ArchIntrinGenerator:
    def __init__(self, arch: Arch, vec_width_bits: int, scalar: str):
        self.arch = arch
        self.vec_width_bits = vec_width_bits
        self.scalar = scalar
        assert arch.supports_vec_width_bits(vec_width_bits)
        assert arch.supports_scalar(scalar)

    def supports_masks(self):
        return self.arch.supports_masks()

    def vec_type(self):
        return self.arch.vec_type(self.scalar, self.vec_width_bits)

    def mask_type(self):
        return self.arch.mask_type(self.scalar, self.vec_width_bits)

    def load_intrin(self, ptr, aligned=False, mask=None):
        if mask is None:
            return self.arch.load_intrin(self.scalar, self.vec_width_bits, aligned)(ptr)
        else:
            assert self.supports_masks()
            return self.arch.masked_load_intrin(self.scalar, self.vec_width_bits, aligned)(ptr, mask=mask)

    def store_intrin(self, ptr, val, aligned=False, mask=None):
        if mask is None:
            return self.arch.store_intrin(self.scalar, self.vec_width_bits, aligned)(ptr, val)
        else:
            assert self.supports_masks()
            return self.arch.masked_store_intrin(self.scalar, self.vec_width_bits, aligned)(ptr, val, mask=mask)

    def fma_intrin(self, a, b, c, fused_broadcast=False, lane=None):
        return self.arch.fma_intrin(self.scalar, self.vec_width_bits, fused_broadcast=fused_broadcast, lane=lane)(a, b, c)

    def setzero_intrin(self):
        return self.arch.setzero_intrin(self.scalar, self.vec_width_bits)()

    def broadcast_intrin(self, val):
        return self.arch.broadcast_intrin(self.scalar, self.vec_width_bits)(val)

    def broadcast_from_ptr_intrin(self, ptr):
        return self.arch.broadcast_from_ptr_intrin(self.scalar, self.vec_width_bits)(ptr)

    def min_intrin(self, a, b):
        return self.arch.min_intrin(self.scalar, self.vec_width_bits)(a, b)

    def max_intrin(self, a, b):
        return self.arch.max_intrin(self.scalar, self.vec_width_bits)(a, b)

    def alignr_intrin(self, a, b, offset):
        return self.arch.alignr_intrin(self.scalar, self.vec_width_bits)(a, b, offset)


class AVX(Arch, ABC):
    scalar_char = {
        "float": ("", "s"),
        "double": ("d", "d"),
    }

    def __init__(self):
        super(AVX, self).__init__()

    def supports_masks(self) -> bool:
        return True

    @staticmethod
    def _intrin(*args, vec_width_bits, scalar, func, masked=False, deref=False, **kwargs):
        args = list(args)
        m_reg_char, mm_func_char = AVX.scalar_char[scalar]
        mm = f'_mm{vec_width_bits}'

        if scalar == "double" and vec_width_bits == 128:
            mm = '_mm'

        if deref:
            args[0] = '*(' + args[0] + ')'

        mask_prefix = ""
        if masked:
            if "load" in func:
                mask_prefix = "maskz_"
            else:
                mask_prefix = "mask_"

            if "store" in func:
                args = (args[0],) + (kwargs["mask"],) + args[1:]
            else:
                # print(args)
                args = [kwargs["mask"]] + args[0:]

        if "alignr" in func:
            return f'{mm}_{mask_prefix}{func}_epi32({", ".join(args)})'
        return f'{mm}_{mask_prefix}{func}_p{mm_func_char}({", ".join(args)})'

    def intrin_include(self):
        return '#include <immintrin.h>'

    def vec_type(self, scalar, vec_width_bits):
        m_reg_char, _ = AVX.scalar_char[scalar]
        return f'__m{vec_width_bits}{m_reg_char}'

    def load_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX._intrin, func='load' if aligned else 'loadu', vec_width_bits=vec_width_bits, scalar=scalar)
    
    def masked_load_intrin(self, scalar, vec_width_bits, aligned=False): #
        return partial(AVX._intrin, func='load' if aligned else 'loadu', vec_width_bits=vec_width_bits, scalar=scalar, masked=True)

    def store_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX._intrin, func='store' if aligned else 'storeu', vec_width_bits=vec_width_bits, scalar=scalar)

    def fma_intrin(self, scalar, vec_width_bits, fused_broadcast=False, lane=None):
        assert lane is None
        return partial(AVX._intrin, func='fmadd', vec_width_bits=vec_width_bits, scalar=scalar)

    def setzero_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='setzero', vec_width_bits=vec_width_bits, scalar=scalar)

    def broadcast_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='set1', vec_width_bits=vec_width_bits, scalar=scalar)

    def broadcast_from_ptr_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='set1', vec_width_bits=vec_width_bits, scalar=scalar, deref=True)

    def max_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='max', vec_width_bits=vec_width_bits, scalar=scalar)

    def min_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='min', vec_width_bits=vec_width_bits, scalar=scalar)

    def alignr_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='alignr', vec_width_bits=vec_width_bits, scalar=scalar)


class AVX512(AVX):
    def __init__(self):
        super(AVX, self).__init__()

    def supports_masks(self) -> bool:
        return True

    def supports_vec_width_bits(self, vec_width_bits) -> bool:
        return vec_width_bits in [128, 256, 512]

    def supports_scalar(self, scalar) -> bool:
        return scalar in ["float", "double"]

    def preprocessor_guard(self):
        return "#ifdef __AVX512F__"

    def mask_type(self, scalar, vec_width_bits):
        return f'__mmask{int(vec_width_bits / SCALAR_SIZE_BITS[scalar])}'

    def load_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX512._intrin, func='load' if aligned else 'loadu', vec_width_bits=vec_width_bits, scalar=scalar)
    
    def masked_load_intrin(self, scalar, vec_width_bits, aligned=False): #
        return partial(self.load_intrin(scalar, vec_width_bits, aligned=aligned), masked=True)

    def masked_store_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(self.store_intrin(scalar, vec_width_bits, aligned=aligned), masked=True)
    
    def alignr_intrin(self, scalar, vec_width_bits):
        return partial(AVX512._intrin, func='alignr', vec_width_bits=vec_width_bits, scalar=scalar)


class AVX2(AVX):
    def __init__(self):
        super(AVX, self).__init__()

    def supports_masks(self) -> bool:
        return True

    def supports_vec_width_bits(self, vec_width_bits) -> bool:
        return vec_width_bits in [128, 256]

    def supports_scalar(self, scalar) -> bool:
        return scalar in ["float", "double"]

    def preprocessor_guard(self):
        return "#ifdef __AVX2__"

    def mask_type(self, scalar, vec_width_bits):
        return f'uint32_t'

    def masked_load_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(self.load_intrin(scalar, vec_width_bits, aligned=aligned), masked=True)

    def masked_store_intrin(self, scalar, vec_width_bits, aligned=False):
        raise NotImplementedError()
    
    def alignr_intrin(self, scalar, vec_width_bits):
        return partial(AVX2._intrin, func='alignr', vec_width_bits=vec_width_bits, scalar=scalar)


class NEON(Arch, ABC):
    scalar_convert = {
        "float": "float32",
        "double": "float64",
    }

    instruction_suffix = {
        "float16": "f16",
        "float": "f32"
    }

    def __init__(self):
        super(NEON, self).__init__()

    def supports_vec_width_bits(self, vec_width_bits) -> bool:
        return vec_width_bits in [64, 128]

    def supports_scalar(self, scalar) -> bool:
        return scalar in ["float16", "float"]

    def supports_masks(self) -> bool:
        return False

    def preprocessor_guard(self):
        return "#if defined(__ARM_NEON) || defined(__ARM_NEON__)"

    def intrin_include(self):
        return '#include <arm_neon.h>'

    def mask_type(self, scalar, vec_width_bits):
        return f'uint32_t'

    def vec_type(self, scalar, vec_width_bits):
        vec_width_ele = int(vec_width_bits / SCALAR_SIZE_BITS[scalar])
        return f'{self.scalar_convert[scalar]}x{vec_width_ele}_t'

    def load_intrin(self, scalar, vec_width_bits, aligned=False):
        return lambda src: f'vld1q_{NEON.instruction_suffix[scalar]}({src})'

    def store_intrin(self, scalar, vec_width_bits, aligned=False):
        return lambda dst, val: f'vst1q_{NEON.instruction_suffix[scalar]}({dst}, {val})'

    def fma_intrin(self, scalar, vec_width_bits, fused_broadcast=False, lane=None):

        if lane is None:
            return lambda a, b, c: f'vfmaq{"_n" if fused_broadcast else ""}_{NEON.instruction_suffix[scalar]}({c}, {a}, {b})'
        else:
            assert fused_broadcast is False
            return lambda a, b, c: f'vfmaq_laneq_{NEON.instruction_suffix[scalar]}({c}, {a}, {b}, {lane})'

    def setzero_intrin(self, scalar, vec_width_bits):
        return lambda: f'vmovq_n_{NEON.instruction_suffix[scalar]}(0)'

    def broadcast_intrin(self, scalar, vec_width_bits):
        return lambda src: f'vld1q_dup_{NEON.instruction_suffix[scalar]}(&({src}))'

    def broadcast_from_ptr_intrin(self, scalar, vec_width_bits):
        return lambda src: f'vld1q_dup_{NEON.instruction_suffix[scalar]}({src})'

    def max_intrin(self, scalar, vec_width_bits):
        return lambda a, b: f'vmaxq_{NEON.instruction_suffix[scalar]}({a}, {b})'

    def min_intrin(self, scalar, vec_width_bits):
        return lambda a, b: f'vminq_{NEON.instruction_suffix[scalar]}({a}, {b})'


instruction_set_reg_width = {
    "AVX512": 512,
    "AVX2": 256,
    "NEON": 128,
}

