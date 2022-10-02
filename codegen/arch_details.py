from abc import ABC, abstractmethod
from functools import partial

SCALAR_SIZE_BITS = {
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
    def mask_type(self, vec_width_bits):
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
        return self.arch.mask_type(self.vec_width_bits)

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

    def fma_intrin(self, a, b, c):
        return self.arch.fma_intrin(self.scalar, self.vec_width_bits)(a, b, c)

    def setzero_intrin(self):
        return self.arch.setzero_intrin(self.scalar, self.vec_width_bits)()

    def broadcast_intrin(self, val):
        return self.arch.broadcast_intrin(self.scalar, self.vec_width_bits)(val)


class AVX(Arch, ABC):
    scalar_char = {
        "float": ("", "s"),
        "double": ("d", "d"),
    }

    def __init__(self):
        super(AVX, self).__init__()

    @staticmethod
    def _intrin(*args, vec_width_bits, scalar, func, masked=False, **kwargs):
        m_reg_char, mm_func_char = AVX.scalar_char[scalar]
        mm = f'_mm{vec_width_bits}{m_reg_char}'

        mask_prefix = ""
        if masked:
            if "load" in func:
                mask_prefix = "maskz_"
            else:
                mask_prefix = "mask_"

            if "store" in func:
                args = (args[0],) + (kwargs["mask"],) + args[1:]
            else:
                args = (kwargs["mask"],) + args[0:]

        return f'{mm}_{mask_prefix}{func}_p{mm_func_char}({", ".join(args)})'

    def vec_type(self, scalar, vec_width_bits):
        return f'__m{vec_width_bits}'

    def load_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX._intrin, func='load' if aligned else 'loadu', vec_width_bits=vec_width_bits, scalar=scalar)

    def store_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX._intrin, func='store' if aligned else 'storeu', vec_width_bits=vec_width_bits, scalar=scalar)

    def fma_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='fmadd', vec_width_bits=vec_width_bits, scalar=scalar)

    def setzero_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='setzero', vec_width_bits=vec_width_bits, scalar=scalar)

    def broadcast_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='set1', vec_width_bits=vec_width_bits, scalar=scalar)


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
        return "#ifdef __AVX512VL__"

    def mask_type(self, vec_width_bits):
        return f'__mmask{vec_width_bits}'

    def masked_load_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(self.load_intrin(scalar, vec_width_bits, aligned=aligned), masked=True)

    def masked_store_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(self.store_intrin(scalar, vec_width_bits, aligned=aligned), masked=True)


class AVX2(AVX):
    def __init__(self):
        super(AVX, self).__init__()

    def supports_masks(self) -> bool:
        return False

    def supports_vec_width_bits(self, vec_width_bits) -> bool:
        return vec_width_bits in [128, 256]

    def supports_scalar(self, scalar) -> bool:
        return scalar in ["float", "double"]

    def preprocessor_guard(self):
        return "#ifdef __AVX512VL__"

    def mask_type(self, vec_width_bits):
        return f'uint32_t'

    def masked_load_intrin(self, scalar, vec_width_bits, aligned=False):
        raise NotImplementedError()

    def masked_store_intrin(self, scalar, vec_width_bits, aligned=False):
        raise NotImplementedError()

class AVX(Arch, ABC):
    scalar_char = {
        "float": ("", "s"),
        "double": ("d", "d"),
    }

    def __init__(self):
        super(AVX, self).__init__()

    @staticmethod
    def _intrin(*args, vec_width_bits, scalar, func, masked=False, **kwargs):
        m_reg_char, mm_func_char = AVX.scalar_char[scalar]
        mm = f'_mm{vec_width_bits}{m_reg_char}'

        mask_prefix = ""
        if masked:
            if "load" in func:
                mask_prefix = "maskz_"
            else:
                mask_prefix = "mask_"

            if "store" in func:
                args = (args[0],) + (kwargs["mask"],) + args[1:]
            else:
                args = (kwargs["mask"],) + args[0:]

        return f'{mm}_{mask_prefix}{func}_p{mm_func_char}({", ".join(args)})'

    def vec_type(self, scalar, vec_width_bits):
        return f'__m{vec_width_bits}'

    def load_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX._intrin, func='load' if aligned else 'loadu', vec_width_bits=vec_width_bits, scalar=scalar)

    def store_intrin(self, scalar, vec_width_bits, aligned=False):
        return partial(AVX._intrin, func='store' if aligned else 'storeu', vec_width_bits=vec_width_bits, scalar=scalar)

    def fma_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='fmadd', vec_width_bits=vec_width_bits, scalar=scalar)

    def setzero_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='setzero', vec_width_bits=vec_width_bits, scalar=scalar)

    def broadcast_intrin(self, scalar, vec_width_bits):
        return partial(AVX._intrin, func='set1', vec_width_bits=vec_width_bits, scalar=scalar)


instruction_set_reg_width = {
    "AVX512": 512,
    "AVX2": 256,
}

vec_type_info = {
    ("float", 512): (16, '', 's'),
    ("float", 256): (8, '', 's'),
    ("double", 512): (8, 'd', 'd'),
    ("double", 256): (4, 'd', 'd'),
}

min_instruction_sets = {
    512: "__AVX512VL__",
    256: "__AVX2__",
}
