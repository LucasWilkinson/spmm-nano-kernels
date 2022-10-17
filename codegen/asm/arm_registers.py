import copy

# 32 128bit floating point vectors (overlap with floating point registers)
MAX_VECTOR_REGISTERS = 32
# 32 - 2 (stack pointer and frame pointer), we can use those if absolutely necessary, with modifications
MAX_GENERAL_PURPOSE_REGISTERS = 30


class FloatingPointOrVectorRegister:
    allocated = set()
    clobbered = set()

    # See
    #  https://developer.arm.com/documentation/dui0801/a/programmers-model/registers-and-the-stack-frame?lang=en
    v64 = ['8B', '4H', '2S']
    v128 = ['16B', '8H', '4S', '2D']
    vtype = v64 + v128
    ftype = ['H', 'S', 'D', 'Q']

    def __init__(self, default_type="4S", allocation_reverse=False):
        if default_type not in self.vtype and default_type not in self.ftype:
            raise ValueError(f"Invalid type {default_type}")

        if len(self.allocated) >= MAX_VECTOR_REGISTERS:
            raise Exception("Too many vector registers allocated")

        if not allocation_reverse:
            reg_list = range(MAX_VECTOR_REGISTERS)
        else:
            reg_list = reversed(range(MAX_VECTOR_REGISTERS))

        for reg_num in reg_list:
            if reg_num not in self.allocated:
                self.reg_num = reg_num
                break

        self.default_type = default_type
        self.allocated.add(self.reg_num)
        self.clobbered.add(self.reg_num)
        self.is_view = False

    def bytes(self):
        if self.default_type in self.v64:
            return 6
        elif self.default_type in self.v128:
            return 16
        elif self.default_type in self.ftype:
            return {"H": 2, "S": 4, "D": 8, "Q": 16}[self.default_type]

    def dealloc(self):
        if self.reg_num in self.allocated:
            self.allocated.remove(self.reg_num)

    def __del__(self):
        if not self.is_view and self.reg_num in self.allocated:
            self.allocated.remove(self.reg_num)

    def __str__(self):
        if self.default_type in self.vtype:
            return f"V{self.reg_num}.{self.default_type}"
        elif self.default_type in self.ftype:
            return f"{self.default_type.lower()}{self.reg_num}"

    def __repr__(self):
        return str(self)

    def view_as(self, type):
        view = copy.copy(self)
        view.default_type = type
        view.is_view = True
        return view


class GeneralPurposeRegister:
    allocated = set()
    clobbered = set()

    def __init__(self, pin=None, default_width="x"):
        if len(self.allocated) >= MAX_GENERAL_PURPOSE_REGISTERS:
            raise Exception("Too many general purpose registers allocated")

        if pin is not None:
            if pin in self.allocated:
                raise Exception("General purpose register already allocated")
            self.reg_num = pin
        else:
            for reg_num in range(MAX_GENERAL_PURPOSE_REGISTERS):
                if reg_num not in self.allocated:
                    self.reg_num = reg_num
                    break
        self.default_width = default_width
        self.allocated.add(self.reg_num)
        self.clobbered.add(self.reg_num)
        self.is_view = False

    def dealloc(self):
        if self.reg_num in self.allocated:
            self.allocated.remove(self.reg_num)

    def __del__(self):
        if not self.is_view and self.reg_num in self.allocated:
            self.allocated.remove(self.reg_num)

    def __str__(self):
        return f"{self.default_width}{self.reg_num}"

    def __repr__(self):
        return str(self)

    def view_as(self, type):
        view = copy.copy(self)
        view.default_width = type
        view.is_view = True
        return view
