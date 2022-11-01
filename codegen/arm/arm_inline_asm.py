from arm_instructions import *
import copy
import math


class Label:
    index = 0

    def __init__(self, name=None, here=False):
        self.index = Label.index
        if name is None:
            self.name = f"lbl_{self.index}"
        else:
            self.name = name

        Label.index += 1
        import arm_instructions
        if arm_instructions.active_function is not None and here:
            arm_instructions.active_function += f"{self.name}:"

    def __str__(self):
        return self.name


class Argument:
    def __init__(self, name, type, asm_type, floating_point=False):
        self.name = name
        self.type = type
        self.asm_type = asm_type
        self.floating_point = floating_point


class LocalVariable:
    offset = 0

    def __init__(self, name, size_bytes, floating_point=False):
        self.name = name
        self.floating_point = floating_point
        self.offset += (size_bytes + 3 // 4) * 4    # Stay 4 byte aligned


class InlineASMFunction:
    def __init__(self, name, args, ret_type="void"):
        self.name = name
        self.args = args
        self.ret_type = ret_type
        self.asm = []

        self.function_signature = f"{ret_type} {name}("
        for arg in args:
            self.function_signature += f"{arg.type} {arg.name}, "
        self.function_signature = self.function_signature[:-2] + ")"

        self.floating_point_input_reg_nums = set()
        self.general_purpose_input_reg_nums = set()

        self.input_registers_dict = {}
        self.input_registers = []
        for arg in self.args:
            if arg.floating_point:
                reg = FloatingPointOrVectorRegister(default_type="S")
                self.floating_point_input_reg_nums.add(reg.reg_num)
            else:
                reg = GeneralPurposeRegister()
                self.general_purpose_input_reg_nums.add(reg.reg_num)
            self.input_registers.append(reg)
            self.input_registers_dict[arg.name] = reg

        self.function_init = self.function_signature + "{\n"
        for input_num, (arg, reg) in enumerate(zip(self.args, self.input_registers)):
            self.function_init += f"\tregister {arg.type}  i{input_num} asm(\"{reg}\") = {arg.name};\n"

        self.input_list = []
        for input_num, (arg, reg) in enumerate(zip(self.args, self.input_registers)):
            self.input_list.append(f"\"{arg.asm_type}\"(i{input_num})")
        self.input_list = "\t: " + ", ".join(self.input_list) + "\n"

    def __add__(self, other):
        self.asm.append(other)
        return self

    def __enter__(self):
        LocalVariable.offset = 0
        import arm_instructions
        arm_instructions.active_function = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        import arm_instructions
        arm_instructions.active_function = None

    def del_last_instruction(self):
        self.asm = self.asm[:-1]

    def insert_label(self, label):
        self.asm.append(f"{label}:")

    def emit(self):
        clobberd_floating_point_regs = FloatingPointOrVectorRegister.clobbered - self.floating_point_input_reg_nums
        clobberd_floating_point_regs = [f"q{reg}" for reg in clobberd_floating_point_regs]
        clobberd_general_purpose_regs = GeneralPurposeRegister.clobbered - self.general_purpose_input_reg_nums
        clobberd_general_purpose_regs = [f"x{reg}" for reg in clobberd_general_purpose_regs]
        clobberd_regs = clobberd_floating_point_regs + clobberd_general_purpose_regs
        clobberd_regs_str = ','.join(f'\"{reg}\"' for reg in clobberd_regs)

        asm_str = ""
        for line in self.asm:
            if ":" not in line:
                asm_str += f"\n\t\"{line:50} \\n\\t\""
            else: # LABEL
                asm_str += f"\n\"{line:50} \t\\n\""

        out = self.function_init
        out += "\tasm volatile ("
        out += asm_str
        out += "\n\t: \n"
        out += self.input_list
        out += f"\t: {clobberd_regs_str});\n"
        out += "}\n"
        return out
