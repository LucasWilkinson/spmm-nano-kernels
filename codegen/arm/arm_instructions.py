from arm_registers import *

active_function = None


def instruction(func):
    def wrapper(*args, **kwargs):
        global active_function
        ret = func(*args, **kwargs)
        if active_function is not None:
            active_function += ret
        return ret
    return wrapper


@instruction
def FMA(dst, src1, src2, lane=None):
    if lane is not None:
        return f"FMLA {dst}, {src1}, {src2}[{lane}]"
    else:
        return f"FMLA {dst}, {src1}, {src2}"


@instruction
def MOV(dst, src):
    return f"MOV {dst}, {src}"


@instruction
def ADD(dst, src1, src2):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    return f"ADD {dst}, {src1}, {src2}"


@instruction
def SUB(dst, src1, src2, shift=0):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    if shift > 0:
        src2 += f", LSL #{shift}"
    return f"SUB {dst}, {src1}, {src2}"


@instruction
def SUBS(dst, src1, src2, shift=0):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    if shift > 0:
        src2 += f", LSL #{shift}"
    return f"SUBS {dst}, {src1}, {src2}"


@instruction
def CMP(src1, src2, shift=0):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    if shift > 0:
        src2 += f", LSL #{shift}"
    return f"CMP {src1}, {src2}"

@instruction
def CCMP(src1, src2, flags, shift=0, cond=""):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    if shift > 0:
        src2 += f", LSL #{shift}"
    return f"CCMP {src1}, {src2}, {flags}, {cond}"



@instruction
def B(label, cond="AL"):
    return f"B{cond} {label}"


@instruction
def MUL(dst, src1, src2):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    return f"MUL {dst}, {src1}, {src2}"


@instruction
def MOVI(dst, value):
    return f"MOVI {dst}, #{value}"


@instruction
def UMADDL(dst, src1, src2, add):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    else:
        src2 = src2.view_as("w")
    return f"UMADDL {dst}, {src1.view_as('w')}, {src2}, {add}"


@instruction
def UMULL(dst, src1, src2):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    else:
        src2 = src2.view_as("w")
    return f"UMULL {dst.view_as('x')}, {src1.view_as('w')}, {src2}"


@instruction
def LSL(dst, src1, src2):
    if isinstance(src2, int):
        src2 = f"#{src2}"
    return f"LSL {dst}, {src1}, {src2}"


def address(base, mode="offset", offset=0, shift=0):
    if isinstance(offset, int) and offset > 0:
        offset = f"#{offset}"
    if shift > 0:
        offset += f", LSL #{shift}"

    if mode == "post":
        return f"[{base}], {offset}"
    elif mode == "pre":
        return f"[{base}, {offset}]!"
    else:
        return f"[{base}, {offset}]"


@instruction
def LDR(dst, address):
    return f"LDR {dst}, {address}"


@instruction
def FMAX(dst, src1, src2):
    return f"FMAX {dst}, {src1}, {src2}"


@instruction
def FMIN(dst, src1, src2):
    return f"FMIN {dst}, {src1}, {src2}"


@instruction
def LD1R(dsts, src, inc=False):
    if isinstance(dsts, FloatingPointOrVectorRegister):
        dsts = [dsts]

    type = dsts[0].default_type
    for dst in dsts:
        assert dst.default_type == type

    assert len(dsts) <= 4
    dsts_str = ", ".join([str(dst) for dst in dsts])
    ret = f"LD1R {{ {dsts_str} }}, [{src}]"
    if inc:
        ret += f", #{len(dsts) * dsts[0].bytes()}"
    return ret


@instruction
def LD1(dsts, src, inc=False):
    if isinstance(dsts, FloatingPointOrVectorRegister):
        dsts = [dsts]

    type = dsts[0].default_type
    for dst in dsts:
        assert dst.default_type == type

    assert len(dsts) <= 4
    dsts_str = ", ".join([str(dst) for dst in dsts])
    ret = f"LD1 {{ {dsts_str} }}, [{src}]"
    if inc:
        ret += f", #{len(dsts) * dsts[0].bytes()}"
    return ret


@instruction
def ST1(dsts, src, inc=False):
    if isinstance(dsts, FloatingPointOrVectorRegister):
        dsts = [dsts]

    type = dsts[0].default_type
    for dst in dsts:
        assert dst.default_type == type

    assert len(dsts) <= 4
    dsts_str = ", ".join([str(dst) for dst in dsts])
    ret = f"ST1 {{ {dsts_str} }}, [{src}]"
    if inc:
        ret += f", #{len(dsts) * dsts[0].bytes()}"
    return ret