import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import glob
import json
from .arch_details import *

supported_archs = {
    'AVX512': AVX512(),
    'AVX2': AVX2(),
    'NEON': NEON(),
}

def generate_ukernel_executor_registration(output_root):
    factory_files = glob.glob(f'{output_root}/*/factories/**/executor_*.cpp', recursive=True)
    factory_descs = []

    for factory_file in factory_files:
        with open(factory_file, 'r') as f:
            for line in f.readlines():
                if "factory_desc" in line:
                    factory_descs.append(json.loads(line.split('|')[-1]))

    kernel_descs_pairs = set()
    for factory_desc in factory_descs:
        kernel_descs_pairs.add((factory_desc["kernel_desc"], factory_desc["scalar"], factory_desc["datatransform"]))

    with open(f'{output_root}/ukernel_executor_registration.cpp', 'w') as f:
        f.write('#include "KernelDesc.h"\n')
        f.write('#include "ExecutorFactory.h"\n\n')
        f.write('namespace sop {\n\n')

        for factory_desc in factory_descs:
            f.write(f'{supported_archs[factory_desc["arch"]].preprocessor_guard()}\n')
            f.write(f'#if defined(ENABLE_{factory_desc["arch"]})\n')
            f.write(f'extern ExecutorFactory<{factory_desc["kernel_desc"]}<{factory_desc["scalar"]}>, {factory_desc["datatransform"]}>* ')
            f.write(f'{factory_desc["func"]}();\n')
            f.write(f'#endif\n')
            f.write(f'#endif\n')

        f.write('\n')

        for kernel_descs, scalar, datatransform in kernel_descs_pairs:
            f.write(f'struct ExecutorFactory{kernel_descs}_scalar_{scalar}_datatransform_{datatransform}: public ExecutorFactory<{kernel_descs}<{scalar}>, {datatransform}> {{\n')
            f.write(f'ExecutorFactory{kernel_descs}_scalar_{scalar}_datatransform_{datatransform}(){{\n')

            for factory_desc in factory_descs:
                if factory_desc["kernel_desc"] != kernel_descs: continue
                if factory_desc["scalar"] != scalar: continue
                if factory_desc["datatransform"] != datatransform: continue
                #f.write(f'  ExecutorFactory<{factory_desc["kernel_desc"]}>::')
                f.write(f'{supported_archs[factory_desc["arch"]].preprocessor_guard()}\n')
                f.write(f'#if defined(ENABLE_{factory_desc["arch"]})\n')
                f.write(f'  register_factory("{factory_desc["id"]}", {factory_desc["func"]}());\n')
                f.write(f'#endif\n')
                f.write(f'#endif\n')

            f.write(f'}}\n')
            f.write(f'}};\n\n')
            f.write(f'ExecutorFactory{kernel_descs}_scalar_{scalar}_datatransform_{datatransform} trip_registration_for_{kernel_descs}_{scalar}_{datatransform};\n\n')
        f.write('}\n')


def generate_ukernel_packer_registration(output_root):
    factory_files = glob.glob(f'{output_root}/*/factories/**/packer_*.cpp', recursive=True)
    factory_descs = []

    for factory_file in factory_files:
        with open(factory_file, 'r') as f:
            for line in f.readlines():
                if "factory_desc" in line:
                    factory_descs.append(json.loads(line.split('|')[-1]))

    scalars = set()
    for factory_desc in factory_descs:
        scalars.add(factory_desc["scalar"])

    with open(f'{output_root}/ukernel_packer_registration.cpp', 'w') as f:
        f.write('#include "MicroKernelPackerFactory.h"\n\n')
        f.write('namespace sop {\n\n')

        for scalar in scalars:
            for factory_desc in filter(lambda x: x["scalar"] == scalar, factory_descs):
                f.write(f'{supported_archs[factory_desc["arch"]].preprocessor_guard()}\n')
                f.write(f'#if defined(ENABLE_{factory_desc["arch"]})\n')
                f.write(f'extern MicroKernelPackerFactory<{scalar}>* ')
                f.write(f'{factory_desc["func"]}();\n')
                f.write(f'#endif\n')
                f.write(f'#endif\n')

            f.write('\n')

            f.write(f'struct MicroKernelPackerFactory{scalar.capitalize()}: '
                    f'public MicroKernelPackerFactory<{scalar}> {{\n')
            f.write(f'MicroKernelPackerFactory{scalar.capitalize()}(): '
                    f'MicroKernelPackerFactory<{scalar}>({factory_desc["M_r"]}) {{\n')

            for factory_desc in filter(lambda x: x["scalar"] == scalar, factory_descs):
                f.write(f'{supported_archs[factory_desc["arch"]].preprocessor_guard()}\n')
                f.write(f'#if defined(ENABLE_{factory_desc["arch"]})\n')
                f.write(f'  register_factory("{factory_desc["id"]}", {factory_desc["func"]}());\n')
                f.write(f'#endif\n')
                f.write(f'#endif\n')

            f.write(f'}}\n')
            f.write(f'}};\n\n')
            f.write(f'MicroKernelPackerFactory{scalar.capitalize()} trip_registration_for_{scalar.capitalize()};\n\n')

        f.write('}\n')


def generate_ukernel_registration(output_root):
    generate_ukernel_executor_registration(output_root)
    generate_ukernel_packer_registration(output_root)
