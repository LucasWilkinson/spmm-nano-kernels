import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import glob
import json
from SOP.codegen.arch_details import *


def generate_ukernel_executor_registration(output_root):
    factory_files = glob.glob(f'{output_root}/*/factories/**/executor_*.cpp', recursive=True)
    factory_descs = []

    for factory_file in factory_files:
        with open(factory_file, 'r') as f:
            for line in f.readlines():
                if "factory_desc" in line:
                    factory_descs.append(json.loads(line.split('|')[-1]))

    kernel_descs = set()
    for factory_desc in factory_descs:
        kernel_descs.add(factory_desc["kernel_desc"])

    with open(f'{output_root}/ukernel_executor_registration.cpp', 'w') as f:
        f.write('#include "KernelDesc.h"\n')
        f.write('#include "ExecutorFactory.h"\n\n')
        f.write('namespace sop {\n\n')

        for factory_desc in factory_descs:
            f.write(f'#if defined({min_instruction_sets[factory_desc["reg_width_bits"]]})')
            f.write(f' && defined(ENABLE_{factory_desc["arch"]})\n')
            f.write(f'extern ExecutorFactory<{factory_desc["kernel_desc"]}>* ')
            f.write(f'{factory_desc["func"]}();\n')
            f.write(f'#endif // {min_instruction_sets[factory_desc["reg_width_bits"]]}\n')

        f.write('\n')

        for kernel_descs in kernel_descs:
            f.write(f'struct ExecutorFactory{kernel_descs} : public ExecutorFactory<{kernel_descs}> {{\n')
            f.write(f'ExecutorFactory{kernel_descs}(){{\n')

            for factory_desc in factory_descs:
                if factory_desc["kernel_desc"] != kernel_descs: continue
                #f.write(f'  ExecutorFactory<{factory_desc["kernel_desc"]}>::')
                f.write(f'#if defined({min_instruction_sets[factory_desc["reg_width_bits"]]})')
                f.write(f' && defined(ENABLE_{factory_desc["arch"]})\n')
                f.write(f'  register_factory("{factory_desc["id"]}", {factory_desc["func"]}());\n')
                f.write(f'#endif // {min_instruction_sets[factory_desc["reg_width_bits"]]}\n')

            f.write(f'}}\n')
            f.write(f'}};\n\n')
            f.write(f'ExecutorFactory{kernel_descs} trip_registration_for_{kernel_descs};\n\n')
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

        for factory_desc in factory_descs:
            f.write(f'#if defined({min_instruction_sets[factory_desc["reg_width_bits"]]})')
            f.write(f' && defined(ENABLE_{factory_desc["arch"]})\n')
            f.write(f'extern MicroKernelPackerFactory<{factory_desc["scalar"]}>* ')
            f.write(f'{factory_desc["func"]}();\n')
            f.write(f'#endif // {min_instruction_sets[factory_desc["reg_width_bits"]]}\n')

        f.write('\n')

        for scalar in scalars:
            f.write(f'struct MicroKernelPackerFactory{scalar.capitalize()}: '
                    f'public MicroKernelPackerFactory<{scalar}> {{\n')
            f.write(f'MicroKernelPackerFactory{scalar.capitalize()}(): '
                    f'MicroKernelPackerFactory<{factory_desc["scalar"]}>({factory_desc["M_r"]}) {{\n')

            for factory_desc in factory_descs:
                f.write(f'#if defined({min_instruction_sets[factory_desc["reg_width_bits"]]})')
                f.write(f' && defined(ENABLE_{factory_desc["arch"]})\n')
                f.write(f'  register_factory("{factory_desc["id"]}", {factory_desc["func"]}());\n')
                f.write(f'#endif // {min_instruction_sets[factory_desc["reg_width_bits"]]}\n')

            f.write(f'}}\n')
            f.write(f'}};\n\n')
            f.write(f'MicroKernelPackerFactory{scalar.capitalize()} trip_registration_for_{scalar.capitalize()};\n\n')

        f.write('}\n')


def generate_ukernel_registration(output_root):
    generate_ukernel_executor_registration(output_root)
    generate_ukernel_packer_registration(output_root)
