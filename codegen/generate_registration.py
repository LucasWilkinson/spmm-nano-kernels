import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import glob
import json

def generate_ukernel_registration(output_root):
    factory_files = glob.glob(f'{output_root}/factories/**/*.cpp', recursive=True)
    factory_descs = []

    for factory_file in factory_files:
        with open(factory_file, 'r') as f:
            for line in f.readlines():
                if "factory_desc" in line:
                    factory_descs.append(json.loads(line.split('|')[-1]))

    kernel_descs = set()
    for factory_desc in factory_descs:
        kernel_descs.add(factory_desc["kernel_desc"])

    with open(f'{output_root}/ukernel_registration.cpp', 'w') as f:
        f.write('#include "KernelDesc.h"\n')
        f.write('#include "ExecutorFactory.h"\n\n')

        for factory_desc in factory_descs:
            f.write(f'extern ExecutorFactory<{factory_desc["kernel_desc"]}> ')
            f.write(f'{factory_desc["func"]}();\n')

        f.write('\n')




        for kernel_descs in kernel_descs:
            f.write(f'struct ExecutorFactory{kernel_descs} : public ExecutorFactory<{kernel_descs}> {{\n')
            f.write(f'ExecutorFactory{kernel_descs}(){{\n')

            for factory_desc in factory_descs:
                if factory_desc["kernel_desc"] != kernel_descs: continue
                #f.write(f'  ExecutorFactory<{factory_desc["kernel_desc"]}>::')
                f.write(f'  register_factory("{factory_desc["id"]}", {factory_desc["func"]}());\n')

            f.write(f'}}\n')
            f.write(f'}};\n\n')
            f.write(f'ExecutorFactory{kernel_descs} trip_registration_for_{kernel_descs};\n\n')