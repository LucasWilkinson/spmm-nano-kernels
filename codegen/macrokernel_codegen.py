from os import sched_get_priority_max


def get_inner_mn_loop(sched):
    if sched.endswith('nmKNM') or sched.endswith('nmNKM') or sched.endswith('nmMNK') or sched.endswith('nmNMK') or sched.endswith('nmMKN') or sched.endswith('nmKMN'):
        return f'_inner_nm_loop(tii, jjj, tiles[tii][tkk], partial_Nc_loop, final_store);'
    elif sched.endswith('nmKM') or sched.endswith('nmMK'):
        return f'_inner_nm_loop(tii, 0, tiles[tii][tkk], partial_Nc_loop, final_store);'
    elif sched.endswith('nmKN') or sched.endswith('nmNK'):
        return f'_inner_nm_loop(0, tjj * N_c, tiles[0][tkk], partial_Nc_loop, final_store);'
    elif sched.endswith('nmNM') or sched.endswith('nmMN'):
        return f'_inner_nm_loop(tii, tjj * N_c, tiles[tii][0], partial_Nc_loop, final_store);'
    elif sched.endswith('nmK'):
        return f'_inner_nm_loop(0, 0, tiles[0][tkk], partial_Nc_loop, final_store);'
    elif sched.endswith('nmN'):
        return f'_inner_nm_loop(0, tjj * N_c, tiles[0][0], partial_Nc_loop, final_store);'
    elif sched.endswith('nmM'):
        return f'_inner_nm_loop(tii, 0, tiles[tii][0], partial_Nc_loop, final_store);'

def get_mn_loop_init(sched):
    if sched.endswith('nmKNM'):
        code = f'{{\n'
        code += f'int tii = p_tile;\n'
        code += f'int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;\n'
        code += f'const int iii = tii * M_c;\n'
        code += f'// N_c loop\n'
        code += f'int tjj = 0, jjj = 0;\n'
        code += f'for (; tjj < Nb_full; tjj++, jjj += N_c) {{\n'
        code += f'    for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'        bool final_store = (tkk == Kb - 1);\n'
        code += f'        bool partial_Nc_loop = false;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'if (partial_N_c_loop || partial_N_r_loop) {{\n'
        code += f'    for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'        bool final_store = (tkk == Kb - 1);\n'
        code += f'        bool partial_Nc_loop = true;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code

    elif sched.endswith('nmNKM'):
        code = f'{{\n'
        code += f'int tii = p_tile;\n'
        code += f'int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;\n'
        code += f'const int iii = tii * M_c;\n'
        code += f'// K_c loop\n'
        code += f'for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'    bool final_store = (tkk == Kb - 1);\n'
        code += f'    int tjj = 0, jjj = 0;\n'
        code += f'    for (; tjj < Nb_full; tjj++, jjj += N_c) {{\n'
        code += f'        bool partial_Nc_loop = false;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'    if (partial_N_c_loop || partial_N_r_loop) {{\n'
        code += f'        bool partial_Nc_loop = true;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmMNK'):
        code = f'{{\n'
        code += f'int tkk = p_tile;\n'
        code += f'bool final_store = (tkk == Kb - 1);\n'
        code += f'int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;\n'
        code += f'const int kkk = tkk * K_c;\n'
        code += f'// N_c loop\n'
        code += f'int tjj = 0, jjj = 0;\n'
        code += f'for (; tjj < Nb_full; tjj++, jjj += N_c) {{\n'
        code += f'    for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'        bool partial_Nc_loop = false;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'if (partial_N_c_loop || partial_N_r_loop) {{\n'
        code += f'    for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'        bool partial_Nc_loop = true;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmNMK'):
        code = f'{{\n'
        code += f'int tkk = p_tile;\n'
        code += f'bool final_store = (tkk == Kb - 1);\n'
        code += f'int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;\n'
        code += f'const int kkk = tkk * K_c;\n'
        code += f'// M_c loop\n'
        code += f'for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'    int tjj = 0, jjj = 0;\n'
        code += f'    for (; tjj < Nb_full; tjj++, jjj += N_c) {{\n'
        code += f'        bool partial_Nc_loop = false;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'    if (partial_N_c_loop || partial_N_r_loop) {{\n'
        code += f'        bool partial_Nc_loop = true;\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmKMN'):
        code = f'{{\n'
        code += f'int tjj = p_tile;\n'
        code += f'bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'const int jjj = tjj * N_c;\n'
        code += f'// M_c loop\n'
        code += f'for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'    for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'        bool final_store = (tkk == Kb - 1);\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmMKN'):
        code = f'{{\n'
        code += f'int tjj = p_tile;\n'
        code += f'bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'const int jjj = tjj * N_c;\n'
        code += f'// K_c loop\n'
        code += f'for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'    bool final_store = (tkk == Kb - 1);\n'
        code += f'    for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'        {get_inner_mn_loop(sched=sched)}\n'
        code += f'    }}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code

    elif sched.endswith('nmKM'):
        code = f'{{\n'
        code += f'int tii = p_tile;\n'
        code += f'// K_c loop\n'
        code += f'for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'    bool final_store = (tkk == Kb - 1);\n'
        code += f'    bool partial_Nc_loop = partial_N_c_loop || partial_N_r_loop; // since Nc == N\n'
        code += f'    {get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code

    elif sched.endswith('nmMK'):
        code = f'{{\n'
        code += f'int tkk = p_tile;\n'
        code += f'bool final_store = (tkk == Kb - 1);\n'
        code += f'// M_c loop\n'
        code += f'for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'    bool partial_Nc_loop = partial_N_c_loop || partial_N_r_loop; // since Nc == N\n'
        code += f'    {get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmKN'):
        code = f'{{\n'
        code += f'int tjj = p_tile;\n'
        code += f'bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'for (int tkk = 0; tkk < Kb; tkk++) {{\n'
        code += f'    bool final_store = (tkk == Kb - 1);\n'
        code += f'    {get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code

    elif sched.endswith('nmNK'):
        code = f'{{\n'
        code += f'int tkk = p_tile;\n'
        code += f'bool final_store = (tkk == Kb - 1);\n'
        code += f'for (int tjj = 0; tjj < Nb; tjj++) {{\n'
        code += f'    bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'    {get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmMN'):
        code = f'{{\n'
        code += f'int tjj = p_tile;\n'
        code += f'bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'bool final_store = true;\n'
        code += f'for (int tii = 0; tii < Mb; tii++) {{\n'
        code += f'    {get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmNM'):
        code = f'{{\n'
        code += f'int tii = p_tile;\n'
        code += f'bool final_store = true;\n'
        code += f'for (int tjj = 0; tjj < Nb; tjj++) {{\n'
        code += f'    bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'    {get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmK'):
        code = f'{{\n'
        code += f'int tkk = p_tile;\n'
        code += f'bool final_store = (tkk == Kb - 1);\n'
        code += f'bool partial_Nc_loop = partial_N_c_loop || partial_N_r_loop;\n'
        code += f'{get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmN'):
        code = f'{{\n'
        code += f'int tjj = p_tile;\n'
        code += f'bool partial_Nc_loop = (partial_N_c_loop || partial_N_r_loop) && (tjj == Nb - 1);\n'
        code += f'bool final_store = true;\n'
        code += f'{get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        return code
    
    elif sched.endswith('nmM'):
        code = f'{{\n'
        code += f'int tii = p_tile;\n'
        code += f'bool final_store = true;\n'
        code += f'bool partial_Nc_loop = partial_N_c_loop || partial_N_r_loop;\n'
        code += f'{get_inner_mn_loop(sched=sched)}\n'
        code += f'}}\n'
        return code

    else:
        raise ValueError(f'Unknown schedule: {sched}')


def inner_mn_loop(partial_N_c_loop):
    code = ''
    code += f'                    # /* nmKN */'
    code += f'                    bool partial_N_c_loop = {partial_N_c_loop};'
    code += f'                    bool final_store = (tkk == Kb - 1);'
    code += f'                        Scalar* __restrict__ C_p = nullptr;'
    code += f'                        const Scalar* __restrict__ B_p = nullptr;'
    code += f'                        int tii = 0;'
    code += f'                        int jjj = tjj * N_c;'
    code += f'                        const PackedTile& pt = tiles[tii][tkk];'
    code += f'                    int _c_N_r = (partial_N_c_loop) ? final_N_c_loop_N_r_count : c_N_r;'
    code += f'                    // M_r loop'
    code += f'                    for (int pi = 0; pi < pt.sop.num_panels; pi++) {{'
    code += f'                        int tj = 0, jj = jjj;'
    code += f'                        const auto& panel_desc = pt.sop.panel_descs[pi];'
    code += f'                        uint32_t* __restrict__  col_inds = (uint32_t*) panel_desc.col_indices;'
    code += f'                        float* __restrict__     values = panel_desc.values;'
    code += f'                        int* __restrict__       nkern_counts = panel_desc.nkern_counts;'
    code += f'                        int global_upanel_id = tii * c_M_r + pi;'
    code += f'                        if constexpr(UPanelOrder != NO_REORDERING && !packed_C)'
    code += f'                            global_upanel_id = upanel_swizzle[global_upanel_id];'
    code += f'                        int ii = (global_upanel_id * M_r); // Row start of ukernel'
    code += f'                        for (; tj < _c_N_r; tj++, jj += N_r) {{ // N_r loop'
    code += f'                            C_p = C + jj + ii * N;'
    code += f'                            B_p = B + jj;'
    code += f'                            ukernel.vectorized('
    code += f'                                C_p, N,'
    code += f'                                B_p, N,'
    code += f'                                nkern_counts, col_inds, values,'
    code += f'                                pt.load_c, final_store,'
    code += f'                                bias ? bias + ii : nullptr'
    code += f'                            );'
    code += f'                        }}'
    code += f'                        if (partial_N_c_loop && partial_N_r_loop) {{'
    code += f'                            C_p = C + jj + ii * N;'
    code += f'                            B_p = B + jj;'
    code += f'                            ukernel.cleanup('
    code += f'                                final_N_r_loop_rem,'
    code += f'                                C_p, N,'
    code += f'                                B_p, N,'
    code += f'                                nkern_counts, col_inds, values,'
    code += f'                                pt.load_c, final_store,'
    code += f'                                bias ? bias + ii : nullptr'
    code += f'                            );'
    code += f'                        }}'
    code += f'                    }}'
    return code

def _execute_row_panel_KN(p_tile):
    code = ''
    code += f'        {{'
    code += f'            int tii = p_tile;'
    code += f'            using std::min;'
    code += f''
    code += f'            ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);'
    code += f'            int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;'
    code += f'            const int iii = tii * M_c;'
    code += f'            // K_c loop'
    code += f'            int tjj = 0, jjj = 0;'
    code += f'            for (; tjj < Nb_full; tjj++, jjj += N_c) {{'
    code += f'                for (int tkk = 0; tkk < Kb; tkk++) {{'
    code += f'                    bool final_store = (tkk == Kb - 1);'
    code += f'                    _inner_nm_loop(tii, jjj, tiles[tii][tkk], false, final_store);'
    code += f'                }}'
    code += f'            }}'
    code += f'            if (partial_N_c_loop || partial_N_r_loop) {{'
    code += f'                for (int tkk = 0; tkk < Kb; tkk++) {{'
    code += f'                    bool final_store = (tkk == Kb - 1);'
    code += f'                    _inner_nm_loop(tii, jjj, tiles[tii][tkk], true, final_store);'
    code += f'                }}'
    code += f'            }}'
    code += f'        }}'

def _execute_row_panel_NK(p_tile):
    code = ''
    code += f'        {{'
    code += f'            int tii = p_tile;'
    code += f'            using std::min;'
    code += f'            ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td);'
    code += f'            int Nb_full = partial_N_c_loop || partial_N_r_loop ? Nb - 1 : Nb;'
    code += f'            const int iii = tii * M_c;'
    code += f'            // K_c loop'
    code += f'            for (int tkk = 0; tkk < Kb; tkk++) {{'
    code += f'                bool final_store = (tkk == Kb - 1);'
    code += f'                int tjj = 0, jjj = 0;'
    code += f'                for (; tjj < Nb_full; tjj++, jjj += N_c) {{'
    code += f'                    _inner_nm_loop(tii, jjj, tiles[tii][tkk], false, final_store);'
    code += f'                }}'
    code += f'                if (partial_N_c_loop || partial_N_r_loop) {{'
    code += f'                    _inner_nm_loop(tii, jjj, tiles[tii][tkk], true, final_store);'
    code += f'                }}'
    code += f'            }}'
    code += f'        }}'

def spmm():
    code = generate_macrokernel(inner_mn_loop, )

class Node:
    def __init__(self, code=""):
        self.code = code

    def add_loop(self, identifier, dimension, tile_size=0, parallel_mode=None):
        result = ''
        if parallel_mode != None:
            result += f'#pragma omp parallel for schedule({parallel_mode})\n'
        forVar = 't'
        if identifier == 'M':
            forVar = 'ti'
        elif identifier == 'N':
            forVar = 'tj'
        elif identifier == 'K':
            forVar = 'tk'
        elif identifier == 'm':
            forVar = 'tii'
        elif identifier == 'n':
            forVar = 'tjj'
        elif identifier == 'k':
            forVar = 'tkk'
        result += f'for (int {forVar} = 0; {forVar} < {dimension}; {identifier}+={tile_size}) {{\n'
        result += self.code
        result += f'}}\n'
        self.code = result
        return self

def generate_macrokernel(microkernel, loopSizes, loop_order,
    tile_sizes, parallel_modes):
    
    if loopSizes == None:
        loopSizes = loop_order
    macrokernel = Node(microkernel())
    for identifier, dimension, tile_size, parallel_mode in zip(loop_order, loopSizes, tile_sizes, parallel_modes):
        macrokernel.add_loop(identifier, dimension, tile_size, parallel_mode)
    return macrokernel.code

def MK():
    return '_inner_nm_loop(ti, tj, pt, partial_Nc_loop, final_store);\n'

for sched in ['nmKNM', 'nmKMN', 'nmNKM', 'nmNMK', 'nmMKN', 'nmMNK', 'nmMK', 'nmKM', 'nmNM', 'nmMN', 'nmKN', 'nmNK', 'nmK', 'nmN', 'nmM']:
    print(f'For {sched}, the code is:')
    print(get_mn_loop_init(sched))
    print('\n\n')

# print(generate_macrokernel(microkernel=MK, loopSizes=None, loop_order='MNK', tile_sizes=[2, 2, 2], parallel_modes=[None, 'static', 'dynamic']))

# print(generate_macrokernel(microkernel=MK, loopSizes=None, loop_order='mnMNK', tile_sizes=[2, 2, 16, 16, 16], parallel_modes=[None, None, None, None, 'auto']))

# for order in ['MNK', 'MKN', 'NKM', 'NMK', 'KMN', 'KNM']:
#     for tile_size1 in 1, 4, 16, 64, 256:
#         for tile_size2 in 1, 4, 16, 64, 256:
#             for tile_size3 in 1, 4, 16, 64, 256:
#                 for parallel_modes in [['auto', None, None], [None, 'auto', None], [None, None, 'auto']]:
#                     print('\n\n')
#                     print(generate_macrokernel(microkernel=MK, loopSizes=None, loop_order=order, tile_sizes=[tile_size1, tile_size2, tile_size3], parallel_modes=parallel_modes))

# for order in ['mnMNK', 'nmMNK', 'mnMKN', 'nmMKN', 'mnNMK', 'nmNMK', 'mnKNM', 'nmKNM', 'mnKMN', 'nmKMN', 'mnNKM', 'nmNKM']:
#     for tile_size1 in 1, 4, 16, 64, 256:
#         for tile_size2 in 1, 4, 16, 64, 256:
#             for tile_size3 in 1, 4, 16, 64, 256:
#                 for tile_size4 in 1, 4, 16, 64, 256:
#                     for tile_size5 in 1, 4, 16, 64, 256:
#                         for parallel_modes in [['auto', None, None, None, None], [None, 'auto', None, None, None], [None, None, 'auto', None, None],\
#                         [None, None, None, None, 'auto']]:
#                             print('\n\n')
#                             print(generate_macrokernel(microkernel=MK, loopSizes=None, loop_order=order, tile_sizes=[tile_size1, tile_size2, tile_size3, tile_size4, tile_size5], parallel_modes=parallel_modes))