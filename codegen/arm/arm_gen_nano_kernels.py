from arm_inline_asm import *



def gen_nano_kernel_asm(Mr, Nr, patterns):
    def popcount(x):
        return bin(x).count("1")


    def reg_tile_alloc(m, n):
        reg_tile = []
        for i in range(m):
            reg_tile.append([])
            for j in range(n):
                reg_tile[i].append(FloatingPointOrVectorRegister())
        return reg_tile


    def reg_vec_alloc(n):
        reg_vec = []
        for _ in range(n):
            reg_vec.append(FloatingPointOrVectorRegister())
        return reg_vec


    args = [
        Argument("C_in", "float*", "p"),  Argument("C_in_stride", "int", "r"),
        Argument("C_out", "float*", "p"), Argument("C_out_stride", "int", "r"),
        Argument("B", "float*", "p"),     Argument("B_stride", "int", "r"),
        Argument("nkern_counts", "int*", "p"),
        Argument("col_inds", "int*", "p"),
        Argument("values", "float*", "p"),
        Argument("load_c", "bool", "r"),
        Argument("final_store", "bool", "r"),
        Argument("bias", "float*", "p"),
        Argument("minmax", "bool", "p"),
        Argument("min", "float*", "p"),
        Argument("max", "float*", "p"),
    ]


    with InlineASMFunction("test", args) as func:
        ##
        #   Parse arguments
        ##

        C_in = func.input_registers_dict["C_in"]
        C_out = func.input_registers_dict["C_out"]

        C_in_inc = func.input_registers_dict["C_in_stride"]
        C_out_inc = func.input_registers_dict["C_out_stride"]

        B_base = func.input_registers_dict["B"]
        B_stride = func.input_registers_dict["B_stride"]

        nkern_counts_ptr = func.input_registers_dict["nkern_counts"]
        col_inds_ptr = func.input_registers_dict["col_inds"]
        values_ptr = func.input_registers_dict["values"]

        bias_ptr = func.input_registers_dict["bias"]

        minmax = func.input_registers_dict["minmax"]
        max_ptr = func.input_registers_dict["max"]
        min_ptr = func.input_registers_dict["min"]

        final_store = func.input_registers_dict["final_store"]
        load_c = func.input_registers_dict["load_c"]

        nkern_count = GeneralPurposeRegister(default_width="w")         # 32 bit, since nkern_counts is int
        nkern_count_next = GeneralPurposeRegister(default_width="w")    # 32 bit, since nkern_counts is int


        ##
        # setup ctile
        ##
        ctile = reg_tile_alloc(Mr, Nr)
        C_temp = GeneralPurposeRegister()
        Nr_bytes = Nr * ctile[0][0].bytes()

        ##
        #  Load first nkern count
        ##
        LDR(nkern_count, address(nkern_counts_ptr, mode="post", offset=4))

        ##
        #  Precompute strides
        ##
        LSL(C_in_inc, C_in_inc, 2)              # C_out_inc *= 4 (i.e. size of float)
        SUB(C_in_inc, C_in_inc, Nr_bytes)       # C_in_inc -= Nr_bytes
        LSL(C_out_inc, C_out_inc, 2)            # C_out_inc *= 4 (i.e. size of float)
        SUB(C_out_inc, C_out_inc, Nr_bytes)     # C_out_inc -= Nr_bytes
        LSL(B_stride, B_stride, 2)              # B_stride *= 4 (i.e. size of float)

        ##
        #  Assume a non-empty kernel, so preemptively calculate B_curr
        ##
        B_curr = GeneralPurposeRegister()
        col_ind = GeneralPurposeRegister(default_width="x")
        LDR(col_ind.view_as("w"), address(col_inds_ptr, mode="post", offset=4))
        UMADDL(B_curr, col_ind, B_stride, B_base)
        del col_ind     # Free the register

        zero_c = Label("zero_c")
        bias_load = Label("bias_load")
        nkerns_start = Label("nkerns_start")

        CMP(load_c, 1)
        B(bias_load, "GE")
        del load_c

        ##
        #  Load C
        ##
        MOV(C_temp, C_in)
        for i in range(Mr):
            for j in range(0, Nr, 4):
                LD1(ctile[i][j:j+4], C_temp, inc=True)
            ADD(C_temp, C_temp, C_in_inc)
        func.del_last_instruction()     # Remove last ADD
        del C_temp, C_in_inc    # Free the registers
        B(nkerns_start, "")

        ##
        #   Load bias
        ##
        func.insert_label(bias_load)
        CMP(bias_ptr, 1)
        B(zero_c, "LT")
        for i in range(Mr):
            for j in range(Nr):
                LD1R(ctile[i][j], bias_ptr)
            ADD(bias_ptr, bias_ptr, 4)
        func.del_last_instruction()     # Remove last ADD
        del bias_ptr
        B(nkerns_start, "")

        ##
        #   Zero C
        ##
        func.insert_label(zero_c)
        for i in range(Mr):
            for j in range(Nr):
                MOVI(ctile[i][j], 0)

        ##
        #  Compute nano-kernels
        ##
        patterns = list(range(1, 2**Mr))
        func.insert_label(nkerns_start)
        for nkern_id, nkern_pat in enumerate(patterns):
            nkern_exit = Label(name=f"nkern_{nkern_pat:08b}_exit")
            nkern_enter = Label(name=f"nkern_{nkern_pat:08b}_enter")
            nnz = popcount(nkern_pat)

            A_vecs = int(math.ceil(nnz / 4))
            A_values = reg_vec_alloc(A_vecs)

            # Prempetively load next count for all but the last nkern
            if nkern_id < len(patterns) - 1:
                LDR(nkern_count_next, address(nkern_counts_ptr, mode="post", offset=4))

            # the first nkern count is preloaded
            if nkern_id != 0:
                MOV(nkern_count, nkern_count_next)

            # Check if nano-kernel should be run
            CMP(nkern_count, 0)
            B(nkern_exit, cond="EQ")

            func.insert_label(nkern_enter)

            B_unroll = {6: 6, 4: 4, 3: 3, 2: 2, 1: 1}.get(Nr)

            # Load B + FMA
            for j in range(0, Nr, B_unroll):
                b_vecs = reg_vec_alloc(min(B_unroll, Nr - j))

                for i in range(0, Nr, 4):
                    LD1(b_vecs[i:i+4], B_curr, inc=True)

                if j == 0:
                    # Load A

                    if nnz > 1:
                        for i in range(A_vecs - 1):
                            LD1(A_values[i], values_ptr, inc=True)
                        LD1(A_values[-1], values_ptr)
                    else:
                        LD1R(A_values[0], values_ptr)
                    ADD(values_ptr, values_ptr, int(nnz % 4) * 4)

                    # Preload/Compute for next iteration
                    SUBS(nkern_count, nkern_count, 1)
                    col_ind = GeneralPurposeRegister(default_width="x")
                    LDR(col_ind.view_as("w"), address(col_inds_ptr, mode="post", offset=4))
                    UMADDL(B_curr, col_ind, B_stride, B_base)
                    del col_ind     # Free the register


                val_idx = 0
                row_idx = 0
                nkern_pat_copy = nkern_pat
                while nkern_pat_copy:
                    if nkern_pat_copy & 1:
                        for _j, b_vec in enumerate(b_vecs):
                            FMA(ctile[row_idx][j+_j], b_vec, A_values[val_idx // 4],
                                lane=val_idx % 4 if nnz > 1 else None)
                        val_idx += 1
                    nkern_pat_copy >>= 1
                    row_idx += 1

                for b_vec in b_vecs: b_vec.dealloc()
                del b_vecs

            for a_vec in A_values: a_vec.dealloc()
            del A_values

            # Loop if required
            B(nkern_enter, cond="GT")
            func.insert_label(nkern_exit)

        store_c = Label("store_c")
        exit = Label("exit")

        CMP(final_store, 1)
        CCMP(minmax, 1, 0, cond="EQ")
        B(store_c, "NE")                # if (minmax && final_store), store C

        ##
        #  Minmax + Store C
        ##

        min_vec = FloatingPointOrVectorRegister()
        max_vec = FloatingPointOrVectorRegister()
        LD1R(max_vec, max_ptr)
        LD1R(min_vec, min_ptr)

        for i in range(Mr):
            for j in range(Nr):
                FMAX(ctile[i][j], min_vec, ctile[i][j])
            for j in range(Nr):
                FMIN(ctile[i][j], max_vec, ctile[i][j])
            for jj in range(0, Nr, 4):
                ST1(ctile[i][jj:jj+4], C_out, inc=True)
            ADD(C_out, C_out, C_out_inc)
        func.del_last_instruction()     # Remove last ADD
        del min_vec, max_vec
        B(exit, "")

        ##
        #  Store C
        ##
        func.insert_label(store_c)
        for i in range(Mr):
            for j in range(0, Nr, 4):
                ST1(ctile[i][j:j+4], C_out, inc=True)
            ADD(C_out, C_out, C_out_inc)
        func.del_last_instruction()     # Remove last ADD
        del C_out      # Free the register

        func.insert_label(exit)
        del C_out_inc   # Free the register

    return func.emit()

if __name__ == "__main__":
    with open('main.cpp', 'w+') as f:
        f.write('''
    #include <iostream>
    
    FUNC
        
    int main() {
      int M = 4;
      int N = 32;
    
      int Mr = 4;
      int Nr = 2*4;
    
      int nkern_counts[] = {2, 1};
      int col_inds[] = {0, 1, 1};
    
    
      float* C = new float[M*N];
      float* C_orig = new float[M*N];
      float* B = new float[M*N];
      float* values = new float[M*N];
      float* bias = new float[M*N];
    
      float min = 0.0;
      float max = 10.0;
    
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          C[i*N + j] = i + j;
          C_orig[i*N + j] = C[i*N + j];
          B[i*N + j] = i + j;
          values[i*N + j] = i + j;
          bias[i*N + j] = -0.5;
        }
      }
    
    
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          std::cout << C[i*N + j] << " ";
        }
        std::cout << std::endl;
      }
    
      std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
      test(C, N,
           C, N,
           B, N,
           nkern_counts, col_inds, values,
           true, true, 
           bias,
           true,
           &min, &max);
    
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          std::cout << C[i*N + j] << " ";
        }
        std::cout << std::endl;
      }
    
    
      std::cout << "<<< DIFF <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          std::cout << C[i*N + j] - C_orig[i*N + j] << " ";
        }
        std::cout << std::endl;
      }
    
    
      return 0;
    }    
       
    '''.replace('FUNC', gen_nano_kernel_asm(4, 6, [0b0011, 0b0100]))