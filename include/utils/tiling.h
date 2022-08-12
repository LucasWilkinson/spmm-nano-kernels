//
// Created by lwilkinson on 7/27/22.
//

#pragma once

#include <math.h>

template<int _M_r, int _N_r>
struct RegTiles {
    static const int M_r = _M_r;
    static const int N_r = _N_r;
};

struct CacheTiles {
    int M_c;
    int N_c;
};

template<int _M_r, int _N_r>
struct TileDims {
  int M, K, N;
  int M_c, K_c, N_c;
  int Mb, Kb, Nb;
  int pMb;
  int p;

  bool M_pad, K_pad, N_pad;
  int  M_padded, K_padded, N_padded;

  // Compile Time Constants
  static constexpr int M_r = _M_r;
  static constexpr int N_r = _N_r;

  TileDims(int M, int K, int N,
           int M_c, int K_c, int N_c,
           int p) :
        M(M), K(K), N(N),
        M_c(M_c), N_c(N_c), K_c(K_c) {

    Mb = std::ceil(M / double(M_c));
    Kb = std::ceil(K / double(K_c));
    Nb = std::ceil(N / double(N_c));

    M_pad = (M % M_c);
    K_pad = (K % K_c);
    N_pad = (N % N_c);

    M_padded = Mb * M_c;
    K_padded = Kb * K_c;
    N_padded = Nb * N_c;

    pMb = std::ceil(Mb / p);
  }
};

#define ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td) \
  int M_c = td.M_c, K_c = td.K_c, N_c = td.N_c;                               \
  static constexpr int M_r = TileDims::M_r;                                   \
  static constexpr int N_r = TileDims::N_r;                                   \
  int Mb = td.Mb, Kb = td.Kb, Nb = td.Nb;                                     \
  int pMb = td.pMb;                                                           \
  bool M_pad = td.M_pad, K_pad = td.K_pad, N_pad = td.N_pad;                  \
  int  M_padded = td.M_padded, K_padded = td.K_padded, N_padded = td.N_padded;


#define ALIAS_TILE_DIMS(TileDims, td)                                         \
  int M = td.M, K = td.K, N = td.N;                                           \
  ALIAS_TILE_DIMS_EXCLUDING_MKN(TileDims, td)
