//
// Created by lwilkinson on 8/1/22.
//    From: https://github.com/vnatesh/CAKE_on_CPU
//

#include "cake_block_dims.h"
#include "utils/bmath.h"

#include <iostream>

int get_cache_size(int level);
int get_num_physical_cores();
int lcm(int n1, int n2);
enum sched derive_schedule(int M, int N, int K, int p,
                           int mc_ret, cake_cntx_t* cake_cntx);


    cake_cntx_t* cake_query_cntx_torch(int L2, int L3) {

  cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
  double alpha_n = 1.0;
  ret->alpha_n = alpha_n;

  // query block size for the microkernel
#ifdef USE_BLIS
  cntx_t* blis_cntx = bli_gks_query_cntx();
  ret->blis_cntx = (void*) blis_cntx;
  ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
  ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);

#elif USE_CAKE_HASWELL
  ret->blis_cntx = NULL;
  ret->mr = 6;
  ret->nr = 16;

#elif USE_CAKE_ARMV8
  ret->blis_cntx = NULL;
  ret->mr = 8;
  ret->nr = 12;
#endif

  ret->L2 = L2;
  ret->L3 = L3;
  return ret;
}


cake_cntx_t* cake_query_cntx() {

  cake_cntx_t* ret = (cake_cntx_t*) malloc(sizeof(cake_cntx_t));
  double alpha_n = 1.0;
  ret->alpha_n = alpha_n;
  ret->L2 = get_cache_size(2);
  ret->L3 = get_cache_size(3);
  ret->ncores = get_num_physical_cores();
  ret->peak_dram_bw = 32 * 1e9; // TODO : hardcoded bw and flops on i9 for now
  ret->peak_flops = 600 * 1e9;

  // query block size for the microkernel
#ifdef USE_BLIS
  cntx_t* blis_cntx = bli_gks_query_cntx();
  ret->blis_cntx = (void*) blis_cntx;
  ret->mr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, blis_cntx);
  ret->nr = (int) bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, blis_cntx);

#elif USE_CAKE_HASWELL
  ret->blis_cntx = NULL;
  ret->mr = 6;
  ret->nr = 16;
  ret->m_map = (ret->mr/2) - 1;
  ret->n_map = (ret->nr/16) - 1;

#elif USE_CAKE_ARMV8
  ret->blis_cntx = NULL;
  ret->mr = 8;
  ret->nr = 12;
  ret->m_map = (ret->mr/2) - 1;
  ret->n_map = (ret->nr/12) - 1;
#endif

  return ret;
}


void update_mr_nr(cake_cntx_t* cake_cntx, int m_r, int n_r) {
  cake_cntx->mr = m_r;
  cake_cntx->nr = n_r;
#ifdef USE_CAKE_HASWELL
  cake_cntx->m_map = (m_r/2) - 1;
  cake_cntx->n_map = (n_r/16) - 1;
#elif USE_CAKE_ARMV8
  cake_cntx->m_map = (m_r/2) - 1;
  cake_cntx->n_map = (n_r/12) - 1;
#endif
}


int get_num_physical_cores() {

  FILE *fp;
  char ret1[16];
  char command1[128];

  sprintf(command1, "grep -c ^processor /proc/cpuinfo");
  fp = popen(command1, "r");

  if (fp == NULL) {
    printf("Failed to run proc/cpuinfo command\n" );
    exit(1);
  }

  if(fgets(ret1, sizeof(ret1), fp) == NULL) {
    printf("cpuinfo error\n");
  }


  char ret2[16];
  char command2[128];

  sprintf(command2, "lscpu | grep Thread -m 1 | tr -dc '0-9'");
  fp = popen(command2, "r");

  if (fp == NULL) {
    printf("Failed to run lscpu1 command\n" );
    exit(1);
  }

  if(fgets(ret2, sizeof(ret2), fp) == NULL) {
    printf("lscpu error\n");
  }


  pclose(fp);
  return atoi(ret1) / atoi(ret2);
}



// find cache size at levels L1d,L1i,L2,and L3 using lscpu
int get_cache_size(int level) {

  int model_id, len, size = 0;
  FILE *fp;
  char ret[16];
  char command[128];

  sprintf(command, "lscpu | grep Model \
                          | head -1 \
                          | tr -dc '0-9'");
  fp = popen(command, "r");

  if (fp == NULL) {
    printf("Failed to run lscpu2 command\n" );
    exit(1);
  }

  if(fgets(ret, sizeof(ret), fp) == NULL) {
    printf("lscpu error\n");
  }

  pclose(fp);
  model_id = atoi(ret);

  if(level == 1) {
    switch(model_id) {
      case 79:
      case 85:
        return (32 * (1 << 10));
      default:
        break;
    }
  }

  if(level == 2) {
    switch(model_id) {
      case 1:
        return (512 * (1 << 10));
      case 3:
        return (32 * (1 << 10));
      case 49:
        return (512 * (1 << 10));
      case 69:
      case 79:
        return (256 * (1 << 10));
      case 85:
        return (1024 * (1 << 10));
      case 142:
        return (256 * (1 << 10));
      case 165:
        return (256 * (1 << 10));
      default:
        break;
    }
  }

  if(level == 3) {
    switch(model_id) {
      case 1:
        return (64 * (1 << 20));
      case 3:
        return (1 * (1 << 20));
      case 49:
        return (128 * (1 << 20));
      case 69:
      case 79:
        return (4 * (1 << 20));
      case 85:
        return (14080 * (1 << 10));
      case 142:
        return (8 * (1 << 20));
      case 165:
        return (20 * (1 << 20));
      default:
        break;
    }
  }


  if(level < 3) {
    sprintf(command, "lscpu --caches=NAME,ONE-SIZE"
                     "  | grep L%d"
                     "  | grep -Eo '[\\.0-9]*M|[\\.0-9]*K|0-9*G'"
                     "  | tr -d '\n'", level);
    fp = popen(command, "r");
  } else {
    sprintf(command, "lscpu --caches=NAME,ALL-SIZE"
                     "  | grep L%d"
                     "  | grep -Eo '[\\.0-9]*M|[\\.0-9]*K|0-9*G'"
                     "  | tr -d '\n'", level);
    fp = popen(command, "r");
  }

  if (fp == NULL) {
    printf("Failed to run lscpu3 command\n" );
    exit(1);
  }

  if(fgets(ret, sizeof(ret), fp) == NULL) {
    printf("lscpu error\n");
    // quick hack for raspberry pi 3 cache sizes (32 KiB L1, 512 KiB L2 shared)
    if(level == 2) {
      return (32 * (1 << 10));
    } else if(level == 3) {
      return (512 * (1 << 10));
    }
  }

  len = strlen(ret) - 1;

  // set cache size variables
  if(ret[len] == 'K') {
    ret[len] = '\0';
    size = atof(ret) * (1 << 10);
  } else if(ret[len] == 'M') {
    ret[len] = '\0';
    size = atof(ret) * (1 << 20);
  } else if(ret[len] == 'G') {
    ret[len] = '\0';
    size = atof(ret) * (1 << 30);
  }


  std::cout << ret << std::endl;
  return size;
}


cache_dims_t* get_cache_dims(int M, int N, int K, int p,
                             cake_cntx_t* cake_cntx, enum sched sch,
                             char* argv[], float density) {

  int mc, mc_ret, nc_ret, a, mc_L2 = 0, mc_L3 = 0;
  int max_threads = cake_cntx->ncores; // 2-way hyperthreaded
  int mn_lcm = lcm(cake_cntx->mr, cake_cntx->nr);
  // int mn_lcm = m_r;

  // solve for optimal mc,kc based on L2 size
  // L2_size >= 2*(mc*kc + kc*nr) + 2*(mc*nr)     (solve for x = m_c = k_c)
  int b = 2*cake_cntx->nr;
  mc_L2 = (int)  ((-b + sqrt(b*b + 4*(((double) cake_cntx->L2) / (2*sizeof(float))))) / 2.0) ;
  // mc_L2 -= (mc_L2 % mn_lcm);
  mc_L2 -= (mc_L2 % cake_cntx->mr);
  // printf("mc_L2 = %d\n", mc_L2);


  // solve for the optimal block size m_c and k_c based on the L3 size
  // L3_size >= 2*(p*mc*kc + alpha*p*mc*kc) + 2*(p*mc*alpha*p*mc)     (solve for x = m_c = k_c)
  // We only use ~ half of the each cache to prevent our working blocks from being evicted
  // and to allow for double buffering of partial results in L3
  mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (2*sizeof(float)))
                    / (max_threads * (1 + 1.0 + 1.0*max_threads)));
  mc_L3 -= (mc_L3 % cake_cntx->mr);
  // printf("mc_L3 = %d\n", mc_L3);

  // if mc_L3 is too small, reduce alpha. likewise if mc_L2 is too small, increase alpha
  // This will reduce/increase L3 tile size and utilize mor/less DRAM bandwidth
  cake_cntx->alpha_n = ((double) mc_L3) / mc_L2;
  mc =  mc_L2;


  mc_ret = mc;
  if(M < p*cake_cntx->mr) {
    mc_ret = cake_cntx->mr;
  } else if(M < p*mc) {

    a = (M / p);
    if(a < cake_cntx->mr) {
      mc_ret = cake_cntx->mr;
    } else {
      a += (cake_cntx->mr - (a % cake_cntx->mr));
      mc_ret = a;
    }
  }

  cache_dims_t* blk_ret = (cache_dims_t*) malloc(sizeof(cache_dims_t));

  // set schedule to MEMA-derived optimal value or user-defined
  blk_ret->sch = (sch == NA ?
                            derive_schedule(M, N, K, p, mc_ret, cake_cntx) :
                            sch);


  // user-defined tile sizes
  int ss = 0;
  if(argv) {
    ss = atoi(argv[5]);
  }

  if(ss) {
    printf("user-defined tile sizes\n");
    blk_ret->m_c = atoi(argv[6]);
    blk_ret->k_c = atoi(argv[7]);
    blk_ret->n_c = atoi(argv[8]);
    // sparsity-aware tiling when A matrix is sparse
  } else if(density > 0.0000001) {

    printf("sparsity-aware tiling\n");
    double a_coeff = (density/cake_cntx->mr) * ((int) ceil(density * cake_cntx->mr)) ;
    printf("a_coeff %f (%f, %d)\n", a_coeff, density/cake_cntx->mr, (int) ceil(density * cake_cntx->mr));

    mc_L2 = (int)  ((-b + sqrt(b*b + 4*a_coeff*(((double) cake_cntx->L2) / (sizeof(float))))) / (2.0*a_coeff)) ;
    mc_L2 -= (mc_L2 % cake_cntx->mr);

    mc_L3 = (int) sqrt((((double) cake_cntx->L3) / (sizeof(float)))
                      / (max_threads * (a_coeff + cake_cntx->alpha_n + cake_cntx->alpha_n*max_threads)));
    mc_L3 -= (mc_L3 % cake_cntx->mr);


    mc_ret = mc_L3;
    if(M < p*cake_cntx->mr) {
      mc_ret = cake_cntx->mr;
    } else if(M < p*mc) {

      a = (M / p);
      if(a < cake_cntx->mr) {
        mc_ret = cake_cntx->mr;
      } else {
        a += (cake_cntx->mr - (a % cake_cntx->mr));
        mc_ret = a;
      }
    }

    // spMM is always K-first so using nc_ret from KMN
    nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
    nc_ret -= (nc_ret % cake_cntx->nr);
    nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;

    blk_ret->m_c = mc_L3 < M ? mc_L3 : cake_cntx->mr;
    blk_ret->k_c = mc_L2 < K ? mc_L2 : K;
    blk_ret->n_c = nc_ret;

    // CAKE tiling for dense MM
  } else {

    switch(blk_ret->sch) {

      case KMN: {
        nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
        nc_ret -= (nc_ret % cake_cntx->nr);
        nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;
        break;
      }

      case MKN: {
        nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
        nc_ret -= (nc_ret % cake_cntx->nr);
        nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;
        break;
      }

      case NKM: {
        nc_ret = (int) mc_ret;
        nc_ret -= (nc_ret % cake_cntx->nr);
        nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;

        mc_ret = (int) (cake_cntx->alpha_n*mc_ret);
        mc_ret -= (mc_ret % cake_cntx->mr);
        mc_ret = mc_ret == 0 ? cake_cntx->mr : mc_ret;
        break;
      }

      default: {
        printf("unknown schedule\n");
        exit(1);
      }
    }

    blk_ret->m_c = mc_ret;
    blk_ret->k_c = mc_ret;
    blk_ret->n_c = nc_ret;
  }


  return blk_ret;
}



// derive and set schedule according to MEMA analysis
enum sched derive_schedule(int M, int N, int K, int p,
                           int mc_ret, cake_cntx_t* cake_cntx) {

  float m,k,n, K_cut_M, K_cut_N, N_cut_M, M_cut_N;

  m = (float) (p*mc_ret);
  k = (float) (p*mc_ret);
  n = (float) (p*mc_ret);

  K_cut_M = (2.0*M) / (1.0 + (M * ((2.0/k) - (1.0/m))));
  K_cut_N = (2.0*N) / (1.0 + (N * ((2.0/k) - (1.0/n))));

  // N/M dim cutoffs for M vs N choice
  m = (float) (cake_cntx->alpha_n*p*mc_ret);
  n = (float) (cake_cntx->alpha_n*p*mc_ret);
  N_cut_M = M / (1.0 + (M * ((1.0/n) - (1.0/m))));
  M_cut_N = N / (1.0 + (N * ((1.0/m) - (1.0/n))));

  printf("K_cut_M %f K_cut_N %f N_cut_M %f M_cut_N %f\n",K_cut_M,
         K_cut_N,N_cut_M,M_cut_N);
  // IO optimal schedule based on input parameters M,K,N,m,k,n
  if((N <= N_cut_M) && (K <= K_cut_M)) {
    return MKN;
  } else if((M <= M_cut_N) && (K <= K_cut_N)) {
    return NKM;
  } else {
    return KMN;
  }
}


void init_block_dims(int M, int N, int K, int p,
                     blk_dims_t* x, cake_cntx_t* cake_cntx, enum sched sch,
                     char* argv[], float density) {

  int m_r = cake_cntx->mr;
  int n_r = cake_cntx->nr;
  cache_dims_t* cache_dims = get_cache_dims(M, N, K, p,
                                            cake_cntx, sch, argv, density);
  x->m_c = cache_dims->m_c;
  x->k_c = cache_dims->k_c;
  x->n_c = cache_dims->n_c;
  x->sch = cache_dims->sch;
  free(cache_dims);

  switch(x->sch) {

    case KMN: {

      x->k_pad = (K % x->k_c) ? 1 : 0;
      x->n_pad = (N % x->n_c) ? 1 : 0;
      x->m_pad = (M % (p*x->m_c)) ? 1 : 0;

      x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r) ;
      int mr_per_core = (int) ceil( ((double) x->mr_rem) / p );

      if(mr_per_core)
        x->p_l = (int) ceil( ((double) x->mr_rem) / mr_per_core);
      else
        x->p_l = 0;

      x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
      x->n_c1 = x->nr_rem * n_r;

      x->m_c1 = mr_per_core * m_r;
      x->m_c1_last_core = (mr_per_core - (x->p_l*mr_per_core - x->mr_rem)) * m_r;
      x->k_c1 = K % x->k_c;

      //number of CB blocks in the M, N, and K dims
      x->Mb = (M / (p*x->m_c)) + x->m_pad;
      x->Nb = (N / x->n_c) + x->n_pad;
      x->Kb = (K / x->k_c) + x->k_pad;

      x->M_padded = (m_r*x->mr_rem + (M / (p*x->m_c))*p*x->m_c);
      x->N_padded = (N - (N % x->n_c)) + x->n_c1;

      break;
    }


    case MKN: {

      x->k_pad = (K % (p*x->k_c)) ? 1 : 0;
      x->m_pad = (M % x->m_c) ? 1 : 0;
      x->n_pad = (N % x->n_c) ? 1 : 0;

      x->k_rem = K % (p*x->k_c);
      x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

      if(x->k_c1)
        x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
      else
        x->p_l = 0;

      x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
      x->n_c1 = x->nr_rem * n_r;

      x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
      x->mr_rem = (int) ceil( ((double) (M % x->m_c)) / m_r);
      x->m_c1 = x->mr_rem * m_r;

      // number of CB blocks in the M, N, and K dims
      x->Mb = (M / x->m_c) + x->m_pad;
      x->Kb = (K / (p*x->k_c)) + x->k_pad;
      x->Nb = (N / x->n_c) + x->n_pad;

      x->M_padded = (M / x->m_c)*x->m_c + x->m_c1;
      x->N_padded = (N - (N % x->n_c)) + x->n_c1;


      break;
    }


    case NKM: {

      x->k_pad = (K % (p*x->k_c)) ? 1 : 0;
      x->m_pad = (M % (p*x->m_c)) ? 1 : 0;
      x->n_pad = (N % x->n_c) ? 1 : 0;

      x->k_rem = K % (p*x->k_c);
      x->k_c1 = (int) ceil( ((double) x->k_rem) / p);

      if(x->k_c1)
        x->p_l = (int) ceil( ((double) x->k_rem) / x->k_c1);
      else
        x->p_l = 0;

      x->nr_rem = (int) ceil( ((double) (N % x->n_c) / n_r)) ;
      x->n_c1 = x->nr_rem * n_r;

      x->k_c1_last_core = x->k_rem - x->k_c1*(x->p_l-1);
      x->mr_rem = (int) ceil( ((double) (M % (p*x->m_c))) / m_r);
      x->m_c1 = x->mr_rem * m_r;

      // number of CB blocks in the M, N, and K dims
      x->Mb = (M / (p*x->m_c)) + x->m_pad;
      x->Kb = (K / (p*x->k_c)) + x->k_pad;
      x->Nb = (N / x->n_c) + x->n_pad;

      x->M_padded = (M / (p*x->m_c))*(p*x->m_c) + x->m_c1;
      x->N_padded = (N - (N % x->n_c)) + x->n_c1;

      break;
    }


    default: {
      printf("unknown schedule\n");
      exit(1);
    }
  }
}


// least common multiple
int lcm(int n1, int n2) {
  int max = (n1 > n2) ? n1 : n2;
  while (1) {
    if (max % n1 == 0 && max % n2 == 0) {
      break;
    }
    ++max;
  }
  return max;
}

cache_dims_t* get_cache_dims_3(int M, int N, int K, int p,
                               cake_cntx_t* cake_cntx, enum sched sch,
                               char* argv[], float density,
                               bool mc_must_divide_M,
                               bool nc_must_divide_N) {

  int mc, mc_ret, nc_ret, a, mc_L2 = 0, mc_L3 = 0, kc_L2 = 0, kc_L3 = 0;
  int max_threads = cake_cntx->ncores; // 2-way hyperthreaded
  int mn_lcm = lcm(cake_cntx->mr, cake_cntx->nr);

  cache_dims_t* blk_ret = (cache_dims_t*) malloc(sizeof(cache_dims_t));

  //  // set schedule to MEMA-derived optimal value or user-defined
  //  blk_ret->sch = (sch == NA ?
  //                            derive_schedule(M, N, K, p, mc_ret, cake_cntx) :
  //                            sch);
  //
  blk_ret->sch = KMN;

  // solve for optimal mc,kc based on L2 size
  // L2_size >= 2*(mc*2*kc + 2*kc*nr) + 2*(mc*nr)     (solve for x = m_c = beta*k_c)
  //    The A mc*kc tile needs to fit into L2 but we can conservatively estimate
  //    this as (nnz * 3 * sizeof(float/int)) = (mc * kc *density * 3 * sizeof(float/int)),
  //    the B mc*nr tile is dense so se can leave that as is
  double beta = 1.;

  {
    double a = (3. / beta) * density;
    double b = ((1. + beta) / beta) * cake_cntx->nr;
    double c = -double(cake_cntx->L2) / 2*(sizeof(float));

    int x = (int)(-b + sqrt(b * b - 4.f * a * c) / (2.f * a));
    mc_L2 = x - (x % cake_cntx->mr);
    kc_L2 = (int)(x * beta);
  }


  // solve for the optimal block size m_c and k_c based on the L3 size
  // L3_size >= 2*(A tile + B tile) + 2*(C tile)
  // L3_size >= 2*(p*mc*kc + alpha*p*mc*kc) + 2*(p*mc*alpha*p*mc)     (solve for x = m_c = beta*k_c)
  //    The A mc*kc tile needs to fit into L2 but we can conservatively estimate
  //    this as (nnz * 3 * sizeof(float/int)) = (mc * nr *density * 3 * sizeof(float/int)),
  //    the B mc*nr tile is dense so se can leave that as is
  {
    double a = (p*3*density / beta) *(1 + cake_cntx->alpha_n) + p*p*cake_cntx->alpha_n;
    double b = 0;
    double c = -double(cake_cntx->L3) / (2* sizeof(float));

    int x = (int)(-b + sqrt(b * b - 4.f * a * c) / (2.f * a));
    mc_L3 = x - (x % cake_cntx->mr);
    kc_L3 = (int)(x * beta);
  }

  mc_ret = mc_L3;
  if(M < p*cake_cntx->mr) {
    mc_ret = cake_cntx->mr;
  } else if(M < p*mc_ret) {

    a = (M / p);
    if(a < cake_cntx->mr) {
      mc_ret = cake_cntx->mr;
    } else {
      a += (cake_cntx->mr - (a % cake_cntx->mr));
      mc_ret = a;
    }
  }

  // spMM is always K-first so using nc_ret from KMN
  nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
  nc_ret -= (nc_ret % cake_cntx->nr);
  nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;

  blk_ret->m_c = mc_L3 < M ? mc_L3 : M;
  blk_ret->k_c = kc_L2 < K ? kc_L2 : K;
  blk_ret->n_c = nc_ret;

  if (mc_must_divide_M) {
    while (M % blk_ret->m_c) {
      blk_ret->m_c--;
    }
  }

  if (nc_must_divide_N) {
    while (N % blk_ret->n_c) {
      blk_ret->n_c--;
    }
  }

  blk_ret->n_c  = std::max(blk_ret->n_c / cake_cntx->nr, 1) * cake_cntx->nr;
  blk_ret->m_c  = std::max(blk_ret->m_c / cake_cntx->mr, 1) * cake_cntx->mr;

  blk_ret->m_c = blk_ret->m_c*p > M ? std::ceil(M / p) : blk_ret->m_c;
  blk_ret->n_c = blk_ret->n_c > N ? N : blk_ret->n_c;

  blk_ret->m_c  = std::max(blk_ret->m_c / cake_cntx->mr, 1) * cake_cntx->mr;

  return blk_ret;
}


cache_dims_t* get_cache_dims_4(int M, int N, int K, int p,
                               cake_cntx_t* cake_cntx,
                               enum sched sch,
                               char* argv[],
                               bool sparse_a,
                               float density,
                               float beta,
                               bool mc_must_divide_M,
                               bool nc_must_divide_N) {

  int mc, mc_ret, nc_ret, a, mc_L2 = 0, mc_L3 = 0, kc_L2 = 0, kc_L3 = 0;
  int max_threads = cake_cntx->ncores; // 2-way hyperthreaded
  int mn_lcm = lcm(cake_cntx->mr, cake_cntx->nr);

  cache_dims_t* blk_ret = (cache_dims_t*) malloc(sizeof(cache_dims_t));

  //  // set schedule to MEMA-derived optimal value or user-defined
  //  blk_ret->sch = (sch == NA ?
  //                            derive_schedule(M, N, K, p, mc_ret, cake_cntx) :
  //                            sch);
  //
  blk_ret->sch = KMN;

  double alpha = cake_cntx->alpha_n;
  double lambda = sparse_a ? 3 * density : 1;

  // solve for optimal mc,kc based on L2 size
  // L2_size >= 2*(mc*kc + kc*nr) + 2*(mc*nr)      (solve for x = beta * m_c = k_c )
  //    For the A mc*kc tile:
  //      If not sparsity_aware,
  //        then use: mc*kc * sizeof(float/int)
  //      Else:
  //        we can conservatively estimate
  //        this as (nnz * 3 * sizeof(float/int))
  //          = (mc * kc *density * 3 * sizeof(float/int)),
  //    For the B kc*nr and C mc*nr:
  //      The tile is dense so se can leave them as is
  {
    double a = lambda * beta;
    double b = (alpha + beta) * cake_cntx->nr;
    double c = -double(cake_cntx->L2) / 2*(sizeof(float));

    double x = (int)(-b + sqrt(b * b - 4.f * a * c) / (2.f * a));
    mc_L2 = (int) x / beta;
    mc_L2 -= (mc_L2 % cake_cntx->mr);
    kc_L2 = (int)(x);
  }


  // solve for the optimal block size m_c and k_c based on the L3 size
  // L3_size >= 2*(A tile + B tile) + 2*(C tile)
  // L3_size >= 2*(p*mc*kc + alpha*p*mc*kc) + 2*(p*mc*alpha*p*mc)     (solve for x = m_c = beta*k_c)
  //    For the A mc*kc tile:
  //      If not sparsity_aware,
  //        then use: mc*kc * sizeof(float/int)
  //      Else:
  //        we can conservatively estimate
  //        this as (nnz * 3 * sizeof(float/int))
  //          = (mc * kc *density * 3 * sizeof(float/int)),
  //    For the B kc*nr and C mc*nr:
  //      The tile is dense so se can leave them as is
  {
    double a = (lambda + alpha) * (p * beta) + alpha*beta*p*p;
    double b = 0;
    double c = -double(cake_cntx->L3) / (2* sizeof(float));

    double x = (int)(-b + sqrt(b * b - 4.f * a * c) / (2.f * a));
    mc_L3 = (int) x / beta;
    mc_L3 -= (mc_L3 % cake_cntx->mr);
    kc_L3 = (int)(x);
  }

//  std::cout << std::endl << "p=" << p << " nr=" << cake_cntx->nr << " alpha=" << alpha << " beta=" << beta << " lambda=" << lambda << " L2=" << cake_cntx->L2 << std::endl;
//  std::cout << "mc_L3: " << mc_L3 << " kc_L2: " << kc_L2 << std::endl;

  mc_ret = mc_L3;
  if(M < p*cake_cntx->mr) {
    mc_ret = cake_cntx->mr;
  } else if(M < p*mc_ret) {

    a = (M / p);
    if(a < cake_cntx->mr) {
      mc_ret = cake_cntx->mr;
    } else {
      a += (cake_cntx->mr - (a % cake_cntx->mr));
      mc_ret = a;
    }
  }

  // spMM is always K-first so using nc_ret from KMN
  nc_ret = (int) (cake_cntx->alpha_n*p*mc_ret);
  nc_ret -= (nc_ret % cake_cntx->nr);
  nc_ret = nc_ret == 0 ? cake_cntx->nr : nc_ret;

  blk_ret->m_c = mc_L3 < M ? mc_L3 : M;
  blk_ret->k_c = kc_L2 < K ? kc_L2 : K;
  blk_ret->n_c = nc_ret;

  blk_ret->m_c = blk_ret->m_c*p > M ? std::ceil(M / p) : blk_ret->m_c;
//  blk_ret->n_c = blk_ret->n_c > N ? N : blk_ret->n_c;

  if (mc_must_divide_M) {
    blk_ret->m_c -= (blk_ret->m_c % cake_cntx->mr);
    while (blk_ret->m_c > 0 && M % blk_ret->m_c) blk_ret->m_c -= cake_cntx->mr; // Must also be a multiple of Mr

    if (blk_ret->m_c <= 0) {
      blk_ret->m_c += cake_cntx->mr;
      while (M % blk_ret->m_c) blk_ret->m_c += cake_cntx->mr;
    }
  }


  if (blk_ret->n_c >= N) {
    blk_ret->n_c = N; // next_largest_multiple(N, 16);
  }

//  if (nc_must_divide_N) {
//    while (N % blk_ret->n_c) {
//      blk_ret->n_c--;
//    }
//  }

  blk_ret->n_c  = std::max(next_multiple(blk_ret->n_c, cake_cntx->nr), cake_cntx->nr);
  blk_ret->m_c  = std::max(blk_ret->m_c / cake_cntx->mr, 1) * cake_cntx->mr;

  //blk_ret->n_c = std::min(blk_ret->n_c, 128);
  //std::cout << "N_c: " << blk_ret->n_c << std::endl;

  return blk_ret;
}
