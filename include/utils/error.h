//
// Created by lwilkinson on 8/26/22.
//

#ifndef DNN_SPMM_BENCH_ERROR_H
#define DNN_SPMM_BENCH_ERROR_H

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#define ERROR_AND_EXIT(message) do {                                  \
  std::cerr << "Error: " << message << std::endl;                     \
  std::cerr << "   " __FILE__ ":" TOSTRING(__LINE__) << std::endl;    \
  exit(-1);                                                           \
} while(0)

#define ERROR_AND_EXIT_IF(condition, message)                         \
if (condition) {                                                      \
  ERROR_AND_EXIT(message);                                            \
}

#endif // DNN_SPMM_BENCH_ERROR_H
