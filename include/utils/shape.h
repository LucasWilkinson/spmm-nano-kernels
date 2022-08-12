//
// Created by lwilkinson on 8/11/22.
//

#ifndef DNN_SPMM_BENCH_SHAPE_H
#define DNN_SPMM_BENCH_SHAPE_H

struct IntRange     { int start = 0; int end = 0; int size() const { return end - start; }};
struct Shape        { int rows = 0; int cols = 0; int area() const { return rows * cols; }};
struct SubmatrixLoc { IntRange rows; IntRange cols; Shape shape() const { return { rows.size(), cols.size() }; } };

#endif // DNN_SPMM_BENCH_SHAPE_H
