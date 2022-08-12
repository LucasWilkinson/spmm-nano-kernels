//
// Created by lwilkinson on 11/2/21.
//

#ifndef DDT_MATRIXUTILS_H
#define DDT_MATRIXUTILS_H

#include <algorithm>
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>
#include <valarray>
#include <stdexcept>

#include "utils/shape.h"

typedef std::vector<std::tuple<int,int,float>> RawMatrix;

class SparsityPattern {
public:
    enum layout { CSR, CSC } layout_;
    int * indices_;
    int * ptrs_;
    size_t num_rows_;
    size_t num_cols_;

    SparsityPattern(int * ptrs_, int * indices_, int num_rows_, int num_cols_,  enum layout layout_) :
            ptrs_(ptrs_), indices_(indices_), num_rows_(num_rows_), num_cols_(num_cols_), layout_(layout_) { }

};

template<typename Scalar>
class DenseMatrix {
    int rows = 0;
    int cols = 0;
    int stride = 0;

    std::valarray<Scalar> values;

public:
    DenseMatrix() {}
    DenseMatrix(int rows, int cols) : rows(rows), cols(cols), stride(cols), values(rows * cols) {};
    DenseMatrix(Shape shape) : rows(shape.rows), cols(shape.cols), stride(cols), values(rows * cols) {};


    std::valarray<Scalar> row(int row) {
        return values[std::slice(row * stride, cols, 1)];
    }

    std::slice_array<Scalar> row_slice(int row) {
        return values[std::slice(row * stride, cols, 1)];
    }

    Scalar& at(int row, int col) { return values[row * stride + col]; }

    // Move Constructor
    DenseMatrix(DenseMatrix &&lhs) noexcept {
        delete[] this->values;

        this->rows = lhs.rows;
        this->cols = lhs.cols;
        this->stride = lhs.stride;
        this->values = lhs.values;

        lhs.rows = 0;
        lhs.cols = 0;
        lhs.stride = 0;
        lhs.values = nullptr;
    }

    // Assignment operator
    DenseMatrix &operator=(DenseMatrix &&lhs) noexcept {
        this->rows = lhs.rows;
        this->cols = lhs.cols;
        this->stride = lhs.stride;
        this->values = lhs.values;

        lhs.rows = 0;
        lhs.cols = 0;
        lhs.stride = 0;

        return *this;
    }
};

template<typename Scalar>
class Matrix {
public:
    Matrix() = default;

    Matrix(int r, int c, int nz) : r(r), c(c), nz(nz), Lp(nullptr), Li(nullptr), Lx(nullptr) {}

    virtual ~Matrix() {
      delete[] this->Lp;
      delete[] this->Li;
      delete[] this->Lx;
    };

    // Assignment operator
    Matrix &operator=(Matrix &&lhs) noexcept {
      this->nz = lhs.nz;
      this->r = lhs.r;
      this->c = lhs.c;

      this->Lp = lhs.Lp;
      this->Li = lhs.Li;
      this->Lx = lhs.Lx;

      lhs.Lp = nullptr;
      lhs.Li = nullptr;
      lhs.Lx = nullptr;

      return *this;
    }

    // Assignment operator
    Matrix &operator=(const Matrix &lhs) {
      if (this == &lhs) {
        return *this;
      }
      this->nz = lhs.nz;
      this->r = lhs.r;
      this->c = lhs.c;

      this->Lp = new int[lhs.r + 1]();
      this->Li = new int[lhs.nz]();
      this->Lx = new Scalar[lhs.nz]();

      std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r + 1);
      std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
      std::memcpy(this->Lx, lhs.Lx, sizeof(Scalar) * lhs.nz);

      return *this;
    }

    // Copy Constructor
    Matrix(const Matrix &lhs) {
      this->r = lhs.r;
      this->c = lhs.c;
      this->nz = lhs.nz;

      this->Lp = new int[lhs.r + 1]();
      this->Li = new int[lhs.nz]();
      this->Lx = new Scalar[lhs.nz]();

      std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r + 1);
      std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
      std::memcpy(this->Lx, lhs.Lx, sizeof(Scalar) * lhs.nz);
    }

    // Move Constructor
    Matrix(Matrix &&lhs) noexcept {
      this->r = lhs.r;
      this->c = lhs.c;
      this->nz = lhs.nz;

      this->Lp = lhs.Lp;
      this->Lx = lhs.Lx;
      this->Li = lhs.Li;

      lhs.Lp = nullptr;
      lhs.Li = nullptr;
      lhs.Lx = nullptr;
    }

    int r;
    int c;
    int nz;

    int *Lp;
    int *Li;
    Scalar *Lx;

    void print() {
      for (int i = 0; i < this->r; ++i) {
        for (int j = this->Lp[i]; j < this->Lp[i + 1]; ++j) {
          std::cout << i << "," << this->Li[j] << std::endl;
        }
      }
    }

    virtual SparsityPattern sparsity_pattern() throw() {
        // Assume CSR, todo fix
        return SparsityPattern(this->Lp, this->Li, this->r, this->c, SparsityPattern::CSR);
//        throw std::runtime_error("sparsity_pattern called on Matrix base class");
//        return SparsityPattern(nullptr, nullptr, 0, 0, SparsityPattern::CSR);
    }
};

template<typename _Scalar>
class CSR : public Matrix<_Scalar> {
 public:
    using Scalar = _Scalar;

    CSR(int r, int c, int nz) : Matrix<Scalar>(r, c, nz) {
      this->Lp = new int[r + 1]();
      this->Li = new int[nz]();
      this->Lx = new Scalar[nz]();
    }

    CSR(int r, int c, int nz, RawMatrix m) : Matrix<Scalar>(r, c, nz) {
      this->Lp = new int[r + 1]();
      this->Li = new int[nz]();
      this->Lx = new Scalar[nz]();

      std::sort(m.begin(), m.end(), [](std::tuple<int, int, Scalar> lhs, std::tuple<int, int, Scalar> rhs) {
          bool c0 = std::get<0>(lhs) == std::get<0>(rhs);
          bool c1 = std::get<0>(lhs) < std::get<0>(rhs);
          bool c2 = std::get<1>(lhs) < std::get<1>(rhs);
          return c0 ? c2 : c1;
      });

      // Parse CSR Matrix
      for (int i = 0, LpCnt = 0; i < nz; i++) {
        auto &v = m[i];

        int ov = std::get<0>(v);
        Scalar im = std::get<2>(v);
        int iv = std::get<1>(v);

        this->Li[i] = iv;
        this->Lx[i] = im;

        if (i == 0) {
          this->Lp[LpCnt] = i;
        }
        if (i != 0 && std::get<0>(m[i - 1]) != ov) {
          while (LpCnt != ov) {
            this->Lp[++LpCnt] = i;
          }
        }
        if (nz - 1 == i) {
          this->Lp[++LpCnt] = i + 1;
        }
      }
    }

    // Copy constructor
    CSR(const CSR &lhs) : Matrix<Scalar>(lhs.r, lhs.c, lhs.nz) {
      this->Lp = new int[lhs.r + 1]();
      this->Li = new int[lhs.nz]();
      this->Lx = new float[lhs.nz]();

      std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r + 1);
      std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
      std::memcpy(this->Lx, lhs.Lx, sizeof(Scalar) * lhs.nz);
    }

    // Assignment operator
    CSR &operator=(const CSR &lhs) {
      this->nz = lhs.nz;
      this->r = lhs.r;
      this->c = lhs.c;

      this->Lp = new int[lhs.r + 1]();
      this->Li = new int[lhs.nz]();
      this->Lx = new float[lhs.nz]();

      std::memcpy(this->Lp, lhs.Lp, sizeof(int) * lhs.r + 1);
      std::memcpy(this->Li, lhs.Li, sizeof(int) * lhs.nz);
      std::memcpy(this->Lx, lhs.Lx, sizeof(Scalar) * lhs.nz);

      return *this;
    }

    // Move Constructor
    CSR(CSR &&lhs) noexcept: Matrix<Scalar>(lhs.r, lhs.c, lhs.nz) {
      this->Lp = lhs.Lp;
      this->Lx = lhs.Lx;
      this->Li = lhs.Li;

      lhs.Lp = nullptr;
      lhs.Li = nullptr;
      lhs.Lx = nullptr;
    }

    SparsityPattern sparsity_pattern() throw() override {
        return SparsityPattern(this->Lp, this->Li, this->r, this->c, SparsityPattern::CSR);
    }
};

template<typename _Scalar>
class CSC : public Matrix<_Scalar> {
public:
    using Scalar = _Scalar;

    CSC(int r, int c, int nz) : Matrix<Scalar>(r, c, nz) {
      this->Lp = new int[c + 1]();
      this->Li = new int[nz]();
      this->Lx = new Scalar[nz]();
    }

    CSC(int r, int c, int nz, RawMatrix m) : Matrix<Scalar>(r, c, nz) {
      this->Lp = new int[c + 1]();
      this->Li = new int[nz]();
      this->Lx = new Scalar[nz]();

      std::sort(m.begin(), m.end(), [](std::tuple<int, int, Scalar> lhs, std::tuple<int, int, Scalar> rhs) {
          bool c0 = std::get<1>(lhs) == std::get<1>(rhs);
          bool c1 = std::get<0>(lhs) < std::get<0>(rhs);
          bool c2 = std::get<1>(lhs) < std::get<1>(rhs);
          return c0 ? c1 : c2;
      });

      // Parse CSR Matrix
      for (int i = 0, LpCnt = 0; i < nz; i++) {
        auto &v = m[i];

        int ov = std::get<0>(v);
        Scalar im = std::get<2>(v);
        int iv = std::get<1>(v);

        this->Li[i] = ov;
        this->Lx[i] = im;

        if (i == 0) {
          this->Lp[LpCnt] = i;
        }
        if (i != 0 && std::get<1>(m[i - 1]) != iv) {
          while (LpCnt != iv) {
            this->Lp[++LpCnt] = i;
          }
        }
        if (nz - 1 == i) {
          this->Lp[++LpCnt] = i + 1;
        }
      }
    }

    void make_full() {
      auto lpc = new int[this->c + 1]();
      auto lxc = new Scalar[this->nz * 2 - this->c]();
      auto lic = new int[this->nz * 2 - this->c]();

      auto ind = new int[this->c]();

      for (size_t i = 0; i < this->c; i++) {
        for (size_t p = this->Lp[i]; p < this->Lp[i + 1]; p++) {
          int row = this->Li[p];
          ind[i]++;
          if (row != i)
            ind[row]++;
        }
      }
      lpc[0] = 0;
      for (size_t i = 0; i < this->c; i++)
        lpc[i + 1] = lpc[i] + ind[i];

      for (size_t i = 0; i < this->c; i++)
        ind[i] = 0;
      for (size_t i = 0; i < this->c; i++) {
        for (size_t p = this->Lp[i]; p < this->Lp[i + 1]; p++) {
          int row = this->Li[p];
          int index = lpc[i] + ind[i];
          lic[index] = row;
          lxc[index] = this->Lx[p];
          ind[i]++;
          if (row != i) {
            index = lpc[row] + ind[row];
            lic[index] = i;
            lxc[index] = this->Lx[p];
            ind[row]++;
          }
        }
      }
      delete[]ind;
      delete this->Lp;
      delete this->Li;
      delete this->Lx;

      this->nz = this->nz * 2 - this->c;
      this->Lx = lxc;
      this->Li = lic;
      this->Lp = lpc;
    }

    // Assignment copy operator
    CSC &operator=(CSC &&lhs) noexcept {
      this->Lp = lhs.Lp;
      this->Lx = lhs.Lx;
      this->Li = lhs.Li;

      lhs.Lp = nullptr;
      lhs.Li = nullptr;
      lhs.Lx = nullptr;

      return *this;
    }

    // Assignment operator
    CSC &operator=(const CSC &lhs) {
      if (&lhs == this) {
        return *this;
      }
      this->r = lhs.r;
      this->c = lhs.c;
      this->nz = lhs.nz;

      this->Lp = new int[lhs.r + 1]();
      this->Li = new int[lhs.nz]();
      this->Lx = new Scalar[lhs.nz]();

      std::copy(lhs.Lp, lhs.Lp + lhs.r + 1, this->Lp);
      std::copy(lhs.Lx, lhs.Lx + lhs.nz, this->Lx);
      std::copy(lhs.Li, lhs.Li + lhs.nz, this->Li);

      return *this;
    }

    // Copy Constructor
    CSC(CSC &lhs) : Matrix<Scalar>(lhs.r, lhs.c, lhs.nz) {
      this->Lp = new int[lhs.r + 1]();
      this->Li = new int[lhs.nz]();
      this->Lx = new Scalar[lhs.nz]();

      std::copy(lhs.Lp, lhs.Lp + lhs.r + 1, this->Lp);
      std::copy(lhs.Lx, lhs.Lx + lhs.nz, this->Lx);
      std::copy(lhs.Li, lhs.Li + lhs.nz, this->Li);
    }

    // Move Constructor
    CSC(CSC &&lhs) noexcept: Matrix<Scalar>(lhs.r, lhs.c, lhs.nz) {
      this->Lp = lhs.Lp;
      this->Lx = lhs.Lx;
      this->Li = lhs.Li;

      lhs.Lp = nullptr;
      lhs.Li = nullptr;
      lhs.Lx = nullptr;
    }

    SparsityPattern sparsity_pattern() throw() override {
        return SparsityPattern(this->Lp, this->Li, this->r, this->c, SparsityPattern::CSC);
    }
};

#endif //DDT_MATRIXUTILS_H
