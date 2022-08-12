//
// Created by lwilkinson on 7/7/22.
//

#ifndef DNN_SPMM_BENCH_MATRIXMANIPULATION_H
#define DNN_SPMM_BENCH_MATRIXMANIPULATION_H

#include "Matrix.h"
#include <utility>
#include <algorithm>
#include <iterator> // For std::forward_iterator_tag
#include <cstddef>  // For std::ptrdiff_t


enum CSRPtrType {
    COUNT,
    OFFSET
};

template<typename _Scalar>
class COO {
public:
    struct NonZero {
        int row;
        int col;
        _Scalar value;
    };

private:
    bool m_sorted_csr_style = false;

    std::vector<NonZero> m_non_zeros;
    int m_rows = 0;
    int m_cols = 0;


    void sort_csr_style() {
        if (m_sorted_csr_style) return;

        std::stable_sort(m_non_zeros.begin(), m_non_zeros.end(),
        [](NonZero const &a, NonZero const &b) {
            return (a.row != b.row) ? (a.row < b.row) : (a.col < b.col);
        });
        m_sorted_csr_style = true;
    }

public:
    struct SubMatrixIterator
    {

    private:
        IntRange row_range;
        IntRange col_range;
        const std::vector<NonZero>& non_zeros;

        int offset = 0;

        inline bool offset_inside_submatrix() {
            return ((non_zeros[offset].row < row_range.start || non_zeros[offset].row >= row_range.end)
                ||  (non_zeros[offset].col < col_range.start || non_zeros[offset].col >= col_range.end));
        }

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using value_type        = const NonZero;
        using pointer           = const NonZero*;  // or also value_type*
        using reference         = const NonZero&;  // or also value_type&

        // End Iterator Initialization
        SubMatrixIterator(const std::vector<NonZero>& non_zeros, int offset) :
                non_zeros(non_zeros), offset(offset) {};

        SubMatrixIterator(const std::vector<NonZero>& non_zeros, IntRange row_range, IntRange col_range) :
                non_zeros(non_zeros), row_range(row_range), col_range(col_range) {
            while (offset < non_zeros.size() && offset_inside_submatrix()) { offset++; }
            if (offset >= non_zeros.size()) { offset = -1; } // end
        }

        reference operator*() const { return non_zeros[offset]; }
        pointer operator->() { return &non_zeros[offset]; }

        // Prefix increment
        SubMatrixIterator& operator++() {
            do { offset++; } while(offset < non_zeros.size() && offset_inside_submatrix());
            if (offset >= non_zeros.size()) { offset = -1; } // end

            return *this;
        }

        // Postfix increment
        SubMatrixIterator operator++(int)
            { SubMatrixIterator tmp = *this; ++(*this); return tmp; }

        friend bool operator== (const SubMatrixIterator& a, const SubMatrixIterator& b)
            { return a.offset == b.offset; };
        friend bool operator!= (const SubMatrixIterator& a, const SubMatrixIterator& b)
            { return a.offset != b.offset; };

    };

    COO(int rows, int cols) : m_rows(rows), m_cols(cols) {}

    COO(CSR<_Scalar>& csr): COO(csr.r, csr.c, csr.Lp, csr.Li, csr.Lx) {}

    template<typename Ptr, typename Indices>
    COO(int rows, int cols, Ptr* row_ptrs, Indices* col_inds, _Scalar* values) {
        m_non_zeros.resize(0);
        m_non_zeros.reserve(row_ptrs[rows]);

        m_rows = rows;
        m_cols = cols;

        for (int i = 0; i < rows; i++) {
            for (int p = row_ptrs[i]; p < row_ptrs[i + 1]; p++) {
                if constexpr(std::numeric_limits<Indices>::max() > std::numeric_limits<int>::max())
                    if (col_inds[p] > std::numeric_limits<int>::max())
                        std::cerr << "Index outside of int limit " << col_inds[i] << std::endl;

                m_non_zeros.push_back({i, (int) col_inds[p], values[p]});
            }
        }

        m_sorted_csr_style = true;
    }

    void reserve(size_t num_nnz) { m_non_zeros.reserve(num_nnz); }

    void append_nnz(const NonZero& nnz) {
        m_sorted_csr_style = false;
        m_non_zeros.push_back(nnz);
    }

    void merge(COO<_Scalar>&& other) {
        if (other.m_rows != m_rows || other.m_cols != m_cols) {
            std::cerr << "Cannot merge matrices of different dimensions" << std::endl;
            exit(-1);
        }

        m_sorted_csr_style = false;
        m_non_zeros.insert(
            m_non_zeros.end(),
            std::make_move_iterator(other.m_non_zeros.begin()),
            std::make_move_iterator(other.m_non_zeros.end())
        );
    }

    int rows() const { return m_rows; }
    int cols() const { return m_cols; }
    int nnz()  const { return m_non_zeros.size(); }

    typename std::vector<NonZero>::iterator begin() { return m_non_zeros.begin(); };
    typename std::vector<NonZero>::iterator end()   { return m_non_zeros.end();   };

    SubMatrixIterator submatrix_begin(IntRange row_range, IntRange col_range) const
        { return SubMatrixIterator(m_non_zeros, row_range, col_range); }
    SubMatrixIterator submatrix_end() const
        { return SubMatrixIterator(m_non_zeros, -1); }


    COO<_Scalar> submatrix_extract(IntRange row_range, IntRange col_range, bool preserve_location = false) {
        Shape new_shape;
        sort_csr_style();

        if (preserve_location) {
            new_shape = { this->m_rows, this->m_cols };
        } else {
            new_shape = { std::min(row_range.end, this->m_rows) - row_range.start,
                          std::min(col_range.end, this->m_cols) - col_range.start };
        }

        COO<_Scalar> _submatrix(new_shape.rows, new_shape.cols);
        _submatrix.reserve(m_non_zeros.size());

        for (auto iter = submatrix_begin(row_range, col_range); iter != submatrix_end(); ++iter) {
            if (preserve_location) {
                _submatrix.append_nnz(*iter);
            } else {
                auto nz = *iter;
                nz.row -= row_range.start;
                nz.col -= col_range.start;
                _submatrix.append_nnz(nz);
            }
        }

        _submatrix.m_sorted_csr_style = true;
        return _submatrix;
    }

    COO<_Scalar> submatrix_extract(SubmatrixLoc submatrix_loc, bool preserve_location = false) {
        return submatrix_extract(submatrix_loc.rows, submatrix_loc.cols, preserve_location);
    }

    int submatrix_nnz_count(IntRange row_range, IntRange col_range) const {
        int nnz_count = 0;

        for (auto iter = submatrix_begin(row_range, col_range); iter != submatrix_end(); ++iter) {
            nnz_count++;
        }

        return nnz_count;
    }

    int submatrix_nnz_count(SubmatrixLoc submatrix_loc) const {
        return submatrix_nnz_count(submatrix_loc.rows, submatrix_loc.cols);
    }

    double submatrix_density(IntRange row_range, IntRange col_range, bool pad_boundary = true) const {
        Shape shape = { row_range.size(), col_range.size() } ;
        if (!pad_boundary) {
            shape = { std::min(row_range.end, this->m_rows) - row_range.start,
                      std::min(col_range.end, this->m_cols) - col_range.start };
        }

        return double(submatrix_nnz_count(row_range, col_range)) / double(shape.area());
    }

    double submatrix_density(SubmatrixLoc submatrix_loc, bool pad_boundary = true) {
        return submatrix_density(submatrix_loc.rows, submatrix_loc.cols, pad_boundary);
    }

    CSR<_Scalar> csr() {
        sort_csr_style();

        CSR<_Scalar> csr(m_rows, m_cols, m_non_zeros.size());

        for (int idx = 0; idx < m_non_zeros.size(); idx++) {
            const auto& nz = m_non_zeros[idx];
            csr.Lp[nz.row + 1] = idx + 1;
            csr.Li[idx] = nz.col;
            csr.Lx[idx] = nz.row;
        }

        return csr;
    }

    template<typename PtrType, typename IndexType>
    int populate_csr(_Scalar* values, PtrType* ptrs, IndexType* indices, enum CSRPtrType ptr_type = OFFSET) {
        sort_csr_style();

        for (int i = 0; i < (ptr_type == OFFSET ? m_rows + 1 : m_rows); i++) { ptrs[i] = 0; }

        for (int idx = 0; idx < m_non_zeros.size(); idx++) {
            const auto& nz = m_non_zeros[idx];

            if (ptr_type == COUNT) {
                if ((ptrs[nz.row] + 1) > std::numeric_limits<PtrType>::max()) return -1;
                ptrs[nz.row]++;
            } else {
                if ((idx + 1) > std::numeric_limits<PtrType>::max()) return -1;
                ptrs[nz.row + 1] = idx + 1;
            }

            indices[idx] = nz.col;
            values[idx] = nz.value;
        }

        return m_non_zeros.size();
    }

    int populate_dense(_Scalar* values, int stride = -1) {
        if (stride < 0) { stride = m_cols; }

        for (int i = 0; i < m_rows; i++) {
            for (int j = 0; j < m_cols; j++) {
                values[(i * stride) + j] = 0;
            }
        }

        for (int idx = 0; idx < m_non_zeros.size(); idx++) {
            const auto& nz = m_non_zeros[idx];
            values[(nz.row * stride) + nz.col] = nz.value;
        }

        return m_non_zeros.size();
    }
};

#endif //DNN_SPMM_BENCH_MATRIXMANIPULATION_H
