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

    // TODO: Clean this up
    template <typename MicroKernelDesc> class PanelPacker;

private:
    bool m_sorted_csr_style = false;
    bool m_precompute_row_offsets = false;

    std::vector<NonZero> m_non_zeros;
    std::vector<int> m_row_offsets;
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
        const int* row_offsets = nullptr;

        inline bool offset_outside_submatrix() {
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

        SubMatrixIterator(const std::vector<NonZero>& non_zeros, IntRange row_range, IntRange col_range, const int* row_offsets = nullptr) :
                non_zeros(non_zeros), row_range(row_range), col_range(col_range), offset(row_offsets ? row_offsets[row_range.start] : 0), row_offsets(row_offsets) {
            while(offset < non_zeros.size() && offset_outside_submatrix()) offset++;
            if (offset >= non_zeros.size()) offset = -1; // end 
        }

        reference operator*() const { return non_zeros[offset]; }
        pointer operator->() { return &non_zeros[offset]; }

        // Prefix increment
        SubMatrixIterator& operator++() {
            do { 
                offset++;
                if (offset >= non_zeros.size()) break;
                if (non_zeros[offset].row >= row_range.end) { offset = non_zeros.size(); break; } 
                // if (row_offsets && non_zeros[offset].col >= col_range.end) {
                //     int next_row = non_zeros[offset].row + 1;
                //     while (next_row < non_zeros.back().row && row_offsets[next_row] == offset) next_row++;
                //     std::cout << "next_row: " << next_row << std::endl;
                //     offset = row_offsets[next_row];
                // } else {
                //     offset++;
                // }
            } while (offset_outside_submatrix());
            if (offset >= non_zeros.size()) offset = -1; // end
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
    COO(int rows, int cols, const Ptr* row_ptrs, const Indices* col_inds, const _Scalar* values) {
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


    void precompute_row_offsets() {
        if (m_precompute_row_offsets) return;
        sort_csr_style();

        m_row_offsets.resize(m_rows + 1);
        m_row_offsets[0] = 0;
        int last_row = 0;
        int curr_offset = -1;
        for (auto const& nnz : m_non_zeros) {
            curr_offset++;
            if (nnz.row != last_row) {
                for (int i = last_row + 1; i <= nnz.row; i++) {
                    m_row_offsets[i] = curr_offset;
                }
                last_row = nnz.row;
            }   
        }

        for (int i = last_row + 1; i <= m_rows; i++) {
            m_row_offsets[i] = curr_offset;
        }
        m_precompute_row_offsets = true;
    }

    void reserve(size_t num_nnz) { m_non_zeros.reserve(num_nnz); }

    void pad_to_multiple_of(int m_r) {
        m_rows += (m_r - m_rows % m_r);
    }

    void append_nnz(const NonZero& nnz) {
        m_sorted_csr_style = false;
        m_precompute_row_offsets = false;
        m_non_zeros.push_back(nnz);
    }

    void merge(COO<_Scalar>&& other) {
        if (other.m_rows != m_rows || other.m_cols != m_cols) {
            std::cerr << "Cannot merge matrices of different dimensions" << std::endl;
            exit(-1);
        }

        m_sorted_csr_style = false;
        m_precompute_row_offsets = false;
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

    SubMatrixIterator submatrix_begin(SubmatrixLoc loc, const int* row_offsets = nullptr) const
        { return SubMatrixIterator(m_non_zeros, loc.rows, loc.cols, row_offsets); }
    SubMatrixIterator submatrix_begin(IntRange row_range, IntRange col_range, const int* row_offsets = nullptr) const
        { return SubMatrixIterator(m_non_zeros, row_range, col_range, row_offsets); }
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
        _submatrix.m_precompute_row_offsets = false;
        return _submatrix;
    }

    COO<_Scalar> submatrix_extract(SubmatrixLoc submatrix_loc, bool preserve_location = false) {
        return submatrix_extract(submatrix_loc.rows, submatrix_loc.cols, preserve_location);
    }

    int submatrix_nnz_count(IntRange row_range, IntRange col_range) const {
        int nnz_count = 0;

        if (m_precompute_row_offsets && col_range.start == 0 && col_range.end >= m_cols) {
            return m_row_offsets[row_range.end] - m_row_offsets[row_range.start];
        }


        const int* row_offsets = m_precompute_row_offsets ? m_row_offsets.data() : nullptr;
        for (auto iter = submatrix_begin(row_range, col_range, row_offsets); iter != submatrix_end(); ++iter) {
            nnz_count++;
        }

        return nnz_count;
    }

    std::pair<int, int> submatrix_working_set_size(IntRange row_range, IntRange col_range, int bcols) const {
        int nnz_count = 0; 

        std::vector <uint8_t> col_active(col_range.size(), 0);
        std::vector <uint8_t> row_active(row_range.size(), 0);

        const int* row_offsets = m_precompute_row_offsets ? m_row_offsets.data() : nullptr;
        for (auto iter = submatrix_begin(row_range, col_range, row_offsets); iter != submatrix_end(); ++iter) {
            nnz_count++;
            col_active[iter->col - col_range.start] = 1;
            row_active[iter->row - row_range.start] = 1;
        }

        int col_active_count = 0;
        for (int i = 0; i < col_active.size(); i++) {
            col_active_count += col_active[i];
        }
        int row_active_count = 0;
        for (int i = 0; i < row_active.size(); i++) {
            row_active_count += row_active[i];
        }

        return {nnz_count, nnz_count + col_active_count * bcols + row_active_count * bcols};
    }



    int submatrix_nnz_count(SubmatrixLoc submatrix_loc, int bcols) const {
        return submatrix_nnz_count(submatrix_loc.rows, submatrix_loc.cols);
    }

    std::pair<int, int> submatrix_working_set_size(SubmatrixLoc submatrix_loc, int bcols) const {
        return submatrix_working_set_size(submatrix_loc.rows, submatrix_loc.cols, bcols);
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
    int populate_csr(_Scalar* values, PtrType* ptrs, IndexType* indices,
                     enum CSRPtrType ptr_type = OFFSET,
                     int col_offset = 0, int idx_offset = 0) {
        sort_csr_style();

        for (int i = 0; i < (ptr_type == OFFSET ? m_rows + 1 : m_rows); i++) { ptrs[i] = 0; }

        for (int idx = 0; idx < m_non_zeros.size(); idx++) {
            const auto& nz = m_non_zeros[idx];

            if (ptr_type == COUNT) {
                if ((ptrs[nz.row] + 1) > std::numeric_limits<PtrType>::max()) return -1;
                ptrs[nz.row]++;
            } else {
                if ((idx + 1) > std::numeric_limits<PtrType>::max()) return -1;
                ptrs[nz.row + 1] = idx_offset + idx + 1;
            }

            indices[idx] = nz.col + col_offset;
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
