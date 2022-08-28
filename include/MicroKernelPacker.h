//
// Created by lwilkinson on 8/25/22.
//

#ifndef DNN_SPMM_BENCH_SOPPACKING_H
#define DNN_SPMM_BENCH_SOPPACKING_H

#include "COO.h"
#include "utils/error.h"
#include "utils/misc.h"

#include "MicroKernelBase.h"

#include <assert.h>
#include <memory>
#include <numeric>
#include <vector>
#include <array>


namespace sop {

using std::vector;
using std::array;

template<typename Scalar>
struct MicroKernelPacker {
  MicroKernelPacker(int M_r): M_r(M_r) {}

  const int M_r = 0;
  virtual void pack(MicroKernelPackedData& panel_desc, SubmatrixLoc loc, const COO<Scalar>& coo) = 0;
  virtual uint8_t* repack_coalesced(MicroKernelPackedData& panel_desc, uint8_t* buffer, bool free=true) = 0;
};

template<typename MicroKernelDesc>
struct MicroKernelPackerSpeaclized:
      public MicroKernelPacker<typename MicroKernelDesc::Scalar> {

  using Scalar = typename MicroKernelDesc::Scalar;
  using Super = MicroKernelPacker<Scalar>;
  using MicroKernel = typename MicroKernelDesc::MicroKernel;
  using NonZero = typename COO<Scalar>::NonZero;

  template <typename T>
  int countbits(T ch) {
    int n = 0;
    if (ch) {
      do
        n++;
      while (0 != (ch = ch & (ch - 1)));
    }
    return n;
  }

  std::shared_ptr<NanoKernelMapping> m_pattern_mapping;

  MicroKernelPackerSpeaclized(
    int M_r, std::shared_ptr<NanoKernelMapping> pattern_mapping):
      Super(M_r), m_pattern_mapping(pattern_mapping) { }

  vector<Pattern> map_to_supported_patterns(Pattern pat) {
    return m_pattern_mapping->at(pat);
  }

  template <typename T>
  vector<T> permute(const vector<size_t>& order, vector<T>& v) {
    assert(order.size() == v.size());
    vector<T> v_permuted(v.size());

    for (int i = 0; i < order.size(); i++) {
      assert(order[i] < v.size());
      v_permuted[i] = v[order[i]];
    }

    return v_permuted;
  }

  template <typename T>
  vector<size_t> sort_encoded_patterns(vector<T>& v) {
    vector<size_t> idx(v.size());

    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {
      return (v[i1] == ZERO_PATTERN_ID || v[i1] < v[i2]) &&
          (v[i2] != ZERO_PATTERN_ID);
    });

    v = permute(idx, v);
    return idx;
  }

 public:
  std::tuple<int, int, int> compute_required_storage(
      const MicroKernelPackedData& panel_desc) {
    int nnz_count = 0;

    for (int pat_id = 0; pat_id < panel_desc.num_nkern; pat_id++) {
      int nnz_count_pat = MicroKernel::nnz_count(pat_id);
      nnz_count += nnz_count_pat * panel_desc.nkern_counts[pat_id];
    }

    return {nnz_count, panel_desc.num_col_indices, MicroKernel::num_patterns()};
  }

  size_t compute_required_storage_in_bytes(const MicroKernelPackedData& panel_desc) {
    auto [nnz_count, col_indices_count, num_patterns_total] =
        compute_required_storage(panel_desc);

    return num_patterns_total * sizeof(*MicroKernelPackedData::nkern_counts) +
        col_indices_count * sizeof(*MicroKernelPackedData::col_indices) +
        nnz_count * sizeof(*MicroKernelPackedData::values) + 4 * 64;
  }

  void pack(MicroKernelPackedData& panel_desc,
            SubmatrixLoc loc,
            const COO<Scalar>& coo) {

    ERROR_AND_EXIT_IF(loc.rows.size() != MicroKernelDesc::M_r,
                      "row panel size mismatch");

    const int MAX_NKERN_CALLS = loc.cols.size() * 2;
    const int num_cols = loc.cols.size();

    assert(num_cols == loc.cols.size());

    vector<Pattern> panel_patterns;
    vector<int> panel_col_indices;
    vector<array<Scalar, MicroKernelDesc::M_r>> panel_values;

    panel_patterns.reserve(MAX_NKERN_CALLS);
    panel_col_indices.reserve(MAX_NKERN_CALLS);
    panel_values.reserve(MAX_NKERN_CALLS);

    assert(num_cols == loc.cols.size());

    panel_patterns.resize(num_cols);
    panel_col_indices.resize(num_cols);
    panel_values.resize(num_cols);

    assert(num_cols == loc.cols.size());

    for (auto& vec : panel_values) vec.fill(0);

    assert(num_cols == loc.cols.size());

    for (auto iter = coo.submatrix_begin(loc); iter != coo.submatrix_end(); ++iter) {
      auto nnz = *iter;
      nnz.row -= loc.rows.start;
      nnz.col -= loc.cols.start;

      assert(nnz.col < panel_patterns.size());
      assert(nnz.col < panel_values.size());
      assert(nnz.col < panel_col_indices.size());
      assert(nnz.row < panel_values[nnz.col].size());

      panel_patterns[nnz.col] |= (1 << nnz.row);
      panel_values[nnz.col][nnz.row] = nnz.value;
      panel_col_indices[nnz.col] = nnz.col;
    }

    int total_mapped_nnz = 0;
    for (int c = 0; c < num_cols; c++) {
      assert(c < panel_patterns.size());

      int nnz = countbits(panel_patterns[c]);

      vector<Pattern> mapped_patterns =
          map_to_supported_patterns(panel_patterns[c]);

      assert(mapped_patterns.size() > 0);
      assert(c < panel_values.size());

      panel_patterns[c] = mapped_patterns[0];
      int mapped_nnz = countbits(mapped_patterns[0]);

      for (int i = 1; i < mapped_patterns.size(); i++) {
        panel_patterns.push_back(mapped_patterns[i]);
        panel_values.push_back(panel_values[c]); // Copy matching m_values for ease of packing
        panel_col_indices.push_back(c);

        mapped_nnz += countbits(mapped_patterns[i]);
      }

      ERROR_AND_EXIT_IF(mapped_nnz < nnz, "mapped_nnz < nnz");
      total_mapped_nnz += mapped_nnz;
    }

    vector<Pattern> encoded_patterns(panel_patterns.size());

    // Encode the patterns
    std::transform(
      panel_patterns.begin(),
      panel_patterns.end(),
      encoded_patterns.begin(),
      [](Pattern pat) { return MicroKernel::encode_nkern_pattern(pat); });

    auto order = sort_encoded_patterns(encoded_patterns);
    auto permuted_values = permute(order, panel_values);
    auto permuted_col_indices = permute(order, panel_col_indices);

    int num_values = 0;
    int num_col_indices = 0;

    for (const auto& encoded_pattern : encoded_patterns) {
      num_values += MicroKernel::nnz_count_for_nkern_code(encoded_pattern);
      if (encoded_pattern != ZERO_PATTERN_ID) num_col_indices++;
    }

    assert(num_values == total_mapped_nnz);

    panel_desc.nkern_counts = new int[MicroKernel::num_nkern_patterns()];
    panel_desc.col_indices = new int[num_col_indices];
    panel_desc.values = new Scalar[num_values];

    std::fill(
      panel_desc.nkern_counts,
      panel_desc.nkern_counts + MicroKernel::num_nkern_patterns(),
      0);

    int curr_value_offset = 0;
    int curr_col_indices_offset = 0;
    for (int p = 0; p < encoded_patterns.size(); p++) {
      assert(p < permuted_values.size());
      assert(p < permuted_col_indices.size());

      const auto& encoded_pattern = encoded_patterns[p];
      const auto& values = permuted_values[p];

      if (encoded_pattern == ZERO_PATTERN_ID) continue;

      assert(encoded_pattern < MicroKernel::num_nkern_patterns());
      assert(curr_col_indices_offset < permuted_col_indices.size());
      assert(p < permuted_col_indices.size());

      panel_desc.nkern_counts[encoded_pattern]++;
      panel_desc.col_indices[curr_col_indices_offset++] = permuted_col_indices[p];
      auto pattern = MicroKernel::decode_nkern_pattern(encoded_pattern);

      int row = 0;
      while (pattern) {
        if (pattern & 1) {
          assert(curr_value_offset < num_values);
          assert(row < values.size());

          panel_desc.values[curr_value_offset++] = values[row];
        }
        pattern >>= 1;
        row++;
      }
    }

    ERROR_AND_EXIT_IF(curr_value_offset != num_values,
                      "curr_value_offset != num_values");

    // Reapply offset
    for (int i = 0; i < curr_col_indices_offset; i++) {
      panel_desc.col_indices[i] += loc.cols.start;
    }

    panel_desc.num_col_indices = curr_col_indices_offset;
    panel_desc.num_nnz = curr_value_offset;
    panel_desc.num_nkern = MicroKernel::num_nkern_patterns();
  }

  uint8_t* repack_coalesced(
      MicroKernelPackedData& panel_desc,
      uint8_t* buffer,
      bool free = true
  ) {
    MicroKernelPackedData panel_desc_orig = panel_desc;

    buffer = cacheline_align_ptr(buffer);
    panel_desc.nkern_counts = (int*)buffer;
    buffer = std::copy(
        panel_desc_orig.nkern_counts,
        panel_desc_orig.nkern_counts + panel_desc_orig.num_nkern,
        buffer);

    buffer = cacheline_align_ptr(buffer);
    panel_desc.col_indices = (int*)buffer;
    buffer = std::copy(
        panel_desc_orig.col_indices,
        panel_desc_orig.col_indices + panel_desc_orig.num_col_indices,
        buffer);

    buffer = cacheline_align_ptr(buffer);
    panel_desc.values = (float*)buffer;
    buffer = std::copy(
        panel_desc_orig.values,
        panel_desc_orig.values + panel_desc_orig.num_nnz,
        buffer);

    if (free) panel_desc_orig.free();
    return buffer;
  }
};

} // namespace sop
#endif // DNN_SPMM_BENCH_SOPPACKING_H
