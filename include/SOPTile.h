//
// Created by lwilkinson on 7/12/22.
//

#pragma once


#include <assert.h>
#include <vector>
#include <numeric>

#include "utils/misc.h"

#include "COO.h"
#include "SOPStorage.h"
#include "SOPMicroKernelBase.h"

using std::vector;
using std::array;
using std::pair;

namespace sop {

template <typename Scalar, typename Executor>
class SOPTile {
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

  using NonZero = typename COO<Scalar>::NonZero;
  using Pattern = uint16_t;

  static constexpr int MAX_PANEL_HEIGHT =
      16; // Max panel height 16, TODO template this

  static constexpr double PATTERN_BASE_COST = 4.1;
  static constexpr double PATTERN_NNZ_COST = 1.0;

  vector<vector<Pattern>> m_patterns;
  vector<vector<array<Scalar, MAX_PANEL_HEIGHT>>> m_values;
  vector<vector<int>> m_col_indices;

  const Pattern* m_supported_patterns;
  int m_num_supported_patterns;

  COO<Scalar>& m_coo_tile;
  int m_panel_height;
  int m_num_panels;
  int m_col_offset;

  int m_num_padded_nnz = 0;
  int m_num_total_patterns = 0;

  double compute_cost(const vector<Pattern>& patterns) {
    double _cost = 0;
    for (const auto& pat : patterns)
      _cost += PATTERN_BASE_COST + PATTERN_NNZ_COST * countbits(pat);
    return _cost;
  }

  double compute_cost(const Pattern* patterns, int num_patterns) {
    double _cost = 0;
    for (int i = 0; i < num_patterns; i++) {
      int bitcount = countbits(patterns[i]);
      _cost += PATTERN_BASE_COST + PATTERN_NNZ_COST * bitcount;
    }
    return _cost;
  }

  vector<Pattern> map_to_supported_patterns_old(Pattern pat) {
    int pattern_bits = countbits(pat);

    if (m_panel_height == 4) {
      if (pattern_bits < 2) {
        return {pat};
      } else {
        // Merge
        static uint16_t merge_patterns[] = {
            0b00001110, 0b00001101, 0b00001011, 0b00000111, 0b00001111};

        static constexpr int num_merge_patterns =
            sizeof(merge_patterns) / sizeof(merge_patterns[0]);

        bool pattern_merged = false;
        for (int i = 0; i < num_merge_patterns; i++) {
          if ((~merge_patterns[i] & pat) == 0) {
            return {merge_patterns[i]};
          }
        }

        if (!pattern_merged) {
          std::cerr << "Dangling pattern " << pat << std::endl;
          exit(-1);
        };
      }

    } else if (m_panel_height == 8) {
      if (pattern_bits < 2) {
        return {pat};
      } else if (pattern_bits == 3 && false) {
        // Split
        int idx = 0;
        while (((1 << idx) & pat) == 0) {
          idx++;
        }

        Pattern pat1 = 1 << idx;
        Pattern pat2 = ~(1 << idx) & pat;
        return {pat1, pat2};
      } else {
        vector<Pattern> patterns;

        // Merge
        static uint16_t merge_patterns[] = {
            // 0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b01010101,
            0b10101010,
            0b11000011,
            0b00111100,
            0b00001111,
            0b11110000,
            0b11111100,
            0b11110011,
            0b11001111,
            0b00111111,
            0b11111111};

        static constexpr int num_merge_patterns =
            sizeof(merge_patterns) / sizeof(merge_patterns[0]);

        static constexpr double split_cost = 2.1;
        static constexpr double merge_cost = 1;

        double min_cost = 100000;
        int best_pattern = -1;

        for (int i = 0; i < num_merge_patterns; i++) {
          int target_pattern = merge_patterns[i];

          int padding =
              countbits(target_pattern) - countbits(pat & target_pattern);
          int split = countbits(pat & ~target_pattern);
          double cost = split_cost * split + merge_cost * padding;

          if (cost < min_cost) {
            min_cost = cost;
            best_pattern = target_pattern;
          }
        }

        if (best_pattern == -1) {
          std::cerr << "Failed to find pattern" << std::endl;
          exit(-1);
        }

        patterns.push_back(best_pattern);

        // Split
        if (pat & ~best_pattern) {
          int split_pattern = pat & ~best_pattern;
          int idx = 0;

          while (split_pattern) {
            if (split_pattern & 1) {
              patterns.push_back(1 << idx);
            }

            idx++;
            split_pattern >>= 1;
          }
        }

        return patterns;
      }
    }

    return {};
  }

  vector<Pattern> map_to_supported_patterns(Pattern pat) {
    // Fast outs, assume
    if (pat == 0)
      return {pat};
    if (countbits(pat) == 1)
      return {pat};

    // First simply just check if matches one of the supported patterns exactly, if so just return that
    for (int i = 0; i < m_num_supported_patterns; i++) {
      if (pat == m_supported_patterns[i])
        return {m_supported_patterns[i]};
    }

    static constexpr int MAX_SEARCH_DEPTH = 8;

    int num_patterns = 0;
    int curr_stack_size = 0;
    int stack_pattern_idx[MAX_SEARCH_DEPTH] = {0};
    Pattern stack_pattern_set[MAX_SEARCH_DEPTH] = {0};
    Pattern stack_pattern[MAX_SEARCH_DEPTH] = {0};

    double best_pattern_set_cost = std::numeric_limits<double>::max();
    Pattern best_pattern_set[MAX_SEARCH_DEPTH] = {0};
    int best_pattern_set_len = 0;

    auto push = [&, this](int idx, Pattern pat) -> Pattern {
      stack_pattern_idx[curr_stack_size] = idx;
      stack_pattern_set[curr_stack_size] = m_supported_patterns[idx];
      stack_pattern[curr_stack_size] = pat;
      curr_stack_size++;

      return pat & ~m_supported_patterns[idx];
    };

    auto pop = [&, this]() -> pair<Pattern, int> {
      curr_stack_size--;
      pair<int, Pattern> result = {
          stack_pattern_idx[curr_stack_size], stack_pattern[curr_stack_size]};
      return result;
    };

    Pattern pat_remaining = pat;
    //        std::cout << "=== " << pat_remaining << std::endl;
    for (int p = 0; p < m_num_supported_patterns; p++) {
      if (pat & m_supported_patterns[p]) {
        pat_remaining = push(p, pat);

        // DFS
        int i = p;
        while (curr_stack_size) {
          // Prune the search space
          double curr_cost = compute_cost(stack_pattern_set, curr_stack_size);
          if (curr_cost > best_pattern_set_cost) {
            std::tie(i, pat_remaining) = pop();
            continue;
          }; // Prune

          for (; i < m_num_supported_patterns && pat_remaining &&
               curr_stack_size < MAX_SEARCH_DEPTH;
               i++) {
            if (pat_remaining & m_supported_patterns[i]) {
              pat_remaining = push(i, pat_remaining);
            }
          }

          if (!pat_remaining) {
            // Check cost
            double cost = compute_cost(stack_pattern_set, curr_stack_size);
            //                        std::cout << cost << " " << curr_stack_size << std::endl;
            if (cost < best_pattern_set_cost) {
              //                            std::cout << " >> " << cost << " " << curr_stack_size; for (int t = 0; t < curr_stack_size; t++) {
              //                                std::cout << " " << stack_pattern_set[t];
              //                            }
              //                            std::cout << std::endl;
              std::copy(
                  stack_pattern_set,
                  &stack_pattern_set[curr_stack_size],
                  best_pattern_set);
              best_pattern_set_cost = cost;
              best_pattern_set_len = curr_stack_size;
            }
          }

          std::tie(i, pat_remaining) = pop();
          i++;
        }
        if (pat_remaining == 0) {
          std::cerr << "pop failed" << std::endl;
        }
      }
      curr_stack_size = 0;
    }

    if (best_pattern_set_len == 0) {
      // TODO: Could just return infinite cost here
      std::cerr << "Failed to map pattern" << std::endl;
      exit(-1);
    }

    vector<Pattern> mapped_patterns(best_pattern_set_len);
    std::copy(
        best_pattern_set,
        &best_pattern_set[best_pattern_set_len],
        mapped_patterns.begin());

    Pattern coverage = 0;
    for (const auto& mapped_pattern : mapped_patterns)
      coverage |= mapped_pattern;
    assert((pat & ~coverage) == 0);

    return mapped_patterns;
  }

  void compute_vec_patterns_for_row_panel(int panel_id) {
    auto panel = m_coo_tile.submatrix_extract(
        {panel_id * m_panel_height, (panel_id + 1) * m_panel_height},
        {0, m_coo_tile.cols()});

    vector<Pattern>& panel_patterns = m_patterns[panel_id];
    vector<array<Scalar, MAX_PANEL_HEIGHT>>& panel_values = m_values[panel_id];
    vector<int>& panel_col_indices = m_col_indices[panel_id];

    panel_patterns.resize(m_coo_tile.cols(), 0);
    panel_values.resize(m_coo_tile.cols());
    panel_col_indices.resize(m_coo_tile.cols());

    for (auto& vec : panel_values)
      vec.fill(0);

    for (const NonZero& nnz : panel) {
      assert(nnz.col < panel_patterns.size());
      assert(nnz.col < panel_values.size());
      assert(nnz.col < panel_col_indices.size());
      assert(nnz.row < panel_values[nnz.col].size());

      panel_patterns[nnz.col] |= (1 << nnz.row);
      panel_values[nnz.col][nnz.row] = nnz.value;
      panel_col_indices[nnz.col] = nnz.col;
    }

    int num_cols = m_coo_tile.cols();
    for (int c = 0; c < num_cols; c++) {
      assert(c < panel_patterns.size());

      int nnz = countbits(panel_patterns[c]);

      vector<Pattern> mapped_patterns =
          map_to_supported_patterns_old(panel_patterns[c]);
      assert(mapped_patterns.size() > 0);
      assert(c < panel_values.size());

      panel_patterns[c] = mapped_patterns[0];
      int mapped_nnz = countbits(mapped_patterns[0]);

      for (int i = 1; i < mapped_patterns.size(); i++) {
        panel_patterns.push_back(mapped_patterns[i]);
        panel_values.push_back(
            panel_values[c]); // Copy matching m_values for ease of packing
        panel_col_indices.push_back(c);

        mapped_nnz += countbits(mapped_patterns[i]);
      }

      m_num_padded_nnz += mapped_nnz - nnz;
      m_num_total_patterns += (nnz) ? mapped_patterns.size() : 0;

      if (mapped_nnz < nnz) {
        std::cerr << "Mapped patterns do not all nnz " << mapped_nnz << " "
                  << nnz << std::endl;
        exit(-1);
      }
    }
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
  // Scalar* m_values;

  SOPTile(COO<Scalar>& coo_tile, int col_offset = 0)
      : m_coo_tile(coo_tile),
        m_supported_patterns(Executor::supported_patterns()),
        m_num_supported_patterns(Executor::number_of_patterns()),
        m_panel_height(Executor::panel_height()),
        m_num_panels(std::ceil(coo_tile.rows() / m_panel_height)),
        m_col_offset(col_offset) {
    m_patterns.resize(m_num_panels);
    m_values.resize(m_num_panels);
    m_col_indices.resize(m_num_panels);
  }

  int num_panels() {
    return m_num_panels;
  }
  int num_padded_nnz() {
    return m_num_padded_nnz;
  }
  int num_patterns_total() {
    return m_num_total_patterns;
  }

  std::tuple<int, int, int> compute_required_storage() {
    int nnz_count = 0;
    int col_indices_count = 0;
    int num_patterns_total = 0;

    for (int panel_id = 0; panel_id < m_num_panels; panel_id++) {
      assert(panel_id < m_values.size());
      assert(panel_id < m_patterns.size());
      assert(panel_id < m_col_indices.size());

      const auto& panel_patterns = m_patterns[panel_id];

      vector<uint16_t> encoded_patterns(panel_patterns.size());

      // Encode the patterns
      std::transform(
          panel_patterns.begin(),
          panel_patterns.end(),
          encoded_patterns.begin(),
          [](Pattern pat) { return Executor::encode_pattern(pat); });

      for (const auto& pattern : encoded_patterns) {
        if (pattern == ZERO_PATTERN_ID)
          continue;

        nnz_count += Executor::nnz_count(pattern);
        col_indices_count++;
      }

      num_patterns_total += m_num_supported_patterns;
    }

    return {nnz_count, col_indices_count, num_patterns_total};
  }

  size_t compute_required_storage_in_bytes() {
    auto [nnz_count, col_indices_count, num_patterns_total] =
        compute_required_storage();

    return num_patterns_total * sizeof(*PanelUsingCounts::pattern_counts) +
        col_indices_count * sizeof(*PanelUsingCounts::col_indices) +
        nnz_count * sizeof(*PanelUsingCounts::values) + 4 * 64 * m_num_panels;
  }

  void pack_patterns(PanelUsingCounts* panel_descs, uint8_t* buffer = nullptr) {
    for (int panel_id = 0; panel_id < m_num_panels; panel_id++) {
      assert(panel_id < m_values.size());
      assert(panel_id < m_patterns.size());
      assert(panel_id < m_col_indices.size());

      const auto& panel_patterns = m_patterns[panel_id];

      vector<uint16_t> encoded_patterns(panel_patterns.size());

      // Encode the patterns
      std::transform(
          panel_patterns.begin(),
          panel_patterns.end(),
          encoded_patterns.begin(),
          [](Pattern pat) { return Executor::encode_pattern(pat); });

      auto order = sort_encoded_patterns(encoded_patterns);
      auto permuted_values = permute(order, m_values[panel_id]);
      auto permuted_col_indices = permute(order, m_col_indices[panel_id]);

      int num_values = 0;
      int num_col_indices = 0;

      for (const auto& encoded_pattern : encoded_patterns) {
        num_values += Executor::nnz_count(encoded_pattern);
        if (encoded_pattern != ZERO_PATTERN_ID)
          num_col_indices++;
      }

      auto& panel_desc = panel_descs[panel_id];
      panel_desc.num_patterns = m_num_supported_patterns;

      if (buffer) {
        buffer = cacheline_align_ptr(buffer);
        panel_desc.pattern_counts = (int*)buffer;
        buffer += m_num_supported_patterns * sizeof(*panel_desc.pattern_counts);

        buffer = cacheline_align_ptr(buffer);
        panel_desc.col_indices = (int*)buffer;
        buffer += num_col_indices * sizeof(*panel_desc.col_indices);

        buffer = cacheline_align_ptr(buffer);
        panel_desc.values = (Scalar*)buffer;
        buffer += num_values * sizeof(*panel_desc.values);
      } else {
        panel_desc.pattern_counts = new int[m_num_supported_patterns]();
        panel_desc.col_indices = new int[permuted_col_indices.size()]();
        panel_desc.values = new Scalar[num_values]();
      }

      std::fill(
          panel_desc.pattern_counts,
          &panel_desc.pattern_counts[m_num_supported_patterns],
          0);

      int curr_value_offset = 0;
      int curr_col_indices_offset = 0;
      for (int p = 0; p < encoded_patterns.size(); p++) {
        const auto& encoded_pattern = encoded_patterns[p];
        const auto& values = permuted_values[p];

        if (encoded_pattern == ZERO_PATTERN_ID)
          continue;

        assert(encoded_pattern < m_num_supported_patterns);
        assert(curr_col_indices_offset < permuted_col_indices.size());
        assert(p < permuted_col_indices.size());

        panel_desc.pattern_counts[encoded_pattern]++;
        panel_desc.col_indices[curr_col_indices_offset++] =
            permuted_col_indices[p] + m_col_offset;

        auto pattern = Executor::decode_pattern(encoded_pattern);

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

      panel_desc.num_col_indices = curr_col_indices_offset;
      panel_desc.num_nnz = curr_value_offset;
    }
  }

  void inspect() {
    //#pragma parallel for num_threads(16)
    for (int panel_id = 0; panel_id < m_num_panels; panel_id++) {
      compute_vec_patterns_for_row_panel(panel_id);
    }
  }
};

}; // namespace sop
