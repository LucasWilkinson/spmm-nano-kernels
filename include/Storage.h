//
// Created by lwilkinson on 8/11/22.
//

#pragma once

#include <iostream>
#include <cstring>

#include "utils/tiling.h"
#include "utils/shape.h"

#include "Enums.h"

namespace sop {

using Pattern = uint16_t;
using NanoKernel = uint16_t;
using NanoKernelMapping = std::vector<std::vector<NanoKernel>>;

struct MicroKernelPackedData {
  int*   nkern_counts   = nullptr;
  int*   col_indices    = nullptr;
  float* values         = nullptr;

  int num_nnz = 0;
  int num_nkern = 0;
  int num_col_indices = 0;

  void free() {
    delete[] values;
    delete[] col_indices;
    delete[] nkern_counts;

    values = nullptr;
    col_indices = nullptr;
    nkern_counts = nullptr;
  }
};

struct PanelUsingCodelets {
  struct Codelet {
    uint16_t pat_code;
    uint16_t count;
  };

  float*   values         = nullptr;
  int*     col_indices    = nullptr;
  Codelet* codelets       = nullptr;

  int num_nnz = 0;
  int num_codeletes = 0;
  int num_col_indices = 0;

  void free() {
    delete[] values;
    delete[] col_indices;
    delete[] codelets;

    values = nullptr;
    col_indices = nullptr;
    codelets = nullptr;
  }
};

template <typename Scalar>
struct PackedTile {
  using CSRPtr = int;
  using CSRIndex = int;

  TileType type;
  SubmatrixLoc loc;
  Shape shape;
  int row_panel_offset;
  bool load_c = true;
  bool free_on_destruction = true;

  union {
    struct {
      Scalar* values;
      CSRPtr* ptrs;
      CSRIndex* inds;
    } csr;

    struct {
      Scalar* values;
    } dense;

    struct {
      int num_panels;
      struct MicroKernelPackedData* panel_descs;
    } sop;
  };

  PackedTile() {
    zero_union();
  }

  PackedTile(PackedTile&& other) {
    type = other.type;
    loc = other.loc;
    shape = other.shape;
    load_c = other.load_c;
    free_on_destruction = other.free_on_destruction;
    std::memcpy(&dense, &other.dense, size_of_union());

    other.type = SPARSE_CSR;
    other.loc = {{0, 0}, {0, 0}};
    other.shape = {0, 0};
    other.load_c = false;
    other.free_on_destruction = false;
    zero_union();
  }

  PackedTile& operator=(PackedTile&& other) {
    free();

    type = other.type;
    loc = other.loc;
    shape = other.shape;
    load_c = other.load_c;
    free_on_destruction = other.free_on_destruction;
    std::memcpy((uint8_t*)&dense, (uint8_t*)&other.dense, size_of_union());

    other.type = SPARSE_CSR;
    other.loc = {{0, 0}, {0, 0}};
    other.shape = {0, 0};
    other.load_c = false;
    other.free_on_destruction = false;
    other.zero_union();
    return *this;
  }

  ~PackedTile() {
    free();
  }

  int linear_size_in_bytes() const {
    switch (type) {
      case SPARSE_CSR: {
        int nnz = 0;
        for (int i = 0; i < shape.rows; i++)
          nnz += csr.ptrs[i];
        int size = 0;
        size += sizeof(CSRPtr) * (shape.rows + 1);
        size += sizeof(CSRIndex) * nnz;
        size += sizeof(Scalar) * nnz;
        return size + 3 * 64; // 3*64 for cacheline alignment
      }
      case SPARSE_SOP: {
        int size = 0;
        size += sizeof(sop.panel_descs[0]) * sop.num_panels;

        for (int i = 0; i < sop.num_panels; i++) {
          size +=
              sizeof(sop.panel_descs[i].values) * sop.panel_descs[i].num_nnz;
          size += sizeof(sop.panel_descs[i].col_indices) *
              sop.panel_descs[i].num_col_indices;
          size += sizeof(sop.panel_descs[i].nkern_counts) *
              sop.panel_descs[i].num_nkern;
        }
        return size + (3) * 64; // 3*64 for cacheline alignment
      }

      case SPARSE_MKL:
        std::cerr << "Not implemented " __FILE__ ": " << __LINE__ << std::endl;
        exit(-1);

      case DENSE:
        return sizeof(Scalar) * shape.area() + 64;
    }
    return 0;
  }

  void* cacheline_align_ptr(void* ptr) {
    return (void*)(((uintptr_t(ptr) + 63) / 64) * 64);
  }

  PackedTile pack_linear(void** buffer_ptr) {
    PackedTile updated_tile;

    updated_tile.type = this->type;
    updated_tile.loc = this->loc;
    updated_tile.shape = this->shape;
    updated_tile.load_c = this->load_c;
    updated_tile.free_on_destruction = false;

    void* buffer = *buffer_ptr;
    buffer = cacheline_align_ptr(buffer);

    switch (type) {
      case SPARSE_CSR: {
        int nnz = 0;
        for (int i = 0; i < shape.rows; i++)
          nnz += csr.ptrs[i];

        updated_tile.csr.ptrs = (decltype(csr.ptrs))buffer;
        buffer = std::copy(
            csr.ptrs, &csr.ptrs[shape.rows + 1], (decltype(csr.ptrs))buffer);

        buffer = cacheline_align_ptr(buffer);
        updated_tile.csr.inds = (decltype(csr.inds))buffer;
        buffer =
            std::copy(csr.inds, &csr.inds[nnz], (decltype(csr.inds))buffer);

        buffer = cacheline_align_ptr(buffer);
        updated_tile.csr.values = (decltype(csr.values))buffer;
        buffer = std::copy(
            csr.values, &csr.values[nnz], (decltype(csr.values))buffer);

        break;
      }
      case SPARSE_SOP: {
        updated_tile.sop.num_panels = sop.num_panels;
        updated_tile.sop.panel_descs = (decltype(sop.panel_descs))buffer;
        buffer = std::copy(
            sop.panel_descs,
            &sop.panel_descs[sop.num_panels],
            (decltype(sop.panel_descs))buffer);

        buffer = cacheline_align_ptr(buffer);
        for (int i = 0; i < sop.num_panels; i++) {
          updated_tile.sop.panel_descs[i].nkern_counts = (int*)buffer;
          buffer = std::copy(
              sop.panel_descs[i].nkern_counts,
              sop.panel_descs[i].nkern_counts + sop.panel_descs[i].num_nkern,
              (int*)buffer);

          updated_tile.sop.panel_descs[i].col_indices = (int*)buffer;
          buffer = std::copy(
              sop.panel_descs[i].col_indices,
              &sop.panel_descs[i]
                   .col_indices[sop.panel_descs[i].num_col_indices],
              (int*)buffer);

          updated_tile.sop.panel_descs[i].values = (Scalar*)buffer;
          buffer = std::copy(
              sop.panel_descs[i].values,
              &sop.panel_descs[i].values[sop.panel_descs[i].num_nnz],
              (Scalar*)buffer);
        }
        break;
      }

      case SPARSE_MKL:
        std::cerr << "Not implemented " __FILE__ ": " << __LINE__ << std::endl;
        exit(-1);

      case DENSE:
        updated_tile.dense.values = (Scalar*)buffer;
        buffer = std::copy(
            dense.values, &dense.values[shape.area()], (Scalar*)buffer);
        break;
    }

    *buffer_ptr = buffer;
    return updated_tile;
  }

 private:
  void zero_union() {
    std::memset((uint8_t*)&this->dense, 0, size_of_union());
  }

  size_t size_of_union() {
    return ((uint8_t*)this + sizeof(*this)) - (uint8_t*)&this->dense;
  }

  void free() {
    if (free_on_destruction) {
      switch (type) {
        case SPARSE_CSR:
          delete[] csr.values;
          delete[] csr.ptrs;
          delete[] csr.inds;
          break;

        case SPARSE_SOP:
          for (int i = 0; i < sop.num_panels; i++)
            sop.panel_descs[i].free();
          delete[] sop.panel_descs;
          break;

        case DENSE:
          delete[] dense.values;
          break;
      }
    }

    zero_union();
    free_on_destruction = false;
  }
};

template <typename PackedTile>
struct TileGroup {
  Shape C_tile; // TODO: split-n parallelism, split-k?
  int buffer_offset_in_row_panels;
  std::vector<int> row_panels;
  std::vector<PackedTile> tiles;
};


};