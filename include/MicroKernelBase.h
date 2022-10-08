//
// Created by lwilkinson on 8/11/22.
//

#pragma once

#include "Storage.h"

#define __ALWAYS_INLINE __inline __attribute__((__always_inline__))

namespace sop {

enum Activation {
    NONE,
    MINMAX
};

static const uint16_t ZERO_PATTERN_ID = std::numeric_limits<uint16_t>::max();

} // namespace sop