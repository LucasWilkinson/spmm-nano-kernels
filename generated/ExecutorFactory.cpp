//
// Created by lwilkinson on 8/25/22.
//

#include "ExecutorFactory.h"
#include "KernelDesc.h"

#include "microkernel_13fd3_float_512.h"
#include "microkernel_924ca_float_512.h"

namespace sop {

REGISTER_FACTORY_MicroKernel_float_13fd3_512_8x2(KDFloatNoPacking);
REGISTER_FACTORY_MicroKernel_float_13fd3_512_8x2(KDFloatCPartialPacking);

REGISTER_FACTORY_MicroKernel_float_924ca_512_4x4(KDFloatNoPacking);
REGISTER_FACTORY_MicroKernel_float_924ca_512_4x4(KDFloatCPartialPacking);

} // namespace sop
