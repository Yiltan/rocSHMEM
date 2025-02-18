/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_DEVICE_PROPERTIES_HPP_
#define LIBRARY_SRC_DEVICE_PROPERTIES_HPP_

#include "util.hpp"

namespace rocshmem {

static int hip_device_id;
static int wavefront_size;

__device__ static int wavefront_size_d;

static inline void init_device_properties() {
  hipDeviceProp_t deviceProp{};

  CHECK_HIP(hipGetDevice(&hip_device_id));
  CHECK_HIP(hipGetDeviceProperties(&deviceProp, hip_device_id));

  wavefront_size = deviceProp.warpSize;

  CHECK_HIP(hipMemcpy(&wavefront_size_d, &wavefront_size, sizeof(int), hipMemcpyHostToDevice));
}

}  // namespace rocshmem

#endif /* LIBRARY_SRC_DEVICE_PROPERTIES_HPP_ */
