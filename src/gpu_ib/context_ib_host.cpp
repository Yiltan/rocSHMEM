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

#include "context_ib_host.hpp"

#include <mpi.h>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "../backend_type.hpp"
#include "../context_incl.hpp"
#include "backend_ib.hpp"
#include "../host/host.hpp"

namespace rocshmem {

__host__ GPUIBHostContext::GPUIBHostContext(Backend *backend,
                                            [[maybe_unused]] int64_t options)
    : Context(backend, true) {
  GPUIBBackend *b{static_cast<GPUIBBackend *>(backend)};

  host_interface = b->host_interface;

  context_window_info = host_interface->acquire_window_context();
}

__host__ GPUIBHostContext::~GPUIBHostContext() {
  host_interface->release_window_context(context_window_info);
}

__host__ void GPUIBHostContext::putmem_nbi(void *dest, const void *source,
                                           size_t nelems, int pe) {
  host_interface->putmem_nbi(dest, source, nelems, pe, context_window_info);
}

__host__ void GPUIBHostContext::getmem_nbi(void *dest, const void *source,
                                           size_t nelems, int pe) {
  host_interface->getmem_nbi(dest, source, nelems, pe, context_window_info);
}

__host__ void GPUIBHostContext::putmem(void *dest, const void *source,
                                       size_t nelems, int pe) {
  host_interface->putmem(dest, source, nelems, pe, context_window_info);
}

__host__ void GPUIBHostContext::getmem(void *dest, const void *source,
                                       size_t nelems, int pe) {
  host_interface->getmem(dest, source, nelems, pe, context_window_info);
}

__host__ void GPUIBHostContext::fence() {
  host_interface->fence(context_window_info);
}

__host__ void GPUIBHostContext::quiet() {
  host_interface->quiet(context_window_info);
}

__host__ void GPUIBHostContext::sync_all() {
  host_interface->sync_all(context_window_info);
}

__host__ void GPUIBHostContext::barrier_all() {
  host_interface->barrier_all(context_window_info);
}

}  // namespace rocshmem
