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

#ifndef LIBRARY_SRC_IPC_POLICY_HPP_
#define LIBRARY_SRC_IPC_POLICY_HPP_

#include <hip/hip_runtime.h>
#include <mpi.h>

#include <atomic>
#include <vector>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory/hip_allocator.hpp"
#include "util.hpp"

namespace rocshmem {

class Backend;
class Context;

class IpcImpl {
  using HEAP_BASES_T = std::vector<char *, StdAllocatorHIP<char *>>;

 public:
  int shm_rank{0};

  int shm_size{0};

  char **ipc_bases{nullptr};

  __host__ void ipcHostInit(int my_pe, const HEAP_BASES_T &heap_bases);

  __host__ void ipcHostStop();

  __device__ bool isIpcAvailable(int my_pe, int target_pe) {
    return my_pe / shm_size == target_pe / shm_size;
  }
  __device__ void ipcGpuInit(Backend *gpu_backend, Context *ctx, int thread_id);

  __device__ void ipcCopy(void *dst, void *src, size_t size);

  __device__ void ipcCopy_wg(void *dst, void *src, size_t size);

  __device__ void ipcCopy_wave(void *dst, void *src, size_t size);

  __device__ void ipcFence() { __threadfence_system(); }

  template <typename T>
  __device__ T ipcAMOFetchAdd(T *val, T value) {
    return __hip_atomic_fetch_add(val, value, __ATOMIC_SEQ_CST,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ T ipcAMOFetchCas(T *val, T cond, T value) {
    __hip_atomic_compare_exchange_strong(val, &cond, value, __ATOMIC_SEQ_CST,
                                         __ATOMIC_SEQ_CST,
                                         __HIP_MEMORY_SCOPE_SYSTEM);
    return cond;
  }

  template <typename T>
  __device__ void ipcAMOAdd(T *val, T value) {
    __hip_atomic_fetch_add(val, value, __ATOMIC_SEQ_CST,
                           __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ void ipcAMOCas(T *val, T cond, T value) {
    __hip_atomic_compare_exchange_strong(val, &cond, value, __ATOMIC_SEQ_CST,
                                         __ATOMIC_SEQ_CST,
                                         __HIP_MEMORY_SCOPE_SYSTEM);
  }

  template <typename T>
  __device__ void ipcAMOSet(T *val, T value) {
    __hip_atomic_store(val, value, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  __device__ void zero_byte_read(int pe) {
    int local_pe = pe % shm_size;
    uint32_t *pe_ipc_base = reinterpret_cast<uint32_t *>(ipc_bases[local_pe]);
    volatile uint32_t read_value = __hip_atomic_load(
        pe_ipc_base, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_SYSTEM);
  }
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_IPC_POLICY_HPP_
