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

#ifndef ROCSHMEM_IPC_IMPL_SIMPLE_FINE_GTEST_HPP
#define ROCSHMEM_IPC_IMPL_SIMPLE_FINE_GTEST_HPP

#include "gtest/gtest.h"

#include <numeric>

#include <mpi.h>
#include "../src/memory/symmetric_heap.hpp"
#include "../src/ipc/ipc_policy.hpp"

namespace rocshmem {

enum TestType {
    READ = 0,
    WRITE = 1
};

__global__
void
kernel_simple_fine_copy(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes, TestType test) {
    if (!threadIdx.x) {
      ipc_impl->ipcCopy(dest, src, bytes);
      ipc_impl->ipcFence();
      if (test == WRITE) {
        ipc_impl->ipc
      }
    }
    __syncthreads();
}

__global__
void
kernel_simple_fine_copy_signal_validate(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    if (!threadIdx.x) {
      ipc_impl->ipcCopy(dest, src, bytes);
      ipc_impl->ipcFence();
    }
    __syncthreads();
}

__global__
void
kernel_simple_fine_copy_wg(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    ipc_impl->ipcCopy_wg(dest, src, bytes);
    ipc_impl->ipcFence();
    __syncthreads();
}

__global__
void
kernel_simple_fine_copy_wg_signal_validate(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    ipc_impl->ipcCopy_wg(dest, src, bytes);
    ipc_impl->ipcFence();
    __syncthreads();
}

__global__
void
kernel_simple_fine_copy_wave(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    ipc_impl->ipcCopy_wave(dest, src, bytes);
    ipc_impl->ipcFence();
    __syncthreads();
}

__global__
void
kernel_simple_fine_copy_wave_signal_validate(IpcImpl *ipc_impl, int *src, int *dest, size_t bytes) {
    ipc_impl->ipcCopy_wave(dest, src, bytes);
    ipc_impl->ipcFence();
    __syncthreads();
}

class IPCImplSimpleFineTestFixture : public ::testing::Test {

    using HEAP_T = HeapMemory<HIPDefaultFinegrainedAllocator>;

    using MPI_T = RemoteHeapInfo<CommunicatorMPI>;

    using FN_T = void (*)(IpcImpl*, int*, int*, size_t);

  public:
    IPCImplSimpleFineTestFixture() {
        ipc_impl_.ipcHostInit(mpi_.my_pe(), mpi_.get_heap_bases() , MPI_COMM_WORLD);

        assert(ipc_impl_dptr_ == nullptr);
        hip_allocator_.allocate((void**)&ipc_impl_dptr_, sizeof(IpcImpl));

        CHECK_HIP(hipMemcpy(ipc_impl_dptr_, &ipc_impl_,
                            sizeof(IpcImpl), hipMemcpyHostToDevice));
    }

    ~IPCImplSimpleFineTestFixture() {
        if (ipc_impl_dptr_) {
            hip_allocator_.deallocate(ipc_impl_dptr_);
        }

        ipc_impl_.ipcHostStop();
    }

    void launch(FN_T f, const dim3 grid, const dim3 block, int* src, int* dest, size_t bytes) {
        f<<<grid, block>>>(ipc_impl_dptr_, src, dest, bytes);
        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    void write(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(WRITE);
        copy(WRITE, grid, block);
        check_device_validation_errors(WRITE);
    }

    void write_wg(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(WRITE);
        copy_wg(WRITE, grid, block);
        check_device_validation_errors(WRITE);
    }

    void write_wave(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(WRITE);
        copy_wave(WRITE, grid, block);
        check_device_validation_errors(WRITE);
    }

    void read(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(READ);
        copy(READ, grid, block);
        check_device_validation_errors(READ);
    }

    void read_wg(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(READ);
        copy_wg(READ, grid, block);
        check_device_validation_errors(READ);
    }

    void read_wave(const dim3 grid, const dim3 block, size_t elems) {
        iota_golden(elems);
        initialize_src_buffer(READ);
        copy_wave(READ, grid, block);
        check_device_validation_errors(READ);
    }

    void iota_golden(size_t elems) {
        golden_.resize(elems);
        std::iota(golden_.begin(), golden_.end(), 0);
    }

    void validate_golden(size_t elems) {
        ASSERT_EQ(golden_.size(), elems);
        for (int i{0}; i < golden_.size(); i++) {
            ASSERT_EQ(golden_[i], i);
        }
    }

    void initialize_src_buffer(TestType test) {
        if (!pe_initializes_src_buffer(test)) {
            return;
        }
        size_t bytes = golden_.size() * sizeof(int);
        auto dev_src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[mpi_.my_pe()]);
        CHECK_HIP(hipMemcpy(dev_src, golden_.data(), bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipStreamSynchronize(nullptr));
    }

    __host__ __device__
    bool pe_initializes_src_buffer(TestType test) {
        bool is_write_test = test;
        bool is_read_test = !test;
        return (is_write_test && mpi_.my_pe() == 0) ||
               (is_read_test && mpi_.my_pe() == 1);
    }

    void execute(TestType test, FN_T fn, const dim3 grid, const dim3 block) {
        if (mpi_.my_pe()) {
            mpi_.barrier();
            mpi_.barrier();
            return;
        }
        int *src{nullptr};
        int *dest{nullptr};
        if (test == WRITE) {
            src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[0]);
            dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[1]);
        } else {
            src = reinterpret_cast<int*>(ipc_impl_.ipc_bases[1]);
            dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[0]);
        }
        size_t bytes = golden_.size() * sizeof(int);
        mpi_.barrier();
        launch(fn, grid, block, src, dest, bytes, test);
        mpi_.barrier();
    }

    void copy(TestType test, dim3 grid, dim3 block) {
        execute(test, kernel_simple_fine_copy, grid, block);
    }

    void copy_wg(TestType test, dim3 grid, dim3 block) {
        execute(test, kernel_simple_fine_copy_wg, grid, block);
    }

    void copy_wave(TestType test, dim3 grid, dim3 block) {
        execute(test, kernel_simple_fine_copy_wave, grid, block);
    }

    void check_device_validation_errors(TestType test) {
        if (!pe_validates_dest_buffer(test)) {
            return;
        }
        ASSERT_EQ(validation_error, false);
    }

    void validate_dest_buffer(TestType test) {
        if (!pe_validates_dest_buffer(test)) {
            return;
        }

        auto dev_dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[mpi_.my_pe()]);
        for (int i{0}; i < golden_.size(); i++) {
            ASSERT_EQ(golden_[i], dev_dest[i]);
        }
    }

    __device__
    void validate_dest_buffer(TestType test) {
        if (!pe_validates_dest_buffer(test)) {
            return;
        }

        auto dev_dest = reinterpret_cast<int*>(ipc_impl_.ipc_bases[mpi_.my_pe()]);
        for (int i {get_flat_id()}; i < golden_.size(); i += get_flat_grid_size()) {
            if (dev_golden_[i] != dev_dest[i]) {
                validation_error = true;
            }
        }
    }

    __host__ __device__
    bool pe_validates_dest_buffer(TestType test) {
        return !pe_initializes_src_buffer(test);
    }

  protected:
    std::vector<int> golden_;

    std::vector<int> device_golden_;

    HEAP_T heap_mem_ {};

    MPI_T mpi_ {heap_mem_.get_ptr(), heap_mem_.get_size()};

    IpcImpl ipc_impl_ {};

    IpcImpl *ipc_impl_dptr_ {nullptr};

    HIPAllocator hip_allocator_ {};

    bool validation_error {false};
};

} // namespace rocshmem

#endif  // ROCSHMEM_IPC_IMPL_SIMPLE_FINE_GTEST_HPP
