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

#include "mpi_init_singleton.hpp"
#include <iostream>

namespace rocshmem {

MPIInitSingleton* MPIInitSingleton::instance{nullptr};

void MPIInitSingleton::init() {
  if (nullptr == instance) {
    instance = new MPIInitSingleton();
  }
}

MPIInitSingleton::MPIInitSingleton() {
  MPI_Initialized(&initialized);

  if (0 == initialized) {
    int provided;

    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);

    if (provided != MPI_THREAD_MULTIPLE) {
      std::cerr
        << "MPI_THREAD_MULTIPLE support is not available. rocSHMEM may not perform correctly."
        << std::endl;
    }
  }

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
}

MPIInitSingleton::~MPIInitSingleton() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized && !initialized) {
    MPI_Finalize();
  }
}

MPIInitSingleton* MPIInitSingleton::GetInstance() {
  MPIInitSingleton::init();
  return instance;
}

int MPIInitSingleton::get_world_rank() { return world_rank; }

int MPIInitSingleton::get_world_size() { return world_size; }

}  // namespace rocshmem
