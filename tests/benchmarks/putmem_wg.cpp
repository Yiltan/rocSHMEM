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

#include "roc_shmem_benchmark_util.hpp"

__global__ void kernel(char *source, char *target, size_t nelems_per_thread, int iters)
{
  __shared__ roc_shmem_ctx_t ctx;
  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(8, &ctx); // SHMEM_CTX_WP_PRIVATE

  char* source_off = source + (nelems_per_thread * get_flat_block_id());
  char* target_off = target + (nelems_per_thread * get_flat_block_id());

  for (int  i=0; i<iters; i++)
  {
    roc_shmemx_ctx_putmem_wg(ctx, target_off, source_off, nelems_per_thread, 1);
    roc_shmem_ctx_quiet(ctx);
  }

  roc_shmem_wg_ctx_destroy(&ctx);
  roc_shmem_wg_finalize();
}

int main(int argc, char **argv)
{
  benchmark_args_t args;
  int ndevices;
  int n_pes;
  int my_pe;
  int my_device;
  char *source_buf;
  char *target_buf;
  hipEvent_t start, stop;
  hipStream_t stream;

  my_pe = roc_shmem_my_pe();

  CHECK_HIP(hipGetDeviceCount(&ndevices));
  my_device = my_pe % ndevices;
  CHECK_HIP(hipSetDevice(my_device));

  roc_shmem_init();

  n_pes = roc_shmem_n_pes();

  if (n_pes != 2)
  {
    printf("This test requires two PEs\n");
    roc_shmem_global_exit(1);
  }

  parse_args(argc, argv, &args, my_pe);

  source_buf = (char*) roc_shmem_malloc(args.max_bytes);
  target_buf = (char*) roc_shmem_malloc(args.max_bytes);

  if (NULL == source_buf || NULL == target_buf)
  {
    printf("Error allocating memory from symmetric heap\n");
    roc_shmem_global_exit(1);
  }

  memset(target_buf, my_pe, args.max_bytes);
  memset(source_buf, my_pe, args.max_bytes);

  if (my_pe == 0)
  {
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
    CHECK_HIP(hipStreamCreate(&stream));
    print_header("putmem_wg", &args);
  }

  dim3 gridSize(args.num_wgs, 1, 1);
  dim3 blockSize(args.num_threads, 1, 1);

  size_t total_threads = gridSize.x * gridSize.y * gridSize.z;

  for (size_t size=args.min_bytes; size<=args.max_bytes; size *=2)
  {
      CHECK_HIP(hipDeviceSynchronize());

      size_t nelems_per_thread = size / total_threads;

      if (nelems_per_thread == 0) { continue; }

      if (my_pe == 0)
      {
        float time_ms;
        CHECK_HIP(hipEventRecord(start, stream));
        kernel<<<gridSize, blockSize, 1024, stream>>>(source_buf, target_buf, nelems_per_thread, args.num_iters);
        CHECK_HIP(hipEventRecord(stop, stream));
        CHECK_HIP(hipStreamSynchronize(stream));
        CHECK_HIP(hipEventElapsedTime(&time_ms, start, stop));

        double latency_us = (time_ms * 1000.0) / ((double) args.num_iters);

        double total_latency_s = time_ms / 1000.0;
        double bandwidth_Bps = ((double) size * (double) args.num_iters) / total_latency_s;
        double bandwidth_GBps = bandwidth_Bps / (1024.0 * 1024.0 * 1024.0);

        print_measurements(size, latency_us, bandwidth_GBps, 0.0, 0.0);
      }
  }

  roc_shmem_free(source_buf);
  roc_shmem_free(target_buf);
  roc_shmem_finalize();
  return EXIT_SUCCESS;
}
