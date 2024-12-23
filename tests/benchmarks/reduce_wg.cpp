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

__global__ void kernel_allreduce(char *source, char *target, size_t nelems, int iters,
                                 roc_shmem_team_t team)
{
  __shared__ roc_shmem_ctx_t ctx;
  roc_shmem_wg_init();
  roc_shmem_wg_ctx_create(8, &ctx); // SHMEM_CTX_WP_PRIVATE

  for (int  i=0; i<iters; i++)
  {
    roc_shmem_ctx_int_sum_wg_reduce(ctx, team, (int*) target, (int*)source, nelems);
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

  if (n_pes < 2)
  {
    printf("This test requires at least two PEs\n");
    roc_shmem_global_exit(1);
  }

  parse_args(argc, argv, &args, my_pe);

  if (1 != args.num_wgs)
  {
    printf("This test only supports a single workgroup\n");
    roc_shmem_global_exit(1);
  }

  source_buf = (char*) roc_shmem_malloc(args.max_bytes);
  target_buf = (char*) roc_shmem_malloc(args.max_bytes);

  if (NULL == source_buf || NULL == target_buf)
  {
    printf("Error allocating memory from symmetric heap\n");
    roc_shmem_global_exit(1);
  }

  memset(target_buf, my_pe, args.max_bytes);
  memset(source_buf, my_pe, args.max_bytes);

  CHECK_HIP(hipEventCreate(&start));
  CHECK_HIP(hipEventCreate(&stop));
  CHECK_HIP(hipStreamCreate(&stream));

  if (0 == my_pe)
  {
    print_header("int_sum_wg_reduce", &args);
  }

  dim3 gridSize(1, 1, 1);
  dim3 blockSize(args.num_threads, 1, 1);

  size_t total_threads = blockSize.x * blockSize.y * blockSize.z;

  roc_shmem_team_t team_reduce_world_dup;
  team_reduce_world_dup = ROC_SHMEM_TEAM_INVALID;
  roc_shmem_team_split_strided(ROC_SHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &team_reduce_world_dup);

  if (ROC_SHMEM_TEAM_INVALID == team_reduce_world_dup)
  {
    printf("Error creating roc_shmem_team\n");
    roc_shmem_global_exit(1);
  }

  for (size_t size=args.min_bytes; size<=args.max_bytes; size *=2)
  {
      if ((size / (args.num_threads * n_pes)) < 1) { continue; }

      CHECK_HIP(hipDeviceSynchronize());

      float time_ms;
      CHECK_HIP(hipEventRecord(start, stream));
      kernel_allreduce<<<gridSize, blockSize, 1024, stream>>>(source_buf, target_buf, (size / sizeof(int)), args.num_iters, team_reduce_world_dup);
      CHECK_HIP(hipEventRecord(stop, stream));
      CHECK_HIP(hipStreamSynchronize(stream));
      CHECK_HIP(hipEventElapsedTime(&time_ms, start, stop));

      double latency_us = (time_ms * 1000.0) / ((double) args.num_iters);

      double total_latency_s = time_ms / 1000.0;
      // Reduce sends to all PEs and recives from all PEs
      // total data volume per iteration becomes 2 * n_pes * size
      double bandwidth_Bps = (2.0 * (double) n_pes * (double) size * (double) args.num_iters)
                           / total_latency_s;
      double bandwidth_GBps = bandwidth_Bps / (1024.0 * 1024.0 * 1024.0);

      // Calculate Averages accross ranks
      double global_latency_us_sum;
      double global_bandwidth_GBps_sum;
      MPI_Reduce(&latency_us, &global_latency_us_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(&bandwidth_GBps, &global_bandwidth_GBps_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (0 == my_pe)
      {
        latency_us = global_latency_us_sum / ((double) n_pes);
        bandwidth_GBps = global_bandwidth_GBps_sum / ((double) n_pes);

        print_measurements(size, latency_us, bandwidth_GBps, 0.0, 0.0);
      }

      MPI_Barrier(MPI_COMM_WORLD);
  }

  roc_shmem_free(source_buf);
  roc_shmem_free(target_buf);
  roc_shmem_finalize();
  return EXIT_SUCCESS;
}
