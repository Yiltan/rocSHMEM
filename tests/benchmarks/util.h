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

#ifndef __UTIL_H__
#define __UTIL_H__

#include <getopt.h>
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include <rocm-core/rocm_version.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

#define FIELD_WIDTH (20)
#define FLOAT_PRECISION (2)

#define CHECK_HIP(cmd)                                            \
{                                                                 \
  hipError_t error = (cmd);                                       \
  if (error != hipSuccess) {                                      \
    fprintf(stderr,"HIP error: %d line: %d\n", error,  __LINE__); \
    rocshmem_global_exit(1);                                      \
  }                                                               \
}

struct benchmark_args_t {
  int num_threads;
  int num_wgs;
  size_t max_bytes;
  size_t min_bytes;
  int num_iters;
  int verbose;
};
typedef struct benchmark_args_t benchmark_args_t;

static double parsesize(const char *value)
{
    // From RCCL-Tests
    long long int units;
    double size;
    char size_lit[2];

    int count = sscanf(value, "%lf %1s", &size, size_lit);

    switch (count) {
    case 2:
      switch (size_lit[0]) {
      case 'G':
      case 'g':
        units = 1024*1024*1024;
        break;
      case 'M':
      case 'm':
        units = 1024*1024;
        break;
      case 'K':
      case 'k':
        units = 1024;
        break;
      default:
        return -1.0;
      };
      break;
    case 1:
      units = 1;
      break;
    default:
      return -1.0;
    }

    return size * units;
}

static inline void parse_args(int argc, char **argv, benchmark_args_t *benchmark_args, int my_pe)
{
  int c;
  int longindex;
  double parsed;

  struct option longopts[] = {
      {"nthreads",   required_argument, 0, 't'},
      {"workgroups", required_argument, 0, 'w'},
      {"minbytes",   required_argument, 0, 'b'},
      {"maxbytes",   required_argument, 0, 'e'},
      {"iters",      required_argument, 0, 'n'},
      {"verbose",          no_argument, 0, 'v'},
      {"help",       required_argument, 0, 'h'},
  };

  // Set Defaults
  memset(benchmark_args, 0, sizeof(benchmark_args_t));
  benchmark_args->num_threads = 1;
  benchmark_args->num_wgs = 1;
  benchmark_args->max_bytes = 4 * 1024 * 1024;
  benchmark_args->min_bytes = 1;
  benchmark_args->num_iters = 1;

  // Parse Arguments
  while (1)
  {
    c = getopt_long(argc, argv, "t:w:b:e:n:vh:", longopts, &longindex);

    if (-1 == c) { break; }

    switch (c)
    {
      case 't':
        benchmark_args->num_threads = strtol(optarg, NULL, 0);
        break;
      case 'w':
        benchmark_args->num_wgs = strtol(optarg, NULL, 0);
        break;
      case 'b':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'minbytes'\n");
          rocshmem_global_exit(1);
        }
        benchmark_args->min_bytes = parsed;
        break;
      case 'e':
        parsed = parsesize(optarg);
        if (parsed < 0) {
          fprintf(stderr, "invalid size specified for 'maxbytes'\n");
          rocshmem_global_exit(1);
        }
        benchmark_args->max_bytes = parsed;
        printf("max bytes = %zu\n", benchmark_args->max_bytes);
        break;
      case 'n':
        benchmark_args->num_iters = strtol(optarg, NULL, 0);
        break;
      case 'v':
        benchmark_args->verbose = 1;
        break;
      case 'h':
      default:
        if (0 == my_pe)
        {
          if (c != 'h')
          {
            printf("Invalid option '%c'\n", c);
          }
          printf("USAGE: %s \n"
                 "\t[-t,--nthreads   <num threads per workgroup>]\n"
                 "\t[-w,--workgroups <num workgroups>]\n"
                 "\t[-b,--minbytes   <minimum number of bytes>]\n"
                 "\t[-e,--maxbytes   <maximum number of bytes>]\n"
                 "\t[-n,--iters      <iteration count>]\n"
                 "\t[-v,--verbose]\n"
                 "\t[-h,--help]\n",
                 basename(argv[0]));
          rocshmem_global_exit(1);
        }
        break;
    }
  }


  if (benchmark_args->max_bytes < benchmark_args->min_bytes)
  {
      fprintf(stderr, "maxbytes must be greater or equal to minbytes'\n");
      rocshmem_global_exit(1);
  }
}


static inline void print_dash()
{
  printf("#");
  for (int i=0; i<90; i++)
  {
    printf("-");
  }
  printf("\n");
}

static void print_header(const char *test_name, benchmark_args_t *args)
{
  print_dash();
  printf("# rocSHMEM Micro-Benchmarks: %s\n", test_name);

  if (1 == args->verbose)
  {
      print_dash();
      printf("# HIP: %d.%d.%d-%s\n",
              HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH, HIP_VERSION_GITHASH);
      printf("# ROCm: %s\n", ROCM_BUILD_INFO);
#ifdef OMPI_MAJOR_VERSION
      printf("# Open MPI: %d.%d.%d\n",
             OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION, OMPI_RELEASE_VERSION);
#endif /* OMPI_MAJOR_VERSION */

      print_dash();
      printf("# Workgroup(s): %d\n",            args->num_wgs);
      printf("# Thread(s) Per Workgroup: %d\n", args->num_threads);
      printf("# Iteration(s): %d\n",            args->num_iters);
  }

  print_dash();

  printf("%-*s%*s%*s%*s%*s\n",
         10, "#",
         (int)(FIELD_WIDTH * 1.5), "Kernel Total",
         FIELD_WIDTH, "",
         FIELD_WIDTH, "Intra-Kernel",
         FIELD_WIDTH, "");

  printf("%-*s%*s%*s%*s%*s\n",
         10, "# Size (B)",
         FIELD_WIDTH, "Latency (us)",
         FIELD_WIDTH, "Bandwidth (GB/s)",
         FIELD_WIDTH, "Latency (us)",
         FIELD_WIDTH, "Bandwidth (GB/s)");

  print_dash();

  fflush(stdout);
}

static void print_measurements(size_t size,
                               double total_kernel_latency,
                               double total_kernel_bandwidth,
                               double intra_kernel_latency,
                               double intra_kernel_bandwidth)
{
  printf("%-*lu%*.*f%*.*f%*.*f%*.*f\n",
         10, size,
         FIELD_WIDTH, FLOAT_PRECISION, total_kernel_latency,
         FIELD_WIDTH, FLOAT_PRECISION, total_kernel_bandwidth,
         FIELD_WIDTH, FLOAT_PRECISION, intra_kernel_latency,
         FIELD_WIDTH, FLOAT_PRECISION, intra_kernel_bandwidth);
  fflush(stdout);
}

/*
 * Returns the flattened thread index of the calling thread within its
 * thread block.
 */
__device__ __forceinline__ int get_flat_block_id() {
  return hipThreadIdx_x + hipThreadIdx_y * hipBlockDim_x +
         hipThreadIdx_z * hipBlockDim_x * hipBlockDim_y;
}

/*
 * Returns the flattened block index that the calling thread is a member of in
 * in the grid. Callers from the same block will have the same index.
 */
__device__ __forceinline__ int get_flat_grid_id() {
  return hipBlockIdx_x + hipBlockIdx_y * hipGridDim_x +
         hipBlockIdx_z * hipGridDim_x * hipGridDim_y;
}

/*
 * Returns the flattened thread index of the calling thread within the grid.
 */
__device__ __forceinline__ int get_flat_id() {
  return get_flat_grid_id() * (hipBlockDim_x * hipBlockDim_y * hipBlockDim_z)
       + get_flat_block_id();
}

#endif /* __UTIL_H__ */
