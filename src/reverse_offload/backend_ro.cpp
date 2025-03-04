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

#include "backend_ro.hpp"

#include <immintrin.h>
#include <smmintrin.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <thread>  // NOLINT

#include "rocshmem/rocshmem.hpp"
#include "../atomic_return.hpp"
#include "../backend_type.hpp"
#include "../context_incl.hpp"
#include "mpi_transport.hpp"
#include "ro_net_team.hpp"
#include "../util.hpp"

namespace rocshmem {

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

ROBackend::ROBackend(MPI_Comm comm)
    : Backend(BackendType::RO_BACKEND) {

  if (auto maximum_num_contexts_str = getenv("ROCSHMEM_MAX_NUM_CONTEXTS")) {
    std::stringstream sstream(maximum_num_contexts_str);
    sstream >> maximum_num_contexts_;
  }
  poll_block_count_ = maximum_num_contexts_;

  profiler_proxy_ = ProfilerProxyT(maximum_num_contexts_);

  int device_id;
  hipDeviceProp_t device_props;

  CHECK_HIP(hipGetDevice(&device_id));
  CHECK_HIP(hipGetDeviceProperties(&device_props, device_id));

  max_wg_size_ = device_props.maxThreadsPerBlock;

  queue_ = Queue(maximum_num_contexts_, max_wg_size_, queue_size_);

  transport_ = new MPITransport(comm, &queue_);
  num_pes = transport_->getNumPes();
  my_pe = transport_->getMyPe();

  auto *bp{backend_proxy.get()};

  bp->hdp_policy = hdp_proxy_.get();

  bp->profiler = profiler_proxy_.get();

  bp->worker_thread_exit = false;

  bp->heap_ptr = &heap;

  ro_window_proxy_ = new WindowProxyT(&heap, transport_->get_world_comm(),
                                      num_windows_);

  bp->heap_window_info = ro_window_proxy_->get();

  initIPC();

  init_g_ret(&heap, transport_->get_world_comm(), maximum_num_contexts_, &bp->g_ret);

  allocate_atomic_region(&bp->atomic_ret, maximum_num_contexts_);

  transport_->initTransport(maximum_num_contexts_, &backend_proxy);

  host_interface = transport_->host_interface;

  default_host_ctx = std::make_unique<ROHostContext>(this, 0);

  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();

  team_world_proxy_ = new ROTeamProxy<HIPAllocator>(
      this, transport_->get_world_comm(), my_pe, num_pes);
  team_tracker.set_team_world(team_world_proxy_->get());

  ROCSHMEM_TEAM_WORLD =
      reinterpret_cast<rocshmem_team_t>(team_world_proxy_->get());

  default_block_handle_proxy_ = DefaultBlockHandleProxyT(
      bp->g_ret, bp->atomic_ret, &queue_, &ipcImpl, hdp_proxy_.get());

  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;
  default_context_proxy_ = DefaultContextProxyT(this, tinfo);

  block_handle_proxy_ = BlockHandleProxyT(bp->g_ret, bp->atomic_ret, &queue_,
                                          &ipcImpl, hdp_proxy_.get(),
                                          maximum_num_contexts_);
  setup_ctxs();

  worker_thread = std::thread(&ROBackend::ro_net_poll, this);

  *done_init = 1;
}

void ROBackend::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(ROContext) * maximum_num_contexts_));
  for (int i = 0; i < maximum_num_contexts_; i++) {
    new (&ctx_array[i]) ROContext(this, i);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

ROBackend::~ROBackend() {
  ro_net_free_runtime();
  CHECK_HIP(hipFree(ctx_array));
}

__device__ bool ROBackend::create_ctx(int64_t options, rocshmem_ctx_t *ctx) {
  ROContext *ctx_;

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;
  return true;
}

__device__ void ROBackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<ROContext *>(ctx->ctx_opaque));
}

void ROBackend::team_destroy(rocshmem_team_t team) {
  ROTeam *team_obj{get_internal_ro_team(team)};

  team_obj->~ROTeam();
  // CHECK_HIP(hipFree(team_obj));
}

void ROBackend::create_new_team(Team *parent_team,
                                TeamInfo *team_info_wrt_parent,
                                TeamInfo *team_info_wrt_world, int num_pes,
                                int my_pe_in_new_team, MPI_Comm team_comm,
                                rocshmem_team_t *new_team) {
  transport_->createNewTeam(this, parent_team, team_info_wrt_parent,
                            team_info_wrt_world, num_pes, my_pe_in_new_team,
                            team_comm, new_team);
}

void ROBackend::ctx_create(int64_t options, void **ctx) {
  ROHostContext *new_ctx{nullptr};
  new_ctx = new ROHostContext(this, options);
  *ctx = new_ctx;
}

ROHostContext *get_internal_ro_net_ctx(Context *ctx) {
  return reinterpret_cast<ROHostContext *>(ctx);
}

void ROBackend::ctx_destroy(Context *ctx) {
  ROHostContext *ro_net_host_ctx{get_internal_ro_net_ctx(ctx)};
  delete ro_net_host_ctx;
}

void ROBackend::reset_backend_stats() {
  auto *bp{backend_proxy.get()};

  for (size_t i{0}; i < MAX_NUM_BLOCKS; i++) {
    bp->profiler[i].resetStats();
  }
}

void ROBackend::dump_backend_stats() {
  uint64_t total{0};
  for (int i = 0; i < NUM_STATS; i++) {
    total += globalStats.getStat(i);
  }

  uint64_t gpu_frequency_mhz{wallClk_freq_mhz()};

  uint64_t us_wait_slot{0};
  uint64_t us_pack{0};
  uint64_t us_fence1{0};
  uint64_t us_fence2{0};
  uint64_t us_wait_host{0};

  auto *bp{backend_proxy.get()};

  for (size_t i{0}; i < MAX_NUM_BLOCKS; i++) {
    // Average latency as perceived from a thread
    const ROStats &prof{bp->profiler[i]};
    us_wait_slot += prof.getStat(WAITING_ON_SLOT) / gpu_frequency_mhz;
    us_pack += prof.getStat(PACK_QUEUE) / gpu_frequency_mhz;
    us_fence1 += prof.getStat(THREAD_FENCE_1) / gpu_frequency_mhz;
    us_fence2 += prof.getStat(THREAD_FENCE_2) / gpu_frequency_mhz;
    us_wait_host += prof.getStat(WAITING_ON_HOST) / gpu_frequency_mhz;
  }

  constexpr int FIELD_WIDTH{20};
  constexpr int FLOAT_PRECISION{2};

  printf("%*s%*s%*s%*s%*s\n", FIELD_WIDTH + 1, "Wait On Slot (us)",
         FIELD_WIDTH + 1, "Pack Queue (us)", FIELD_WIDTH + 1, "Fence 1 (us)",
         FIELD_WIDTH + 1, "Fence 2 (us)", FIELD_WIDTH + 1, "Wait Host (us)");

  printf("%*.*f %*.*f %*.*f %*.*f %*.*f\n\n", FIELD_WIDTH, FLOAT_PRECISION,
         static_cast<double>(us_wait_slot) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_pack) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_fence1) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_fence2) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_wait_host) / total);
}

void ROBackend::ro_net_free_runtime() {
  /*
   * Validate that a handle was passed that is not a nullptr.
   */
  auto *bp{backend_proxy.get()};
  assert(bp);

  /*
   * Set this flag to denote that the runtime is being torn down.
   */
  bp->worker_thread_exit = true;

  /*
   * Tear down the worker threads.
   */
  worker_thread.join();

  /*
   * Tear down the transport object.
   */
  while (!transport_->readyForFinalize()) {
  }
  transport_->finalizeTransport();

  ro_window_proxy_->~WindowProxyT();
  team_world_proxy_->~ROTeamProxy<HIPAllocator>();
  transport_->~MPITransport();
  /*
   * Free the profiler statistics structure.
   */
  // CHECK_HIP(hipFree(bp->profiler));

  /*
   * Tear down team_world
   */
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  // CHECK_HIP(hipFree(team_world));

  /*
   * Free the gpu_handle.
   */
  // CHECK_HIP(hipHostFree(bp));
}

void ROBackend::ro_net_poll() {
  auto *bp{backend_proxy.get()};
  while (!bp->worker_thread_exit) {
    for (size_t i{0}; i < poll_block_count_; i++) {
      int16_t request_count{0};
      const int16_t max_count{64};
      bool processed_req{true};
      while (processed_req && (request_count < max_count)) {
        processed_req = queue_.process(i, transport_);
        request_count++;
      }
    }
  }
}

void ROBackend::initIPC() {
  const auto &heap_bases{heap.get_heap_bases()};

  ipcImpl.ipcHostInit(transport_->getMyPe(), heap_bases);
}

void ROBackend::global_exit(int status) { transport_->global_exit(status); }

}  // namespace rocshmem
