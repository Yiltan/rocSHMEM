/*
 *  Copyright (c) 2017 Intel Corporation. All rights reserved.
 *  This software is available to you under the BSD license below:
 *
 *      Redistribution and use in source and binary forms, with or
 *      without modification, are permitted provided that the following
 *      conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* For systems without the PThread barrier API (e.g. MacOS) */
//#include "pthread_barrier.h"

#define N_THREADS 8
#define N_ELEMS 10

static long *source;
static long *target;
pthread_barrier_t fencebar;

static void *roundrobin(void *tparam) {
  ptrdiff_t tid = (ptrdiff_t)tparam;
  int offset = tid * N_ELEMS;
  /* fprintf(stderr,"Starting thread %lu with offset %d\n",tid,offset); */

  int nextpe = (rocshmem_my_pe() + 1) % rocshmem_n_pes();
  int prevpe = (rocshmem_my_pe() - 1 + rocshmem_n_pes()) % rocshmem_n_pes();
  rocshmem_long_put(target + offset, source + offset, N_ELEMS, nextpe);

  /* fprintf(stderr,"Thread %lu done first put\n",tid); */
  pthread_barrier_wait(&fencebar);
  if (tid == 0) rocshmem_barrier_all();
  pthread_barrier_wait(&fencebar);

  rocshmem_long_get(source + offset, target + offset, N_ELEMS, prevpe);

  /* fprintf(stderr,"Thread %lu done first get\n",tid); */
  pthread_barrier_wait(&fencebar);
  if (tid == 0) rocshmem_barrier_all();
  pthread_barrier_wait(&fencebar);

  rocshmem_long_get(target + offset, source + offset, N_ELEMS, nextpe);

  /* fprintf(stderr,"Thread %lu done second get\n",tid); */
  pthread_barrier_wait(&fencebar);
  if (tid == 0) rocshmem_barrier_all();
  pthread_barrier_wait(&fencebar);
  /* fprintf(stderr,"Done thread %lu\n",tid); */

  return 0;
}

int main(int argc, char *argv[]) {
  int i;

  int tl;
  rocshmem_init_thread(ROCSHMEM_THREAD_MULTIPLE, &tl);

  if (tl != ROCSHMEM_THREAD_MULTIPLE) {
    printf("Init failed (requested thread level %d, got %d)\n",
           ROCSHMEM_THREAD_MULTIPLE, tl);
    rocshmem_global_exit(1);
  }

  if (rocshmem_n_pes() == 1) {
    printf("%s: Requires number of PEs > 1\n", argv[0]);
    rocshmem_finalize();
    return 0;
  }

  source = (long *)rocshmem_malloc(N_THREADS * N_ELEMS * sizeof(long));
  target = (long *)rocshmem_malloc(N_THREADS * N_ELEMS * sizeof(long));

  for (i = 0; i < N_THREADS * N_ELEMS; ++i) {
    source[i] = i + 1;
  }

  pthread_t threads[N_THREADS];

  pthread_barrier_init(&fencebar, NULL, N_THREADS);

  fprintf(stderr, "Starting threads\n");
  for (i = 0; i < N_THREADS; ++i) {
    /* fprintf(stderr,"Starting thread %d\n",i); */
    ptrdiff_t tid = i;
    pthread_create(&threads[i], NULL, &roundrobin, (void *)tid);
  }

  for (i = 0; i < N_THREADS; ++i) {
    pthread_join(threads[i], NULL);
  }
  pthread_barrier_destroy(&fencebar);

  if (0 != memcmp(source, target, sizeof(long) * N_THREADS * N_ELEMS)) {
    fprintf(stderr, "[%d] Src & Target mismatch?\n", rocshmem_my_pe());
    for (i = 0; i < 10; ++i) {
      printf("%ld,%ld ", source[i], target[i]);
    }
    printf("\n");
    rocshmem_global_exit(1);
  }

  rocshmem_free(source);
  rocshmem_free(target);

  rocshmem_finalize();

  return 0;
}
