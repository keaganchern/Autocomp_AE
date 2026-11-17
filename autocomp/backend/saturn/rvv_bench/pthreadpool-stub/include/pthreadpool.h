#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct pthreadpool* pthreadpool_t;

/* Real pthreadpool.h has 30+ distinct task signatures; none are called
 * at runtime in the spike single-kernel build, but xnnpack/compute.h
 * declares struct fields of each type so the typedef names must exist.
 * Aliasing all to one void(*)(void*) makes field decls parse. Operator
 * sources that would assign function pointers here (operator-run.c,
 * runtime.c) are excluded by XNNPACK_BUILD_LIBRARY=OFF. */
typedef void (*pthreadpool_generic_task_t)(void*);

#define _PTP_STUB_TASK(name) typedef pthreadpool_generic_task_t name

_PTP_STUB_TASK(pthreadpool_task_1d_t);
_PTP_STUB_TASK(pthreadpool_task_1d_with_thread_t);
_PTP_STUB_TASK(pthreadpool_task_1d_tile_1d_t);
_PTP_STUB_TASK(pthreadpool_task_1d_tile_1d_dynamic_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_1d_tile_1d_dynamic_with_id_with_thread_t);

_PTP_STUB_TASK(pthreadpool_task_2d_t);
_PTP_STUB_TASK(pthreadpool_task_2d_with_thread_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_1d_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_1d_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_1d_dynamic_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_1d_dynamic_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_1d_dynamic_with_id_with_thread_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_2d_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_2d_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_2d_dynamic_t);
_PTP_STUB_TASK(pthreadpool_task_2d_tile_2d_dynamic_with_id_t);

_PTP_STUB_TASK(pthreadpool_task_3d_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_1d_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_1d_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_1d_with_thread_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_1d_with_id_with_thread_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_1d_dynamic_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_1d_dynamic_with_id_with_thread_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_2d_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_2d_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_2d_dynamic_t);
_PTP_STUB_TASK(pthreadpool_task_3d_tile_2d_dynamic_with_id_t);

_PTP_STUB_TASK(pthreadpool_task_4d_t);
_PTP_STUB_TASK(pthreadpool_task_4d_tile_2d_t);
_PTP_STUB_TASK(pthreadpool_task_4d_tile_2d_with_id_t);
_PTP_STUB_TASK(pthreadpool_task_4d_tile_2d_dynamic_t);
_PTP_STUB_TASK(pthreadpool_task_4d_tile_2d_dynamic_with_id_t);

_PTP_STUB_TASK(pthreadpool_task_5d_t);
_PTP_STUB_TASK(pthreadpool_task_5d_tile_2d_t);
_PTP_STUB_TASK(pthreadpool_task_6d_tile_2d_t);

#undef _PTP_STUB_TASK

#ifdef __cplusplus
}
#endif
