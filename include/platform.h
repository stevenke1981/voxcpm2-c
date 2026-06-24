#ifndef PLATFORM_H
#define PLATFORM_H

/*
 * platform.h — Platform abstraction layer
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * Provides cross-platform wrappers for:
 *   - Memory-mapped file I/O
 *   - Thread/Task parallelism
 *   - High-resolution timing
 *   - Dynamic library loading
 *   - File system operations
 */

#include "tensor.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Memory-Mapped File I/O
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    void*  addr;        /* mapped address */
    size_t size;        /* file size */
    void*  _impl;       /* platform-specific handle */
} MmapFile;

/* mmap_open: Map a file into memory.
 * Returns NULL on error; sets *size to file size on success.
 * The mapping is read-only. */
MmapFile* mmap_open(const char* path);

/* mmap_close: Unmap the file. Safe to call with NULL. */
void mmap_close(MmapFile* mf);

/* Convenience: read entire file into heap-allocated buffer.
 * Caller must free() the returned buffer. Returns NULL on error. */
void* file_read_all(const char* path, size_t* out_size);

/* ═══════════════════════════════════════════════════════════════
 * High-Resolution Timer
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    double   start_sec;     /* start time in seconds */
    void*    _impl;
} Timer;

/* Start the timer. Returns current time as a handle. */
Timer timer_start(void);

/* Get elapsed time in seconds since timer_start. */
double timer_elapsed(const Timer* timer);

/* Get current wall time in seconds (monotonic). */
double time_now_sec(void);

/* ═══════════════════════════════════════════════════════════════
 * Thread Pool (simple pthreads/Win32 wrapper)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct ThreadPool ThreadPool;

/* Create a thread pool with n_threads worker threads.
 * n_threads = 0 means use hardware concurrency. */
ThreadPool* thread_pool_create(int n_threads);

/* Enqueue a task:
 *   func(arg) will be called by a worker thread.
 *   Returns a task ID (>= 0) on success, -1 on error. */
typedef void (*thread_func_t)(void* arg);
int thread_pool_enqueue(ThreadPool* pool, thread_func_t func, void* arg);

/* Wait for all enqueued tasks to complete. */
void thread_pool_wait(ThreadPool* pool);

/* Get number of threads. */
int thread_pool_count(const ThreadPool* pool);

/* Destroy the thread pool. Safe to call with NULL. */
void thread_pool_free(ThreadPool* pool);

/* ═══════════════════════════════════════════════════════════════
 * Simple Mutex / Lock
 * ═══════════════════════════════════════════════════════════════ */

typedef struct Mutex Mutex;

Mutex* mutex_create(void);
void mutex_lock(Mutex* m);
void mutex_unlock(Mutex* m);
void mutex_free(Mutex* m);

/* ═══════════════════════════════════════════════════════════════
 * File System Utilities
 * ═══════════════════════════════════════════════════════════════ */

/* Check if a file exists. */
bool file_exists(const char* path);

/* Get file size. Returns 0 on error. */
size_t file_size(const char* path);

/* Get the file extension (pointer within path, or empty string). */
const char* file_extension(const char* path);

/* Join path components (allocates new string, caller must free). */
char* path_join(const char* dir, const char* file);

/* ═══════════════════════════════════════════════════════════════
 * Byte Order
 * ═══════════════════════════════════════════════════════════════ */

/* Detect if system is little-endian. */
bool is_little_endian(void);

/* Host-to-little-endian conversions (no-op on little-endian). */
uint16_t htole16(uint16_t x);
uint32_t htole32(uint32_t x);
uint64_t htole64(uint64_t x);
uint16_t le16toh(uint16_t x);
uint32_t le32toh(uint32_t x);
uint64_t le64toh(uint64_t x);

/* ═══════════════════════════════════════════════════════════════
 * Logging
 * ═══════════════════════════════════════════════════════════════ */

/* Log a message with the current log level setting.
 * Uses voxcpm_set_log_level to control verbosity. */
void platform_log(int level, const char* file, int line,
                   const char* fmt, ...);

#define LOG_ERROR(fmt, ...) \
    platform_log(VOXCPM_LOG_ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) \
    platform_log(VOXCPM_LOG_WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    platform_log(VOXCPM_LOG_INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) \
    platform_log(VOXCPM_LOG_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_H */
