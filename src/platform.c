// platform.c — Platform abstraction implementation
// VoxCPM2-C Project
// License: Apache-2.0

#include "platform.h"
#include "voxcpm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─── Platform Detection ───

#if defined(_WIN32) || defined(_WIN64)
    #define PLATFORM_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#elif defined(__APPLE__)
    #define PLATFORM_MACOS
    #include <mach/mach_time.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <pthread.h>
#else
    #define PLATFORM_LINUX
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
    #include <pthread.h>
#endif

/* ═══════════════════════════════════════════════════════════════
 * Memory-Mapped File I/O
 * ═══════════════════════════════════════════════════════════════ */

// Internal structure to store Windows mmap handles
#ifdef PLATFORM_WINDOWS
typedef struct {
    HANDLE hMap;
    HANDLE hFile;
} MmapImpl;
#endif

#ifdef PLATFORM_WINDOWS

MmapFile* mmap_open(const char* path) {
    if (!path) return NULL;

    HANDLE hFile = CreateFileA(
        path, GENERIC_READ, FILE_SHARE_READ,
        NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL
    );
    if (hFile == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) {
        CloseHandle(hFile);
        return NULL;
    }

    HANDLE hMap = CreateFileMappingA(
        hFile, NULL, PAGE_READONLY, 0, 0, NULL
    );
    if (!hMap) {
        CloseHandle(hFile);
        return NULL;
    }

    void* addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!addr) {
        CloseHandle(hMap);
        CloseHandle(hFile);
        return NULL;
    }

    MmapFile* mf = (MmapFile*)calloc(1, sizeof(MmapFile));
    if (!mf) {
        UnmapViewOfFile(addr);
        CloseHandle(hMap);
        CloseHandle(hFile);
        return NULL;
    }

    MmapImpl* impl = (MmapImpl*)calloc(1, sizeof(MmapImpl));
    if (!impl) {
        UnmapViewOfFile(addr);
        CloseHandle(hMap);
        CloseHandle(hFile);
        free(mf);
        return NULL;
    }
    impl->hMap = hMap;
    impl->hFile = hFile;

    mf->addr = addr;
    mf->size = (size_t)file_size.QuadPart;
    mf->_impl = impl;
    return mf;
}

void mmap_close(MmapFile* mf) {
    if (!mf) return;
    if (mf->addr) UnmapViewOfFile(mf->addr);
    if (mf->_impl) {
        MmapImpl* impl = (MmapImpl*)mf->_impl;
        if (impl->hMap) CloseHandle(impl->hMap);
        if (impl->hFile) CloseHandle(impl->hFile);
        free(impl);
    }
    free(mf);
}

#else // POSIX (Linux / macOS)

MmapFile* mmap_open(const char* path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    size_t size = (size_t)st.st_size;
    void* addr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (addr == MAP_FAILED) return NULL;

    MmapFile* mf = (MmapFile*)calloc(1, sizeof(MmapFile));
    if (!mf) {
        munmap(addr, size);
        return NULL;
    }

    mf->addr = addr;
    mf->size = size;
    mf->_impl = NULL;
    return mf;
}

void mmap_close(MmapFile* mf) {
    if (!mf) return;
    if (mf->addr) munmap(mf->addr, mf->size);
    free(mf);
}

#endif // PLATFORM_WINDOWS

void* file_read_all(const char* path, size_t* out_size) {
    if (!path) return NULL;

    MmapFile* mf = mmap_open(path);
    if (!mf) return NULL;

    void* buf = malloc(mf->size ? mf->size : 1);
    if (!buf) {
        mmap_close(mf);
        return NULL;
    }

    memcpy(buf, mf->addr, mf->size);
    if (out_size) *out_size = mf->size;
    mmap_close(mf);
    return buf;
}

/* ═══════════════════════════════════════════════════════════════
 * High-Resolution Timer
 * ═══════════════════════════════════════════════════════════════ */

#ifdef PLATFORM_WINDOWS

Timer timer_start(void) {
    Timer t;
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    t.start_sec = (double)count.QuadPart / (double)freq.QuadPart;
    t._impl = NULL;
    return t;
}

double timer_elapsed(const Timer* timer) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    double now = (double)count.QuadPart / (double)freq.QuadPart;
    return now - timer->start_sec;
}

double time_now_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

#elif defined(PLATFORM_MACOS)

Timer timer_start(void) {
    Timer t;
    t.start_sec = time_now_sec();
    t._impl = NULL;
    return t;
}

double timer_elapsed(const Timer* timer) {
    return time_now_sec() - timer->start_sec;
}

double time_now_sec(void) {
    static mach_timebase_info_data_t info = {0};
    if (info.denom == 0) mach_timebase_info(&info);
    uint64_t now = mach_absolute_time();
    return (double)now * (double)info.numer / (double)info.denom / 1e9;
}

#else // Linux

Timer timer_start(void) {
    Timer t;
    t.start_sec = time_now_sec();
    t._impl = NULL;
    return t;
}

double timer_elapsed(const Timer* timer) {
    return time_now_sec() - timer->start_sec;
}

double time_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

#endif

/* ═══════════════════════════════════════════════════════════════
 * Thread Pool (simple implementation)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef PLATFORM_WINDOWS

// Windows thread pool using CreateThread
typedef struct {
    thread_func_t func;
    void* arg;
} Task;

typedef struct ThreadPool {
    int n_threads;
    int running;
    // Simple: just create threads and join
    HANDLE* threads;
} ThreadPool;

typedef struct {
    ThreadPool* pool;
    int id;
} ThreadArg;

static DWORD WINAPI worker_thread(LPVOID arg) {
    // For basic implementation: idle
    return 0;
}

ThreadPool* thread_pool_create(int n_threads) {
    if (n_threads <= 0) {
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        n_threads = (int)sysinfo.dwNumberOfProcessors;
        if (n_threads < 1) n_threads = 1;
    }

    ThreadPool* pool = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    if (!pool) return NULL;

    pool->n_threads = n_threads;
    pool->running = 1;
    pool->threads = (HANDLE*)calloc((size_t)n_threads, sizeof(HANDLE));
    if (!pool->threads) {
        free(pool);
        return NULL;
    }

    for (int i = 0; i < n_threads; i++) {
        ThreadArg* arg = (ThreadArg*)malloc(sizeof(ThreadArg));
        arg->pool = pool;
        arg->id = i;
        pool->threads[i] = CreateThread(NULL, 0, worker_thread, arg, 0, NULL);
    }

    return pool;
}

int thread_pool_enqueue(ThreadPool* pool, thread_func_t func, void* arg) {
    if (!pool || !func) return -1;
    // Basic: just call synchronously for now
    // Full async queue will be implemented in Phase 4
    func(arg);
    return 0;
}

void thread_pool_wait(ThreadPool* pool) {
    if (!pool) return;
    // Threads are already joined in free
}

int thread_pool_count(const ThreadPool* pool) {
    return pool ? pool->n_threads : 1;
}

void thread_pool_free(ThreadPool* pool) {
    if (!pool) return;
    pool->running = 0;
    for (int i = 0; i < pool->n_threads; i++) {
        if (pool->threads[i]) {
            WaitForSingleObject(pool->threads[i], INFINITE);
            CloseHandle(pool->threads[i]);
        }
    }
    free(pool->threads);
    free(pool);
}

#else // POSIX

typedef struct {
    thread_func_t func;
    void* arg;
} Task;

typedef struct ThreadPool {
    int n_threads;
    int running;
    pthread_t* threads;
} ThreadPool;

static void* posix_worker(void* arg) {
    (void)arg;
    return NULL;
}

ThreadPool* thread_pool_create(int n_threads) {
    if (n_threads <= 0) {
        n_threads = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (n_threads < 1) n_threads = 1;
    }

    ThreadPool* pool = (ThreadPool*)calloc(1, sizeof(ThreadPool));
    if (!pool) return NULL;

    pool->n_threads = n_threads;
    pool->running = 1;
    pool->threads = (pthread_t*)calloc((size_t)n_threads, sizeof(pthread_t));
    if (!pool->threads) {
        free(pool);
        return NULL;
    }

    for (int i = 0; i < n_threads; i++) {
        pthread_create(&pool->threads[i], NULL, posix_worker, NULL);
        pthread_detach(pool->threads[i]);
    }

    return pool;
}

int thread_pool_enqueue(ThreadPool* pool, thread_func_t func, void* arg) {
    if (!pool || !func) return -1;
    // Basic: synchronous execution
    // Full async queue in Phase 4
    func(arg);
    return 0;
}

void thread_pool_wait(ThreadPool* pool) {
    (void)pool;
}

int thread_pool_count(const ThreadPool* pool) {
    return pool ? pool->n_threads : 1;
}

void thread_pool_free(ThreadPool* pool) {
    if (!pool) return;
    pool->running = 0;
    free(pool->threads);
    free(pool);
}

#endif

/* ═══════════════════════════════════════════════════════════════
 * Mutex
 * ═══════════════════════════════════════════════════════════════ */

#ifdef PLATFORM_WINDOWS

struct Mutex {
    CRITICAL_SECTION cs;
};

Mutex* mutex_create(void) {
    Mutex* m = (Mutex*)calloc(1, sizeof(Mutex));
    if (m) InitializeCriticalSection(&m->cs);
    return m;
}

void mutex_lock(Mutex* m) {
    if (m) EnterCriticalSection(&m->cs);
}

void mutex_unlock(Mutex* m) {
    if (m) LeaveCriticalSection(&m->cs);
}

void mutex_free(Mutex* m) {
    if (m) {
        DeleteCriticalSection(&m->cs);
        free(m);
    }
}

#else // POSIX

struct Mutex {
    pthread_mutex_t mutex;
};

Mutex* mutex_create(void) {
    Mutex* m = (Mutex*)calloc(1, sizeof(Mutex));
    if (m) pthread_mutex_init(&m->mutex, NULL);
    return m;
}

void mutex_lock(Mutex* m) {
    if (m) pthread_mutex_lock(&m->mutex);
}

void mutex_unlock(Mutex* m) {
    if (m) pthread_mutex_unlock(&m->mutex);
}

void mutex_free(Mutex* m) {
    if (m) {
        pthread_mutex_destroy(&m->mutex);
        free(m);
    }
}

#endif

/* ═══════════════════════════════════════════════════════════════
 * File System
 * ═══════════════════════════════════════════════════════════════ */

bool file_exists(const char* path) {
    if (!path) return false;
#ifdef PLATFORM_WINDOWS
    DWORD attr = GetFileAttributesA(path);
    return (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY));
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
#endif
}

size_t file_size(const char* path) {
    if (!path) return 0;
#ifdef PLATFORM_WINDOWS
    WIN32_FILE_ATTRIBUTE_DATA info;
    if (!GetFileAttributesExA(path, GetFileExInfoStandard, &info))
        return 0;
    LARGE_INTEGER size;
    size.LowPart = info.nFileSizeLow;
    size.HighPart = info.nFileSizeHigh;
    return (size_t)size.QuadPart;
#else
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (size_t)st.st_size;
#endif
}

const char* file_extension(const char* path) {
    if (!path) return "";
    const char* dot = strrchr(path, '.');
    return dot ? dot : "";
}

char* path_join(const char* dir, const char* file) {
    if (!dir || !file) return NULL;
    size_t dlen = strlen(dir);
    size_t flen = strlen(file);
    int needs_sep = (dlen > 0 && dir[dlen-1] != '/' && dir[dlen-1] != '\\');
    char* result = (char*)malloc(dlen + flen + 2);
    if (!result) return NULL;
    strcpy(result, dir);
    if (needs_sep) {
#ifdef PLATFORM_WINDOWS
        result[dlen] = '\\';
#else
        result[dlen] = '/';
#endif
        result[dlen+1] = '\0';
    }
    strcat(result, file);
    return result;
}

/* ═══════════════════════════════════════════════════════════════
 * Byte Order
 * ═══════════════════════════════════════════════════════════════ */

bool is_little_endian(void) {
    uint16_t test = 1;
    return *(uint8_t*)&test == 1;
}

uint16_t htole16(uint16_t x) {
    return is_little_endian() ? x : ((x >> 8) | (x << 8));
}

uint32_t htole32(uint32_t x) {
    if (is_little_endian()) return x;
    return ((x >> 24) & 0xFF) | ((x >> 8) & 0xFF00) |
           ((x << 8) & 0xFF0000) | ((x << 24) & 0xFF000000);
}

uint64_t htole64(uint64_t x) {
    if (is_little_endian()) return x;
    return ((uint64_t)htole32((uint32_t)(x >> 32)) << 32) |
           (uint64_t)htole32((uint32_t)x);
}

uint16_t le16toh(uint16_t x) { return htole16(x); }
uint32_t le32toh(uint32_t x) { return htole32(x); }
uint64_t le64toh(uint64_t x) { return htole64(x); }

/* ═══════════════════════════════════════════════════════════════
 * Logging
 * ═══════════════════════════════════════════════════════════════ */

static VoxCPMLogLevel g_log_level = VOXCPM_LOG_INFO;
static Mutex* g_log_mutex = NULL;

void voxcpm_set_log_level(VoxCPMLogLevel level) {
    g_log_level = level;
}

VoxCPMLogLevel voxcpm_get_log_level(void) {
    return g_log_level;
}

void platform_log(int level, const char* file, int line,
                   const char* fmt, ...)
{
    if (level > (int)g_log_level) return;

    // Lazy init mutex
    if (!g_log_mutex) {
        g_log_mutex = mutex_create();
    }
    mutex_lock(g_log_mutex);

    const char* prefix = "";
    switch (level) {
        case 0: prefix = ""; break;
        case 1: prefix = "[ERROR]"; break;
        case 2: prefix = "[WARN]"; break;
        case 3: prefix = "[INFO]"; break;
        case 4: prefix = "[DEBUG]"; break;
    }

    fprintf(stderr, "%s %s:%d: ", prefix, file, line);
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    fflush(stderr);

    mutex_unlock(g_log_mutex);
}


