// utils.c — Utility functions (Arena allocator, random, debug)
// VoxCPM2-C Project
// License: Apache-2.0

#include "tensor.h"
#include "voxcpm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════
 * Arena Allocator
 *
 * O(1) allocation and reset — ideal for per-forward-pass temporary
 * tensors. No individual free needed; just arena_reset() after each pass.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    char*   buffer;      /* pre-allocated memory pool */
    size_t  capacity;    /* total pool size */
    size_t  offset;      /* current allocation offset */
} Arena;

/* Create a new arena with the given capacity.
 * The buffer is heap-allocated. Caller must call arena_free. */
Arena* arena_create(size_t capacity) {
    Arena* arena = (Arena*)calloc(1, sizeof(Arena));
    if (!arena) return NULL;

    arena->buffer = (char*)calloc(capacity, 1);
    if (!arena->buffer && capacity > 0) {
        free(arena);
        return NULL;
    }

    arena->capacity = capacity;
    arena->offset = 0;
    return arena;
}

/* Initialize a stack-allocated arena. */
void arena_init(Arena* arena, void* buffer, size_t capacity) {
    if (!arena) return;
    arena->buffer = (char*)buffer;
    arena->capacity = capacity;
    arena->offset = 0;
}

/* Allocate size bytes from the arena.
 * Returns NULL if out of capacity. */
void* arena_alloc(Arena* arena, size_t size) {
    if (!arena || size == 0) return NULL;

    // Align to 8 bytes
    size_t aligned = (size + 7) & ~7;

    if (arena->offset + aligned > arena->capacity) {
        fprintf(stderr, "[Arena] OOM: requested %zu, remaining %zu\n",
                aligned, arena->capacity - arena->offset);
        return NULL;
    }

    void* ptr = arena->buffer + arena->offset;
    arena->offset += aligned;
    return ptr;
}

/* Allocate with explicit alignment (e.g., 32 for AVX, 128 for CUDA). */
void* arena_alloc_align(Arena* arena, size_t size, size_t align) {
    if (!arena || size == 0 || align == 0) return NULL;

    // Align current offset
    size_t mask = align - 1;
    size_t aligned_offset = (arena->offset + mask) & ~mask;

    // Align size
    size_t aligned_size = (size + mask) & ~mask;

    if (aligned_offset + aligned_size > arena->capacity) {
        fprintf(stderr, "[Arena] OOM (align): requested %zu, remaining %zu\n",
                aligned_size, arena->capacity - aligned_offset);
        return NULL;
    }

    void* ptr = arena->buffer + aligned_offset;
    arena->offset = aligned_offset + aligned_size;
    return ptr;
}

/* Reset the arena (O(1) — just reset offset). */
void arena_reset(Arena* arena) {
    if (arena) arena->offset = 0;
}

/* Free all arena resources. */
void arena_free(Arena* arena) {
    if (!arena) return;
    free(arena->buffer);
    arena->buffer = NULL;
    arena->capacity = 0;
    arena->offset = 0;
}

/* Get remaining capacity. */
size_t arena_remaining(const Arena* arena) {
    return arena ? arena->capacity - arena->offset : 0;
}

/* ═══════════════════════════════════════════════════════════════
 * Create tensor from arena (convenience)
 *
 * Allocates both the Tensor struct and data from the arena.
 * ═══════════════════════════════════════════════════════════════ */

Tensor* tensor_create_arena(Arena* arena, int ndim, const int* shape) {
    if (!arena || !shape) return NULL;

    // Validate shape
    if (ndim < 0 || ndim > TENSOR_MAX_DIMS) return NULL;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) return NULL;
    }

    // Allocate tensor struct from arena
    Tensor* t = (Tensor*)arena_alloc(arena, sizeof(Tensor));
    if (!t) return NULL;

    // Allocate shape array from arena
    t->shape = (int*)arena_alloc(arena, (size_t)ndim * sizeof(int));
    if (!t->shape) return NULL;

    memcpy(t->shape, shape, (size_t)ndim * sizeof(int));
    t->ndim = ndim;

    // Compute strides
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= shape[i];
    }
    t->size = (size_t)stride;

    // Allocate data from arena
    t->data = (float*)arena_alloc(arena, t->size * sizeof(float));
    if (!t->data && t->size > 0) return NULL;

    t->is_cuda = false;
    t->is_owned = false;  // Arena owns the memory
    t->name[0] = '\0';

    return t;
}

/* ═══════════════════════════════════════════════════════════════
 * Debug assertion macro (debug builds only)
 * ═══════════════════════════════════════════════════════════════ */

#ifdef VOXCPM_DEBUG
#include <assert.h>
#define TENSOR_ASSERT_SHAPE(t, ...) do { \
    int expected[] = { __VA_ARGS__ }; \
    int n = sizeof(expected) / sizeof(expected[0]); \
    for (int _i = 0; _i < n && _i < (t)->ndim; _i++) { \
        assert((t)->shape[_i] == expected[_i]); \
    } \
} while(0)
#else
#define TENSOR_ASSERT_SHAPE(t, ...) ((void)0)
#endif
