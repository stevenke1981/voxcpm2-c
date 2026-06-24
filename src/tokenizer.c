// tokenizer.c — BPE tokenizer implementation
// VoxCPM2-C Project
// License: Apache-2.0
//
// Byte-level BPE tokenizer (GPT-2 style):
//   1. Load vocabulary from file or weight index
//   2. Encode UTF-8 text to token IDs using BPE merge rules
//   3. Decode token IDs back to UTF-8 text
//
// Special tokens: [PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3

#include "tokenizer.h"
#include "model.h"
#include "platform.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>

/* ═══════════════════════════════════════════════════════════════
 * Constants
 * ═══════════════════════════════════════════════════════════════ */

/* Hash table load factor: resizes when count > capacity * LOAD_FACTOR */
#define TOKENIZER_LOAD_FACTOR 0.72f

/* Initial hash capacity (must be prime) */
#define TOKENIZER_HASH_CAPACITY 147073

/* Maximum bytes per word for BPE encoding working buffer */
#define TOKENIZER_MAX_WORD_BYTES 4096

/* Maximum BPE merge iterations per word (safety limit) */
#define TOKENIZER_MAX_MERGE_ITERS 10000

/* ═══════════════════════════════════════════════════════════════
 * Hash Table Entries
 * ═══════════════════════════════════════════════════════════════ */

/* Entry for token string -> ID mapping */
typedef struct {
    char*  token;          /* owned copy of token string */
    int    id;
    bool   occupied;
    bool   deleted;        /* for tombstones (not used currently) */
} TokenHashEntry;

/* Entry for BPE pair -> (rank, new_id) mapping */
typedef struct {
    uint64_t key;          /* packed pair: ((uint64_t)left << 32) | (uint32_t)right */
    int      rank;         /* merge priority (lower = merge earlier) */
    int      new_id;       /* resulting token ID after merge */
    bool     occupied;
} PairHashEntry;

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Structure (opaque)
 * ═══════════════════════════════════════════════════════════════ */

struct Tokenizer {
    TokenizerType type;

    /* --- Vocabulary --- */
    char**  vocab;            /* [vocab_size] null-terminated strings (owned) */
    int     vocab_size;
    int     max_token_len;

    /* Token string -> ID hash table */
    TokenHashEntry* hash_table;
    int             hash_capacity;
    int             hash_count;

    /* Byte value -> initial token ID mapping */
    int     byte_to_token[256];
    bool    has_byte_mapping;

    /* --- BPE Merge Table --- */
    PairHashEntry* pair_ranks;   /* hash table for pair->(rank, new_id) */
    int            pair_capacity;
    int            pair_count;
};

/* ═══════════════════════════════════════════════════════════════
 * Hash Helpers
 * ═══════════════════════════════════════════════════════════════ */

/* Simple string hash (djb2 variant) */
static uint32_t str_hash(const char* s, int len) {
    uint32_t h = 5381;
    for (int i = 0; i < len; i++) {
        h = ((h << 5) + h) + (unsigned char)s[i];
    }
    return h;
}

/* Pair hash: (left * large_prime + right) */
static uint32_t pair_hash(int left, int right) {
    return (uint32_t)((uint64_t)left * 73459u + (uint64_t)right);
}

/* Find next prime >= n (simple, sufficient for our sizes) */
static int next_prime(int n) {
    /* small prime table for increments */
    static const int primes[] = {
        7, 13, 31, 61, 127, 257, 509, 1021, 2053, 4099,
        8209, 16411, 32771, 65537, 131101, 262147, 524309,
        1048583, 2097169, 4194319, 8388617, 16777259,
        33554467, 67108879, 134217757, 268435459
    };
    for (size_t i = 0; i < sizeof(primes) / sizeof(primes[0]); i++) {
        if (primes[i] >= n) return primes[i];
    }
    return n | 1; /* odd fallback */
}

/* ═══════════════════════════════════════════════════════════════
 * Token Hash Table Operations
 * ═══════════════════════════════════════════════════════════════ */

/* Allocate and initialize token hash table */
static TokenHashEntry* token_hash_create(int capacity) {
    TokenHashEntry* ht = (TokenHashEntry*)calloc((size_t)capacity, sizeof(TokenHashEntry));
    return ht;
}

/* Free token hash table (frees owned token strings) */
static void token_hash_free(TokenHashEntry* ht, int capacity) {
    if (!ht) return;
    for (int i = 0; i < capacity; i++) {
        if (ht[i].occupied && ht[i].token) {
            free(ht[i].token);
            ht[i].token = NULL;
        }
    }
    free(ht);
}

/* Insert a token->ID mapping into the hash table (copies token string) */
static bool token_hash_insert(TokenHashEntry** ht_ptr, int* capacity, int* count,
                                const char* token, int id) {
    if (!ht_ptr || !*ht_ptr) return false;

    TokenHashEntry* ht = *ht_ptr;
    int cap = *capacity;

    /* Check load factor and resize if needed */
    if ((float)(*count + 1) > (float)cap * TOKENIZER_LOAD_FACTOR) {
        int new_cap = next_prime(cap * 2);
        TokenHashEntry* new_ht = token_hash_create(new_cap);
        if (!new_ht) return false;

        /* Rehash all existing entries */
        for (int i = 0; i < cap; i++) {
            if (ht[i].occupied && ht[i].token) {
                uint32_t h = str_hash(ht[i].token, (int)strlen(ht[i].token));
                uint32_t idx = h % (uint32_t)new_cap;
                while (new_ht[idx].occupied) {
                    idx = (idx + 1) % (uint32_t)new_cap;
                }
                new_ht[idx].occupied = true;
                new_ht[idx].token = ht[i].token;   /* transfer ownership */
                new_ht[idx].id = ht[i].id;
                ht[i].token = NULL;                /* prevent double-free */
            }
        }

        free(ht);
        *ht_ptr = new_ht;
        ht = new_ht;
        *capacity = new_cap;
        cap = new_cap;
    }

    /* Insert new entry with linear probing */
    int token_len = (int)strlen(token);
    uint32_t h = str_hash(token, token_len);
    uint32_t idx = h % (uint32_t)cap;

    while (ht[idx].occupied) {
        /* If token already exists, update id (shouldn't happen for unique tokens) */
        if (ht[idx].token && strcmp(ht[idx].token, token) == 0) {
            ht[idx].id = id;
            return true;
        }
        idx = (idx + 1) % (uint32_t)cap;
    }

    ht[idx].occupied = true;
    ht[idx].token = (char*)malloc((size_t)(token_len + 1));
    if (!ht[idx].token) return false;
    memcpy(ht[idx].token, token, (size_t)(token_len + 1));
    ht[idx].id = id;
    (*count)++;
    return true;
}

/* Lookup token ID by string. Returns TOKEN_UNK_ID if not found. */
static int token_hash_lookup(const TokenHashEntry* ht, int capacity,
                              const char* token, int token_len) {
    if (!ht || capacity <= 0 || !token || token_len < 0) return TOKEN_UNK_ID;
    if (token_len == 0) token_len = (int)strlen(token);

    uint32_t h = str_hash(token, token_len);
    uint32_t idx = h % (uint32_t)capacity;

    for (int i = 0; i < capacity; i++) {
        const TokenHashEntry* e = &ht[idx];
        if (!e->occupied) break;  /* empty slot = end of probe chain */
        if (e->token && (int)strlen(e->token) == token_len &&
            memcmp(e->token, token, (size_t)token_len) == 0) {
            return e->id;
        }
        idx = (idx + 1) % (uint32_t)capacity;
    }
    return TOKEN_UNK_ID;
}

/* ═══════════════════════════════════════════════════════════════
 * Pair Hash Table Operations
 * ═══════════════════════════════════════════════════════════════ */

static PairHashEntry* pair_hash_create(int capacity) {
    return (PairHashEntry*)calloc((size_t)capacity, sizeof(PairHashEntry));
}

/* Build the merge lookup from merge training data.
 * For each BPE merge, the pair (left_id, right_id) maps to
 * (new_token_id, merge_rank) where lower rank = higher priority. */
static bool pair_hash_insert(PairHashEntry** ht_ptr, int* capacity, int* count,
                               int left, int right, int new_id, int rank) {
    if (!ht_ptr || !*ht_ptr) return false;

    PairHashEntry* ht = *ht_ptr;
    int cap = *capacity;

    if ((float)(*count + 1) > (float)cap * TOKENIZER_LOAD_FACTOR) {
        int new_cap = next_prime(cap * 2);
        PairHashEntry* new_ht = pair_hash_create(new_cap);
        if (!new_ht) return false;

        for (int i = 0; i < cap; i++) {
            if (ht[i].occupied) {
                uint64_t k = ht[i].key;
                uint32_t h = pair_hash((int)(k >> 32), (int)(k & 0xFFFFFFFF));
                uint32_t ni = h % (uint32_t)new_cap;
                while (new_ht[ni].occupied) ni = (ni + 1) % (uint32_t)new_cap;
                new_ht[ni] = ht[i];
            }
        }

        free(ht);
        *ht_ptr = new_ht;
        ht = new_ht;
        *capacity = new_cap;
        cap = new_cap;
    }

    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    uint32_t h = pair_hash(left, right);
    uint32_t idx = h % (uint32_t)cap;

    while (ht[idx].occupied) {
        if (ht[idx].key == key) {
            /* Update existing entry */
            ht[idx].rank = rank;
            ht[idx].new_id = new_id;
            return true;
        }
        idx = (idx + 1) % (uint32_t)cap;
    }

    ht[idx].occupied = true;
    ht[idx].key = key;
    ht[idx].rank = rank;
    ht[idx].new_id = new_id;
    (*count)++;
    return true;
}

/* Lookup a BPE merge. Returns false if pair not found. */
static bool pair_hash_lookup(const PairHashEntry* ht, int capacity,
                               int left, int right, int* out_rank, int* out_new_id) {
    if (!ht || capacity <= 0) return false;

    uint64_t key = ((uint64_t)(uint32_t)left << 32) | (uint32_t)right;
    uint32_t h = pair_hash(left, right);
    uint32_t idx = h % (uint32_t)capacity;

    for (int i = 0; i < capacity; i++) {
        const PairHashEntry* e = &ht[idx];
        if (!e->occupied) return false;
        if (e->key == key) {
            if (out_rank) *out_rank = e->rank;
            if (out_new_id) *out_new_id = e->new_id;
            return true;
        }
        idx = (idx + 1) % (uint32_t)capacity;
    }
    return false;
}

/* ═══════════════════════════════════════════════════════════════
 * Byte-to-Token Mapping
 * ═══════════════════════════════════════════════════════════════ */

/* Build the byte-to-token mapping by scanning the vocabulary.
 * For each token that is exactly 1 byte long, map that byte value to its ID.
 * All unmapped bytes default to TOKEN_UNK_ID. */
static void build_byte_mapping(Tokenizer* tok) {
    for (int i = 0; i < 256; i++) {
        tok->byte_to_token[i] = TOKEN_UNK_ID;
    }
    tok->has_byte_mapping = false;

    for (int id = 0; id < tok->vocab_size; id++) {
        const char* s = tok->vocab[id];
        if (!s) continue;
        size_t len = strlen(s);
        if (len == 1) {
            unsigned char b = (unsigned char)s[0];
            tok->byte_to_token[b] = id;
            tok->has_byte_mapping = true;
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 * BPE Merge Helper
 * ═══════════════════════════════════════════════════════════════ */

/* Apply BPE merges to a sequence of token IDs in-place.
 * Input: cur[0..cur_len-1] = initial token IDs.
 * Output: result[0..*out_len-1] = merged token IDs.
 * Returns the number of output tokens, or -1 on error.
 *
 * Algorithm: greedily find the pair with lowest rank, merge it, repeat. */
static int bpe_merge_sequence(
    const Tokenizer* tok,
    int* cur,              /* [cur_len] input/output tokens */
    int cur_len,
    int* out_tokens,       /* [max_out] output buffer */
    int max_out)
{
    if (!cur || !out_tokens || cur_len <= 0) return 0;

    int iter = 0;
    while (cur_len > 1 && iter < TOKENIZER_MAX_MERGE_ITERS) {
        iter++;

        /* Scan for the best (lowest rank) pair */
        int best_rank = INT_MAX;
        int best_pos = -1;
        int best_new_id = -1;

        for (int i = 0; i < cur_len - 1; i++) {
            int left = cur[i];
            int right = cur[i + 1];
            int rank, new_id;
            if (pair_hash_lookup(tok->pair_ranks, tok->pair_capacity,
                                  left, right, &rank, &new_id)) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_pos = i;
                    best_new_id = new_id;
                }
            }
        }

        if (best_pos < 0) {
            /* No more merges possible */
            break;
        }

        /* Merge: replace cur[best_pos] and cur[best_pos+1] with best_new_id */
        cur[best_pos] = best_new_id;
        /* Shift remaining elements left by 1 */
        int move_count = cur_len - best_pos - 2;
        if (move_count > 0) {
            memmove(&cur[best_pos + 1], &cur[best_pos + 2],
                    (size_t)move_count * sizeof(int));
        }
        cur_len--;
    }

    /* Copy result */
    int n = (cur_len < max_out) ? cur_len : max_out;
    memcpy(out_tokens, cur, (size_t)n * sizeof(int));
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * Pre-tokenization: Simple Whitespace Split
 * ═══════════════════════════════════════════════════════════════ */

/* A simple pre-tokenizer that splits on whitespace and punctuation boundaries.
 * Returns a list of word boundaries (start, end) within the input text.
 * Max out_words pairs. Returns number of words found. */
static int pre_tokenize(const char* text, int text_len,
                         int* out_starts, int* out_ends, int max_words) {
    if (!text || text_len <= 0 || !out_starts || !out_ends) return 0;

    int word_count = 0;
    int i = 0;

    while (i < text_len && word_count < max_words) {
        /* Skip whitespace */
        while (i < text_len) {
            unsigned char c = (unsigned char)text[i];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v')
                i++;
            else
                break;
        }

        if (i >= text_len) break;

        /* Start of word */
        int start = i;
        while (i < text_len) {
            unsigned char c = (unsigned char)text[i];
            /* Split on whitespace */
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v')
                break;
            i++;
        }

        if (i > start) {
            out_starts[word_count] = start;
            out_ends[word_count] = i;
            word_count++;
        }
    }

    return word_count;
}

/* ═══════════════════════════════════════════════════════════════
 * UTF-8 Byte Encoding
 * ═══════════════════════════════════════════════════════════════ */

/* Convert a word (UTF-8 substring) to initial token IDs using byte-to-token mapping.
 * Each byte of the word maps to an initial token ID.
 * Stores result in out_tokens. Returns number of tokens written, or -1 on error. */
static int word_to_byte_tokens(const Tokenizer* tok,
                                 const char* word_start, int word_len,
                                 int* out_tokens, int max_out) {
    if (!word_start || word_len <= 0 || !out_tokens) return 0;

    int n = word_len < max_out ? word_len : max_out;
    for (int i = 0; i < n; i++) {
        unsigned char b = (unsigned char)word_start[i];
        out_tokens[i] = tok->byte_to_token[b];
    }
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Creation — From Vocabulary Array
 * ═══════════════════════════════════════════════════════════════ */

Tokenizer* tokenizer_create_bpe(
    const char** vocab,
    int vocab_size,
    int max_token_len)
{
    if (!vocab || vocab_size <= 0) {
        LOG_ERROR("tokenizer_create_bpe: invalid arguments");
        return NULL;
    }

    Tokenizer* tok = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    if (!tok) {
        LOG_ERROR("tokenizer_create_bpe: OOM");
        return NULL;
    }

    tok->type = TOKENIZER_TYPE_BPE;
    tok->vocab_size = vocab_size;
    tok->max_token_len = max_token_len;

    /* Allocate vocabulary array */
    tok->vocab = (char**)calloc((size_t)vocab_size, sizeof(char*));
    if (!tok->vocab) {
        LOG_ERROR("tokenizer_create_bpe: OOM for vocab array");
        tokenizer_free(tok);
        return NULL;
    }

    /* Copy vocabulary strings */
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i]) {
            size_t len = strlen(vocab[i]);
            tok->vocab[i] = (char*)malloc(len + 1);
            if (!tok->vocab[i]) {
                LOG_ERROR("tokenizer_create_bpe: OOM for vocab[%d]", i);
                tokenizer_free(tok);
                return NULL;
            }
            memcpy(tok->vocab[i], vocab[i], len + 1);
        }
    }

    /* Initialize hash table */
    tok->hash_capacity = TOKENIZER_HASH_CAPACITY;
    tok->hash_table = token_hash_create(tok->hash_capacity);
    if (!tok->hash_table) {
        LOG_ERROR("tokenizer_create_bpe: OOM for hash table");
        tokenizer_free(tok);
        return NULL;
    }

    /* Build token string -> ID hash table */
    for (int i = 0; i < vocab_size; i++) {
        if (tok->vocab[i]) {
            if (!token_hash_insert(&tok->hash_table, &tok->hash_capacity,
                                    &tok->hash_count, tok->vocab[i], i)) {
                LOG_ERROR("tokenizer_create_bpe: failed to insert token %d", i);
                tokenizer_free(tok);
                return NULL;
            }
        }
    }

    /* Build byte-to-token mapping */
    build_byte_mapping(tok);

    /* Initialize pair rank hash table (will be populated later or empty) */
    tok->pair_capacity = next_prime(vocab_size > 256 ? vocab_size / 4 : 257);
    if (tok->pair_capacity < 257) tok->pair_capacity = 257;
    tok->pair_ranks = pair_hash_create(tok->pair_capacity);
    if (!tok->pair_ranks) {
        LOG_ERROR("tokenizer_create_bpe: OOM for pair ranks");
        tokenizer_free(tok);
        return NULL;
    }
    tok->pair_count = 0;

    return tok;
}

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Creation — From File
 * ═══════════════════════════════════════════════════════════════ */

Tokenizer* tokenizer_create_from_file(const char* path) {
    if (!path) {
        LOG_ERROR("tokenizer_create_from_file: NULL path");
        return NULL;
    }

    /* Read entire file into memory */
    size_t file_size = 0;
    uint8_t* data = (uint8_t*)file_read_all(path, &file_size);
    if (!data) {
        LOG_ERROR("tokenizer_create_from_file: failed to read %s", path);
        return NULL;
    }

    /* Check minimum size: header = 16 bytes */
    if (file_size < 16) {
        LOG_ERROR("tokenizer_create_from_file: file too small: %zu bytes", file_size);
        free(data);
        return NULL;
    }

    /* Parse header */
    size_t offset = 0;

    /* Magic: "TOK1" = 0x314B4F54 in LE */
    uint32_t magic = le32toh(*(uint32_t*)(data + offset));
    offset += 4;
    if (magic != 0x314B4F54) {
        LOG_ERROR("tokenizer_create_from_file: invalid magic: 0x%08X", magic);
        free(data);
        return NULL;
    }

    int vocab_size     = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;
    int max_token_len  = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;
    int num_merges     = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;

    if (vocab_size <= 0 || vocab_size > 200000) {
        LOG_ERROR("tokenizer_create_from_file: invalid vocab_size: %d", vocab_size);
        free(data);
        return NULL;
    }
    if (max_token_len <= 0 || max_token_len > 100000) {
        LOG_ERROR("tokenizer_create_from_file: invalid max_token_len: %d", max_token_len);
        free(data);
        return NULL;
    }

    /* Build vocabulary array from file data */
    const char** vocab_array = (const char**)calloc((size_t)vocab_size, sizeof(char*));
    if (!vocab_array) {
        LOG_ERROR("tokenizer_create_from_file: OOM for vocab array");
        free(data);
        return NULL;
    }

    /* Temporary buffer to hold null-terminated strings */
    char* tokens_buf = (char*)calloc((size_t)(vocab_size * max_token_len + 1), 1);
    if (!tokens_buf) {
        free(vocab_array);
        free(data);
        return NULL;
    }
    char* tb_ptr = tokens_buf;

    /* Read vocabulary entries */
    for (int i = 0; i < vocab_size; i++) {
        if (offset + 4 > file_size) {
            LOG_ERROR("tokenizer_create_from_file: truncated at vocab[%d]", i);
            free(tokens_buf);
            free(vocab_array);
            free(data);
            return NULL;
        }

        int token_len = (int)le32toh(*(uint32_t*)(data + offset));
        offset += 4;

        if (token_len < 0 || token_len > max_token_len || offset + (size_t)token_len > file_size) {
            LOG_ERROR("tokenizer_create_from_file: invalid token_len %d at index %d", token_len, i);
            free(tokens_buf);
            free(vocab_array);
            free(data);
            return NULL;
        }

        if (token_len > 0) {
            memcpy(tb_ptr, data + offset, (size_t)token_len);
        }
        tb_ptr[token_len] = '\0';
        vocab_array[i] = tb_ptr;
        tb_ptr += token_len + 1;
        offset += (size_t)token_len;
    }

    /* Create tokenizer with the vocabulary */
    Tokenizer* tok = tokenizer_create_bpe(vocab_array, vocab_size, max_token_len);
    if (!tok) {
        LOG_ERROR("tokenizer_create_from_file: tokenizer_create_bpe failed");
        free(tokens_buf);
        free(vocab_array);
        free(data);
        return NULL;
    }

    /* We can free the temporary vocab array and tokens_buf now -
     * tokenizer_create_bpe made its own copies. */
    free(vocab_array);
    free(tokens_buf);

    /* Read merge entries */
    int merge_insert_count = 0;
    int merge_skip_count = 0;
    for (int i = 0; i < num_merges; i++) {
        if (i % 10000 == 0)
            LOG_INFO("tokenizer merges: %d/%d (inserted=%d skipped=%d)",
                     i, num_merges, merge_insert_count, merge_skip_count);
        if (offset + 16 > file_size) {
            LOG_ERROR("tokenizer_create_from_file: truncated at merge[%d]", i);
            tokenizer_free(tok);
            free(data);
            return NULL;
        }

        int left_id    = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;
        int right_id   = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;
        int new_id     = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;
        int merge_rank = (int)le32toh(*(uint32_t*)(data + offset)); offset += 4;

        /* Validate IDs */
        if (left_id < 0 || left_id >= vocab_size ||
            right_id < 0 || right_id >= vocab_size ||
            new_id < 0 || new_id >= vocab_size) {
            LOG_DEBUG("tokenizer_create_from_file: invalid merge[%d] IDs: "
                      "left=%d right=%d new=%d (vocab_size=%d)",
                      i, left_id, right_id, new_id, vocab_size);
            merge_skip_count++;
            continue; /* skip invalid merges, don't fail entirely */
        }

        if (!pair_hash_insert(&tok->pair_ranks, &tok->pair_capacity,
                               &tok->pair_count, left_id, right_id, new_id, merge_rank)) {
            LOG_WARN("tokenizer_create_from_file: failed to insert merge[%d]", i);
        }
        merge_insert_count++;
    }

    free(data);
    LOG_INFO("Tokenizer loaded: vocab=%d merges=%d max_len=%d (inserted=%d skipped=%d)",
             vocab_size, num_merges, max_token_len, merge_insert_count, merge_skip_count);
    return tok;
}

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Creation — From Weight Index
 * ═══════════════════════════════════════════════════════════════
 *
 * Looks for these entries in the .vxcpm weight file:
 *   - tokenizer.vocab:         Raw bytes = concatenated token strings
 *   - tokenizer.vocab_sizes:   FP32 array of string lengths [vocab_size]
 *   - tokenizer.merges:        Raw bytes of merge entries (each = 3 x int32 LE)
 *
 * If any required entry is missing, returns NULL (caller should fall back).
 * ═══════════════════════════════════════════════════════════════ */

Tokenizer* tokenizer_create_from_weights(
    const uint8_t* mmap_data,
    const WeightIndex* idx)
{
    if (!mmap_data || !idx) return NULL;

    /* Look for tokenizer vocabulary entries */
    const WeightEntry* vocab_entry = weight_index_find(idx, "tokenizer.vocab");
    const WeightEntry* vocab_sizes_entry = weight_index_find(idx, "tokenizer.vocab_sizes");
    const WeightEntry* merges_entry = weight_index_find(idx, "tokenizer.merges");

    /* Fall back: try tokenizer.vocab.0 and tokenizer.merges.0 naming */
    if (!vocab_entry)    vocab_entry    = weight_index_find(idx, "tokenizer.vocab.0");
    if (!vocab_sizes_entry) vocab_sizes_entry = weight_index_find(idx, "tokenizer.vocab_lengths");
    if (!merges_entry)   merges_entry   = weight_index_find(idx, "tokenizer.merges.0");

    if (!vocab_entry || !vocab_sizes_entry) {
        /* Not found in weights — caller should try file mode */
        return NULL;
    }

    /* --- Load vocab sizes --- */
    /* Read as FP32 array of length = vocab_size */
    const float* sizes_f32 = (const float*)(mmap_data + vocab_sizes_entry->data_offset);
    int64_t num_sizes = 1;
    for (int d = 0; d < (int)vocab_sizes_entry->ndim; d++) {
        num_sizes *= (int64_t)vocab_sizes_entry->shape[d];
    }

    int vocab_size = (int)num_sizes;
    if (vocab_size <= 0 || vocab_size > 200000) {
        LOG_ERROR("tokenizer_from_weights: invalid vocab_size from sizes: %d", vocab_size);
        return NULL;
    }

    /* Determine max token length */
    int max_token_len = 0;
    for (int i = 0; i < vocab_size; i++) {
        int len = (int)sizes_f32[i];
        if (len > max_token_len) max_token_len = len;
    }
    if (max_token_len <= 0) {
        LOG_ERROR("tokenizer_from_weights: max_token_len = %d", max_token_len);
        return NULL;
    }

    /* --- Build vocabulary array --- */
    const char** vocab_array = (const char**)calloc((size_t)vocab_size, sizeof(char*));
    if (!vocab_array) {
        LOG_ERROR("tokenizer_from_weights: OOM");
        return NULL;
    }

    /* Allocate a contiguous buffer for all token strings */
    int64_t total_vocab_bytes = vocab_sizes_entry->data_size; /* total raw bytes */
    const uint8_t* vocab_raw = mmap_data + vocab_entry->data_offset;

    char* tokens_buf = (char*)malloc((size_t)total_vocab_bytes + (size_t)vocab_size + 1);
    if (!tokens_buf) {
        free(vocab_array);
        return NULL;
    }

    /* Reconstruct each null-terminated token string */
    {
        const uint8_t* src = vocab_raw;
        char* dst = tokens_buf;
        for (int i = 0; i < vocab_size; i++) {
            int len = (int)sizes_f32[i];
            if (len < 0) len = 0;
            if (len > 0) {
                memcpy(dst, src, (size_t)len);
                src += len;
            }
            dst[len] = '\0';
            vocab_array[i] = dst;
            dst += len + 1;
        }
    }

    Tokenizer* tok = tokenizer_create_bpe(vocab_array, vocab_size, max_token_len);
    free(vocab_array);
    free(tokens_buf);

    if (!tok) {
        LOG_ERROR("tokenizer_from_weights: tokenizer_create_bpe failed");
        return NULL;
    }

    /* --- Load merge entries (if present) --- */
    if (merges_entry) {
        const uint8_t* merge_data = mmap_data + merges_entry->data_offset;
        size_t merge_bytes = (size_t)merges_entry->data_size;
        int merge_entry_size = 16; /* 4 x int32 LE: left, right, new_id, rank */
        int num_merges = (int)(merge_bytes / (size_t)merge_entry_size);

        LOG_DEBUG("tokenizer_from_weights: loading %d merges", num_merges);

        for (int i = 0; i < num_merges; i++) {
            size_t off = (size_t)i * (size_t)merge_entry_size;
            if (off + merge_entry_size > merge_bytes) break;

            int left_id    = (int)le32toh(*(uint32_t*)(merge_data + off));
            int right_id   = (int)le32toh(*(uint32_t*)(merge_data + off + 4));
            int new_id     = (int)le32toh(*(uint32_t*)(merge_data + off + 8));
            int merge_rank = (int)le32toh(*(uint32_t*)(merge_data + off + 12));

            if (left_id < 0 || left_id >= vocab_size ||
                right_id < 0 || right_id >= vocab_size ||
                new_id < 0 || new_id >= vocab_size) {
                continue; /* skip invalid */
            }

            pair_hash_insert(&tok->pair_ranks, &tok->pair_capacity,
                             &tok->pair_count, left_id, right_id, new_id, merge_rank);
        }

        LOG_INFO("Tokenizer loaded from weights: vocab=%d merges=%d",
                 tok->vocab_size, tok->pair_count);
    } else {
        LOG_INFO("Tokenizer loaded from weights: vocab=%d (no merges)", tok->vocab_size);
    }

    return tok;
}

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Free
 * ═══════════════════════════════════════════════════════════════ */

void tokenizer_free(Tokenizer* tok) {
    if (!tok) return;

    /* Free vocabulary strings */
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }

    /* Free hash table (frees owned token strings stored in hash entries) */
    if (tok->hash_table) {
        for (int i = 0; i < tok->hash_capacity; i++) {
            if (tok->hash_table[i].occupied && tok->hash_table[i].token) {
                /* The token strings in hash_table are shared with tok->vocab,
                 * so we should NOT free them here (they are freed above).
                 * Set pointer to NULL to prevent double-free. */
                tok->hash_table[i].token = NULL;
            }
        }
        free(tok->hash_table);
    }

    /* Free pair ranks */
    if (tok->pair_ranks) {
        free(tok->pair_ranks);
    }

    memset(tok, 0, sizeof(Tokenizer));
    free(tok);
}

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Encode
 * ═══════════════════════════════════════════════════════════════ */

int tokenizer_encode(
    const Tokenizer* tok,
    const char* text,
    int32_t* out_tokens,
    int max_tokens)
{
    if (!tok || !text) {
        return -VOXCPM_ERR_INTERNAL;
    }

    int text_len = (int)strlen(text);
    if (text_len == 0) {
        return 0;
    }

    /* Pre-allocate maximum possible tokens (every byte maps to a token) */
    int max_initial_tokens = text_len;
    int* word_buf = NULL;
    int* merged_buf = NULL;
    int32_t* result_tokens = NULL;

    /* Allocate working buffers */
    word_buf = (int*)malloc((size_t)max_initial_tokens * sizeof(int));
    merged_buf = (int*)malloc((size_t)max_initial_tokens * sizeof(int));
    result_tokens = (int32_t*)malloc((size_t)max_initial_tokens * sizeof(int32_t));

    if (!word_buf || !merged_buf || !result_tokens) {
        free(word_buf);
        free(merged_buf);
        free(result_tokens);
        return -VOXCPM_ERR_OOM;
    }

    /* Pre-tokenize into words */
    int max_words = text_len > 0 ? text_len : 1;  /* at most text_len words */
    int* word_starts = (int*)malloc((size_t)max_words * sizeof(int));
    int* word_ends = (int*)malloc((size_t)max_words * sizeof(int));
    if (!word_starts || !word_ends) {
        free(word_starts);
        free(word_ends);
        free(word_buf);
        free(merged_buf);
        free(result_tokens);
        return -VOXCPM_ERR_OOM;
    }

    int num_words = pre_tokenize(text, text_len, word_starts, word_ends, max_words);

    /* Process each word through BPE */
    int total_tokens = 0;

    /* If no words found (e.g., all whitespace), add nothing */
    for (int w = 0; w < num_words; w++) {
        int word_len = word_ends[w] - word_starts[w];
        const char* word_start = text + word_starts[w];

        if (word_len > TOKENIZER_MAX_WORD_BYTES) {
            /* Word too long: split further or just use bytes directly */
            word_len = TOKENIZER_MAX_WORD_BYTES;
        }

        /* Convert word bytes to initial token IDs */
        int num_byte_tokens = word_to_byte_tokens(tok, word_start, word_len,
                                                    word_buf, max_initial_tokens);
        if (num_byte_tokens <= 0) continue;

        /* Apply BPE merges */
        int num_merged = bpe_merge_sequence(tok, word_buf, num_byte_tokens,
                                              merged_buf, max_initial_tokens);

        if (num_merged < 0) {
            /* Error during merge */
            free(word_starts);
            free(word_ends);
            free(word_buf);
            free(merged_buf);
            free(result_tokens);
            return -VOXCPM_ERR_INTERNAL;
        }

        /* Append merged tokens to result */
        if (total_tokens + num_merged > max_initial_tokens) {
            num_merged = max_initial_tokens - total_tokens;
        }
        for (int i = 0; i < num_merged; i++) {
            result_tokens[total_tokens++] = (int32_t)merged_buf[i];
        }
    }

    free(word_starts);
    free(word_ends);

    /* Query mode: return required buffer length */
    if (!out_tokens) {
        free(word_buf);
        free(merged_buf);
        free(result_tokens);
        return total_tokens;
    }

    /* Copy to output buffer */
    int n = (total_tokens < max_tokens) ? total_tokens : max_tokens;
    if (n < total_tokens) {
        /* Buffer too small — return required length as positive, but also
         * check: if caller provided a buffer, we should report error */
        free(word_buf);
        free(merged_buf);
        free(result_tokens);
        /* Return the required length so caller can retry */
        return total_tokens;
    }

    memcpy(out_tokens, result_tokens, (size_t)n * sizeof(int32_t));
    free(word_buf);
    free(merged_buf);
    free(result_tokens);
    return n;
}

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Decode
 * ═══════════════════════════════════════════════════════════════ */

int tokenizer_decode(
    const Tokenizer* tok,
    const int32_t* tokens,
    int num_tokens,
    char* out_text,
    int max_chars)
{
    if (!tok || !tokens) {
        return -VOXCPM_ERR_INTERNAL;
    }
    if (num_tokens <= 0) {
        return 0;
    }

    /* First pass: compute total length */
    int total_len = 0;
    for (int i = 0; i < num_tokens; i++) {
        int id = (int)tokens[i];
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            total_len += (int)strlen(tok->vocab[id]);
        }
    }

    /* Query mode: return required buffer length */
    if (!out_text) {
        return total_len;
    }
    if (max_chars <= 0) {
        return 0;
    }

    /* Second pass: write output */
    int written = 0;
    for (int i = 0; i < num_tokens && written < max_chars - 1; i++) {
        int id = (int)tokens[i];
        if (id >= 0 && id < tok->vocab_size && tok->vocab[id]) {
            const char* s = tok->vocab[id];
            size_t slen = strlen(s);
            size_t to_write = slen;
            if (written + (int)to_write > max_chars - 1) {
                to_write = (size_t)(max_chars - 1 - written);
            }
            if (to_write > 0) {
                memcpy(out_text + written, s, to_write);
                written += (int)to_write;
            }
        }
    }
    out_text[written] = '\0';
    return written;
}

/* ═══════════════════════════════════════════════════════════════
 * Vocabulary Queries
 * ═══════════════════════════════════════════════════════════════ */

int tokenizer_vocab_size(const Tokenizer* tok) {
    if (!tok) return 0;
    return tok->vocab_size;
}

const char* tokenizer_id_to_token(const Tokenizer* tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

int tokenizer_token_to_id(const Tokenizer* tok, const char* token) {
    if (!tok || !token) return TOKEN_UNK_ID;
    return token_hash_lookup(tok->hash_table, tok->hash_capacity, token, -1);
}

bool tokenizer_is_special(const Tokenizer* tok, int id) {
    if (!tok) return false;
    return (id == TOKEN_PAD_ID || id == TOKEN_UNK_ID ||
            id == TOKEN_BOS_ID || id == TOKEN_EOS_ID);
}
