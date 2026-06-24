#ifndef TOKENIZER_H
#define TOKENIZER_H

/*
 * tokenizer.h — BPE tokenizer for VoxCPM2-C
 * VoxCPM2-C Project
 * License: Apache-2.0
 *
 * Byte-level BPE tokenizer (GPT-2 style) for encoding/decoding text.
 * Supports loading from standalone .bin file or from .vxcpm weight index.
 */

#include "voxcpm.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════
 * Tokenizer Types
 * ═══════════════════════════════════════════════════════════════ */
typedef enum {
    TOKENIZER_TYPE_BPE = 0,
    TOKENIZER_TYPE_SENTENCEPIECE = 1,
} TokenizerType;

/* ═══════════════════════════════════════════════════════════════
 * Special Token IDs
 * ═══════════════════════════════════════════════════════════════ */
#define TOKEN_PAD_ID  0
#define TOKEN_UNK_ID  1
#define TOKEN_BOS_ID  2
#define TOKEN_EOS_ID  3

/* WeightIndex is defined in model.h */
typedef struct WeightIndex WeightIndex;

/* ═══════════════════════════════════════════════════════════════
 * Opaque Tokenizer Structure
 * ═══════════════════════════════════════════════════════════════ */
typedef struct Tokenizer Tokenizer;

/* ═══════════════════════════════════════════════════════════════
 * Lifecycle
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Create BPE tokenizer from pre-built vocabulary array.
 * vocab: array of vocab_size null-terminated token strings.
 *        The tokenizer takes ownership of the strings (or copies them).
 * vocab_size: number of tokens.
 * max_token_len: maximum byte length of any token.
 */
Tokenizer* tokenizer_create_bpe(
    const char** vocab,
    int vocab_size,
    int max_token_len
);

/*
 * Create tokenizer from .vxcpm weight file data.
 * Looks for tokenizer.vocab, tokenizer.vocab_sizes, and
 * tokenizer.merges entries in the weight index.
 * Returns NULL if tokenizer data is not found in weights.
 */
Tokenizer* tokenizer_create_from_weights(
    const uint8_t* mmap_data,
    const WeightIndex* idx
);

/*
 * Create tokenizer from a standalone tokenizer.bin file.
 * File format:
 *   [4 bytes] magic: "TOK1" (0x314B4F54)
 *   [4 bytes] vocab_size (int32 LE)
 *   [4 bytes] max_token_len (int32 LE)
 *   [4 bytes] num_merges (int32 LE)
 *   --- vocab entries: vocab_size entries ---
 *     [4 bytes] token_len (int32 LE)
 *     [token_len bytes] token data (not null-terminated)
 *   --- merge entries: num_merges entries ---
 *     [4 bytes] pair_id_0 (int32 LE)
 *     [4 bytes] pair_id_1 (int32 LE)
 *     [4 bytes] new_token_id (int32 LE)
 *     [4 bytes] merge_rank (int32 LE)
 */
Tokenizer* tokenizer_create_from_file(const char* path);

/*
 * Free a tokenizer. Safe to call with NULL.
 */
void tokenizer_free(Tokenizer* tok);

/* ═══════════════════════════════════════════════════════════════
 * Encoding / Decoding
 * ═══════════════════════════════════════════════════════════════ */

/*
 * Encode UTF-8 text to token IDs using BPE.
 * Returns number of tokens written (non-negative), or negative on error.
 * If out_tokens is NULL, returns the required buffer length (query mode).
 * If max_tokens is insufficient, returns -VOXCPM_ERR_INTERNAL.
 *
 * Thread safety: const Tokenizer* allows concurrent read access
 * (no mutable state during encoding).
 */
int tokenizer_encode(
    const Tokenizer* tok,
    const char* text,
    int32_t* out_tokens,
    int max_tokens
);

/*
 * Decode token IDs back to UTF-8 text.
 * Returns number of chars written (excluding NUL), or negative on error.
 * If out_text is NULL, returns the required buffer length (query mode).
 * Writes at most max_chars-1 characters and always NUL-terminates.
 */
int tokenizer_decode(
    const Tokenizer* tok,
    const int32_t* tokens,
    int num_tokens,
    char* out_text,
    int max_chars
);

/* ═══════════════════════════════════════════════════════════════
 * Vocabulary Queries
 * ═══════════════════════════════════════════════════════════════ */

/* Get vocabulary size. */
int tokenizer_vocab_size(const Tokenizer* tok);

/* Convert token ID to string. Returns NULL for invalid IDs. */
const char* tokenizer_id_to_token(const Tokenizer* tok, int id);

/* Convert token string to ID. Returns TOKEN_UNK_ID for unknown tokens. */
int tokenizer_token_to_id(const Tokenizer* tok, const char* token);

/* Check if a token ID is a special token (PAD, UNK, BOS, EOS). */
bool tokenizer_is_special(const Tokenizer* tok, int id);

#ifdef __cplusplus
}
#endif

#endif /* TOKENIZER_H */
