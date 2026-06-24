// test_tokenizer.c — BPE tokenizer unit tests
// VoxCPM2-C Project
// License: Apache-2.0
//
// Tests: tokenizer_create_bpe, tokenizer_free,
//        tokenizer_encode, tokenizer_decode,
//        tokenizer_vocab_size, tokenizer_id_to_token,
//        tokenizer_token_to_id, tokenizer_is_special

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); } while(0)
#define PASS() do { printf("PASS\n"); g_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); g_failed++; } while(0)
#define ASSERT(cond, msg) do { if (!(cond)) { FAIL(msg); return; } } while(0)

/* ═══════════════════════════════════════════════════════════════
 * Helper: Build a minimal BPE vocabulary for testing
 * ═══════════════════════════════════════════════════════════════
 * We construct a vocabulary with:
 *   0: [PAD]   (special)
 *   1: [UNK]   (special)
 *   2: [BOS]   (special)
 *   3: [EOS]   (special)
 *   4: 'a'     (byte token)
 *   5: 'b'     (byte token)
 *   6: 'c'     (byte token)
 *   7: ' '     (space byte)
 *   8: "ab"    (merged token)
 *   9: "bc"    (merged token)
 *  10: "abc"   (merged token)
 *
 * BPE merges (in merge order):
 *   rank=0: (4,5) -> 8   ("a"+"b" -> "ab")
 *   rank=1: (5,6) -> 9   ("b"+"c" -> "bc")
 *   rank=2: (8,6) -> 10  ("ab"+"c" -> "abc")
 */

#define TEST_VOCAB_SIZE 11

static const char* g_test_vocab[TEST_VOCAB_SIZE] = {
    "[PAD]", "[UNK]", "[BOS]", "[EOS]",  /* 0-3: special */
    "a",      /* 4: byte 0x61 */
    "b",      /* 5: byte 0x62 */
    "c",      /* 6: byte 0x63 */
    " ",      /* 7: space byte 0x20 */
    "ab",     /* 8: merged */
    "bc",     /* 9: merged */
    "abc",    /* 10: merged */
};

/* ═══════════════════════════════════════════════════════════════
 * Tests
 * ═══════════════════════════════════════════════════════════════ */

static void test_create_free(void) {
    TEST("create and free BPE tokenizer");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "tokenizer_create_bpe returned NULL");
    ASSERT(tokenizer_vocab_size(tok) == TEST_VOCAB_SIZE, "vocab_size mismatch");
    tokenizer_free(tok);
    PASS();
}

static void test_free_null(void) {
    TEST("tokenizer_free(NULL) is safe");
    tokenizer_free(NULL);
    PASS();
}

static void test_vocab_size(void) {
    TEST("tokenizer_vocab_size");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");
    ASSERT(tokenizer_vocab_size(tok) == TEST_VOCAB_SIZE, "size == 11");
    ASSERT(tokenizer_vocab_size(NULL) == 0, "NULL returns 0");
    tokenizer_free(tok);
    PASS();
}

static void test_id_to_token(void) {
    TEST("tokenizer_id_to_token");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    const char* t0 = tokenizer_id_to_token(tok, 0);
    ASSERT(t0 != NULL && strcmp(t0, "[PAD]") == 0, "id=0 should be [PAD]");

    const char* t4 = tokenizer_id_to_token(tok, 4);
    ASSERT(t4 != NULL && strcmp(t4, "a") == 0, "id=4 should be 'a'");

    const char* t_inv = tokenizer_id_to_token(tok, 999);
    ASSERT(t_inv == NULL, "invalid id returns NULL");

    const char* t_null = tokenizer_id_to_token(NULL, 0);
    ASSERT(t_null == NULL, "NULL tokenizer returns NULL");

    tokenizer_free(tok);
    PASS();
}

static void test_token_to_id(void) {
    TEST("tokenizer_token_to_id");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    ASSERT(tokenizer_token_to_id(tok, "[PAD]") == 0, "[PAD] -> 0");
    ASSERT(tokenizer_token_to_id(tok, "[UNK]") == 1, "[UNK] -> 1");
    ASSERT(tokenizer_token_to_id(tok, "a") == 4, "'a' -> 4");
    ASSERT(tokenizer_token_to_id(tok, "ab") == 8, "'ab' -> 8");
    ASSERT(tokenizer_token_to_id(tok, "nonexistent") == TOKEN_UNK_ID, "unknown -> UNK");
    ASSERT(tokenizer_token_to_id(tok, "") == TOKEN_UNK_ID, "empty -> UNK");
    ASSERT(tokenizer_token_to_id(NULL, "a") == TOKEN_UNK_ID, "NULL tok -> UNK");

    tokenizer_free(tok);
    PASS();
}

static void test_is_special(void) {
    TEST("tokenizer_is_special");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    ASSERT(tokenizer_is_special(tok, 0) == true,  "PAD is special");
    ASSERT(tokenizer_is_special(tok, 1) == true,  "UNK is special");
    ASSERT(tokenizer_is_special(tok, 2) == true,  "BOS is special");
    ASSERT(tokenizer_is_special(tok, 3) == true,  "EOS is special");
    ASSERT(tokenizer_is_special(tok, 4) == false, "'a' is not special");
    ASSERT(tokenizer_is_special(tok, 999) == false, "invalid id not special");
    ASSERT(tokenizer_is_special(NULL, 0) == false, "NULL tok returns false");

    tokenizer_free(tok);
    PASS();
}

static void test_decode_simple(void) {
    TEST("decode single tokens");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    int32_t tokens[] = {4, 5, 6, 7, 4};  /* "a", "b", "c", " ", "a" */
    char buf[64];
    int n = tokenizer_decode(tok, tokens, 5, buf, sizeof(buf));
    ASSERT(n == 5, "length should be 5");
    ASSERT(strcmp(buf, "abc a") == 0, "should decode to 'abc a'");

    tokenizer_free(tok);
    PASS();
}

static void test_decode_query_mode(void) {
    TEST("decode query mode (NULL out_text)");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    int32_t tokens[] = {4, 5, 6};  /* "a", "b", "c" */
    int n = tokenizer_decode(tok, tokens, 3, NULL, 0);
    ASSERT(n == 3, "query mode should return length 3");

    tokenizer_free(tok);
    PASS();
}

static void test_decode_empty(void) {
    TEST("decode empty input");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    char buf[16] = "x";
    int n = tokenizer_decode(tok, NULL, 0, buf, sizeof(buf));
    ASSERT(n == 0, "NULL tokens returns 0");
    ASSERT(buf[0] == 'x', "buffer unchanged");

    n = tokenizer_decode(tok, NULL, 0, NULL, 0);
    ASSERT(n == 0, "NULL tokens, NULL out returns 0");

    tokenizer_free(tok);
    PASS();
}

static void test_decode_partial_buffer(void) {
    TEST("decode with small buffer");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    int32_t tokens[] = {4, 5, 6, 7, 4, 5, 6};  /* "abc abc" */
    char buf[4];  /* tiny buffer */
    int n = tokenizer_decode(tok, tokens, 7, buf, sizeof(buf));
    ASSERT(n == 3, "should write 3 chars (max 3 + null)");
    ASSERT(buf[0] == 'a', "buf[0] = 'a'");
    ASSERT(buf[1] == 'b', "buf[1] = 'b'");
    ASSERT(buf[2] == 'c', "buf[2] = 'c'");
    ASSERT(buf[3] == '\0', "null-terminated");

    tokenizer_free(tok);
    PASS();
}

static void test_encode_query_mode(void) {
    TEST("encode query mode (NULL out_tokens)");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    int n = tokenizer_encode(tok, "a", NULL, 0);
    ASSERT(n > 0, "query mode should return positive length");

    n = tokenizer_encode(tok, "", NULL, 0);
    ASSERT(n == 0, "empty string returns 0");

    tokenizer_free(tok);
    PASS();
}

static void test_encode_invalid_args(void) {
    TEST("encode with invalid args");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    int32_t buf[16];
    int n = tokenizer_encode(NULL, "hello", buf, 16);
    ASSERT(n < 0, "NULL tokenizer returns negative");

    n = tokenizer_encode(tok, NULL, buf, 16);
    ASSERT(n < 0, "NULL text returns negative");

    tokenizer_free(tok);
    PASS();
}

static void test_encode_decode_roundtrip(void) {
    TEST("basic encode->decode roundtrip with single letters");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    /* Encode "abc" — each byte maps to a token */
    int32_t tokens[16];
    int n = tokenizer_encode(tok, "abc", tokens, 16);
    ASSERT(n > 0, "should produce tokens");

    /* Decode back */
    char text[64];
    int m = tokenizer_decode(tok, tokens, n, text, sizeof(text));
    ASSERT(m > 0, "should decode something");
    ASSERT(strcmp(text, "abc") == 0, "roundtrip should match 'abc'");

    tokenizer_free(tok);
    PASS();
}

static void test_special_tokens_in_vocab(void) {
    TEST("special tokens in vocabulary");
    Tokenizer* tok = tokenizer_create_bpe(g_test_vocab, TEST_VOCAB_SIZE, 3);
    ASSERT(tok != NULL, "create");

    /* Check special token strings */
    ASSERT(strcmp(tokenizer_id_to_token(tok, 0), "[PAD]") == 0, "id=0 text");
    ASSERT(strcmp(tokenizer_id_to_token(tok, 1), "[UNK]") == 0, "id=1 text");
    ASSERT(strcmp(tokenizer_id_to_token(tok, 2), "[BOS]") == 0, "id=2 text");
    ASSERT(strcmp(tokenizer_id_to_token(tok, 3), "[EOS]") == 0, "id=3 text");

    /* Check special token lookup */
    ASSERT(tokenizer_token_to_id(tok, "[PAD]") == 0, "[PAD] lookup");
    ASSERT(tokenizer_token_to_id(tok, "[UNK]") == 1, "[UNK] lookup");
    ASSERT(tokenizer_token_to_id(tok, "[BOS]") == 2, "[BOS] lookup");
    ASSERT(tokenizer_token_to_id(tok, "[EOS]") == 3, "[EOS] lookup");

    tokenizer_free(tok);
    PASS();
}

/* ═══════════════════════════════════════════════════════════════
 * Main
 * ═══════════════════════════════════════════════════════════════ */

int main(void) {
    printf("VoxCPM2 Tokenizer Tests\n");
    printf("=======================\n\n");

    test_create_free();
    test_free_null();
    test_vocab_size();
    test_id_to_token();
    test_token_to_id();
    test_is_special();
    test_decode_simple();
    test_decode_query_mode();
    test_decode_empty();
    test_decode_partial_buffer();
    test_encode_query_mode();
    test_encode_invalid_args();
    test_encode_decode_roundtrip();
    test_special_tokens_in_vocab();

    printf("\nResults: %d passed, %d failed out of %d\n",
           g_passed, g_failed, g_passed + g_failed);

    return g_failed > 0 ? 1 : 0;
}
