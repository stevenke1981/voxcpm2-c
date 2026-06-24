#!/usr/bin/env python3
"""
Convert HuggingFace tokenizer.json to VoxCPM2-C tokenizer.bin format.

tokenizer.bin binary format:
  Header (16 bytes):
    magic:          uint32 LE  (0x314B4F54 = "TOK1")
    vocab_size:     uint32 LE
    max_token_len:  uint32 LE
    num_merges:     uint32 LE

  Vocabulary entries (vocab_size × variable):
    token_len: uint32 LE
    token_data: bytes (length = token_len, NOT null-terminated)

  Merge entries (num_merges × 16):
    left_id:    uint32 LE
    right_id:   uint32 LE
    new_id:     uint32 LE
    merge_rank: uint32 LE

Usage:
    python scripts/convert_tokenizer.py models/VoxCPM2/tokenizer.json models/tokenizer.bin
"""

import json
import struct
import sys


def convert_tokenizer(json_path: str, bin_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model = data.get("model", data)

    # Extract vocabulary: {"token_str": id, ...}
    vocab_dict = model.get("vocab", {})
    # Also try "vocab" directly at top level
    if not vocab_dict:
        vocab_dict = data.get("vocab", {})

    # Build id-to-token mapping
    id_to_token = {}
    for token_str, token_id in vocab_dict.items():
        id_to_token[token_id] = token_str

    vocab_size = len(id_to_token)
    print(f"  Vocab size: {vocab_size}")

    # Determine max token length
    max_token_len = 0
    for token_str in id_to_token.values():
        # Token bytes (encode string as UTF-8)
        token_bytes = token_str.encode("utf-8")
        if len(token_bytes) > max_token_len:
            max_token_len = len(token_bytes)
    print(f"  Max token length: {max_token_len} bytes")

    # Extract merge rules
    merges = model.get("merges", [])
    # merges is a list of strings like "token1 token2"
    # Each pair maps to a new token: the merged token string = left_str + right_str
    merge_pairs = []
    skipped_count = 0
    for i, merge_str in enumerate(merges):
        parts = merge_str.strip().split()
        if len(parts) >= 2:
            left_str = parts[0]
            right_str = parts[1]
            if left_str in vocab_dict and right_str in vocab_dict:
                left_id = vocab_dict[left_str]
                right_id = vocab_dict[right_str]
                # The merged result token is the concatenation of left and right.
                # Look up its actual token ID in the vocabulary.
                merged_str = left_str + right_str
                if merged_str in vocab_dict:
                    new_id = vocab_dict[merged_str]
                else:
                    # Fallback: should not happen for well-formed tokenizers
                    new_id = vocab_size + i
                    skipped_count += 1
                merge_pairs.append((left_id, right_id, new_id, i))
            else:
                skipped_count += 1
                print(f"  Warning: merge[{i}] token not in vocab: '{left_str}' '{right_str}'")
        else:
            skipped_count += 1
            print(f"  Warning: merge[{i}] unparseable: '{merge_str}'")

    print(f"  Merges: {len(merge_pairs)} (skipped={skipped_count})")

    print(f"  Merges: {len(merge_pairs)}")

    # Write binary file
    with open(bin_path, "wb") as f:
        # Header
        f.write(struct.pack("<IIII", 0x314B4F54, vocab_size, max_token_len, len(merge_pairs)))

        # Vocabulary entries
        for token_id in range(vocab_size):
            token_str = id_to_token.get(token_id, "")
            token_bytes = token_str.encode("utf-8")
            f.write(struct.pack("<I", len(token_bytes)))
            if token_bytes:
                f.write(token_bytes)

        # Merge entries
        for left_id, right_id, new_id, rank in merge_pairs:
            f.write(struct.pack("<IIII", left_id, right_id, new_id, rank))

    print(f"  Written: {bin_path}")
    file_size = 16 + sum(4 + len(id_to_token.get(i, "").encode("utf-8")) for i in range(vocab_size)) + 16 * len(merge_pairs)
    print(f"  File size: {file_size} bytes")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert tokenizer.json to tokenizer.bin")
    parser.add_argument("json_path", help="Path to tokenizer.json")
    parser.add_argument("bin_path", help="Output path for tokenizer.bin")
    args = parser.parse_args()
    convert_tokenizer(args.json_path, args.bin_path)
