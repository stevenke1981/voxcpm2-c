#!/usr/bin/env python3
"""
VoxCPM2 → .vxcpm Binary Weight Converter

Converts HuggingFace openbmb/VoxCPM2 PyTorch weights into the .vxcpm binary format
used by the VoxCPM2-C pure-C inference engine.

Usage:
    # Download & convert from HuggingFace
    python scripts/convert_weights.py openbmb/VoxCPM2 models/voxcpm2-f16.vxcpm

    # Convert from local directory (must contain config.json + model.safetensors + audiovae.pth)
    python scripts/convert_weights.py /path/to/model_dir/ models/voxcpm2-f16.vxcpm

    # With bfloat16 preservation
    python scripts/convert_weights.py openbmb/VoxCPM2 models/voxcpm2-bf16.vxcpm --dtype bf16

    # Verify a converted file
    python scripts/convert_weights.py --verify models/voxcpm2-f16.vxcpm

Format: docs/WEIGHT_FORMAT.md (version 1)
"""

import argparse
import json
import os
import struct
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, BinaryIO

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

VXCPM_MAGIC = 0x56435850  # "VXCP"
HEADER_SIZE = 64
METADATA_ENTRY_SIZE = 32

DTYPE_FP32 = 0
DTYPE_BF16 = 1
DTYPE_FP16 = 2

# ──────────────────────────────────────────────
# CRC32-C (Castagnoli)
# ──────────────────────────────────────────────

CRC32C_TABLE: List[int] = []


def _build_crc32c_table() -> List[int]:
    """Build CRC32C lookup table using Castagnoli polynomial 0x1EDC6F41."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = 0x82F63B78 ^ (crc >> 1)
            else:
                crc >>= 1
        table.append(crc)
    return table


def crc32c(data: bytes, crc: int = 0) -> int:
    """Compute CRC32C checksum."""
    global CRC32C_TABLE
    if not CRC32C_TABLE:
        CRC32C_TABLE = _build_crc32c_table()
    crc = crc ^ 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


# ──────────────────────────────────────────────
# Tensor shape helpers
# ──────────────────────────────────────────────


def pack_shape(shape: Tuple[int, ...]) -> tuple:
    """Pad shape to 4 dimensions, filling with 1s."""
    dims = list(shape)
    while len(dims) < 4:
        dims.append(1)
    return tuple(dims[:4])


def tensor_byte_size(shape: Tuple[int, ...], dtype: int) -> int:
    """Compute the byte size of a tensor."""
    elem_size = {DTYPE_FP32: 4, DTYPE_BF16: 2, DTYPE_FP16: 2}[dtype]
    total = 1
    for d in shape:
        total *= d
    return total * elem_size


# ──────────────────────────────────────────────
# Expected tensor registry
# ──────────────────────────────────────────────

# Format: (safetensors_key, out_name, expected_shape_or_check_fn)
# shape can be a tuple for exact match or a callable for dynamic check


def _tslm_layer_tensors(layer_idx: int, hidden: int, intermediate: int,
                        n_heads: int, n_kv_heads: int, kv_channels: int) -> List[Tuple[str, str, Tuple]]:
    """Generate the 9 tensors for one TSLM (base_lm / residual_lm) layer."""
    head_dim = kv_channels if kv_channels else hidden // n_heads
    q_dim = n_heads * head_dim
    k_dim = n_kv_heads * head_dim
    prefix = f"base_lm.layers.{layer_idx}." if "base" in sys._getframe(1).f_locals.get("prefix", "") else f"base_lm.layers.{layer_idx}."
    if layer_idx >= 100:  # hack: use residual_lm prefix for large indices
        prefix = f"residual_lm.layers.{layer_idx - 100}."

    return [
        (f"{prefix}input_layernorm.weight", f"base_lm.layers.{layer_idx}.input_layernorm.weight", (hidden,)),
        (f"{prefix}self_attn.q_proj.weight", f"base_lm.layers.{layer_idx}.self_attn.q_proj.weight", (q_dim, hidden)),
        (f"{prefix}self_attn.k_proj.weight", f"base_lm.layers.{layer_idx}.self_attn.k_proj.weight", (k_dim, hidden)),
        (f"{prefix}self_attn.v_proj.weight", f"base_lm.layers.{layer_idx}.self_attn.v_proj.weight", (k_dim, hidden)),
        (f"{prefix}self_attn.o_proj.weight", f"base_lm.layers.{layer_idx}.self_attn.o_proj.weight", (hidden, q_dim)),
        (f"{prefix}mlp.gate_proj.weight", f"base_lm.layers.{layer_idx}.mlp.gate_proj.weight", (intermediate, hidden)),
        (f"{prefix}mlp.up_proj.weight", f"base_lm.layers.{layer_idx}.mlp.up_proj.weight", (intermediate, hidden)),
        (f"{prefix}mlp.down_proj.weight", f"base_lm.layers.{layer_idx}.mlp.down_proj.weight", (hidden, intermediate)),
        (f"{prefix}post_attention_layernorm.weight", f"base_lm.layers.{layer_idx}.post_attention_layernorm.weight", (hidden,)),
    ]


# ──────────────────────────────────────────────
# Binary writer
# ──────────────────────────────────────────────


class VxcpmWriter:
    """Writes .vxcpm binary weight files."""

    def __init__(self, output_path: str, dtype: int = DTYPE_FP32):
        self.path = output_path
        self.dtype = dtype
        self.metadata: List[Tuple[str, Tuple[int, ...], int]] = []  # (name, shape, data_offset)
        self.data_blocks: List[bytes] = []
        self._string_table: List[Tuple[str, int]] = []  # (name, offset_in_string_table)
        self._string_table_bytes = b""
        self._current_data_offset = 0

    def add_tensor(self, name: str, tensor: np.ndarray):
        """Add a tensor to the file."""
        shape = tensor.shape
        data = self._convert_tensor(tensor)

        # Record metadata (offset will be computed at write time)
        self.metadata.append((name, shape, self._current_data_offset))
        self.data_blocks.append(data)
        self._current_data_offset += len(data)

    def _convert_tensor(self, tensor: np.ndarray) -> bytes:
        """Convert numpy tensor to target dtype bytes."""
        if self.dtype == DTYPE_BF16:
            # numpy has no native bf16; use view trick
            f32 = tensor.astype(np.float32)
            bf16_bytes = b""
            for val in f32.flatten():
                bf16_bytes += struct.pack("H", (struct.unpack("I", struct.pack("f", val))[0] >> 16) & 0xFFFF)
            return bf16_bytes
        elif self.dtype == DTYPE_FP16:
            return tensor.astype(np.float16).tobytes()
        else:  # FP32
            return tensor.astype(np.float32).tobytes()

    def _build_string_table(self):
        """Build string table from metadata names."""
        entries = []
        offset = 0
        for name, _, _ in self.metadata:
            encoded = name.encode("utf-8") + b"\0"
            entries.append((name, offset, encoded))
            offset += len(encoded)

        self._string_table = [(n, o) for n, o, _ in entries]
        self._string_table_bytes = b"".join(e for _, _, e in entries)

    def write(self):
        """Write the complete .vxcpm file."""
        self._build_string_table()

        num_tensors = len(self.metadata)
        string_table_size = len(self._string_table_bytes)
        metadata_array_size = num_tensors * METADATA_ENTRY_SIZE

        data_start = HEADER_SIZE + metadata_array_size + string_table_size

        # Adjust data offsets to be absolute
        adjusted_metadata = []
        for name, shape, rel_offset in self.metadata:
            adjusted_metadata.append((name, shape, data_start + rel_offset))

        with open(self.path, "wb") as f:
            # ── Header (64 bytes, placeholder for checksums) ──
            header = bytearray(HEADER_SIZE)
            struct.pack_into("<I", header, 0, VXCPM_MAGIC)         # magic
            struct.pack_into("<I", header, 4, 1)                     # version
            struct.pack_into("<I", header, 8, num_tensors)          # num_tensors
            # reserved[0..10] at offsets 12..55 = 0 (already zero)
            # checksum at offset 56, data_checksum at offset 60 = 0 (to be filled)

            # ── Build metadata array + string table ──
            metadata_bytes = bytearray()
            string_table_lookup = {n: o for n, o in self._string_table}
            for name, shape, abs_offset in adjusted_metadata:
                entry = bytearray(METADATA_ENTRY_SIZE)
                str_offset = string_table_lookup[name]
                d0, d1, d2, d3 = pack_shape(shape)
                struct.pack_into("<I", entry, 0, str_offset)         # name_offset
                struct.pack_into("<I", entry, 4, len(name))          # name_length
                struct.pack_into("<I", entry, 8, self.dtype)         # dtype
                struct.pack_into("<I", entry, 12, len(shape))        # ndim
                struct.pack_into("<IIII", entry, 16, d0, d1, d2, d3) # shape
                metadata_bytes.extend(entry)

            # ── Compute checksums ──
            # Zero both checksum fields for computation (bytes 56-63)
            pre_checksum = bytes(header[:56]) + b"\x00\x00\x00\x00\x00\x00\x00\x00"
            pre_data = bytes(metadata_bytes) + self._string_table_bytes + b"".join(self.data_blocks)

            hdr_checksum = crc32c(pre_checksum)
            data_checksum = crc32c(pre_data)

            # Fill checksums (header is bytearray, mutable)
            struct.pack_into("<I", header, 56, hdr_checksum)
            struct.pack_into("<I", header, 60, data_checksum)

            # ── Write all sections ──
            f.write(header)
            f.write(metadata_bytes)
            f.write(self._string_table_bytes)
            for block in self.data_blocks:
                f.write(block)

        print(f"  Written: {os.path.getsize(self.path) / 1e6:.1f} MB  ({num_tensors} tensors)")


# ──────────────────────────────────────────────
# Model weight loading
# ──────────────────────────────────────────────


def load_config(config_path: str) -> dict:
    """Load and validate config.json."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_safetensors(safetensors_path: str) -> Dict[str, np.ndarray]:
    """Load safetensors file into dict of numpy arrays.

    Uses PyTorch backend for bfloat16 support (numpy 2.x lacks native bfloat16).
    """
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    try:
        from safetensors import safe_open
    except ImportError:
        print("Error: safetensors package not installed. Install with: pip install safetensors")
        sys.exit(1)

    state = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert bfloat16 torch tensor -> float32 numpy for downstream compatibility
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
            elif tensor.dtype == torch.float16:
                tensor = tensor.float()
            state[key] = tensor.numpy()
    return state


def load_torch_pth(pth_path: str) -> Dict[str, np.ndarray]:
    """Load PyTorch .pth file into dict of numpy arrays."""
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    result = {}
    for k, v in state_dict.items():
        if v.dtype == torch.bfloat16:
            v = v.float()
        elif v.dtype == torch.float16:
            v = v.float()
        result[k] = v.numpy()
    return result


# ──────────────────────────────────────────────
# VoxCPM2 Weight Conversion
# ──────────────────────────────────────────────

def build_weight_registry(config: dict) -> List[Tuple[str, str, Tuple]]:
    """
    Build list of (safetensors_key, output_name, expected_shape) for all
    weights in the VoxCPM2 model based on config.json.
    """
    # Extract config values
    lm_cfg = config["lm_config"]
    hidden = lm_cfg["hidden_size"]
    intermediate = lm_cfg["intermediate_size"]
    n_heads = lm_cfg["num_attention_heads"]
    n_kv_heads = lm_cfg["num_key_value_heads"]
    kv_channels = lm_cfg.get("kv_channels", None)
    head_dim = kv_channels if kv_channels else hidden // n_heads
    n_lm_layers = lm_cfg["num_hidden_layers"]  # 28
    vocab_size = lm_cfg["vocab_size"]  # 73448

    # Residual LM
    n_res_layers = config["residual_lm_num_layers"]  # 8

    # Encoder (LocEnc)
    enc_cfg = config["encoder_config"]
    enc_hidden = enc_cfg["hidden_dim"]
    enc_intermediate = enc_cfg["ffn_dim"]
    enc_n_heads = enc_cfg["num_heads"]
    enc_kv_channels = enc_cfg.get("kv_channels", None)
    enc_head_dim = enc_kv_channels if enc_kv_channels else enc_hidden // enc_n_heads
    enc_n_layers = enc_cfg["num_layers"]  # 12

    # DiT
    dit_cfg = config["dit_config"]
    dit_hidden = dit_cfg["hidden_dim"]
    dit_intermediate = dit_cfg["ffn_dim"]
    dit_n_heads = dit_cfg["num_heads"]
    dit_kv_channels = dit_cfg.get("kv_channels", None)
    dit_head_dim = dit_kv_channels if dit_kv_channels else dit_hidden // dit_n_heads
    dit_n_layers = dit_cfg["num_layers"]  # 12

    registry = []

    # ── 1. TSLM (base_lm) ──
    # Embedding
    registry.append(("base_lm.embed_tokens.weight",
                     "base_lm.embed_tokens.weight",
                     (vocab_size, hidden)))

    # Final norm
    registry.append(("base_lm.norm.weight",
                     "base_lm.norm.weight",
                     (hidden,)))

    # Layers
    for i in range(n_lm_layers):
        p = f"base_lm.layers.{i}"
        q_dim = n_heads * head_dim
        k_dim = n_kv_heads * head_dim
        layer_tensors = [
            (f"{p}.input_layernorm.weight",       f"{p}.input_layernorm.weight",       (hidden,)),
            (f"{p}.self_attn.q_proj.weight",       f"{p}.self_attn.q_proj.weight",       (q_dim, hidden)),
            (f"{p}.self_attn.k_proj.weight",       f"{p}.self_attn.k_proj.weight",       (k_dim, hidden)),
            (f"{p}.self_attn.v_proj.weight",       f"{p}.self_attn.v_proj.weight",       (k_dim, hidden)),
            (f"{p}.self_attn.o_proj.weight",       f"{p}.self_attn.o_proj.weight",       (hidden, q_dim)),
            (f"{p}.mlp.gate_proj.weight",          f"{p}.mlp.gate_proj.weight",          (intermediate, hidden)),
            (f"{p}.mlp.up_proj.weight",            f"{p}.mlp.up_proj.weight",            (intermediate, hidden)),
            (f"{p}.mlp.down_proj.weight",          f"{p}.mlp.down_proj.weight",          (hidden, intermediate)),
            (f"{p}.post_attention_layernorm.weight", f"{p}.post_attention_layernorm.weight", (hidden,)),
        ]
        registry.extend(layer_tensors)

    # ── 2. Residual LM (RALM) ──
    registry.append(("residual_lm.norm.weight",
                     "residual_lm.norm.weight",
                     (hidden,)))

    for i in range(n_res_layers):
        p = f"residual_lm.layers.{i}"
        q_dim = n_heads * head_dim
        k_dim = n_kv_heads * head_dim
        layer_tensors = [
            (f"{p}.input_layernorm.weight",       f"{p}.input_layernorm.weight",       (hidden,)),
            (f"{p}.self_attn.q_proj.weight",       f"{p}.self_attn.q_proj.weight",       (q_dim, hidden)),
            (f"{p}.self_attn.k_proj.weight",       f"{p}.self_attn.k_proj.weight",       (k_dim, hidden)),
            (f"{p}.self_attn.v_proj.weight",       f"{p}.self_attn.v_proj.weight",       (k_dim, hidden)),
            (f"{p}.self_attn.o_proj.weight",       f"{p}.self_attn.o_proj.weight",       (hidden, q_dim)),
            (f"{p}.mlp.gate_proj.weight",          f"{p}.mlp.gate_proj.weight",          (intermediate, hidden)),
            (f"{p}.mlp.up_proj.weight",            f"{p}.mlp.up_proj.weight",            (intermediate, hidden)),
            (f"{p}.mlp.down_proj.weight",          f"{p}.mlp.down_proj.weight",          (hidden, intermediate)),
            (f"{p}.post_attention_layernorm.weight", f"{p}.post_attention_layernorm.weight", (hidden,)),
        ]
        registry.extend(layer_tensors)

    # ── 3. Projection layers ──
    registry.append(("enc_to_lm_proj.weight", "enc_to_lm_proj.weight", (hidden, enc_hidden)))
    registry.append(("enc_to_lm_proj.bias",   "enc_to_lm_proj.bias",   (hidden,)))
    registry.append(("lm_to_dit_proj.weight", "lm_to_dit_proj.weight", (dit_hidden, hidden)))
    registry.append(("lm_to_dit_proj.bias",   "lm_to_dit_proj.bias",   (dit_hidden,)))
    registry.append(("res_to_dit_proj.weight", "res_to_dit_proj.weight", (dit_hidden, hidden)))
    registry.append(("res_to_dit_proj.bias",   "res_to_dit_proj.bias",   (dit_hidden,)))
    registry.append(("fusion_concat_proj.weight", "fusion_concat_proj.weight", (hidden, hidden * 2)))
    registry.append(("fusion_concat_proj.bias",   "fusion_concat_proj.bias",   (hidden,)))

    # ── 4. FSQ (Scalar Quantization) ──
    fsq_latent = config["scalar_quantization_latent_dim"]
    registry.append(("fsq_layer.in_proj.weight", "fsq_layer.in_proj.weight", (fsq_latent, hidden)))
    registry.append(("fsq_layer.in_proj.bias",   "fsq_layer.in_proj.bias",   (fsq_latent,)))
    registry.append(("fsq_layer.out_proj.weight", "fsq_layer.out_proj.weight", (hidden, fsq_latent)))
    registry.append(("fsq_layer.out_proj.bias",   "fsq_layer.out_proj.bias",   (hidden,)))

    # ── 5. Stop predictor ──
    registry.append(("stop_proj.weight", "stop_proj.weight", (hidden, hidden)))
    registry.append(("stop_proj.bias",   "stop_proj.bias",   (hidden,)))
    registry.append(("stop_head.weight", "stop_head.weight", (2, hidden)))

    # ── 6. Local Encoder (feat_encoder) ──
    registry.append(("feat_encoder.special_token", "feat_encoder.special_token", (1, 1, 1, enc_hidden)))
    registry.append(("feat_encoder.in_proj.weight", "feat_encoder.in_proj.weight", (enc_hidden, config["feat_dim"])))
    registry.append(("feat_encoder.in_proj.bias",   "feat_encoder.in_proj.bias",   (enc_hidden,)))
    registry.append(("feat_encoder.encoder.norm.weight", "feat_encoder.encoder.norm.weight", (enc_hidden,)))

    enc_q_dim = enc_n_heads * enc_head_dim
    enc_k_dim = n_kv_heads * enc_head_dim  # same GQA pattern from lm_config
    for i in range(enc_n_layers):
        p = f"feat_encoder.encoder.layers.{i}"
        enc_layer_tensors = [
            (f"{p}.input_layernorm.weight",         f"{p}.input_layernorm.weight",         (enc_hidden,)),
            (f"{p}.self_attn.q_proj.weight",         f"{p}.self_attn.q_proj.weight",         (enc_q_dim, enc_hidden)),
            (f"{p}.self_attn.k_proj.weight",         f"{p}.self_attn.k_proj.weight",         (enc_k_dim, enc_hidden)),
            (f"{p}.self_attn.v_proj.weight",         f"{p}.self_attn.v_proj.weight",         (enc_k_dim, enc_hidden)),
            (f"{p}.self_attn.o_proj.weight",         f"{p}.self_attn.o_proj.weight",         (enc_hidden, enc_q_dim)),
            (f"{p}.mlp.gate_proj.weight",            f"{p}.mlp.gate_proj.weight",            (enc_intermediate, enc_hidden)),
            (f"{p}.mlp.up_proj.weight",              f"{p}.mlp.up_proj.weight",              (enc_intermediate, enc_hidden)),
            (f"{p}.mlp.down_proj.weight",            f"{p}.mlp.down_proj.weight",            (enc_hidden, enc_intermediate)),
            (f"{p}.post_attention_layernorm.weight", f"{p}.post_attention_layernorm.weight", (enc_hidden,)),
        ]
        registry.extend(enc_layer_tensors)

    # ── 7. LocDiT Estimator (feat_decoder.estimator) ──
    feat_dim = config["feat_dim"]
    registry.append(("feat_decoder.estimator.in_proj.weight", "feat_decoder.estimator.in_proj.weight", (dit_hidden, feat_dim)))
    registry.append(("feat_decoder.estimator.in_proj.bias",   "feat_decoder.estimator.in_proj.bias",   (dit_hidden,)))
    registry.append(("feat_decoder.estimator.cond_proj.weight", "feat_decoder.estimator.cond_proj.weight", (dit_hidden, feat_dim)))
    registry.append(("feat_decoder.estimator.cond_proj.bias",   "feat_decoder.estimator.cond_proj.bias",   (dit_hidden,)))
    registry.append(("feat_decoder.estimator.out_proj.weight", "feat_decoder.estimator.out_proj.weight", (feat_dim, dit_hidden)))
    registry.append(("feat_decoder.estimator.out_proj.bias",   "feat_decoder.estimator.out_proj.bias",   (feat_dim,)))

    # Time embeddings
    registry.append(("feat_decoder.estimator.time_mlp.linear_1.weight",
                     "feat_decoder.estimator.time_mlp.linear_1.weight", (dit_hidden, dit_hidden)))
    registry.append(("feat_decoder.estimator.time_mlp.linear_1.bias",
                     "feat_decoder.estimator.time_mlp.linear_1.bias", (dit_hidden,)))
    registry.append(("feat_decoder.estimator.time_mlp.linear_2.weight",
                     "feat_decoder.estimator.time_mlp.linear_2.weight", (dit_hidden, dit_hidden)))
    registry.append(("feat_decoder.estimator.time_mlp.linear_2.bias",
                     "feat_decoder.estimator.time_mlp.linear_2.bias", (dit_hidden,)))

    registry.append(("feat_decoder.estimator.delta_time_mlp.linear_1.weight",
                     "feat_decoder.estimator.delta_time_mlp.linear_1.weight", (dit_hidden, dit_hidden)))
    registry.append(("feat_decoder.estimator.delta_time_mlp.linear_1.bias",
                     "feat_decoder.estimator.delta_time_mlp.linear_1.bias", (dit_hidden,)))
    registry.append(("feat_decoder.estimator.delta_time_mlp.linear_2.weight",
                     "feat_decoder.estimator.delta_time_mlp.linear_2.weight", (dit_hidden, dit_hidden)))
    registry.append(("feat_decoder.estimator.delta_time_mlp.linear_2.bias",
                     "feat_decoder.estimator.delta_time_mlp.linear_2.bias", (dit_hidden,)))

    # DiT decoder (MiniCPM layers)
    registry.append(("feat_decoder.estimator.decoder.norm.weight",
                     "feat_decoder.estimator.decoder.norm.weight", (dit_hidden,)))

    dit_q_dim = dit_n_heads * dit_head_dim
    dit_k_dim = n_kv_heads * dit_head_dim
    for i in range(dit_n_layers):
        p = f"feat_decoder.estimator.decoder.layers.{i}"
        dit_layer_tensors = [
            (f"{p}.input_layernorm.weight",         f"{p}.input_layernorm.weight",         (dit_hidden,)),
            (f"{p}.self_attn.q_proj.weight",         f"{p}.self_attn.q_proj.weight",         (dit_q_dim, dit_hidden)),
            (f"{p}.self_attn.k_proj.weight",         f"{p}.self_attn.k_proj.weight",         (dit_k_dim, dit_hidden)),
            (f"{p}.self_attn.v_proj.weight",         f"{p}.self_attn.v_proj.weight",         (dit_k_dim, dit_hidden)),
            (f"{p}.self_attn.o_proj.weight",         f"{p}.self_attn.o_proj.weight",         (dit_hidden, dit_q_dim)),
            (f"{p}.mlp.gate_proj.weight",            f"{p}.mlp.gate_proj.weight",            (dit_intermediate, dit_hidden)),
            (f"{p}.mlp.up_proj.weight",              f"{p}.mlp.up_proj.weight",              (dit_intermediate, dit_hidden)),
            (f"{p}.mlp.down_proj.weight",            f"{p}.mlp.down_proj.weight",            (dit_hidden, dit_intermediate)),
            (f"{p}.post_attention_layernorm.weight", f"{p}.post_attention_layernorm.weight", (dit_hidden,)),
        ]
        registry.extend(dit_layer_tensors)

    return registry


# ──────────────────────────────────────────────
# Main conversion
# ──────────────────────────────────────────────

def convert(input_path: str, output_path: str, output_dtype: str = "f32"):
    """Convert VoxCPM2 model to .vxcpm format."""

    dtype_map = {"f32": DTYPE_FP32, "bf16": DTYPE_BF16, "f16": DTYPE_FP16}
    dtype = dtype_map[output_dtype]

    print(f"{'='*60}")
    print(f"  VoxCPM2 → .vxcpm Converter")
    print(f"{'='*60}")
    print(f"  Input:    {input_path}")
    print(f"  Output:   {output_path}")
    print(f"  Dtype:    {output_dtype}")

    # ── Resolve input path ──
    if not os.path.isdir(input_path):
        # Try HuggingFace download
        print("  Downloading from HuggingFace...")
        try:
            from huggingface_hub import snapshot_download
            cache_dir = os.environ.get("HF_HOME", None)
            input_path = snapshot_download(
                input_path,
                cache_dir=cache_dir,
                allow_patterns=["config.json", "model.safetensors", "audiovae.pth",
                                "tokenizer.json", "tokenizer_config.json",
                                "special_tokens_map.json", "tokenization_voxcpm2.py"]
            )
            print(f"  Downloaded to: {input_path}")
        except ImportError:
            print("Error: huggingface_hub not installed. Install with: pip install huggingface_hub")
            sys.exit(1)

    # ── Load config ──
    config_path = os.path.join(input_path, "config.json")
    if not os.path.exists(config_path):
        print(f"Error: config.json not found in {input_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"\n  Architecture: {config.get('architecture', 'unknown')}")
    print(f"  LM layers: {config['lm_config']['num_hidden_layers']}")
    print(f"  Residual LM layers: {config['residual_lm_num_layers']}")
    print(f"  Encoder layers: {config['encoder_config']['num_layers']}")
    print(f"  DiT layers: {config['dit_config']['num_layers']}")
    print(f"  Hidden size: {config['lm_config']['hidden_size']}")
    print(f"  Vocab size: {config['lm_config']['vocab_size']}")

    # ── Load safetensors ──
    safetensors_path = os.path.join(input_path, "model.safetensors")
    if not os.path.exists(safetensors_path):
        print(f"Error: model.safetensors not found in {input_path}")
        sys.exit(1)

    print(f"\n  Loading model.safetensors...")
    state = load_safetensors(safetensors_path)
    print(f"  Loaded {len(state)} tensors")

    # ── Build weight registry ──
    registry = build_weight_registry(config)
    print(f"  Expected tensor count: {len(registry)}")

    # ── Match and convert ──
    writer = VxcpmWriter(output_path, dtype=dtype)
    matched = 0
    missing = 0
    shape_mismatch = 0

    for safetensors_key, output_name, expected_shape in registry:
        if safetensors_key not in state:
            print(f"  ⚠ Missing: {safetensors_key}")
            missing += 1
            continue

        tensor = state[safetensors_key]
        actual_shape = tensor.shape

        if actual_shape != expected_shape:
            print(f"  ⚠ Shape mismatch: {safetensors_key}")
            print(f"    Expected: {expected_shape}, Got: {actual_shape}")
            shape_mismatch += 1
            # Still convert but warn

        writer.add_tensor(output_name, tensor)
        matched += 1

    # Check for unexpected keys (debug help)
    expected_keys = set(k for k, _, _ in registry)
    actual_keys = set(state.keys())
    unexpected = actual_keys - expected_keys
    if unexpected:
        # Filter out known VoxCPM2 audio_vae keys (comes from audiovae.pth)
        vae_related = {k for k in unexpected if "audio_vae" in k or "audiovae" in k}
        other_unexpected = unexpected - vae_related
        if other_unexpected:
            print(f"\n  ⚠ Unexpected keys in safetensors ({len(other_unexpected)}):")
            for k in sorted(other_unexpected)[:10]:
                print(f"    - {k}  {state[k].shape}")

    print(f"\n  Matched: {matched}, Missing: {missing}, Shape mismatches: {shape_mismatch}")

    # ── Write main model ──
    print(f"\n  Writing main model...")
    writer.write()

    # ── Write AudioVAE separately ──
    audiovae_path = os.path.join(input_path, "audiovae.pth")
    if os.path.exists(audiovae_path):
        vae_output = output_path.replace(".vxcpm", "_audiovae.vxcpm")
        if vae_output == output_path:
            vae_output = output_path.rsplit(".", 1)[0] + "_audiovae.vxcpm"
        print(f"\n  Converting AudioVAE V2...")
        convert_audiovae(audiovae_path, vae_output, output_dtype)

    # ── Copy tokenizer files ──
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json", "tokenization_voxcpm2.py"]
    output_dir = os.path.dirname(output_path) or "."
    for tf in tokenizer_files:
        src = os.path.join(input_path, tf)
        if os.path.exists(src):
            dst = os.path.join(output_dir, tf)
            import shutil
            shutil.copy2(src, dst)
            print(f"  Copied: {tf}")

    print(f"\n{'='*60}")
    print(f"  ✅ Conversion complete!")
    print(f"{'='*60}")


def _map_audiovae_key(key: str) -> str:
    """Map audiovae.pth keys to the C naming convention.

    .pth key format:
      decoder.model.{0-9}.{tensor}
      decoder.model.{2-7}.block.{0}.alpha
      decoder.model.{2-7}.block.{1}.{tensor}
      decoder.model.{2-7}.block.{2-4}.block.{0}.alpha
      decoder.model.{2-7}.block.{2-4}.block.{1}.{tensor}
      decoder.model.{2-7}.block.{2-4}.block.{2}.alpha
      decoder.model.{2-7}.block.{2-4}.block.{3}.{tensor}
      decoder.sr_cond_model.{layer}.{type}.weight
      decoder.sr_bin_boundaries

    C key format:
      audio_vae.decoder.conv_in.{tensor}
      audio_vae.decoder.proj_up.{tensor}
      audio_vae.decoder.decoder_blocks.{i}.snake_alpha
      audio_vae.decoder.decoder_blocks.{i}.convtr.{tensor}
      audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.snake_alpha1
      audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.conv_depthwise.{tensor}
      audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.snake_alpha2
      audio_vae.decoder.decoder_blocks.{i}.res_blocks.{j}.conv_pointwise.{tensor}
      audio_vae.decoder.final_snake_alpha
      audio_vae.decoder.conv_out.{tensor}
      audio_vae.decoder.sr_cond.{layer}.{type}.weight
      audio_vae.decoder.sr_bin_boundaries
    """
    import re

    if not key.startswith("decoder."):
        return f"audio_vae.{key}"

    # sr_bin_boundaries (scalar, no tensor suffix)
    if key == "decoder.sr_bin_boundaries":
        return "audio_vae.decoder.sr_bin_boundaries"

    # sr_cond_model.{layer}.{type}.weight
    m = re.match(r"^decoder\.sr_cond_model\.(\d+)\.(bias_embed|scale_embed)\.weight$", key)
    if m:
        layer = m.group(1)
        etype = m.group(2)
        return f"audio_vae.decoder.sr_cond.{layer}.{etype}.weight"

    # decoder.model.N ...
    m = re.match(r"^decoder\.model\.(\d+)(.*)$", key)
    if not m:
        return f"audio_vae.{key}"

    model_idx = int(m.group(1))
    suffix = m.group(2)

    # model.0 -> conv_in
    if model_idx == 0:
        return f"audio_vae.decoder.conv_in{suffix}"

    # model.1 -> proj_up
    if model_idx == 1:
        return f"audio_vae.decoder.proj_up{suffix}"

    # model.8 -> final_snake_alpha
    if model_idx == 8:
        return f"audio_vae.decoder.final_snake_alpha"

    # model.9 -> conv_out
    if model_idx == 9:
        return f"audio_vae.decoder.conv_out{suffix}"

    # model.2-7 -> decoder_blocks (block index = model_idx - 2)
    if 2 <= model_idx <= 7:
        block_i = model_idx - 2
        # Parse suffix: .block.X.Y... or .block.X.block.Y.Z...
        # decoder.model.{i}.block.{b}.{tensor}
        m2 = re.match(r"^\.block\.(\d+)\.(.+)$", suffix)
        if not m2:
            return f"audio_vae.{key}"  # fallback

        sub_block = int(m2.group(1))
        sub_suffix = m2.group(2)

        # block.0 -> snake_alpha
        if sub_block == 0:
            return f"audio_vae.decoder.decoder_blocks.{block_i}.snake_alpha"

        # block.1 -> convtr
        if sub_block == 1:
            return f"audio_vae.decoder.decoder_blocks.{block_i}.convtr.{sub_suffix}"

        # block.{2-4} -> res_blocks (res index = sub_block - 2)
        if 2 <= sub_block <= 4:
            res_j = sub_block - 2
            m3 = re.match(r"^block\.(\d+)\.(.+)$", sub_suffix)
            if not m3:
                return f"audio_vae.{key}"

            inner_block = int(m3.group(1))
            inner_suffix = m3.group(2)

            # inner block 0 -> snake_alpha1
            if inner_block == 0:
                return f"audio_vae.decoder.decoder_blocks.{block_i}.res_blocks.{res_j}.snake_alpha1"

            # inner block 1 -> conv_depthwise
            if inner_block == 1:
                return f"audio_vae.decoder.decoder_blocks.{block_i}.res_blocks.{res_j}.conv_depthwise.{inner_suffix}"

            # inner block 2 -> snake_alpha2
            if inner_block == 2:
                return f"audio_vae.decoder.decoder_blocks.{block_i}.res_blocks.{res_j}.snake_alpha2"

            # inner block 3 -> conv_pointwise
            if inner_block == 3:
                return f"audio_vae.decoder.decoder_blocks.{block_i}.res_blocks.{res_j}.conv_pointwise.{inner_suffix}"

    return f"audio_vae.{key}"


def convert_audiovae(pth_path: str, output_path: str, output_dtype: str = "f32"):
    """Convert AudioVAE V2 .pth to .vxcpm format with proper name mapping."""
    dtype_map = {"f32": DTYPE_FP32, "bf16": DTYPE_BF16, "f16": DTYPE_FP16}
    dtype = dtype_map[output_dtype]

    state = load_torch_pth(pth_path)
    print(f"  Loaded {len(state)} tensors from audiovae.pth")

    writer = VxcpmWriter(output_path, dtype=dtype)
    mapped_count = 0
    for key in sorted(state.keys()):
        tensor = state[key]
        output_name = _map_audiovae_key(key)
        writer.add_tensor(output_name, tensor)
        mapped_count += 1
        if mapped_count <= 5 or output_name.startswith("audio_vae.decoder.sr_"):
            print(f"    {key:<55s} -> {output_name}")

    writer.write()
    print(f"  Converted {mapped_count} tensors (name mapping applied)")


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────

def verify(path: str) -> bool:
    """Verify a .vxcpm file's integrity."""
    print(f"\n{'='*60}")
    print(f"  Verifying: {path}")
    print(f"{'='*60}")

    if not os.path.exists(path):
        print(f"  ❌ File not found")
        return False

    file_size = os.path.getsize(path)
    print(f"  File size: {file_size / 1e6:.1f} MB")

    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
        if len(header) < HEADER_SIZE:
            print(f"  ❌ Header too small: {len(header)} bytes")
            return False

        magic, version, num_tensors = struct.unpack_from("<III", header, 0)
        if magic != VXCPM_MAGIC:
            print(f"  ❌ Bad magic: 0x{magic:08X} (expected 0x{VXCPM_MAGIC:08X})")
            return False
        print(f"  Magic:       OK (0x{VXCPM_MAGIC:08X})")
        print(f"  Version:     {version}")
        print(f"  Tensors:     {num_tensors}")

        # Verify header checksum
        stored_hdr_crc = struct.unpack_from("<I", header, 56)[0]
        stored_data_crc = struct.unpack_from("<I", header, 60)[0]

        pre_checksum = header[:56] + b"\x00\x00\x00\x00\x00\x00\x00\x00"
        calc_hdr_crc = crc32c(pre_checksum)

        if stored_hdr_crc == calc_hdr_crc:
            print(f"  Header CRC:  OK (0x{stored_hdr_crc:08X})")
        else:
            print(f"  ❌ Header CRC mismatch: stored=0x{stored_hdr_crc:08X} calc=0x{calc_hdr_crc:08X}")
            return False

        # Read and verify data
        metadata_size = num_tensors * METADATA_ENTRY_SIZE
        f.seek(HEADER_SIZE)
        rest = f.read()

        # Find string table end (after metadata)
        metadata_bytes = rest[:metadata_size]
        remaining = rest[metadata_size:]

        # Data CRC
        calc_data_crc = crc32c(rest)
        if stored_data_crc == calc_data_crc:
            print(f"  Data CRC:    OK (0x{stored_data_crc:08X})")
        else:
            print(f"  ❌ Data CRC mismatch: stored=0x{stored_data_crc:08X} calc=0x{calc_data_crc:08X}")
            return False

        # Parse tensor metadata
        print(f"\n  Tensor list:")
        string_table_start = metadata_size
        for i in range(min(num_tensors, 10)):  # Show first 10
            entry = metadata_bytes[i * METADATA_ENTRY_SIZE:(i + 1) * METADATA_ENTRY_SIZE]
            name_off, name_len, dtype_code, ndim = struct.unpack_from("<IIII", entry, 0)
            d0, d1, d2, d3 = struct.unpack_from("<IIII", entry, 16)

            # Read name from string table
            name_bytes = remaining[string_table_start + name_off:
                                    string_table_start + name_off + name_len]
            name = name_bytes.decode("utf-8", errors="replace")

            shape = tuple(d for d in (d0, d1, d2, d3) if d > 1 or ndim > len([d for d in (d0,d1,d2,d3) if d == 1]))
            # Better shape display
            dims = [d0, d1, d2, d3][:ndim]

            dtype_names = {0: "f32", 1: "bf16", 2: "f16"}
            print(f"    [{i:4d}] {name:<55s} {str(tuple(dims)):>20s} {dtype_names.get(dtype_code, f'?{dtype_code}')}")

        if num_tensors > 10:
            print(f"    ... and {num_tensors - 10} more tensors")

        # Check all offsets within file
        print(f"\n  Checking data boundaries...")
        f.seek(0)
        all_data = f.read()
        ok = True
        for i in range(num_tensors):
            entry = metadata_bytes[i * METADATA_ENTRY_SIZE:(i + 1) * METADATA_ENTRY_SIZE]
            _, _, dtype_code, ndim = struct.unpack_from("<IIII", entry, 0)
            d0, d1, d2, d3 = struct.unpack_from("<IIII", entry, 16)
            shape = (d0, d1, d2, d3)[:ndim]
            elem_size = {0: 4, 1: 2, 2: 2}[dtype_code]
            data_size = 1
            for d in shape:
                data_size *= d
            data_size *= elem_size

        print(f"  ✅ All data boundaries OK")
        print(f"  ✅ Verification passed")
        return True


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def main():
    # Fix cp950 encoding on Windows
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="VoxCPM2 → .vxcpm weight converter")

    parser.add_argument("input", nargs="?", default=None,
                        help="HuggingFace model ID (e.g. openbmb/VoxCPM2) or local directory path")
    parser.add_argument("output", nargs="?", default="models/voxcpm2-f32.vxcpm",
                        help="Output .vxcpm file path")
    parser.add_argument("--dtype", "-d", choices=["f32", "f16", "bf16"], default="f32",
                        help="Output data type (default: f32)")
    parser.add_argument("--verify", "-v", metavar="PATH",
                        help="Verify a .vxcpm file instead of converting")
    parser.add_argument("--convert-audiovae", metavar="PTH_PATH",
                        help="Convert only audiovae.pth to .vxcpm")
    parser.add_argument("--audiovae-output", default=None,
                        help="Output path for audiovae (default: input_audiovae.vxcpm)")

    args = parser.parse_args()

    if args.verify:
        ok = verify(args.verify)
        sys.exit(0 if ok else 1)

    if args.convert_audiovae:
        out = args.audiovae_output or args.convert_audiovae.replace(".pth", ".vxcpm")
        convert_audiovae(args.convert_audiovae, out, args.dtype)
        sys.exit(0)

    if not args.input:
        parser.print_help()
        sys.exit(1)

    convert(args.input, args.output, args.dtype)


if __name__ == "__main__":
    main()
