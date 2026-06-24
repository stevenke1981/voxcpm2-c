#!/usr/bin/env python3
"""
VoxCPM2 .vxcpm 權重量化工具

將已轉換的 .vxcpm 權重在不同精度之間轉換，支援:
  - f32 → f16
  - f32 → q4
  - f32 → q8
  - f16 → f32 (解除量化)
  - f16 → q4

使用方式:
    python scripts/quantize.py input.vxcpm output.vxcpm --quant q4
    python scripts/quantize.py input.vxcpm output.vxcpm --quant f16
    python scripts/quantize.py input.vxcpm output.vxcpm --quant q8
"""

import argparse
import os
import struct
import sys
import hashlib
import numpy as np

VOXCPM_MAGIC = b"VXCPM2\0"
VOXCPM_HEADER_SIZE = 256
Q4_BLOCK_SIZE = 32
QUANT_TYPES = {"f32": 0, "f16": 1, "q4": 2, "q8": 3}

# ═══════════════════════════════════════════════════════════════
# Tensor Operations
# ═══════════════════════════════════════════════════════════════

def float16_to_float32(f16_bytes: bytes) -> np.ndarray:
    """FP16 bytes → FP32 numpy array"""
    return np.frombuffer(f16_bytes, dtype=np.float16).astype(np.float32)

def float32_to_float16(f32: np.ndarray) -> bytes:
    """FP32 → FP16 bytes"""
    return f32.astype(np.float16).tobytes()

def quantize_q4_block(block: np.ndarray) -> bytes:
    """Q4 block quantization (32 FP32 → 16+2+2=20 bytes)"""
    w_min = float(block.min())
    w_max = float(block.max())
    scale = (w_max - w_min) / 15.0 if w_max != w_min else 1.0
    q = np.round((block - w_min) / scale).clip(0, 15).astype(np.uint8)
    
    packed = bytearray()
    for i in range(0, Q4_BLOCK_SIZE, 2):
        packed.append((q[i] << 4) | q[i + 1])
    packed.extend(np.float16(scale).tobytes())
    packed.extend(np.float16(w_min).tobytes())
    return bytes(packed)

def dequantize_q4_block(data: bytes) -> np.ndarray:
    """Q4 block dequantization (20 bytes → 32 FP32)"""
    q_packed = np.frombuffer(data[:16], dtype=np.uint8)
    scale = np.frombuffer(data[16:18], dtype=np.float16)[0]
    w_min = np.frombuffer(data[18:20], dtype=np.float16)[0]
    
    q = np.zeros(Q4_BLOCK_SIZE, dtype=np.uint8)
    for i in range(16):
        q[i * 2] = (q_packed[i] >> 4) & 0xF
        q[i * 2 + 1] = q_packed[i] & 0xF
    
    return q.astype(np.float32) * scale + w_min

def quantize_q8_block(block: np.ndarray) -> bytes:
    """Q8 block quantization (32 FP32 → 32+4+4=40 bytes)"""
    w_min = float(block.min())
    w_max = float(block.max())
    scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
    q = np.round((block - w_min) / scale).clip(0, 255).astype(np.int8)
    return q.tobytes() + struct.pack('f', scale) + struct.pack('f', w_min)

def dequantize_q8_block(data: bytes) -> np.ndarray:
    """Q8 block dequantization (40 bytes → 32 FP32)"""
    q = np.frombuffer(data[:32], dtype=np.int8).astype(np.float32)
    scale = struct.unpack('f', data[32:36])[0]
    w_min = struct.unpack('f', data[36:40])[0]
    return q * scale + w_min

# ═══════════════════════════════════════════════════════════════
# Tensor Size Calculation
# ═══════════════════════════════════════════════════════════════

def compute_tensor_size(nelements: int, quant_type: int) -> int:
    """計算 tensor 在給定量化類型下的位元組大小"""
    # 每個 tensor 有 20 bytes header
    header_size = 20
    
    if quant_type == 0:  # f32
        return header_size + nelements * 4
    elif quant_type == 1:  # f16
        return header_size + nelements * 2
    elif quant_type == 2:  # q4
        n_blocks = (nelements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
        return header_size + n_blocks * 20
    elif quant_type == 3:  # q8
        n_blocks = (nelements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
        return header_size + n_blocks * 40
    else:
        raise ValueError(f"Unknown quant type: {quant_type}")

# ═══════════════════════════════════════════════════════════════
# Conversion
# ═══════════════════════════════════════════════════════════════

def convert_quant(input_path: str, output_path: str, target_quant: str):
    """轉換 .vxcpm 權重的量化類型"""
    
    print(f"🔄 Quantizing {input_path} → {target_quant}")
    
    with open(input_path, 'rb') as fin:
        data = fin.read()
    
    # ─── Parse header ───
    header = data[:VOXCPM_HEADER_SIZE]
    magic = header[:8]
    
    if magic != VOXCPM_MAGIC:
        print(f"❌ Invalid magic: {magic}")
        sys.exit(1)
    
    version = struct.unpack_from('I', header, 8)[0]
    n_layers = struct.unpack_from('I', header, 12)[0]
    n_layers_enc = struct.unpack_from('I', header, 16)[0]
    n_ralm_layers = struct.unpack_from('I', header, 20)[0]
    n_dit_layers = struct.unpack_from('I', header, 24)[0]
    d_model = struct.unpack_from('I', header, 28)[0]
    d_ff = struct.unpack_from('I', header, 32)[0]
    n_heads = struct.unpack_from('I', header, 36)[0]
    n_kv_heads = struct.unpack_from('I', header, 40)[0]
    head_dim = struct.unpack_from('I', header, 44)[0]
    max_seq_len = struct.unpack_from('I', header, 48)[0]
    vocab_size = struct.unpack_from('I', header, 52)[0]
    src_quant = struct.unpack_from('I', header, 56)[0]
    sample_rate = struct.unpack_from('I', header, 60)[0]
    audio_vae_dim = struct.unpack_from('I', header, 64)[0]
    rope_theta = struct.unpack_from('f', header, 68)[0]
    n_layer_groups = struct.unpack_from('I', header, 80)[0]
    
    quant_names = {v: k for k, v in QUANT_TYPES.items()}
    target_type = QUANT_TYPES[target_quant]
    
    print(f"  Source quant: {quant_names.get(src_quant, src_quant)}")
    print(f"  Target quant: {target_quant}")
    print(f"  Layers: {n_layers}, Groups: {n_layer_groups}")
    
    if src_quant == target_type:
        print("⚠ Source and target quant are the same, copying file.")
        import shutil
        shutil.copy2(input_path, output_path)
        return
    
    # ─── Parse layer index ───
    index_size = (n_layer_groups + 1) * 8
    layer_offsets = struct.unpack(
        f'{n_layer_groups + 1}Q',
        data[VOXCPM_HEADER_SIZE:VOXCPM_HEADER_SIZE + index_size]
    )
    
    # ─── Build new file ───
    with open(output_path, 'wb') as fout:
        # Header (placeholder)
        fout.write(b'\0' * VOXCPM_HEADER_SIZE)
        
        # Layer index (placeholder)
        index_start = fout.tell()
        fout.write(b'\0' * index_size)
        
        new_offsets = []
        total_input_size = 0
        total_output_size = 0
        
        for g in range(n_layer_groups):
            new_offsets.append(fout.tell())
            start = layer_offsets[g]
            end = layer_offsets[g + 1]
            
            if end <= start:
                continue
            
            chunk = data[start:end]
            pos = 0
            
            while pos < len(chunk):
                if pos + 20 > len(chunk):
                    break
                
                # Read tensor header
                name_hash = struct.unpack_from('I', chunk, pos)[0]
                ndim = struct.unpack_from('I', chunk, pos + 4)[0]
                dims = struct.unpack_from('III', chunk, pos + 8)
                
                # Compute n_elements
                actual_dims = dims[:ndim]
                nelements = 1
                for d in actual_dims:
                    nelements *= d
                
                hdr_size = 20
                
                # Read source tensor data
                src_size = compute_tensor_size(nelements, src_quant) - hdr_size
                tensor_data = chunk[pos + hdr_size:pos + hdr_size + src_size]
                
                if len(tensor_data) < src_size:
                    break  # Truncated, stop
                
                # Dequantize to FP32
                if src_quant == 1:  # f16
                    f32_data = float16_to_float32(tensor_data)
                elif src_quant == 0:  # f32
                    f32_data = np.frombuffer(tensor_data, dtype=np.float32)
                elif src_quant == 2:  # q4
                    blocks = []
                    n_blocks = (nelements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
                    for b in range(n_blocks):
                        bstart = b * 20
                        blocks.append(dequantize_q4_block(tensor_data[bstart:bstart+20]))
                    f32_data = np.concatenate(blocks)[:nelements]
                elif src_quant == 3:  # q8
                    blocks = []
                    n_blocks = (nelements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
                    for b in range(n_blocks):
                        bstart = b * 40
                        blocks.append(dequantize_q8_block(tensor_data[bstart:bstart+40]))
                    f32_data = np.concatenate(blocks)[:nelements]
                else:
                    raise ValueError(f"Unsupported source quant: {src_quant}")
                
                # Re-quantize to target
                if target_type == 0:  # f32
                    new_data = f32_data.tobytes()
                elif target_type == 1:  # f16
                    new_data = float32_to_float16(f32_data)
                elif target_type == 2:  # q4
                    packed = bytearray()
                    n_blocks = (nelements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
                    for b in range(n_blocks):
                        start_b = b * Q4_BLOCK_SIZE
                        end_b = min(start_b + Q4_BLOCK_SIZE, nelements)
                        block = f32_data[start_b:end_b]
                        if len(block) < Q4_BLOCK_SIZE:
                            block = np.pad(block, (0, Q4_BLOCK_SIZE - len(block)), 'constant')
                        packed.extend(quantize_q4_block(block))
                    new_data = bytes(packed)
                elif target_type == 3:  # q8
                    packed = bytearray()
                    n_blocks = (nelements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE
                    for b in range(n_blocks):
                        start_b = b * Q4_BLOCK_SIZE
                        end_b = min(start_b + Q4_BLOCK_SIZE, nelements)
                        block = f32_data[start_b:end_b]
                        if len(block) < Q4_BLOCK_SIZE:
                            block = np.pad(block, (0, Q4_BLOCK_SIZE - len(block)), 'constant')
                        packed.extend(quantize_q8_block(block))
                    new_data = bytes(packed)
                else:
                    raise ValueError(f"Unsupported target quant: {target_quant}")
                
                # Write tensor header + data
                fout.write(chunk[pos:pos + hdr_size])
                fout.write(new_data)
                
                total_input_size += src_size
                total_output_size += len(new_data)
                
                pos += hdr_size + src_size
            
            # Print per-group progress
            print(f"  Group {g}: written {fout.tell() - new_offsets[-1]} bytes")
        
        # Final offset
        new_offsets.append(fout.tell())
        
        # ─── Write header ───
        fout.seek(0)
        new_header = bytearray(VOXCPM_HEADER_SIZE)
        struct.pack_into('8s', new_header, 0, VOXCPM_MAGIC)
        struct.pack_into('I', new_header, 8, version)
        struct.pack_into('I', new_header, 12, n_layers)
        struct.pack_into('I', new_header, 16, n_layers_enc)
        struct.pack_into('I', new_header, 20, n_ralm_layers)
        struct.pack_into('I', new_header, 24, n_dit_layers)
        struct.pack_into('I', new_header, 28, d_model)
        struct.pack_into('I', new_header, 32, d_ff)
        struct.pack_into('I', new_header, 36, n_heads)
        struct.pack_into('I', new_header, 40, n_kv_heads)
        struct.pack_into('I', new_header, 44, head_dim)
        struct.pack_into('I', new_header, 48, max_seq_len)
        struct.pack_into('I', new_header, 52, vocab_size)
        struct.pack_into('I', new_header, 56, target_type)
        struct.pack_into('I', new_header, 60, sample_rate)
        struct.pack_into('I', new_header, 64, audio_vae_dim)
        struct.pack_into('f', new_header, 68, rope_theta)
        struct.pack_into('Q', new_header, 72, 0)  # checksum placeholder
        struct.pack_into('I', new_header, 80, len(new_offsets) - 1)
        fout.write(bytes(new_header))
        
        # ─── Write layer index ───
        fout.seek(index_start)
        for offset in new_offsets:
            fout.write(struct.pack('Q', offset))
        
        # ─── Compute checksum ───
        fout.seek(0)
        file_data = fout.read()
        checksum = hashlib.sha256(file_data[:72] + b'\0'*8 + file_data[80:]).digest()
        checksum_int = int.from_bytes(checksum[:8], 'little')
        
        fout.seek(72)
        fout.write(struct.pack('Q', checksum_int))
    
    # Statistics
    ratio = total_output_size / total_input_size if total_input_size else 1.0
    print(f"\n✅ Quantization complete!")
    print(f"  Input size:  {total_input_size / 1e6:.2f} MB")
    print(f"  Output size: {total_output_size / 1e6:.2f} MB")
    print(f"  Ratio:       {ratio:.2%}")
    print(f"  File:        {output_path}")

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VoxCPM2 .vxcpm weight quantization tool")
    parser.add_argument('input', type=str, help='Input .vxcpm file')
    parser.add_argument('output', type=str, help='Output .vxcpm file')
    parser.add_argument('--quant', '-q', type=str, required=True,
        choices=['f32', 'f16', 'q4', 'q8'],
        help='Target quantization type')
    parser.add_argument('--verify', action='store_true',
        help='Verify output after conversion')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(1)
    
    convert_quant(args.input, args.output, args.quant)
    
    if args.verify:
        print("\n🔍 Verifying output...")
        import subprocess
        subprocess.run([sys.executable,
            os.path.join(os.path.dirname(__file__), "convert_weights.py"),
            "--verify", args.output])

if __name__ == "__main__":
    main()
