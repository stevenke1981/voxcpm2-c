#!/usr/bin/env python3
"""
VoxCPM2-C 測試資料產生器

產生參考測試資料，用於 C 程式的數值驗證對照。
所有輸出使用 numpy 二進位格式 (.npy)，確保跨平台一致性。

使用方式:
    # 產生所有測試資料
    python scripts/gen_test_data.py
    
    # 產生特定測試資料
    python scripts/gen_test_data.py --only matmul
    python scripts/gen_test_data.py --only rmsnorm
    python scripts/gen_test_data.py --only quantize
    
    # 指定輸出目錄
    python scripts/gen_test_data.py --output tests/test_data/
"""

import argparse
import os
import struct
import sys
import numpy as np

SEED = 42
RNG = np.random.RandomState(SEED)

OUTPUT_DIR = "tests/test_data"

# ═══════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════

def save_npy(name: str, data: np.ndarray):
    """儲存為 .npy 格式"""
    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(path, data)
    print(f"  ✓ {path}  ({data.shape}, {data.dtype})")

def save_bin(name: str, data: bytes):
    """儲存為原始二進位格式"""
    path = os.path.join(OUTPUT_DIR, f"{name}.bin")
    with open(path, 'wb') as f:
        f.write(data)
    print(f"  ✓ {path}  ({len(data)} bytes)")

def save_txt(name: str, data):
    """儲存為文字格式（用於 tokenizer 測試）"""
    path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(str(data))
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
# Tensor Operations (matmul, add, mul, etc.)
# ═══════════════════════════════════════════════════════════════

def gen_tensor_tests():
    """產生張量運算測試資料"""
    print("\n=== Tensor Operations ===")
    
    # Simple matmul: (M,K) @ (K,N) → (M,N)
    for name, a_shape, b_shape in [
        ("matmul_small",  (4, 8),   (8, 4)),
        ("matmul_medium", (32, 64), (64, 32)),
        ("matmul_large",  (128, 256), (256, 128)),
    ]:
        a = RNG.randn(*a_shape).astype(np.float32)
        b = RNG.randn(*b_shape).astype(np.float32)
        c = a @ b
        save_npy(f"matmul_a_{name}", a)
        save_npy(f"matmul_b_{name}", b)
        save_npy(f"matmul_c_{name}", c)
    
    # Batched matmul: (B,M,K) @ (B,K,N) → (B,M,N)
    b, m, k, n = 4, 8, 16, 8
    a = RNG.randn(b, m, k).astype(np.float32)
    bm = RNG.randn(b, k, n).astype(np.float32)
    c = np.matmul(a, bm)
    save_npy("batch_matmul_a", a)
    save_npy("batch_matmul_b", bm)
    save_npy("batch_matmul_c", c)
    
    # Elementwise ops
    x = RNG.randn(16).astype(np.float32)
    y = RNG.randn(16).astype(np.float32)
    save_npy("elem_add_x", x)
    save_npy("elem_add_y", y)
    save_npy("elem_add_result", x + y)
    save_npy("elem_mul_result", x * y)
    save_npy("elem_sub_result", x - y)
    save_npy("elem_div_result", x / (y + 1.0))
    
    # Relu
    save_npy("relu_input", x)
    save_npy("relu_output", np.maximum(0, x))
    
    # Silu
    def silu(x):
        return x / (1.0 + np.exp(-x))
    save_npy("silu_input", x)
    save_npy("silu_output", silu(x))
    
    # Transpose
    t = RNG.randn(6, 10).astype(np.float32)
    save_npy("transpose_input", t)
    save_npy("transpose_output", t.T)
    
    # Softmax
    logits = RNG.randn(4, 8).astype(np.float32)
    logits_exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    softmax = logits_exp / logits_exp.sum(axis=-1, keepdims=True)
    save_npy("softmax_logits", logits)
    save_npy("softmax_output", softmax)

# ═══════════════════════════════════════════════════════════════
# Neural Network Primitives
# ═══════════════════════════════════════════════════════════════

def gen_nn_tests():
    """產生神經網路層測試資料"""
    print("\n=== Neural Network Layers ===")
    
    # RMSNorm
    d_model = 64
    rms_weight = RNG.randn(d_model).astype(np.float32)
    rms_input = RNG.randn(8, d_model).astype(np.float32)
    rms_eps = 1e-6
    
    rms_mean = np.mean(rms_input ** 2, axis=-1, keepdims=True)
    rms_out = rms_input / np.sqrt(rms_mean + rms_eps) * rms_weight
    
    save_npy("rmsnorm_weight", rms_weight)
    save_npy("rmsnorm_input", rms_input)
    save_npy("rmsnorm_output", rms_out)
    
    # RoPE
    seq_len, n_heads, head_dim = 8, 4, 64
    rope_base = 10000.0
    
    # Generate position indices
    pos = np.arange(seq_len)
    
    # Generate sin/cos tables
    inv_freq = 1.0 / (rope_base ** (np.arange(0, head_dim, 2) / head_dim))
    freqs = np.outer(pos, inv_freq)  # (seq_len, head_dim/2)
    cos_t = np.cos(freqs)
    sin_t = np.sin(freqs)
    
    # Interleave: repeat each column twice
    cos_full = np.repeat(cos_t, 2, axis=1)  # (seq_len, head_dim)
    sin_full = np.repeat(sin_t, 2, axis=1)
    
    # Query/key before RoPE
    q = RNG.randn(seq_len, n_heads, head_dim).astype(np.float32)
    k = RNG.randn(seq_len, n_heads, head_dim).astype(np.float32)
    
    # Apply RoPE
    q_rot = q.copy()
    k_rot = k.copy()
    for s in range(seq_len):
        for h in range(n_heads):
            for d in range(0, head_dim, 2):
                d1, d2 = d, d + 1
                qc = q[s, h, d1].copy()
                qs = q[s, h, d2].copy()
                q_rot[s, h, d1] = qc * cos_t[s, d//2] - qs * sin_t[s, d//2]
                q_rot[s, h, d2] = qs * cos_t[s, d//2] + qc * sin_t[s, d//2]
                
                kc = k[s, h, d1].copy()
                ks = k[s, h, d2].copy()
                k_rot[s, h, d1] = kc * cos_t[s, d//2] - ks * sin_t[s, d//2]
                k_rot[s, h, d2] = ks * cos_t[s, d//2] + kc * sin_t[s, d//2]
    
    save_npy("rope_q", q)
    save_npy("rope_k", k)
    save_npy("rope_cos", cos_full)
    save_npy("rope_sin", sin_full)
    save_npy("rope_q_rotated", q_rot)
    save_npy("rope_k_rotated", k_rot)
    
    # Scaled Dot-Product Attention
    d_k = 64
    scale = 1.0 / np.sqrt(d_k)
    
    attn_q = RNG.randn(2, n_heads, 8, d_k).astype(np.float32)
    attn_k = RNG.randn(2, n_heads, 8, d_k).astype(np.float32)
    attn_v = RNG.randn(2, n_heads, 8, d_k).astype(np.float32)
    
    # Compute attention
    scores = np.matmul(attn_q, attn_k.transpose(0, 1, 3, 2)) * scale
    
    # Causal mask
    mask = np.triu(np.full((8, 8), -np.inf), k=1)
    scores = scores + mask
    
    # Softmax
    scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
    
    # Output
    attn_out = np.matmul(attn_weights, attn_v)
    
    save_npy("attention_q", attn_q)
    save_npy("attention_k", attn_k)
    save_npy("attention_v", attn_v)
    save_npy("attention_weights", attn_weights)
    save_npy("attention_output", attn_out)

# ═══════════════════════════════════════════════════════════════
# Quantization
# ═══════════════════════════════════════════════════════════════

def gen_quant_tests():
    """產生量化測試資料"""
    print("\n=== Quantization ===")
    
    # Q4 block quantization
    block_size = 32
    n_blocks = 8
    
    # Random block data
    for name, scale_val in [
        ("q4_uniform", 1.0),
        ("q4_spread", 10.0),
        ("q4_extreme", 100.0),
        ("q4_zero", 0.0),
    ]:
        data = RNG.randn(n_blocks * block_size).astype(np.float32) * scale_val
        if name == "q4_zero":
            data = np.zeros_like(data)
        
        # Quantize
        packed = bytearray()
        for b in range(n_blocks):
            block = data[b * block_size:(b + 1) * block_size]
            w_min = float(block.min())
            w_max = float(block.max())
            scale = (w_max - w_min) / 15.0 if w_max != w_min else 1.0
            
            q = np.round((block - w_min) / scale).clip(0, 15).astype(np.uint8)
            
            # Pack to 4-bit
            q_packed = np.zeros(16, dtype=np.uint8)
            for i in range(0, block_size, 2):
                q_packed[i // 2] = (q[i] << 4) | q[i + 1]
            
            packed.extend(q_packed.tobytes())
            packed.extend(np.float16(scale).tobytes())
            packed.extend(np.float16(w_min).tobytes())
        
        save_npy(f"q4_input_{name}", data)
        save_bin(f"q4_packed_{name}", bytes(packed))
    
    # Q8 block quantization
    for name, scale_val in [
        ("q8_uniform", 1.0),
        ("q8_spread", 10.0),
    ]:
        data = RNG.randn(n_blocks * block_size).astype(np.float32) * scale_val
        
        packed = bytearray()
        for b in range(n_blocks):
            block = data[b * block_size:(b + 1) * block_size]
            w_min = float(block.min())
            w_max = float(block.max())
            scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
            
            q = np.round((block - w_min) / scale).clip(0, 255).astype(np.int8)
            
            packed.extend(q.tobytes())
            packed.extend(struct.pack('f', scale))
            packed.extend(struct.pack('f', w_min))
        
        save_npy(f"q8_input_{name}", data)
        save_bin(f"q8_packed_{name}", bytes(packed))

# ═══════════════════════════════════════════════════════════════
# Audio / WAV
# ═══════════════════════════════════════════════════════════════

def gen_audio_tests():
    """產生音訊處理測試資料"""
    print("\n=== Audio Processing ===")
    
    sample_rate = 48000
    duration = 1.0  # 1秒
    n_samples = int(sample_rate * duration)
    
    # 正弦波 (440Hz)
    t = np.arange(n_samples) / sample_rate
    sine = 0.5 * np.sin(2 * np.pi * 440 * t)
    save_npy("sine_440hz", sine)
    
    # 複合波
    multi = 0.3 * np.sin(2 * np.pi * 440 * t) + \
            0.2 * np.sin(2 * np.pi * 880 * t) + \
            0.1 * np.sin(2 * np.pi * 1320 * t)
    save_npy("multi_tone", multi)
    
    # 白噪聲
    noise = RNG.randn(n_samples).astype(np.float32) * 0.1
    save_npy("white_noise", noise)
    
    # WAV 格式繞行測試
    def save_wav_mono(filename: str, data: np.ndarray, sr: int = 48000):
        """手動建立 WAV file for testing"""
        import struct
        
        n = len(data)
        bits_per_sample = 16
        byte_rate = sr * 2
        block_align = 2
        data_size = n * 2
        
        # Clip to [-1, 1]
        data_clipped = np.clip(data, -1.0, 1.0)
        data_int16 = (data_clipped * 32767).astype(np.int16)
        
        with open(os.path.join(OUTPUT_DIR, filename), 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))
            f.write(struct.pack('<H', 1))  # PCM
            f.write(struct.pack('<H', 1))  # Mono
            f.write(struct.pack('<I', sr))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            f.write(data_int16.tobytes())
        
        print(f"  ✓ {filename}  ({len(data_int16)} samples, {sr} Hz)")
    
    save_wav_mono("test_sine.wav", sine)
    save_wav_mono("test_multi.wav", multi)
    save_wav_mono("test_noise.wav", noise)

# ═══════════════════════════════════════════════════════════════
# Model Weights (tiny test model)
# ═══════════════════════════════════════════════════════════════

def gen_model_tests():
    """產生小型測試模型權重"""
    print("\n=== Model Weights (Tiny Test Model) ===")
    
    # 一個極小的 TSLM 層用於測試
    d_model = 32
    n_heads = 4
    n_kv_heads = 2
    head_dim = d_model // n_heads
    d_ff = 64
    
    # 簡化模型：1層 TSLM，無 RALM/DiT/VAE
    weights = {}
    
    # Token embedding
    weights["token_embed"] = RNG.randn(100, d_model).astype(np.float32)
    
    # Attention weights
    weights["wq"] = RNG.randn(d_model, d_model).astype(np.float32)
    weights["wk"] = RNG.randn(d_model, n_kv_heads * head_dim).astype(np.float32)
    weights["wv"] = RNG.randn(d_model, n_kv_heads * head_dim).astype(np.float32)
    weights["wo"] = RNG.randn(d_model, d_model).astype(np.float32)
    
    # FFN weights (SwiGLU)
    weights["w1"] = RNG.randn(d_model, d_ff).astype(np.float32)   # gate
    weights["w2"] = RNG.randn(d_ff, d_model).astype(np.float32)   # down
    weights["w3"] = RNG.randn(d_model, d_ff).astype(np.float32)   # up
    
    # RMSNorm
    weights["rms_att"] = RNG.randn(d_model).astype(np.float32)
    weights["rms_ffn"] = RNG.randn(d_model).astype(np.float32)
    
    # Output norm
    weights["rms_out"] = RNG.randn(d_model).astype(np.float32)
    
    for name, w in weights.items():
        save_npy(f"tiny_{name}", w)
    
    # Save combined as .vxcpm format (test)
    import struct
    import hashlib
    
    with open(os.path.join(OUTPUT_DIR, "tiny_model.vxcpm"), 'wb') as f:
        # Header (placeholder)
        f.write(b'\0' * 256)
        
        # Layer index (placeholder)
        n_layers = 3  # token_embed, 1 layer, rms_out
        index_start = f.tell()
        index_size = (n_layers + 1) * 8
        f.write(b'\0' * index_size)
        
        offsets = []
        
        # Layer 0: Token embed
        offsets.append(f.tell())
        t = weights["token_embed"]
        f.write(struct.pack('I', hash("token_embed") & 0xFFFFFFFF))
        f.write(struct.pack('I', 2))
        f.write(struct.pack('III', t.shape[0], t.shape[1], 1))
        f.write(t.astype(np.float32).tobytes())
        
        # Layer 1: Single TSLM layer
        offsets.append(f.tell())
        for name in ["wq", "wk", "wv", "wo", "w1", "w2", "w3", "rms_att", "rms_ffn"]:
            t = weights[name]
            f.write(struct.pack('I', hash(name) & 0xFFFFFFFF))
            ndim = len(t.shape)
            dims = list(t.shape)
            while len(dims) < 3:
                dims.append(1)
            f.write(struct.pack('I', ndim))
            f.write(struct.pack('III', dims[0], dims[1], dims[2]))
            f.write(t.astype(np.float32).tobytes())
        
        # Layer 2: Output norm
        offsets.append(f.tell())
        t = weights["rms_out"]
        f.write(struct.pack('I', hash("rms_out") & 0xFFFFFFFF))
        f.write(struct.pack('I', 1))
        f.write(struct.pack('III', t.shape[0], 1, 1))
        f.write(t.astype(np.float32).tobytes())
        
        # End offset
        offsets.append(f.tell())
        
        # Write header
        f.seek(0)
        header = bytearray(256)
        struct.pack_into('8s', header, 0, b'VXCPM2\0')
        struct.pack_into('I', header, 8, 1)    # version
        struct.pack_into('I', header, 12, 1)   # n_layers
        struct.pack_into('I', header, 16, 0)   # n_layers_enc
        struct.pack_into('I', header, 20, 0)   # n_ralm_layers
        struct.pack_into('I', header, 24, 0)   # n_dit_layers
        struct.pack_into('I', header, 28, d_model)
        struct.pack_into('I', header, 32, d_ff)
        struct.pack_into('I', header, 36, n_heads)
        struct.pack_into('I', header, 40, n_kv_heads)
        struct.pack_into('I', header, 44, head_dim)
        struct.pack_into('I', header, 48, 512)  # max_seq_len
        struct.pack_into('I', header, 52, 100)  # vocab_size
        struct.pack_into('I', header, 56, 0)    # quant_type (f32)
        struct.pack_into('I', header, 60, 48000)
        struct.pack_into('I', header, 64, 64)   # audio_vae_dim
        struct.pack_into('f', header, 68, 10000.0)
        struct.pack_into('Q', header, 72, 0)    # checksum placeholder
        struct.pack_into('I', header, 80, len(offsets) - 1)
        f.write(bytes(header))
        
        # Write layer index
        f.seek(index_start)
        for offset in offsets:
            f.write(struct.pack('Q', offset))
        
        # Compute checksum
        f.seek(0)
        file_data = f.read()
        checksum = hashlib.sha256(file_data[:72] + b'\0'*8 + file_data[80:]).digest()
        checksum_int = int.from_bytes(checksum[:8], 'little')
        
        f.seek(72)
        f.write(struct.pack('Q', checksum_int))
    
    file_size = os.path.getsize(os.path.join(OUTPUT_DIR, "tiny_model.vxcpm"))
    print(f"  ✓ tiny_model.vxcpm  ({file_size} bytes)")

# ═══════════════════════════════════════════════════════════════
# Tokenizer Test Data
# ═══════════════════════════════════════════════════════════════

def gen_tokenizer_tests():
    """產生 tokenizer 測試資料"""
    print("\n=== Tokenizer ===")
    
    # Test sentences
    sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "VoxCPM2 supports 30 languages.",
        "你好，世界！",
        "こんにちは、世界！",
        "Bonjour le monde!",
    ]
    
    # Expected byte-level tokenization (BPE)
    save_txt("tokenizer_input", "\n".join(sentences))
    
    # Byte-pair encoding mapping (sample)
    # 一個最小化的 BPE merges 表
    merge_rules = [
        ("h", "e", "he"),
        ("he", "ll", "hell"),
        ("hell", "o", "hello"),
        ("w", "o", "wo"),
        ("wo", "rl", "worl"),
        ("worl", "d", "world"),
    ]
    save_txt("tokenizer_merges", "\n".join(
        f"{a} {b} -> {c}" for a, b, c in merge_rules
    ))
    
    # Vocabulary (sample)
    vocab = {
        0: "<unk>",
        1: "<s>",
        2: "</s>",
        100: "hello",
        101: "world",
        102: " ",
        103: "!",
        104: "the",
        105: "quick",
    }
    save_txt("tokenizer_vocab", "\n".join(
        f"{k}\t{v}" for k, v in sorted(vocab.items())
    ))

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VoxCPM2-C test data generator")
    parser.add_argument('--output', '-o', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    parser.add_argument('--only', type=str, default=None,
                        choices=['tensor', 'nn', 'quant', 'audio', 'model', 'tokenizer'],
                        help='Generate only specific test data')
    args = parser.parse_args()
    
    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"VoxCPM2-C Test Data Generator")
    print(f"Output: {OUTPUT_DIR}")
    
    generators = {
        'tensor':    gen_tensor_tests,
        'nn':        gen_nn_tests,
        'quant':     gen_quant_tests,
        'audio':     gen_audio_tests,
        'model':     gen_model_tests,
        'tokenizer': gen_tokenizer_tests,
    }
    
    if args.only:
        if args.only in generators:
            generators[args.only]()
        else:
            print(f"Unknown test type: {args.only}")
            sys.exit(1)
    else:
        for gen in generators.values():
            gen()
    
    # Summary
    npy_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    bin_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.bin')]
    wav_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.wav')]
    vxcpm_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.vxcpm')]
    
    print(f"\n=== Summary ===")
    print(f"  .npy files:  {len(npy_files)}")
    print(f"  .bin files:  {len(bin_files)}")
    print(f"  .wav files:  {len(wav_files)}")
    print(f"  .vxcpm files: {len(vxcpm_files)}")
    print(f"  Total size: {sum(os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in os.listdir(OUTPUT_DIR)) / 1024:.1f} KB")
    print(f"\n✅ Test data generation complete!")

if __name__ == "__main__":
    main()
