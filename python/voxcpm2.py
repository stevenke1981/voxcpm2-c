"""
VoxCPM2-C Python Binding (ctypes)

讓 Python 可以直接呼叫 VoxCPM2-C 的推理引擎，無需透過 subprocess。

使用方式:
    from voxcpm2 import VoxCPM2
    
    # 載入模型
    tts = VoxCPM2("models/voxcpm2-q4.vxcpm")
    
    # 基本 TTS
    audio = tts.generate("Hello, world!")
    tts.save_wav("hello.wav", audio)
    
    # 語音設計 (Voice Design)
    audio = tts.voice_design(
        text="This is a calm and warm voice.",
        description="A soft, gentle female voice with a warm tone"
    )
    
    # 語音複製
    audio = tts.ultimate_cloning(
        text="This is a cloned voice.",
        reference_audio="speaker.wav",
        reference_text="This is the reference speaker."
    )
    
    # 串流
    for chunk in tts.generate_stream("Streaming output..."):
        process_chunk(chunk)
"""

import ctypes
import ctypes.util
import os
import platform
import struct
from typing import Optional, List, Generator, BinaryIO

# ═══════════════════════════════════════════════════════════════
# C 資料結構對應
# ═══════════════════════════════════════════════════════════════

class VoxCPMModelConfig(ctypes.Structure):
    _fields_ = [
        ("model_path",      ctypes.c_char_p),
        ("use_gpu",         ctypes.c_bool),
        ("num_threads",     ctypes.c_int),
        ("quant_type",      ctypes.c_int),
        ("max_seq_len",     ctypes.c_int),
    ]

class VoxCPMGenerateOptions(ctypes.Structure):
    _fields_ = [
        ("input_ids",       ctypes.POINTER(ctypes.c_int32)),
        ("n_input_ids",     ctypes.c_int),
        ("output_audio",    ctypes.POINTER(ctypes.c_float)),
        ("output_len",      ctypes.POINTER(ctypes.c_int)),
        ("use_gpu",         ctypes.c_bool),
        ("max_audio_len",   ctypes.c_int),
        ("temperature",     ctypes.c_float),
        ("cfg_scale",       ctypes.c_float),
        ("ddim_steps",      ctypes.c_int),
        ("seed",            ctypes.c_uint32),
    ]

class VoxCPMStreamOptions(ctypes.Structure):
    _fields_ = [
        ("callback",         ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)),
        ("user_data",        ctypes.c_void_p),
        ("chunk_size_frames", ctypes.c_int),
        ("temperature",      ctypes.c_float),
        ("cfg_scale",        ctypes.c_float),
        ("ddim_steps",       ctypes.c_int),
        ("seed",             ctypes.c_uint32),
    ]

# 回傳碼
VOXCPM_SUCCESS           = 0
VOXCPM_ERR_LOAD_MODEL    = -1
VOXCPM_ERR_GENERATION    = -2
VOXCPM_ERR_INVALID_ARGS  = -3
VOXCPM_ERR_MEMORY        = -4
VOXCPM_ERR_CUDA          = -5
VOXCPM_ERR_FILE_NOT_FOUND = -6

# ═══════════════════════════════════════════════════════════════
# VoxCPM2 Python Wrapper
# ═══════════════════════════════════════════════════════════════

class VoxCPM2Error(Exception):
    """VoxCPM2 引擎錯誤"""
    pass

class VoxCPM2:
    """VoxCPM2-C TTS 引擎的 Python 綁定"""
    
    def __init__(self, model_path: str, use_gpu: bool = False,
                 num_threads: int = 0):
        """初始化 TTS 引擎
        
        Args:
            model_path: .vxcpm 權重檔案路徑
            use_gpu: 是否使用 CUDA 加速
            num_threads: CPU 執行緒數 (0=auto)
        """
        self._lib = self._load_library()
        self._setup_functions()
        
        # 建立模型設定
        config = VoxCPMModelConfig()
        config.model_path = model_path.encode('utf-8')
        config.use_gpu = use_gpu
        config.num_threads = num_threads
        
        # 載入模型
        err = ctypes.c_int(0)
        self._model = self._lib.voxcpm_create(
            ctypes.byref(config),
            ctypes.byref(err)
        )
        
        if not self._model:
            err_val = err.value
            err_str = self._lib.voxcpm_error_string(err)
            raise VoxCPM2Error(
                f"Failed to load model (err={err_val}): {err_str}"
            )
        
        self._sample_rate = 48000
    
    def __del__(self):
        """釋放模型資源"""
        if hasattr(self, '_lib') and hasattr(self, '_model') and self._model:
            self._lib.voxcpm_free(self._model)
            self._model = None
    
    # ─── 函式庫載入 ────────────────────────────────────────
    
    @staticmethod
    def _load_library():
        """載入 VoxCPM2-C 共享函式庫"""
        
        # 搜尋順序
        lib_names = []
        system = platform.system()
        
        if system == "Windows":
            lib_names = ["voxcpm2.dll", "libvoxcpm2.dll", "voxcpm2-c.dll"]
        elif system == "Darwin":
            lib_names = ["libvoxcpm2.dylib", "libvoxcpm2-c.dylib"]
        else:  # Linux
            lib_names = ["libvoxcpm2.so", "libvoxcpm2-c.so"]
        
        # 嘗試在常見路徑搜尋
        search_paths = [
            os.path.dirname(__file__) or ".",
            os.path.join(os.path.dirname(__file__), "..", "build"),
            os.path.join(os.path.dirname(__file__), "..", "build", "Release"),
            os.path.join(os.path.dirname(__file__), ".."),
            os.getcwd(),
        ]
        
        # 先嘗試系統路徑
        for name in lib_names:
            path = ctypes.util.find_library(name)
            if path:
                return ctypes.CDLL(path)
        
        # 再嘗試搜尋路徑
        for search_path in search_paths:
            for name in lib_names:
                path = os.path.join(search_path, name)
                if os.path.exists(path):
                    return ctypes.CDLL(path)
        
        raise VoxCPM2Error(
            "Cannot find VoxCPM2-C library. "
            f"Searched: {lib_names} in {search_paths}"
        )
    
    def _setup_functions(self):
        """設定 ctypes 函式簽名"""
        
        self._lib.voxcpm_create.argtypes = [
            ctypes.POINTER(VoxCPMModelConfig),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.voxcpm_create.restype = ctypes.c_void_p
        
        self._lib.voxcpm_free.argtypes = [ctypes.c_void_p]
        self._lib.voxcpm_free.restype = None
        
        self._lib.voxcpm_generate.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(VoxCPMGenerateOptions),
        ]
        self._lib.voxcpm_generate.restype = ctypes.c_int
        
        self._lib.voxcpm_generate_stream.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.POINTER(VoxCPMStreamOptions),
        ]
        self._lib.voxcpm_generate_stream.restype = ctypes.c_int
        
        self._lib.voxcpm_memory_used.argtypes = [ctypes.c_void_p]
        self._lib.voxcpm_memory_used.restype = ctypes.c_size_t
        
        self._lib.voxcpm_error_string.argtypes = [ctypes.c_int]
        self._lib.voxcpm_error_string.restype = ctypes.c_char_p
        
        self._lib.voxcpm_have_gpu.restype = ctypes.c_bool
        
        self._lib.voxcpm_last_error.argtypes = [ctypes.c_void_p]
        self._lib.voxcpm_last_error.restype = ctypes.c_int
    
    # ─── Tokenizer ─────────────────────────────────────────
    
    def tokenize(self, text: str) -> List[int]:
        """將文字轉換為 token IDs
        
        Args:
            text: 輸入文字
            
        Returns:
            token ID 列表
        """
        # 使用簡單的 byte-level 編碼（實作時應使用 BPE）
        tokens = []
        for byte in text.encode('utf-8'):
            tokens.append(int(byte) + 4)  # +4 for special tokens
        return tokens
    
    # ─── TTS 核心 ──────────────────────────────────────────
    
    def generate(self, text: str, temperature: float = 0.7,
                 cfg_scale: float = 1.0, ddim_steps: int = 10,
                 seed: Optional[int] = None) -> List[float]:
        """文字轉語音
        
        Args:
            text: 輸入文字
            temperature: 取樣溫度 (0.0-2.0)
            cfg_scale: Classifier-Free Guidance 強度 (1.0=關閉)
            ddim_steps: DDIM 採樣步數 (5-50)
            seed: 隨機種子 (None=隨機)
            
        Returns:
            音訊 samples (float32, [-1.0, 1.0], 48kHz)
        """
        tokens = self.tokenize(text)
        
        # 準備音訊緩衝區 (max 30 seconds)
        max_audio_len = 48000 * 30
        audio_buf = (ctypes.c_float * max_audio_len)()
        audio_len = ctypes.c_int(0)
        
        # 準備輸入
        input_ids = (ctypes.c_int32 * len(tokens))(*tokens)
        
        opts = VoxCPMGenerateOptions()
        opts.input_ids = input_ids
        opts.n_input_ids = len(tokens)
        opts.output_audio = audio_buf
        opts.output_len = ctypes.byref(audio_len)
        opts.use_gpu = False
        opts.max_audio_len = max_audio_len
        opts.temperature = temperature
        opts.cfg_scale = cfg_scale
        opts.ddim_steps = ddim_steps
        opts.seed = seed if seed is not None else 0
        
        err = self._lib.voxcpm_generate(self._model, ctypes.byref(opts))
        if err != 0:
            raise VoxCPM2Error(
                f"Generation failed: {self._lib.voxcpm_error_string(err)}"
            )
        
        return list(audio_buf[:audio_len.value])
    
    def generate_stream(self, text: str, temperature: float = 0.7,
                        cfg_scale: float = 1.0, ddim_steps: int = 10,
                        chunk_size_frames: int = 4800
                        ) -> Generator[List[float], None, None]:
        """串流 TTS（逐段產生音訊）
        
        Args:
            text: 輸入文字
            temperature: 取樣溫度
            cfg_scale: CFG 強度
            ddim_steps: DDIM 步數
            chunk_size_frames: 每段 frame 數 (預設 4800 = 0.1s)
            
        Yields:
            音訊 chunks
        """
        tokens = self.tokenize(text)
        input_ids = (ctypes.c_int32 * len(tokens))(*tokens)
        
        chunks = []
        
        def chunk_callback(data: ctypes.c_void_p,
                           n_frames: ctypes.c_int,
                           user_data: ctypes.c_void_p) -> bool:
            """串流回呼"""
            if n_frames <= 0 or not data:
                return False
            
            frame_array = (ctypes.c_float * n_frames).from_address(
                ctypes.addressof(data.contents)
            )
            chunks.append(list(frame_array))
            return True  # continue streaming
        
        CALLBACK_TYPE = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p
        )
        callback = CALLBACK_TYPE(chunk_callback)
        
        stream_opts = VoxCPMStreamOptions()
        stream_opts.callback = callback
        stream_opts.user_data = None
        stream_opts.chunk_size_frames = chunk_size_frames
        stream_opts.temperature = temperature
        stream_opts.cfg_scale = cfg_scale
        stream_opts.ddim_steps = ddim_steps
        stream_opts.seed = 42
        
        err = self._lib.voxcpm_generate_stream(
            self._model, input_ids, len(tokens),
            ctypes.byref(stream_opts)
        )
        
        if err != VOXCPM_SUCCESS:
            raise VoxCPM2Error(
                f"Stream generation failed: {self._lib.voxcpm_error_string(err)}"
            )
        
        for chunk in chunks:
            yield chunk
    
    # ─── 進階功能 ──────────────────────────────────────────
    
    def voice_design(self, text: str, description: str,
                     temperature: float = 0.8,
                     ddim_steps: int = 15) -> List[float]:
        """Voice Design: 用自然語言描述產生新語音
        
        Args:
            text: 要朗讀的文字
            description: 語音描述 (e.g. "calm, warm female voice")
            temperature: 取樣溫度 (稍高以增加多樣性)
            ddim_steps: DDIM 步數
            
        Returns:
            音訊 samples
        """
        # Voice Design 使用 special token 嵌入描述
        design_text = f"[VOICE_DESIGN]{description}[TEXT]{text}"
        return self.generate(design_text, temperature=temperature,
                             ddim_steps=ddim_steps)
    
    def controllable_cloning(self, text: str,
                              reference_audio_path: str,
                              similarity: float = 0.7,
                              emotion: str = "neutral"
                              ) -> List[float]:
        """Controllable Cloning: 參考音檔 + 風格控制
        
        Args:
            text: 要朗讀的文字
            reference_audio_path: 參考音檔路徑
            similarity: 相似度 (0.0-1.0)
            emotion: 情緒 (neutral, happy, sad, angry)
        """
        clone_text = (
            f"[CLONE]{reference_audio_path}"
            f"[SIMILARITY]{similarity}"
            f"[EMOTION]{emotion}"
            f"[TEXT]{text}"
        )
        return self.generate(clone_text, cfg_scale=1.2)
    
    def ultimate_cloning(self, text: str,
                          reference_audio_path: str,
                          reference_text: str,
                          ddim_steps: int = 20) -> List[float]:
        """Ultimate Cloning: 參考音檔 + 逐字稿
        
        Args:
            text: 要朗讀的文字
            reference_audio_path: 參考音檔路徑 (.wav)
            reference_text: 參考音檔的逐字稿
            ddim_steps: DDIM 步數 (更多 = 更高品質)
        """
        clone_text = (
            f"[ULTIMATE_CLONE]{reference_audio_path}"
            f"[TRANSCRIPT]{reference_text}"
            f"[TEXT]{text}"
        )
        return self.generate(clone_text, ddim_steps=ddim_steps, cfg_scale=1.5)
    
    # ─── 工具函式 ──────────────────────────────────────────
    
    @property
    def memory_used(self) -> float:
        """回傳模型使用的記憶體 (bytes)"""
        return self._lib.voxcpm_memory_used(self._model)
    
    @property
    def has_gpu(self) -> bool:
        """檢查 CUDA 是否可用"""
        return self._lib.voxcpm_have_gpu()
    
    def save_wav(self, path: str, audio: List[float],
                 sample_rate: int = 48000):
        """儲存音訊為 WAV 檔案
        
        Args:
            path: 輸出路徑
            audio: 音訊 samples
            sample_rate: 採樣率 (預設 48000)
        """
        import struct
        
        n_samples = len(audio)
        bits_per_sample = 16
        byte_rate = sample_rate * 2
        block_align = 2
        data_size = n_samples * 2
        
        # Clip to [-1, 1]
        audio_clipped = [max(-1.0, min(1.0, s)) for s in audio]
        audio_int16 = [int(s * 32767) for s in audio_clipped]
        
        with open(path, 'wb') as f:
            # RIFF header
            f.write(b'RIFF')
            f.write(struct.pack('<I', 36 + data_size))
            f.write(b'WAVE')
            # fmt chunk
            f.write(b'fmt ')
            f.write(struct.pack('<I', 16))
            f.write(struct.pack('<H', 1))  # PCM
            f.write(struct.pack('<H', 1))  # Mono
            f.write(struct.pack('<I', sample_rate))
            f.write(struct.pack('<I', byte_rate))
            f.write(struct.pack('<H', block_align))
            f.write(struct.pack('<H', bits_per_sample))
            # data chunk
            f.write(b'data')
            f.write(struct.pack('<I', data_size))
            for sample in audio_int16:
                f.write(struct.pack('<h', sample))
    
    @staticmethod
    def list_backends() -> List[str]:
        """列出可用的後端"""
        backends = ["cpu"]
        # TODO: 檢查 CUDA library 是否存在
        # TODO: 檢查 Vulkan 支援
        return backends


# ═══════════════════════════════════════════════════════════════
# 命令列介面
# ═══════════════════════════════════════════════════════════════

def main():
    """命令列 TTS 工具"""
    import argparse
    
    parser = argparse.ArgumentParser(description="VoxCPM2-C TTS via Python")
    parser.add_argument("-t", "--text", required=True, help="Input text")
    parser.add_argument("-o", "--output", default="output.wav", help="Output WAV path")
    parser.add_argument("-m", "--model", default="models/voxcpm2-q4.vxcpm",
                        help="Model path")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stream", action="store_true", help="Stream output")
    parser.add_argument("--voice-design", type=str, default=None,
                        help="Voice design description")
    parser.add_argument("--clone", type=str, default=None,
                        help="Reference audio for voice cloning")
    
    args = parser.parse_args()
    
    try:
        tts = VoxCPM2(args.model, use_gpu=args.gpu)
    except VoxCPM2Error as e:
        print(f"❌ {e}")
        return 1
    
    try:
        if args.stream:
            print(f"🔄 Streaming TTS: {args.text}")
            for chunk in tts.generate_stream(
                args.text,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                ddim_steps=args.ddim_steps,
            ):
                print(f"  Chunk: {len(chunk)} samples")
            print("✅ Streaming complete")
        elif args.voice_design:
            print(f"🎨 Voice Design: {args.voice_design}")
            audio = tts.voice_design(
                args.text, args.voice_design,
                temperature=args.temperature,
                ddim_steps=args.ddim_steps,
            )
            tts.save_wav(args.output, audio)
            print(f"✅ Saved: {args.output} ({len(audio)} samples)")
        elif args.clone:
            print(f"🔊 Voice Cloning: {args.clone}")
            audio = tts.controllable_cloning(
                args.text, args.clone,
            )
            tts.save_wav(args.output, audio)
            print(f"✅ Saved: {args.output} ({len(audio)} samples)")
        else:
            print(f"🔊 TTS: {args.text}")
            audio = tts.generate(
                args.text,
                temperature=args.temperature,
                cfg_scale=args.cfg_scale,
                ddim_steps=args.ddim_steps,
                seed=args.seed,
            )
            tts.save_wav(args.output, audio)
            duration = len(audio) / tts._sample_rate
            print(f"✅ Saved: {args.output} ({duration:.1f}s, {len(audio)} samples)")
    
    except VoxCPM2Error as e:
        print(f"❌ {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
