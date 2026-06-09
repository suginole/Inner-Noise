"""
phoneme.py — 2bitsパルス ↔ 母音変換モジュール（2bits統一版）

2bits = 母音のみ（子音なし）
  00 = う  (F1低・F2低)
  01 = い  (F1低・F2高)  ← F2/F1比が最大で最も識別しやすい
  10 = お  (F1中・F2低)
  11 = あ  (F1高・F2中)  ← F1が最大で識別しやすい

PhonemeConverter : 2bits → フォルマント合成 → pygame.mixer 出力
PhonemeDecoder   : マイク入力 → F1/F2比率判定 → 2bits 復元

検知方法（比率ベース、音量・話者依存なし）:
  F2/F1 > 4.5  → い (01)
  F1 > 600Hz   → あ (11)
  F2 < 900Hz   → う (00)
  else         → お (10)

5Hz同期読み取り:
  record_frame_sync() で200ms区間を1回だけ録音・判定する。
  呼び出し側がパルスタイミングに合わせて呼ぶこと。
"""
from __future__ import annotations
import numpy as np

_mixer_ready = False

try:
    import pygame
    import pygame.mixer
    import pygame.sndarray
    _pygame_available = True
except ImportError:
    _pygame_available = False

try:
    import pyaudio
    _pyaudio_available = True
except ImportError:
    _pyaudio_available = False

try:
    from scipy.fft import rfft, rfftfreq
    _scipy_available = True
except ImportError:
    _scipy_available = False

from config import (
    PHONEME_VOWEL, PHONEME_FORMANTS, PHONEME_TABLE,
    AUDIO_SAMPLE_RATE, AUDIO_FRAME_MS, AUDIO_FRAME_SAMPLES,
    VOWEL_F2F1_I, VOWEL_F1_A, VOWEL_F2_U,
)


def _ensure_mixer():
    global _mixer_ready
    if _mixer_ready or not _pygame_available:
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(
                frequency=AUDIO_SAMPLE_RATE,
                size=-16,
                channels=1,
                buffer=AUDIO_FRAME_SAMPLES,
            )
        _mixer_ready = True
    except Exception:
        pass


# ================================================================
class PhonemeConverter:
    """
    2bitsパルス → フォルマント合成 → pygame.mixer 出力。

    合成ロジック:
      - F1・F2 のサイン波を重ね合わせ（母音のみ）
      - フェードイン/アウトでクリックノイズを防止
    """

    def __init__(self):
        _ensure_mixer()
        self._channel = None
        if _pygame_available and _mixer_ready:
            try:
                self._channel = pygame.mixer.Channel(0)
            except Exception:
                pass

    # ----------------------------------------------------------------
    def pulse_to_phoneme(self, bits2: int) -> str:
        """2bits → 音素文字列（例: 'あ'）を返す。"""
        return PHONEME_TABLE.get(bits2 & 0x3, '?')

    # ----------------------------------------------------------------
    def synthesize(self, bits2: int) -> np.ndarray:
        """
        2bits → 200ms の音声バッファ（float32, -1.0〜1.0）を返す。

        Returns:
            np.ndarray shape=(AUDIO_FRAME_SAMPLES,) dtype=float32
        """
        bits2 = bits2 & 0x3
        vowel = PHONEME_VOWEL[bits2]
        f1, f2 = PHONEME_FORMANTS[vowel]

        n  = AUDIO_FRAME_SAMPLES
        sr = AUDIO_SAMPLE_RATE
        t  = np.linspace(0, AUDIO_FRAME_MS / 1000.0, n, endpoint=False)

        # F1 + F2 のサイン波（母音のみ）
        buf = (0.6 * np.sin(2 * np.pi * f1 * t) +
               0.4 * np.sin(2 * np.pi * f2 * t))

        # フェードイン/アウト（クリックノイズ防止）
        fade_n = min(int(sr * 0.005), n // 4)   # 5ms
        buf[:fade_n]  *= np.linspace(0, 1, fade_n)
        buf[-fade_n:] *= np.linspace(1, 0, fade_n)

        # 正規化
        peak = np.max(np.abs(buf))
        if peak > 0:
            buf = buf / peak * 0.8

        return buf.astype(np.float32)

    # ----------------------------------------------------------------
    def play(self, bits2: int) -> None:
        """synthesize() の結果を pygame.mixer でリアルタイム出力する。"""
        if not (_pygame_available and _mixer_ready):
            return
        buf = self.synthesize(bits2)
        pcm = (buf * 32767).astype(np.int16)
        try:
            sound = pygame.sndarray.make_sound(pcm)
            if self._channel:
                self._channel.play(sound)
            else:
                sound.play()
        except Exception:
            pass

    def play_all_demo(self, interval_ms: int = 400) -> None:
        """全4音素を順番に再生するデモ（モデル音源確認用）。"""
        import time
        for bits2 in range(4):
            self.play(bits2)
            time.sleep(interval_ms / 1000.0)


# ================================================================
class PhonemeDecoder:
    """
    マイク入力（pyaudio）→ F1/F2比率判定 → 2bits 復元。

    検知方法（比率ベース、音量・話者依存なし）:
      1. FFT でスペクトルを取得
      2. F1帯域（200〜1000Hz）と F2帯域（1000〜2600Hz）のピーク周波数を検出
      3. F2/F1比率と F1絶対値で母音を判定

    5Hz同期読み取り:
      record_frame_sync() を呼ぶと200ms録音して即座に判定を返す。
      パルス発火タイミングに合わせて呼ぶこと。
    """

    def __init__(self):
        self._pa     = None
        self._stream = None
        self._available = False

        if not _pyaudio_available:
            return

        try:
            self._pa = pyaudio.PyAudio()
            self._pa.get_default_input_device_info()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_FRAME_SAMPLES,
            )
            self._available = True
        except Exception:
            self._available = False

    def __del__(self):
        self.close()

    def close(self):
        try:
            if self._stream:
                self._stream.stop_stream()
                self._stream.close()
            if self._pa:
                self._pa.terminate()
        except Exception:
            pass

    @property
    def available(self) -> bool:
        return self._available

    # ----------------------------------------------------------------
    def record_frame(self) -> np.ndarray:
        """
        200ms 分のマイク入力を取得する。
        マイクが利用不可の場合はゼロ配列を返す。
        """
        if not self._available or self._stream is None:
            return np.zeros(AUDIO_FRAME_SAMPLES, dtype=np.float32)
        try:
            raw = self._stream.read(AUDIO_FRAME_SAMPLES, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.float32).copy()
        except Exception:
            return np.zeros(AUDIO_FRAME_SAMPLES, dtype=np.float32)

    # ----------------------------------------------------------------
    def detect_f1_f2(self, frame: np.ndarray) -> tuple[float, float]:
        """
        FFT でスペクトルを取得し、F1・F2 のピーク周波数を返す。

        Returns:
            (f1_hz, f2_hz): F1帯域と F2帯域のピーク周波数
        """
        if not _scipy_available or len(frame) == 0:
            return 0.0, 0.0

        n     = len(frame)
        spec  = np.abs(rfft(frame))
        freqs = rfftfreq(n, d=1.0 / AUDIO_SAMPLE_RATE)

        # スペクトルを平滑化（ノイズ低減）
        from numpy.lib.stride_tricks import sliding_window_view
        win = 5
        if len(spec) > win:
            spec_smooth = np.convolve(spec, np.ones(win) / win, mode='same')
        else:
            spec_smooth = spec

        def peak_freq(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                return 0.0
            return float(freqs[mask][np.argmax(spec_smooth[mask])])

        f1 = peak_freq(200,  1000)
        f2 = peak_freq(1000, 2600)
        return f1, f2

    # ----------------------------------------------------------------
    def detect_vowel(self, frame: np.ndarray) -> int:
        """
        F1/F2比率判定で母音を特定し、2bits（0〜3）を返す。

        判定ツリー:
          F2/F1 > VOWEL_F2F1_I → い (01)
          F1 > VOWEL_F1_A      → あ (11)
          F2 < VOWEL_F2_U      → う (00)
          else                 → お (10)

        Returns:
            int: 0=u, 1=i, 2=o, 3=a
        """
        f1, f2 = self.detect_f1_f2(frame)

        if f1 <= 0:
            return 0   # 無音 → う

        ratio = f2 / f1

        if ratio > VOWEL_F2F1_I:
            return 1   # い
        if f1 > VOWEL_F1_A:
            return 3   # あ
        if f2 < VOWEL_F2_U:
            return 0   # う
        return 2       # お

    # ----------------------------------------------------------------
    def decode(self, frame: np.ndarray) -> int:
        """
        フレームから 2bits 整数を復元する。

        Returns:
            int: 0〜3
        """
        return self.detect_vowel(frame) & 0x3

    # ----------------------------------------------------------------
    def record_frame_sync(self) -> tuple[np.ndarray, int]:
        """
        5Hz同期読み取り用。
        200ms録音して即座に判定し (frame, bits2) を返す。
        パルス発火タイミングに合わせて呼ぶこと。

        Returns:
            (frame, bits2)
        """
        frame = self.record_frame()
        bits2 = self.decode(frame)
        return frame, bits2
