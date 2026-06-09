"""
phoneme.py — 4bitsパルス ↔ 音素変換モジュール

PhonemeConverter : 4bits → 音声合成 → pygame.mixer 出力
PhonemeDecoder   : マイク入力 → FFT → 4bits 復元

このモジュールは NN・GA・フィールドに一切依存しない独立モジュール。
RNNボトルネック実装後も差し替え不要。
リアル物理通信（スピーカー→マイク）への拡張を想定したバッファ設計。
"""
from __future__ import annotations
import numpy as np
import threading
import queue
import time

# pygame.mixer は遅延インポート（ヘッドレス環境対応）
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
    PHONEME_CONSONANT, PHONEME_VOWEL, PHONEME_FORMANTS, PHONEME_TABLE,
    AUDIO_SAMPLE_RATE, AUDIO_FRAME_MS, AUDIO_FRAME_SAMPLES,
    AUDIO_SIBILANT_FREQ, AUDIO_SIBILANT_MS,
    AUDIO_NASAL_FREQ, AUDIO_NASAL_GAIN, AUDIO_BILABIAL_MS,
)


def _ensure_mixer():
    """pygame.mixer を初期化する（一度だけ）。"""
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
    4bitsパルス → フォルマント合成 → pygame.mixer 出力。

    合成ロジック:
      - 母音: F1・F2 のサイン波を重ね合わせ
      - s   : 冒頭 AUDIO_SIBILANT_MS ms に 1500Hz 以上のホワイトノイズを付加
      - n   : 鼻腔共鳴（AUDIO_NASAL_FREQ Hz）を加算
      - m   : n と同様 + 冒頭 AUDIO_BILABIAL_MS ms を無音（両唇閉鎖）
    """

    def __init__(self):
        _ensure_mixer()
        self._channel: "pygame.mixer.Channel | None" = None
        if _pygame_available and _mixer_ready:
            try:
                self._channel = pygame.mixer.Channel(0)
            except Exception:
                pass

    # ----------------------------------------------------------------
    def pulse_to_phoneme(self, bits4: int) -> str:
        """4bits → 音素文字列（例: 'ま'）を返す。"""
        return PHONEME_TABLE.get(bits4 & 0xF, '？')

    # ----------------------------------------------------------------
    def synthesize(self, bits4: int) -> np.ndarray:
        """
        4bits → 200ms の音声バッファ（float32, -1.0〜1.0）を返す。

        Returns:
            np.ndarray shape=(AUDIO_FRAME_SAMPLES,) dtype=float32
        """
        bits4 = bits4 & 0xF
        consonant_bits = (bits4 >> 2) & 0x3
        vowel_bits     = bits4 & 0x3

        consonant = PHONEME_CONSONANT[consonant_bits]
        vowel     = PHONEME_VOWEL[vowel_bits]
        f1, f2    = PHONEME_FORMANTS[vowel]

        n = AUDIO_FRAME_SAMPLES
        sr = AUDIO_SAMPLE_RATE
        t = np.linspace(0, AUDIO_FRAME_MS / 1000.0, n, endpoint=False)

        # ---- 母音サイン波（F1 + F2）----
        buf = (0.6 * np.sin(2 * np.pi * f1 * t) +
               0.4 * np.sin(2 * np.pi * f2 * t))

        # ---- 調音の重ね合わせ ----
        if consonant == 's':
            # 冒頭 AUDIO_SIBILANT_MS ms にホワイトノイズ
            sib_n = int(sr * AUDIO_SIBILANT_MS / 1000)
            noise = np.random.randn(sib_n)
            # 1500Hz 以上のハイパスフィルタ（簡易: 差分で近似）
            noise_hp = np.diff(noise, prepend=noise[0])
            buf[:sib_n] += 0.5 * noise_hp

        elif consonant in ('n', 'm'):
            # 鼻腔共鳴を加算
            nasal = AUDIO_NASAL_GAIN * np.sin(2 * np.pi * AUDIO_NASAL_FREQ * t)
            buf += nasal

            if consonant == 'm':
                # 冒頭 AUDIO_BILABIAL_MS ms を無音
                bilabial_n = int(sr * AUDIO_BILABIAL_MS / 1000)
                buf[:bilabial_n] = 0.0

        # ---- フェードイン/アウト（クリックノイズ防止）----
        fade_n = min(int(sr * 0.005), n // 4)   # 5ms
        buf[:fade_n]  *= np.linspace(0, 1, fade_n)
        buf[-fade_n:] *= np.linspace(1, 0, fade_n)

        # ---- 正規化 ----
        peak = np.max(np.abs(buf))
        if peak > 0:
            buf = buf / peak * 0.8

        return buf.astype(np.float32)

    # ----------------------------------------------------------------
    def play(self, bits4: int) -> None:
        """synthesize() の結果を pygame.mixer でリアルタイム出力する。"""
        if not (_pygame_available and _mixer_ready):
            return
        buf = self.synthesize(bits4)
        # float32 → int16 に変換して Sound オブジェクト化
        pcm = (buf * 32767).astype(np.int16)
        try:
            sound = pygame.sndarray.make_sound(pcm)
            if self._channel:
                self._channel.play(sound)
            else:
                sound.play()
        except Exception:
            pass


# ================================================================
class PhonemeDecoder:
    """
    マイク入力（pyaudio）→ FFT → 4bits 復元。

    処理フロー:
      200ms 分のマイク入力 → FFT → F1/F2 ピーク検出 → 母音特定 → 下位2bits
      1500Hz 以上エネルギー → s 判定
      250〜300Hz 鼻腔共鳴  → n/m 判定
      冒頭無音              → m 確定
      → 4bits 復元

    マイクが利用できない環境ではダミー値を返す。
    """

    def __init__(self):
        self._pa: "pyaudio.PyAudio | None" = None
        self._stream = None
        self._available = False

        if not _pyaudio_available:
            return

        try:
            self._pa = pyaudio.PyAudio()
            # デフォルトマイクが存在するか確認
            info = self._pa.get_default_input_device_info()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=AUDIO_SAMPLE_RATE,
                input=True,
                frames_per_buffer=AUDIO_FRAME_SAMPLES,
            )
            self._available = True
        except Exception:
            # マイクなし環境（ヘッドレス等）ではダミーモードで動作
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

        Returns:
            np.ndarray shape=(AUDIO_FRAME_SAMPLES,) dtype=float32
        """
        if not self._available or self._stream is None:
            return np.zeros(AUDIO_FRAME_SAMPLES, dtype=np.float32)
        try:
            raw = self._stream.read(AUDIO_FRAME_SAMPLES, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.float32).copy()
        except Exception:
            return np.zeros(AUDIO_FRAME_SAMPLES, dtype=np.float32)

    # ----------------------------------------------------------------
    def detect_vowel(self, frame: np.ndarray) -> int:
        """
        FFT で F1/F2 ピークを検出し、母音を特定して下位2bits（0〜3）を返す。

        Returns:
            int: 0=u, 1=i, 2=o, 3=a
        """
        if not _scipy_available:
            return 0

        n = len(frame)
        spectrum = np.abs(rfft(frame))
        freqs    = rfftfreq(n, d=1.0 / AUDIO_SAMPLE_RATE)

        def peak_in_band(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            if not np.any(mask):
                return 0.0
            return float(np.max(spectrum[mask]))

        # F1 帯域（200〜1000Hz）
        f1_energy = {
            'a': peak_in_band(600, 1000),
            'i': peak_in_band(200,  400),
            'u': peak_in_band(200,  400),
            'o': peak_in_band(400,  600),
        }
        # F2 帯域（700〜2500Hz）
        f2_energy = {
            'a': peak_in_band(1000, 1400),
            'i': peak_in_band(2000, 2600),
            'u': peak_in_band( 700,  900),
            'o': peak_in_band( 700,  900),
        }

        scores = {v: f1_energy[v] + f2_energy[v] for v in ('a', 'i', 'u', 'o')}
        best_vowel = max(scores, key=scores.get)

        vowel_to_bits = {'u': 0, 'i': 1, 'o': 2, 'a': 3}
        return vowel_to_bits[best_vowel]

    # ----------------------------------------------------------------
    def detect_consonant(self, frame: np.ndarray) -> int:
        """
        周波数エネルギー分析で調音を判定し、上位2bits（0〜3）を返す。

        Returns:
            int: 0=None, 1=s, 2=n, 3=m
        """
        if not _scipy_available:
            return 0

        n = len(frame)
        spectrum = np.abs(rfft(frame))
        freqs    = rfftfreq(n, d=1.0 / AUDIO_SAMPLE_RATE)

        def band_energy(lo, hi):
            mask = (freqs >= lo) & (freqs <= hi)
            return float(np.mean(spectrum[mask])) if np.any(mask) else 0.0

        total_energy = float(np.mean(spectrum)) + 1e-10

        # s 判定: 1500Hz 以上のエネルギーが高い
        sib_ratio = band_energy(AUDIO_SIBILANT_FREQ, AUDIO_SAMPLE_RATE // 2) / total_energy

        # n/m 判定: 250〜300Hz の鼻腔共鳴
        nasal_ratio = band_energy(250, 300) / total_energy

        # m 判定: 冒頭 10ms が無音
        bilabial_n = int(AUDIO_SAMPLE_RATE * AUDIO_BILABIAL_MS / 1000)
        onset_rms  = float(np.sqrt(np.mean(frame[:bilabial_n] ** 2))) if bilabial_n > 0 else 1.0
        is_bilabial = onset_rms < 0.01

        if sib_ratio > 0.4:
            return 1   # s
        if nasal_ratio > 0.15:
            if is_bilabial:
                return 3   # m
            return 2       # n
        return 0           # クリーン

    # ----------------------------------------------------------------
    def decode(self, frame: np.ndarray) -> int:
        """
        フレームから 4bits 整数を復元する。

        Returns:
            int: 0〜15
        """
        consonant_bits = self.detect_consonant(frame)
        vowel_bits     = self.detect_vowel(frame)
        return ((consonant_bits & 0x3) << 2) | (vowel_bits & 0x3)
