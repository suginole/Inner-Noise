"""
phoneme.py — 2bitsパルス ↔ 母音変換モジュール（LPC改良版 + 方向別ピッチ）

2bits = 母音のみ（子音なし）
  00 = う  (F1低・F2中)
  01 = い  (F1低・F2高)
  10 = お  (F1中・F2低)
  11 = あ  (F1高・F2中)

方向別ピッチ:
  S→M（傾聴ターン）: PITCH_FACTOR_HIGH=1.4 → 高ピッチ（♀）
  M→S（発話ターン）: PITCH_FACTOR_LOW=0.7  → 低ピッチ（♂）

フォルマント推定: LPC（線形予測分析）
  Levinson-Durbin法で声道フィルタの極を求め、偏角からフォルマントを推定。
  FFTピーク検出より精度が高い（倍音の影響を受けない）。

判定ロジック（Mokhtari & Tanaka 2000）:
  F2/F1 > 6.0  → い  (実測F2/F1=8.3)
  F1 > 650Hz   → あ  (実測F1=801Hz)
  F2 > 1200Hz  → う  (実測F2=1550Hz)
  else         → お  (実測F2=811Hz)
"""
from __future__ import annotations
import numpy as np

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
    PITCH_FACTOR_HIGH, PITCH_FACTOR_LOW,
)


def _is_mixer_ready() -> bool:
    """pygame.mixer が使用可能な状態かどうかを確認する。"""
    if not _pygame_available:
        return False
    try:
        info = pygame.mixer.get_init()
        return info is not None
    except Exception:
        return False


def _ensure_mixer() -> bool:
    """
    pygame.mixer を使用可能な状態にする。
    pygame.init() で既に初期化されていればそのまま使う。
    まだ初期化されていなければ mono/22050Hz で初期化する。

    Returns:
        True if mixer is ready
    """
    if not _pygame_available:
        return False

    # 既に初期化済みならそのまま使う
    try:
        if pygame.mixer.get_init() is not None:
            return True
    except Exception:
        pass

    # 未初期化なら初期化を試みる
    try:
        pygame.mixer.init(
            frequency=AUDIO_SAMPLE_RATE,
            size=-16,
            channels=1,
            buffer=AUDIO_FRAME_SAMPLES,
        )
        return True
    except Exception:
        return False


# ================================================================
def lpc_formants(frame: np.ndarray, sr: int, n_formants: int = 4,
                 lpc_order: int = 12) -> list[float]:
    """
    LPC（線形予測分析）でフォルマント周波数を推定する。

    手順:
      1. プリエンファシス（高域強調）
      2. ハミング窓
      3. Levinson-Durbin法でLPC係数を計算
      4. LPC多項式の根（極）を求める
      5. 極の偏角 → フォルマント周波数
      6. 帯域幅フィルタリング（広すぎる極を除外）

    Args:
        frame: 音声フレーム（float32）
        sr: サンプリングレート
        n_formants: 返すフォルマントの数
        lpc_order: LPC次数

    Returns:
        フォルマント周波数のリスト（Hz）、昇順ソート済み
    """
    if not _scipy_available or len(frame) < lpc_order + 1:
        return [0.0] * n_formants

    # 1. プリエンファシス
    pre_emph = 0.97
    frame_pe = np.append(frame[0], frame[1:] - pre_emph * frame[:-1])

    # 2. ハミング窓
    frame_w = frame_pe * np.hamming(len(frame_pe))

    # 3. Levinson-Durbin法
    n = len(frame_w)
    order = min(lpc_order, n - 1)

    r = np.correlate(frame_w, frame_w, mode='full')
    r = r[n - 1:]

    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    if e == 0:
        return [0.0] * n_formants

    for i in range(1, order + 1):
        lam = -np.dot(a[:i], r[i:0:-1]) / e
        a_new = a[:i + 1] + lam * a[:i + 1][::-1]
        a[:i + 1] = a_new
        e = e * (1 - lam ** 2)
        if e <= 0:
            break

    # 4. 多項式の根
    roots = np.roots(a)

    # 5. 単位円内かつ上半平面の極のみ
    roots = roots[np.abs(roots) < 1.0]
    roots = roots[roots.imag >= 0]

    if len(roots) == 0:
        return [0.0] * n_formants

    angles = np.angle(roots)
    freqs  = angles * (sr / (2 * np.pi))

    # 6. 帯域幅フィルタ（< 400Hz）と最低周波数フィルタ（> 50Hz）
    bandwidths = -np.log(np.abs(roots)) * (sr / np.pi)
    valid = (bandwidths < 400) & (freqs > 50)
    freqs = np.sort(freqs[valid])

    result = list(freqs[:n_formants])
    while len(result) < n_formants:
        result.append(0.0)
    return result


# ================================================================
class PhonemeConverter:
    """
    2bitsパルス → フォルマント合成 → pygame.mixer 出力。

    方向別ピッチ:
      direction='S→M': PITCH_FACTOR_HIGH（高ピッチ）
      direction='M→S': PITCH_FACTOR_LOW（低ピッチ）
    """

    def __init__(self):
        self._mixer_ok = _ensure_mixer()
        self._channel  = None
        if self._mixer_ok:
            try:
                # モノラルチャンネルを確保
                pygame.mixer.set_num_channels(max(8, pygame.mixer.get_num_channels()))
                self._channel = pygame.mixer.Channel(0)
            except Exception:
                self._channel = None

    # ----------------------------------------------------------------
    def pulse_to_phoneme(self, bits2: int) -> str:
        return PHONEME_TABLE.get(bits2 & 0x3, '?')

    # ----------------------------------------------------------------
    def synthesize(self, bits2: int, direction: str = 'S→M') -> np.ndarray:
        """
        2bits + 方向 → 200ms の音声バッファ（float32）を返す。

        Args:
            bits2: 0〜3 の母音インデックス
            direction: 'S→M'（高ピッチ）または 'M→S'（低ピッチ）
        """
        bits2 = bits2 & 0x3
        vowel = PHONEME_VOWEL[bits2]
        f1_base, f2_base = PHONEME_FORMANTS[vowel]

        # 方向別ピッチ係数を適用
        pitch = PITCH_FACTOR_HIGH if direction == 'S→M' else PITCH_FACTOR_LOW
        f1 = f1_base * pitch
        f2 = f2_base * pitch

        n  = AUDIO_FRAME_SAMPLES
        sr = AUDIO_SAMPLE_RATE
        t  = np.linspace(0, AUDIO_FRAME_MS / 1000.0, n, endpoint=False)

        buf = (0.6 * np.sin(2 * np.pi * f1 * t) +
               0.4 * np.sin(2 * np.pi * f2 * t))

        # フェードイン/アウト（クリックノイズ防止）
        fade_n = min(int(sr * 0.005), n // 4)
        buf[:fade_n]  *= np.linspace(0, 1, fade_n)
        buf[-fade_n:] *= np.linspace(1, 0, fade_n)

        # 正規化
        peak = np.max(np.abs(buf))
        if peak > 0:
            buf = buf / peak * 0.8

        return buf.astype(np.float32)

    # ----------------------------------------------------------------
    def play(self, bits2: int, direction: str = 'S→M') -> None:
        """
        synthesize() の結果を pygame.mixer でリアルタイム出力する。

        バグ修正:
          - _ensure_mixer() を毎回確認（mixer が後から初期化された場合に対応）
          - make_sound には int16 の 1D 配列を渡す（float64 は無音になる）
          - channels=2 の mixer でも動作するよう stereo に変換
        """
        # mixer の状態を毎回確認（遅延初期化対応）
        if not _ensure_mixer():
            return

        buf = self.synthesize(bits2, direction)

        # int16 に変換
        pcm_mono = (buf * 32767).astype(np.int16)

        # mixer が stereo（channels=2）で初期化されている場合は stereo に変換
        try:
            mixer_info = pygame.mixer.get_init()
            if mixer_info and mixer_info[2] == 2:
                # stereo: (N, 2) に変換
                pcm = np.column_stack([pcm_mono, pcm_mono])
            else:
                pcm = pcm_mono

            sound = pygame.sndarray.make_sound(pcm)

            # チャンネルが取得できない場合は直接再生
            if self._channel is None:
                try:
                    self._channel = pygame.mixer.Channel(0)
                except Exception:
                    pass

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
            self.play(bits2, direction='S→M')
            time.sleep(interval_ms / 1000.0)


# ================================================================
class PhonemeDecoder:
    """
    マイク入力 → LPCフォルマント推定 → 2bits 復元。
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

    def record_frame(self) -> np.ndarray:
        if not self._available or self._stream is None:
            return np.zeros(AUDIO_FRAME_SAMPLES, dtype=np.float32)
        try:
            raw = self._stream.read(AUDIO_FRAME_SAMPLES, exception_on_overflow=False)
            return np.frombuffer(raw, dtype=np.float32).copy()
        except Exception:
            return np.zeros(AUDIO_FRAME_SAMPLES, dtype=np.float32)

    def detect_f1_f2(self, frame: np.ndarray) -> tuple[float, float]:
        """LPCでF1・F2を推定する。"""
        formants = lpc_formants(
            frame, AUDIO_SAMPLE_RATE,
            n_formants=4,
            lpc_order=max(12, int(AUDIO_SAMPLE_RATE / 1000) + 2),
        )
        f1 = formants[0] if len(formants) > 0 else 0.0
        f2 = formants[1] if len(formants) > 1 else 0.0
        return f1, f2

    def detect_vowel(self, frame: np.ndarray) -> int:
        """
        LPC推定F1/F2から母音を判定し、2bits（0〜3）を返す。

        判定ツリー（Mokhtari & Tanaka 2000）:
          F2/F1 > 6.0  → い (01)
          F1 > 650Hz   → あ (11)
          F2 > 1200Hz  → う (00)
          else         → お (10)
        """
        f1, f2 = self.detect_f1_f2(frame)
        if f1 <= 0:
            return 0

        ratio = f2 / f1

        if ratio > VOWEL_F2F1_I:
            return 1   # い
        if f1 > VOWEL_F1_A:
            return 3   # あ
        if f2 > VOWEL_F2_U:
            return 0   # う
        return 2       # お

    def decode(self, frame: np.ndarray) -> int:
        return self.detect_vowel(frame) & 0x3

    def record_frame_sync(self) -> tuple[np.ndarray, int]:
        """5Hz同期読み取り用。"""
        frame = self.record_frame()
        bits2 = self.decode(frame)
        return frame, bits2
