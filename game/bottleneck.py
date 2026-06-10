"""
bottleneck.py — リアルタイム半双方向通信バス（Sage-Brute版）
"""
import numpy as np
from config import *


class Bottleneck:
    def __init__(self, sage, brute):
        self.sage      = sage
        self.brute     = brute
        self.direction = 'S→B'
        self._buf      = []
        self._frame    = 0
        self._turn     = 0
        self._last_pulse  = [0, 0]
        self._last_action = np.array([0.0, 0.5, 0.0])
        self._history     = []
        self._last_obs_sage  = None
        self._last_obs_brute = None
        self.audio_enabled = False
        self.converter     = None

    def enable_audio(self):
        if self.converter is None:
            from game.phoneme import PhonemeConverter
            self.converter = PhonemeConverter()
        self.audio_enabled = True

    def disable_audio(self):
        self.audio_enabled = False

    def toggle_audio(self) -> bool:
        if self.audio_enabled:
            self.disable_audio()
        else:
            self.enable_audio()
        return self.audio_enabled

    def step(self, obs_sage, obs_brute):
        self._last_obs_sage  = obs_sage
        self._last_obs_brute = obs_brute
        f = self._frame
        self._frame += 1

        is_gen  = (f % PULSE_GEN_INTERVAL == 0)
        is_cons = (f % PULSE_CONSUME_RATE == 0)
        is_turn = (f > 0 and f % TURN_FRAMES == 0)

        if is_gen:
            if self.direction == 'S→B':
                pulse = self.sage.forward(obs_sage, is_pulse_frame=True)
            else:
                _, pulse = self.brute.forward(
                    self._inject(obs_brute, 0), is_pulse_frame=True)
            self._buf.append(pulse)
            bits = [(pulse >> 1) & 1, pulse & 1]
            self._last_pulse = bits
            self._history.append(bits)
            if len(self._history) > 80:
                self._history.pop(0)
            if self.audio_enabled and self.converter:
                self.converter.play(pulse, self.direction)
        else:
            # 非生成フレームでも両NNはobsを処理
            self.sage.forward(obs_sage, is_pulse_frame=False)
            self.brute.forward(obs_brute, is_pulse_frame=False)

        if is_cons and self._buf:
            pulse = self._buf.pop(0)
            if self.direction == 'S→B':
                action, _ = self.brute.forward(
                    self._inject(obs_brute, pulse), is_pulse_frame=True)
                self._last_action = action
            else:
                self.sage.forward(
                    self._inject(obs_sage, pulse), is_pulse_frame=True)

        if is_turn:
            self.direction = 'B→S' if self.direction == 'S→B' else 'S→B'
            self._turn += 1

        return self._last_action

    def _inject(self, obs, pulse):
        obs = obs.copy()
        obs[-2] = float((pulse >> 1) & 1)
        obs[-1] = float(pulse & 1)
        return obs

    def reset_episode(self):
        self._buf    = []
        self._frame  = 0
        self._turn   = 0
        self._last_pulse  = [0, 0]
        self._last_action = np.array([0.0, 0.5, 0.0])
        self.sage.reset_episode()
        self.brute.reset_episode()

    # reset エイリアス（旧コード互換）
    def reset(self, prefill=False):
        self.reset_episode()

    # モニター用アクセサ
    def get_current_pulse(self):    return self._last_pulse
    def get_mode(self):             return 'listen' if self.direction == 'S→B' else 'speak'
    def get_display_progress(self): return (self._frame % TURN_FRAMES) / TURN_FRAMES
    def get_turn_progress(self):    return self.get_display_progress()
    def get_display_history(self):  return self._history[-20:]
    def get_pulse_history(self):    return self._history

    def get_display_phoneme(self):
        p = (self._last_pulse[0] << 1) | self._last_pulse[1]
        return PHONEME_TABLE.get(p, '')

    def get_last_phoneme(self):
        return self.get_display_phoneme()
