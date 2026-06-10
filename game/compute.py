"""
compute.py — 計算バックエンド検出ユーティリティ
"""
from config import COMPUTE_MODE, GA_POP_SIZE_GPU, GA_POP_SIZE_CPU


def detect_device() -> str:
    """利用可能なデバイスを返す。'cuda' または 'cpu'。"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
    except ImportError:
        pass
    return 'cpu'


def detect_compute_mode() -> str:
    """計算モードを返す。'torch_gpu' または 'numpy'。"""
    if COMPUTE_MODE != "auto":
        return COMPUTE_MODE
    try:
        import torch
        if torch.cuda.is_available():
            return 'torch_gpu'
    except ImportError:
        pass
    return 'numpy'


def get_pop_size(mode: str) -> int:
    """モードに応じた pop_size を返す。"""
    return GA_POP_SIZE_GPU if mode == 'torch_gpu' else GA_POP_SIZE_CPU
