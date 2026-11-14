"""open_clip wrapper. The model is loaded once, lazily, and runs on CPU (the demo laptop has no NVIDIA GPU)."""

import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from . import config

_lock = threading.Lock()  # torch modules are not guaranteed thread-safe; requests run in a thread pool


@lru_cache(maxsize=1)
def _model():
    model, _, preprocess = open_clip.create_model_and_transforms(config.CLIP_MODEL, pretrained=config.CLIP_PRETRAINED)
    model.eval()
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL)
    return model, preprocess, tokenizer


def _normalize(x: torch.Tensor) -> np.ndarray:
    return (x / x.norm(dim=-1, keepdim=True)).cpu().numpy().astype(np.float32)


def embed_images(paths: list[Path], batch_size: int = 32) -> np.ndarray:
    model, preprocess, _ = _model()
    chunks = []
    with _lock, torch.no_grad():
        for start in range(0, len(paths), batch_size):
            batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch_size]])
            chunks.append(_normalize(model.encode_image(batch)))
    return np.concatenate(chunks) if chunks else np.zeros((0, 512), dtype=np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    model, _, tokenizer = _model()
    with _lock, torch.no_grad():
        return _normalize(model.encode_text(tokenizer(texts)))


def warm_up() -> None:
    embed_texts(["a room"])
