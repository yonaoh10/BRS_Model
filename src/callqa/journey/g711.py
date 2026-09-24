"""G.711 A-law and mu-law, in numpy: telephone audio with no external tool.

Most call recorders store 8 kHz G.711; decoding it is a 256-entry table
lookup, so an NMF file in either law needs neither ffmpeg nor PyAV. The
encoders exist for tests and the demo generator (synthetic recordings).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _alaw_to_linear(a: int) -> int:
    a ^= 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    if seg == 0:
        t += 8
    elif seg == 1:
        t += 0x108
    else:
        t += 0x108
        t <<= seg - 1
    return t if a & 0x80 else -t


def _ulaw_to_linear(u: int) -> int:
    u = ~u & 0xFF
    t = ((u & 0x0F) << 3) + 0x84
    t <<= (u & 0x70) >> 4
    return (0x84 - t) if u & 0x80 else (t - 0x84)


ALAW_TABLE = np.array([_alaw_to_linear(i) for i in range(256)], dtype=np.int16)
ULAW_TABLE = np.array([_ulaw_to_linear(i) for i in range(256)], dtype=np.int16)


def decode_alaw(data: bytes) -> np.ndarray:
    return ALAW_TABLE[np.frombuffer(data, dtype=np.uint8)]


def decode_ulaw(data: bytes) -> np.ndarray:
    return ULAW_TABLE[np.frombuffer(data, dtype=np.uint8)]


_SEG_END = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF]


def _linear_to_alaw(pcm: int) -> int:
    pcm = int(pcm) >> 3
    if pcm >= 0:
        mask = 0xD5
    else:
        mask = 0x55
        pcm = -pcm - 1
    seg = next((i for i, end in enumerate(_SEG_END) if pcm <= end >> 3), 8)
    if seg >= 8:
        return 0x7F ^ mask
    aval = seg << 4
    aval |= (pcm >> 1) & 0x0F if seg < 2 else (pcm >> seg) & 0x0F
    return aval ^ mask


def _linear_to_ulaw(pcm: int) -> int:
    bias, clip = 0x84, 32635
    sign = 0x80 if pcm < 0 else 0
    if pcm < 0:
        pcm = -pcm
    pcm = min(pcm, clip) + bias
    exponent = 7
    mask = 0x4000
    while exponent > 0 and not (pcm & mask):
        exponent -= 1
        mask >>= 1
    mantissa = (pcm >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


@lru_cache(maxsize=2)
def _encoder(law: str) -> np.ndarray:
    fn = _linear_to_alaw if law == "a" else _linear_to_ulaw
    return np.array([fn(v) for v in range(-32768, 32768)], dtype=np.uint8)


def encode_alaw(pcm: np.ndarray) -> bytes:
    idx = np.asarray(pcm, dtype=np.int32) + 32768
    return _encoder("a")[np.clip(idx, 0, 65535)].tobytes()


def encode_ulaw(pcm: np.ndarray) -> bytes:
    idx = np.asarray(pcm, dtype=np.int32) + 32768
    return _encoder("u")[np.clip(idx, 0, 65535)].tobytes()
