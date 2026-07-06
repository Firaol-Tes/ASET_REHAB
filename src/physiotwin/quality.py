"""Movement-quality features and the population envelope.

Each repetition is summarized by six clinically interpretable numbers:
  rom_deg     range of motion: peak rotation angle from rep start
  dur_s       repetition duration
  speed_norm  peak angular speed normalized by ROM (1/s) — tempo
              independent of movement size
  ldj         log dimensionless jerk (smoothness, Balasubramanian 2015)
  hf_ratio    fraction of velocity power above 3 Hz — tremor band
  end_deg     angle left over at the end of the rep — failure to
              return to the start position

The envelope per exercise is a robust interval median +/- K*MAD per
feature, learned from genuine repetitions. Bounds are one-sided where
only one direction is clinically abnormal: smoother than human (ldj),
cleaner than human (hf_ratio), and more-complete return (end_deg) are
not faults.
"""
from __future__ import annotations

import numpy as np

# feature -> (fault sides, interval half-width in robust SDs).
# Wide intervals for features whose fault signatures are huge
# (truncation leaves tens of degrees unreturned; tremor multiplies
# high-frequency power), tight ones where faults are subtler.
FEATURES = {
    "rom_deg": (("lo", "hi"), 3.0),
    "dur_s": (("lo", "hi"), 3.0),
    "speed_norm": (("lo", "hi"), 3.5),
    "ldj": (("hi",), 4.0),
    "hf_ratio": (("hi",), 6.0),
    "end_deg": (("hi",), 6.0),
}
MAD_SCALE = 1.4826


def rep_features(theta: np.ndarray, fs: float = 100.0) -> dict:
    """Quality features for one repetition's rotation-angle profile."""
    theta = theta - theta[0]
    dur = len(theta) / fs
    rom = float(np.degrees(np.max(np.abs(theta))))
    v = np.gradient(theta, 1 / fs)
    vpeak = max(float(np.max(np.abs(v))), 1e-9)
    j = np.gradient(np.gradient(v, 1 / fs), 1 / fs)
    ldj = float(np.log((dur ** 3 / vpeak ** 2) * np.trapz(j ** 2, dx=1 / fs)
                       + 1e-12))
    spec = np.abs(np.fft.rfft(v - v.mean())) ** 2
    freqs = np.fft.rfftfreq(len(v), 1 / fs)
    tot = float(spec.sum()) + 1e-12
    hf = float(spec[freqs > 3.0].sum()) / tot
    end = float(np.degrees(np.abs(theta[-1])))
    return {
        "rom_deg": rom, "dur_s": dur,
        "speed_norm": float(np.degrees(vpeak)) / max(rom, 1e-6),
        "ldj": ldj, "hf_ratio": hf, "end_deg": end,
    }


class Envelope:
    """Robust per-feature intervals learned from genuine repetitions."""

    def __init__(self, rows: list[dict]):
        self.lo, self.hi = {}, {}
        for f, (_sides, k) in FEATURES.items():
            x = np.asarray([r[f] for r in rows], float)
            med = np.median(x)
            mad = MAD_SCALE * np.median(np.abs(x - med)) + 1e-9
            self.lo[f] = med - k * mad
            self.hi[f] = med + k * mad

    def flag(self, feats: dict) -> list[str]:
        """Names of features outside the envelope (empty = pass)."""
        out = []
        for f, (sides, _k) in FEATURES.items():
            if "lo" in sides and feats[f] < self.lo[f]:
                out.append(f)
            elif "hi" in sides and feats[f] > self.hi[f]:
                out.append(f)
        return out
