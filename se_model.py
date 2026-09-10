"""Scale-independent SE0 numerical kernel from the supplied SE precomputer.

Unlike CSE, sigma increases with frequency, each partial has area compensation,
and the spectrum is not convolved. Model base frequency is separate from output tuning.
"""
from __future__ import annotations
import math
import numpy as np

PARAMETER_DEFAULTS = {
    'sigma0_hz_at_100hz': 0.4,
    'q_per_100hz': 0.8,
    'model_base_freq_hz': 100.0,
    'max_freq_hz': 4000.0,
    'resolution_hz': 1.0,
}


def resolve_parameters(overrides=None):
    if overrides is None: overrides = {}
    if not isinstance(overrides, dict): raise ValueError('se_parameters must be an object')
    if set(overrides) - set(PARAMETER_DEFAULTS):
        raise ValueError(f'Unknown SE parameters: {sorted(set(overrides)-set(PARAMETER_DEFAULTS))}')
    out = dict(PARAMETER_DEFAULTS)
    out.update({k:float(v) for k,v in overrides.items()})
    if any(not math.isfinite(v) or v <= 0 for v in out.values()):
        raise ValueError('SE parameters must be finite and positive')
    if out['q_per_100hz'] > 1: raise ValueError('SE q must be in (0,1]')
    if out['resolution_hz'] > out['max_freq_hz']:
        raise ValueError('SE resolution must not exceed max frequency')
    return out


def spectrum_for(step, edo, parameters):
    p = parameters
    return note_spectrum_se0(step, p['sigma0_hz_at_100hz'], p['q_per_100hz'],
        p['model_base_freq_hz'], p['max_freq_hz'], p['resolution_hz'],
        cpp_buffer_size(p['max_freq_hz'], p['resolution_hz']), edo=edo)


def cpp_buffer_size(max_freq: float, resolution: float) -> int:
    """Exact CSE2.cpp SE0 buffer-size rule, even though no FFT is performed."""
    n = 1
    target = (max_freq / resolution) * 2.0
    while n < target:
        n <<= 1
    return n


def note_spectrum_se0(step: int, sigma0: float, q: float, base_freq: float,
                      max_freq: float, resolution: float, n_bins: int, *, edo: int) -> np.ndarray:
    """One-note spectrum matching CSE2.cpp::SE0 sample construction."""
    f0 = base_freq * 2.0 ** (int(step) / edo)
    spectrum = np.zeros(n_bins, dtype=np.float64)
    h = 1
    while True:
        freq = f0 * h
        if freq >= max_freq:
            break

        current_sigma = sigma0 * (freq / 100.0)
        if current_sigma < resolution:
            current_sigma = resolution

        amplitude = q ** ((freq - f0) / 100.0)
        if amplitude < 0.001:
            break
        norm_amplitude = amplitude / current_sigma

        # C++ casts positive values to int, equivalent to truncation/floor here.
        start = max(0, int((freq - 5.0 * current_sigma) / resolution))
        end = min(n_bins - 1, int((freq + 5.0 * current_sigma) / resolution))
        idx = np.arange(start, end + 1, dtype=np.float64)
        current_f = idx * resolution
        spectrum[start:end + 1] += norm_amplitude * np.exp(
            -((current_f - freq) ** 2) / (2.0 * current_sigma * current_sigma)
        )
        h += 1
    return spectrum


def entropy_se0(spectrum: np.ndarray) -> float:
    """Direct spectral entropy; deliberately no convolution/FFT."""
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    p = spectrum / total
    nz = p > 1e-12
    return float(-(p[nz] * np.log(p[nz])).sum())


def score_from_spectra(combo: tuple[int, ...], note_specs,
                       single_entropy, zero) -> float:
    spectrum = zero.copy()
    for step in combo:
        spectrum += note_specs[int(step)]
    mixture_entropy = entropy_se0(spectrum)
    baseline = sum(single_entropy[int(step)] for step in combo) / len(combo)
    return float(mixture_entropy - baseline)


