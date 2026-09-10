"""Unified fixed-width SE/CSE/CCSE, matching CSE_SE_CCSE_unified.cpp.

Columns are SE, CSE, CCSE. Entropies use natural logs and p > 1e-12.
All three use the SAME spectrum. FFT length deliberately matches the C++
cyclic convolution, including any fourth-power wraparound. No octave folding.
"""
from __future__ import annotations
import math
import numpy as np

METRICS = ('se', 'cse', 'ccse')
ALGORITHM = 'UnifiedSpectralEntropy/fixed-sigma-cyclic-powers-1-2-4/v1'
PARAMETER_DEFAULTS = {
    'sigma_hz': 1.0,
    'q_per_100hz': 0.8,
    'model_base_freq_hz': 100.0,
    'max_freq_hz': 4000.0,
    'resolution_hz': 1.0,
}


def resolve_parameters(overrides=None):
    if overrides is None: overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError('spectral_parameters must be an object')
    unknown = set(overrides) - set(PARAMETER_DEFAULTS)
    if unknown: raise ValueError(f'Unknown spectral parameters: {sorted(unknown)}')
    p = {**PARAMETER_DEFAULTS, **overrides}
    p = {k: float(v) for k, v in p.items()}
    if any(not math.isfinite(v) or v <= 0 for v in p.values()):
        raise ValueError('Spectral parameters must be finite and positive')
    if p['q_per_100hz'] > 1: raise ValueError('q must be in (0,1]')
    if p['resolution_hz'] > p['max_freq_hz']:
        raise ValueError('resolution must not exceed max frequency')
    return p


def fft_size(max_freq, resolution):
    n = 1
    while n < 2.0 * max_freq / resolution: n <<= 1
    return n


def spectrum_at_frequency(f0, parameters=None):
    p = resolve_parameters(parameters)
    if not math.isfinite(f0) or f0 <= 0 or f0 >= p['max_freq_hz']:
        raise ValueError(f'Model fundamental {f0} must be positive and below max_freq_hz')
    sigma, q = p['sigma_hz'], p['q_per_100hz']
    res, maximum = p['resolution_hz'], p['max_freq_hz']
    n = fft_size(maximum, res)
    out = np.zeros(n, dtype=np.float64)
    h = 1
    while f0*h < maximum:
        freq = f0*h
        amplitude = q ** ((freq-f0)/100.0)
        if amplitude < 0.001: break
        start = max(0, int((freq-5*sigma)/res))
        end = min(n-1, int((freq+5*sigma)/res))
        x = np.arange(start, end+1, dtype=np.float64)*res - freq
        out[start:end+1] += amplitude*np.exp(-x*x/(2*sigma*sigma))
        h += 1
    return out


def spectrum_for(step, edo, parameters=None):
    p = resolve_parameters(parameters)
    return spectrum_at_frequency(p['model_base_freq_hz']*2.0**(int(step)/int(edo)), p)


def direct_entropy(spectrum):
    total = float(np.sum(spectrum))
    if total <= 0: return 0.0
    p = spectrum/total
    nz = p > 1e-12
    return float(-np.sum(p[nz]*np.log(p[nz])))


def raw_entropies(spectrum, fourier=None):
    """One spectrum and one forward FFT produce H(S), H(S*S), H(S^*4)."""
    f = np.fft.rfft(spectrum) if fourier is None else fourier
    f2 = f*f
    c2 = np.abs(np.fft.irfft(f2, n=len(spectrum)))
    c4 = np.abs(np.fft.irfft(f2*f2, n=len(spectrum)))
    return np.array([direct_entropy(spectrum), direct_entropy(c2), direct_entropy(c4)])


def prepare_notes(steps, edo, parameters=None):
    """Build each absolute note's spectrum/FFT and baseline entropies once."""
    p = resolve_parameters(parameters)
    spectra = {s: spectrum_for(s, edo, p) for s in steps}
    fourier = {s: np.fft.rfft(v) for s, v in spectra.items()}
    singles = {s: raw_entropies(v, fourier[s]) for s, v in spectra.items()}
    return spectra, fourier, singles


def score_from_notes(combo, spectra, fourier, singles):
    if not combo: raise ValueError('Empty chord')
    # Repetitions retain their spectral mass and their share of the baseline.
    if len(set(combo)) == 1: return np.zeros(3, dtype=np.float64)
    spectrum = np.zeros_like(spectra[combo[0]])
    f = np.zeros_like(fourier[combo[0]])
    baseline = np.zeros(3, dtype=np.float64)
    for step in combo:
        spectrum += spectra[step]
        f += fourier[step]
        baseline += singles[step]
    return raw_entropies(spectrum, f) - baseline/len(combo)


def score_steps(steps, edo, parameters=None):
    """Absolute register scoring: model 100 Hz is step zero, not chord bass."""
    xs = tuple(map(int, steps))
    return score_from_notes(xs, *prepare_notes(sorted(set(xs)), edo, parameters))


def score_ratios(ratios, parameters=None, *, normalize_first=True):
    """C++ interactive convention for reference comparisons (first tone = 1)."""
    ratios = tuple(map(float, ratios))
    if not ratios or any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError('Ratios must be finite and positive')
    p = resolve_parameters(parameters)
    # Match the C++ log -> subtract -> exp path, including its rounding at
    # Gaussian-window endpoints. Direct x/root can differ by several 1e-6.
    root = math.log(ratios[0]) if normalize_first else 0.0
    spectra = [spectrum_at_frequency(p['model_base_freq_hz']*math.exp(math.log(x)-root), p) for x in ratios]
    return raw_entropies(np.sum(spectra, axis=0)) - np.mean([raw_entropies(s) for s in spectra], axis=0)
