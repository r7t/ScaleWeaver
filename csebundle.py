#!/usr/bin/env python3
"""Compatibility entry point: all new builds produce SE, CSE and CCSE together.

CSEBundle.entry()/score() still return pure CSE by default.
Old AdaptiveCSE and SE0 caches are intentionally not accepted by this loader.
"""
from spectral_bundle import (SpectralBundle, ensure_bundle, build_bundle,
                             bundle_paths, numerical_signature, colex_rank,
                             FORMAT, main)
from spectral_model import fft_size
CSEBundle = SpectralBundle

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
