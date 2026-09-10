#!/usr/bin/env python3
"""Compatibility filename; use build_native_se_bundle.py --scale <definition.json>."""
import multiprocessing as mp
from build_native_se_bundle import main

if __name__ == '__main__':
    mp.freeze_support()
    main()
