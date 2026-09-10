#!/usr/bin/env python3
"""Compatibility filename; use se_precompute.py --scale <definition.json>."""
import multiprocessing as mp
from se_precompute import main

if __name__ == '__main__':
    mp.freeze_support()
    main()
