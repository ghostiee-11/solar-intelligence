#!/bin/bash
cd /Users/amankumar/Desktop/gsoc-26/solar-intelligence
python -m pytest tests/test_dual_source.py -v --tb=short 2>&1
