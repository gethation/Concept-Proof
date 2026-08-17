"""Shared primitives for the pair-study pipeline.

Nothing in here executes on import and nothing in here is a CLI. Entry points
live in ingest/, features/, backtest/ and report/; they reach this package by
putting scripts/ on sys.path (see lib.bootstrap for the one-liner).

The split exists because these functions were previously copy-pasted across the
pipeline -- format_taipei_timestamps lived in six files, read_ohlcv in four --
and the copies had already drifted (two spellings of the FX staleness column,
two ways of sorting a spliced FX frame). One definition each means a fix lands
everywhere at once.
"""
