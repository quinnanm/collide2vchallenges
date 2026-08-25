"""Shared epsilon for numerically-sensitive divisions in converters.py
(dxy/dxysig ratio) -- kept as its own module so it can't silently drift if
this ever needs to be tuned per-project.
"""
EPS = 1e-4
