#!/usr/bin/env python3
"""Compatibility entry point for the modular IndexTTS web application."""

from indextts_web.main import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
