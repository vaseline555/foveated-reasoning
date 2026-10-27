#!/usr/bin/env python3
"""Standalone experiment entry point; preparation never calls the API."""

from active_vlm.experiment.cli import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, BlockingIOError) as error:
        raise SystemExit(str(error)) from None
