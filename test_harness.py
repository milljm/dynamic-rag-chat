#!/usr/bin/env python3
"""Discover and run every unittest file matching src/*_test.py.

Loads each file as a top-level module (think_tags_test, not src.think_tags_test)
so src/__init__.py is never imported — that module pulls in langchain.

Usage:
    python test_harness.py
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'src'


def discover_paths() -> list[Path]:
    """Return src/*_test.py paths in name order."""
    return sorted(SRC.glob('*_test.py'))


def load_suite(path: Path) -> unittest.TestSuite:
    """Load tests from a file without treating src as a package."""
    name = path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return unittest.defaultTestLoader.loadTestsFromModule(module)


def main() -> int:
    """Run all discovered tests. Return 0 on success."""
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))

    paths = discover_paths()
    if not paths:
        print(f'No tests matching {SRC / "*_test.py"}', flush=True)
        return 1

    print(f'Found {len(paths)} test file(s) in {SRC.relative_to(ROOT)}:', flush=True)
    suite = unittest.TestSuite()
    for path in paths:
        rel = path.relative_to(ROOT)
        print(f'  {rel}', flush=True)
        suite.addTests(load_suite(path))

    print(flush=True)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
