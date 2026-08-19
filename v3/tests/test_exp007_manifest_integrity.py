#!/usr/bin/env python3
"""Regression tests for EXP-007 frozen manifest/result consistency."""

from __future__ import annotations

import unittest

from v3.ci.check_exp007_evidence import require_close


class Exp007ManifestIntegrityTests(unittest.TestCase):
    def test_require_close_accepts_tiny_float_noise(self) -> None:
        require_close(0.4990119071382824 + 1e-13, 0.4990119071382824, "tiny")

    def test_require_close_rejects_material_result_change(self) -> None:
        with self.assertRaises(ValueError):
            require_close(0.51, 0.4990119071382824, "material drift")


if __name__ == "__main__":
    unittest.main()
