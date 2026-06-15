"""Tests for models."""

import unittest

from models.pinn_model import PINNModel

class TestPINNModel(unittest.TestCase):
    """Test cases for PINNModel."""

    def test_init(self) -> None:
        """Test initialization."""
        model = PINNModel()
        self.assertIsInstance(model, PINNModel)