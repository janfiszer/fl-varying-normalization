import unittest
import torch
import torch.nn as nn
from src.deep_learning import models

class TestModels(unittest.TestCase):
    def test_unet2d_forward(self):
        model = models.UNet2D(in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 2, 32, 32))

    def test_unet3d_forward(self):
        model = models.UNet3D(in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 16, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 2, 16, 32, 32))

    def test_simple_cnn2d_forward(self):
        model = models.SimpleCNN2D(in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 2, 32, 32))

    def test_simple_cnn3d_forward(self):
        model = models.SimpleCNN3D(in_channels=1, out_channels=2)
        x = torch.randn(1, 1, 8, 16, 16)
        out = model(x)
        self.assertEqual(out.shape, (1, 2, 8, 16, 16))

    def test_get_model_2d(self):
        model = models.get_model('unet2d', in_channels=1, out_channels=2)
        self.assertIsInstance(model, nn.Module)
        x = torch.randn(1, 1, 32, 32)
        out = model(x)
        self.assertEqual(out.shape[1], 2)

    def test_get_model_3d(self):
        model = models.get_model('unet3d', in_channels=1, out_channels=2)
        self.assertIsInstance(model, nn.Module)
        x = torch.randn(1, 1, 8, 16, 16)
        out = model(x)
        self.assertEqual(out.shape[1], 2)

    def test_get_model_invalid(self):
        with self.assertRaises(ValueError):
            models.get_model('invalid_model', in_channels=1, out_channels=2)

if __name__ == '__main__':
    unittest.main()