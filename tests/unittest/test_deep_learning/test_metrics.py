import unittest
from unittest.mock import patch, mock_open
import torch
import numpy as np

# src/deep_learning/test_metrics.py


from src.deep_learning.metrics import (
    metrics_to_str,
    LossGeneralizedTwoClassDice,
    GeneralizedTwoClassDice,
    BinaryDice,
    JaccardIndex,
    compute_average_std_metric,
    save_metrics_and_std,
    false_positive_ratio,
)

class TestMetrics(unittest.TestCase):
    def test_metrics_to_str(self):
        metrics = {"dice": 0.85, "jaccard": 0.75}
        s = metrics_to_str(metrics, starting_symbol=">", sep="|")
        self.assertIn("dice: 0.850|", s)
        self.assertIn("jaccard: 0.750|", s)
        self.assertTrue(s.startswith(">"))

    def test_LossGeneralizedTwoClassDice_forward(self):
        device = "cpu"
        pred = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
        target = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        # Without BCE
        loss_fn = LossGeneralizedTwoClassDice(device)
        loss = loss_fn(pred, target)
        self.assertIsInstance(loss, torch.Tensor)
        # With BCE
        loss_fn_bce = LossGeneralizedTwoClassDice(device, binary_crossentropy=True)
        loss_bce = loss_fn_bce(pred, target)
        self.assertIsInstance(loss_bce, torch.Tensor)

    def test_GeneralizedTwoClassDice(self):
        dice = GeneralizedTwoClassDice()
        pred = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        target = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        dice.update(pred, target)
        result = dice.compute()
        self.assertTrue(torch.is_tensor(result))
        # Test static method
        num, denom = GeneralizedTwoClassDice.compute_dice_components(pred, target)
        self.assertIsInstance(num, torch.Tensor)
        self.assertIsInstance(denom, torch.Tensor)

    def test_BinaryDice(self):
        dice = BinaryDice(smooth=1.0, binarize_threshold=0.5)
        pred = torch.tensor([[0.6, 0.2], [0.1, 0.9]], dtype=torch.float32)
        target = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        dice.update(pred, target)
        result = dice.compute()
        self.assertTrue(torch.is_tensor(result))
        # Test static method
        val = BinaryDice._compute_smoothed_dice(torch.tensor(2.0), torch.tensor(3.0), 1.0)
        self.assertAlmostEqual(val.item(), (2*2+1)/(3+1))
        # Test compute_dice_components
        inter, denom = BinaryDice.compute_dice_components(torch.tensor([[1,0]]), torch.tensor([[1,1]]))
        self.assertIsInstance(inter, torch.Tensor)
        self.assertIsInstance(denom, torch.Tensor)

    def test_JaccardIndex(self):
        jac = JaccardIndex(smooth=1.0, binarize_threshold=0.5)
        pred = torch.tensor([[0.6, 0.2], [0.1, 0.9]], dtype=torch.float32)
        target = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
        jac.update(pred, target)
        result = jac.compute()
        self.assertTrue(torch.is_tensor(result))
        # Test static method
        val = JaccardIndex._compute_jaccard(torch.tensor(2.0), torch.tensor(3.0), 1.0)
        self.assertAlmostEqual(val.item(), (2+1)/(3+1))
        # Test compute_jaccard_components
        inter, denom = JaccardIndex.compute_jaccard_components(torch.tensor([[1,0]]), torch.tensor([[1,1]]), None)
        self.assertIsInstance(inter, torch.Tensor)
        self.assertIsInstance(denom, torch.Tensor)

    def test_compute_average_std_metric(self):
        metrics = {"dice": [0.8, 0.9, 1.0], "jaccard": [0.7, 0.8, 0.9]}
        avg, std = compute_average_std_metric(metrics)
        self.assertAlmostEqual(avg["dice"], np.mean([0.8, 0.9, 1.0]))
        self.assertAlmostEqual(std["jaccard"], np.std([0.7, 0.8, 0.9]))

    @patch("src.deep_learning.metrics.pickle.dump")
    @patch("src.deep_learning.metrics.open", new_callable=mock_open)
    @patch("src.deep_learning.metrics.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_save_metrics_and_std(self, mock_join, mock_open_fn, mock_pickle):
        averaged_metrics = {"gen_dice": 0.9}
        stds = {"gen_dice": 0.05}
        save_metrics_and_std(averaged_metrics, "pred_dir", stds=stds, filename_prefix="test")
        self.assertTrue(mock_open_fn.called)
        self.assertTrue(mock_pickle.called)

    def test_false_positive_ratio(self):
        preds = np.array([1, 0, 1, 0])
        target = np.array([0, 0, 1, 0])
        ratio = false_positive_ratio(preds, target)
        self.assertIsInstance(ratio, float)
        # If no true negatives, should return 0
        preds2 = np.array([1, 1])
        target2 = np.array([0, 0])
        ratio2 = false_positive_ratio(preds2, target2)
        self.assertEqual(ratio2, 0)

if __name__ == "__main__":
    unittest.main()
