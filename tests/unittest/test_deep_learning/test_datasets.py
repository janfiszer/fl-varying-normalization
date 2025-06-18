import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from src.deep_learning.datasets import SegmentationDataset2DSlices, VolumeEvaluation

# src/deep_learning/test_datasets.py


class TestSegmentationDataset2DSlices(unittest.TestCase):
    @patch("src.deep_learning.datasets.np.load")
    @patch("src.deep_learning.datasets.os.listdir")
    @patch("src.deep_learning.datasets.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_len_and_getitem(self, mock_join, mock_listdir, mock_npload):
        # Setup fake directory structure
        modalities = ["t1", "t2"]
        mask_dir = "mask"
        data_dir = "data"
        # Simulate: data/patient1/slice1, data/patient1/slice2
        mock_listdir.side_effect = [
            ["patient1"],  # patients in data_dir
            ["slice1", "slice2"],  # slices in patient1
            ["slice1", "slice2"],  # for second patient_dir call (redundant, but safe)
        ]
        # Simulate np.load returning arrays
        mock_npload.side_effect = [
            np.ones((2,2)), np.ones((2,2)),  # t1, t2 for slice1
            np.ones((2,2)),                  # mask for slice1
            np.ones((2,2)), np.ones((2,2)),  # t1, t2 for slice2
            np.ones((2,2)),                  # mask for slice2
        ]
        # Patch SLICES_FILE_FORMAT
        with patch("src.deep_learning.datasets.TransformVolumesToNumpySlices") as mock_transform:
            mock_transform.SLICES_FILE_FORMAT = ".npy"
            ds = SegmentationDataset2DSlices(data_dir, modalities, mask_dir)
            self.assertEqual(len(ds), 2)
            img, mask = ds[0]
            self.assertIsInstance(img, torch.Tensor)
            self.assertIsInstance(mask, torch.Tensor)
            self.assertEqual(img.shape[0], 2)  # modalities
            self.assertEqual(mask.shape[0], 1)
            # Test binarize_mask
            ds.binarize_mask = True
            img, mask = ds[0]
            self.assertTrue(((mask == 0) | (mask == 1)).all())

    @patch("src.deep_learning.datasets.os.listdir")
    @patch("src.deep_learning.datasets.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_load_full_paths(self, mock_join, mock_listdir):
        modalities = ["t1", "t2"]
        mask_dir = "mask"
        data_dir = "data"
        mock_listdir.side_effect = [
            ["patient1"],  # patients
            ["slice1"],    # slices
        ]
        with patch("src.deep_learning.datasets.TransformVolumesToNumpySlices") as mock_transform:
            mock_transform.SLICES_FILE_FORMAT = ".npy"
            ds = SegmentationDataset2DSlices.__new__(SegmentationDataset2DSlices)
            ds.modalities_names = modalities
            ds.mask_dir = mask_dir
            modalities_fp, targets_fp = ds.load_full_paths(data_dir)
            self.assertEqual(set(modalities_fp.keys()), set(modalities))
            self.assertEqual(len(targets_fp), 1)
            for m in modalities:
                self.assertEqual(len(modalities_fp[m]), 1)

class TestVolumeEvaluation(unittest.TestCase):
    @patch("src.deep_learning.datasets.np.load")
    @patch("src.deep_learning.datasets.os.listdir")
    @patch("src.deep_learning.datasets.os.path.join", side_effect=lambda *args: "/".join(args))
    def test_len_and_getitem(self, mock_join, mock_listdir, mock_npload):
        gt_path = "gt"
        pred_path = "pred"
        # Simulate: gt/p1/s1, gt/p1/s2; pred/p1/s1, pred/p1/s2
        mock_listdir.side_effect = [
            ["p1"],  # dirs in gt
            ["p1"],  # dirs in pred
            ["s1", "s2"],  # slices in gt/p1
            ["s1", "s2"],  # slices in pred/p1
        ]
        # np.load returns arrays
        mock_npload.side_effect = [
            np.ones((2,2)), np.ones((2,2)),  # gt slices
            np.ones((1,2,2)), np.ones((1,2,2)),  # pred slices
        ]
        ve = VolumeEvaluation(gt_path, pred_path)
        self.assertEqual(len(ve), 1)
        target_vol, pred_vol = ve[0]
        self.assertIsInstance(target_vol, np.ndarray)
        self.assertIsInstance(pred_vol, np.ndarray)
        self.assertEqual(target_vol.shape, (2,2,2))
        self.assertEqual(pred_vol.shape, (2,2,2))
        # Test squeeze_pred False
        ve.squeeze_pred = False
        mock_npload.side_effect = [
            np.ones((2,2)), np.ones((2,2)),
            np.ones((1,2,2)), np.ones((1,2,2)),
        ]
        _, pred_vol2 = ve[0]
        self.assertEqual(pred_vol2.shape, (2,1,2,2))
        # Test binarize_target False
        ve.binarize_target = False
        mock_npload.side_effect = [
            np.ones((2,2)), np.ones((2,2)),
            np.ones((1,2,2)), np.ones((1,2,2)),
        ]
        target_vol2, _ = ve[0]
        self.assertTrue((target_vol2 == 1).all())

if __name__ == "__main__":
    unittest.main()