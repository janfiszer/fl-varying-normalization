import unittest
from unittest.mock import patch, mock_open, MagicMock
import numpy as np
from pathlib import Path
import os

from src.utils.files_operations import (
    TransformVolumesToNumpySlices,
    get_patients_filepaths,
    extract_background_pixel_value,
    trim_image,
    filter_filepaths,
    sort_by_substring_order,
    get_youngest_dir
)


class TestFilesOperations(unittest.TestCase):
    def setUp(self):
        self.target_root_dir = "test_output"
        self.origin_data_dir = "test_input"
        self.transpose_order = (2, 0, 1)
        self.transformer = TransformVolumesToNumpySlices(
            target_root_dir=self.target_root_dir,
            origin_data_dir=self.origin_data_dir,
            transpose_order=self.transpose_order
        )

    def generate_image(self):
        image = np.random.randint(0, 255, (100, 100)).astype(np.uint8)
        # settigng a specific background value
        image[image < 10] = 0  # Assume background pixels are less than 10

        return image

    def get_example_path(self):
        return "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\minmax\\UCSF-PDGM-0429_FU003d_nifti\\t1.npy"

    def load_example_image(self, path=None):
        if path is None:
            path = "C:\\Users\\JanFiszer\\data\\mri\\nomralized-UCSF-PDGM\\minmax\\UCSF-PDGM-0429_FU003d_nifti\\t1.npy"

        image = np.load(path)

        return image

    def test_extract_background_pixel_value(self):
        # a bigger image to have significant part of the background
        image = self.generate_image()
        result = extract_background_pixel_value(image)
        self.assertEqual(result, 0)

        image_name = "test_image"
        with self.assertLogs(level='WARNING') as log:
            extract_background_pixel_value(image, image_name)
            self.assertIn("The method assumes that all the background pixels have the same value.", log.output[0])

    def test_trim_image(self):
        image = np.random.randint(0, 255, (10, 10)).astype(np.uint8)
        target_size = (8, 8)
        trimmed_image = trim_image(image, target_size)

        self.assertEqual(trimmed_image.shape, (8, 8))
        np.testing.assert_array_equal(trimmed_image, image[1:9, 1:9])

        with self.assertRaises(ValueError):
            trim_image(image, (15, 15))

    def test_get_optimal_slice_range(self):
        brain_slices = np.ones((10, 20, 20))
        brain_slices[5:] = 0  # Make last 5 slices background
        target_ratio = 0.8
        result = self.transformer.get_optimal_slice_range(brain_slices, target_ratio)
        self.assertEqual(len(result), 5)  # Should find 5 slices with brain tissue

    def test_get_indices_mask_slices(self):
        mask_volume = np.zeros((10, 20, 20))
        mask_volume[3:6] = 1  # Set slices 3-5 to have mask
        result = self.transformer.get_indices_mask_slices(mask_volume)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(idx in result for idx in [3, 4, 5]))

    def test_smart_load_slices(self):
        # mock_img = self.generate_image()
        # with patch.object(self.transformer, 'load_slice', return_value=mock_img):
        min_index = 50
        max_index = 140
        slices, indices = self.transformer.smart_load_slices(
            self.get_example_path(),
            min_slices_index=min_index,
            max_slices_index=max_index,
            compute_optimal_slice_range=False
        )
        self.assertEqual(len(slices), max_index-min_index+1)
        self.assertEqual(len(indices), max_index-min_index+1)

    def test_filter_filepaths(self):
        test_filepaths = {
            "t1": ["/path/patient1/t1.nii", "/path/patient2/t1.nii"],
            "t2": ["/path/patient1/t2.nii", "/path/patient2/t2.nii"]
        }
        filtered = filter_filepaths(test_filepaths, ["patient1"])
        self.assertEqual(len(filtered["t1"]), 1)
        self.assertTrue("patient1" in filtered["t1"][0])

    def test_get_youngest_dir(self):
        test_path = os.path.join("path", "to", "patient1", "file.nii")
        result = get_youngest_dir(test_path)
        self.assertEqual(result, "patient1")

    def test_create_empty_dirs(self):
        test_dir = "test_dir"
        test_names = ["dir1", "dir2"]
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            self.transformer.create_empty_dirs(test_dir, test_names)
            self.assertEqual(mock_mkdir.call_count, 2)


if __name__ == '__main__':
    unittest.main()