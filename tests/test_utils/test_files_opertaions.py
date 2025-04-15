import os
import unittest
import tempfile
import shutil
import numpy as np
import nibabel as nib
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path

# Import the module to test
from src.utils.files_operations import (
    TransformVolumesToNumpySlices,
    trim_image,
    get_nii_filepaths,
    get_youngest_dir
)


class TestTrimImage(unittest.TestCase):
    def test_trim_image_normal_case(self):
        # Create a test image of size 100x100
        image = np.ones((100, 100))
        # Target size of 60x60
        target_size = (60, 60)
        # Trim the image
        trimmed_image = trim_image(image, target_size)
        # Check if the size is as expected
        self.assertEqual(trimmed_image.shape, target_size)

    def test_trim_image_error_when_target_too_large(self):
        # Create a test image of size 50x50
        image = np.ones((50, 50))
        # Target size larger than image
        target_size = (100, 100)
        # Should raise ValueError
        with self.assertRaises(ValueError):
            trim_image(image, target_size)


# class TestGetIndicesMaskSlices(unittest.TestCase):
#     @patch('numpy.load')
#     @patch('nibabel.load')
#     def test_npy_file(self, mock_nib_load, mock_np_load):
#         # Setup mock data
#         mock_volume = np.zeros((10, 20, 20))
#         # Add some non-zero values in specific slices
#         mock_volume[3] = np.ones((20, 20))
#         mock_volume[7] = np.ones((20, 20))
#
#         mock_np_load.return_value = mock_volume
#
#         # Test with .npy file
#         indices = get_indices_mask_slices("test.npy", transpose_order=None)
#
#         # Expected indices where the mask has non-zero values
#         expected_indices = {3, 7}
#         self.assertEqual(indices, expected_indices)
#         mock_np_load.assert_called_once()
#
#     @patch('numpy.load')
#     @patch('nibabel.load')
#     def test_nii_file(self, mock_nib_load, mock_np_load):
#         # Setup mock data
#         mock_volume = np.zeros((10, 20, 20))
#         # Add some non-zero values in specific slices
#         mock_volume[3] = np.ones((20, 20))
#         mock_volume[7] = np.ones((20, 20))
#
#         # For nib.load, we need to simulate the get_fdata method
#         mock_nib = MagicMock()
#         mock_nib.get_fdata.return_value = mock_volume
#         mock_nib_load.return_value = mock_nib
#
#         # Test with .nii file
#         indices = get_indices_mask_slices("test.nii", transpose_order=None)
#
#         # Expected indices where the mask has non-zero values
#         expected_indices = {3, 7}
#         self.assertEqual(indices, expected_indices)
#         mock_nib_load.assert_called_once()
#
#     def test_unsupported_file_extension(self):
#         with self.assertRaises(ValueError):
#             get_indices_mask_slices("test.txt", transpose_order=None)
#
#     @patch('numpy.load')
#     def test_with_transpose(self, mock_np_load):
#         # Setup mock data - assume original shape is (20, 20, 10)
#         mock_volume = np.zeros((20, 20, 10))
#         # Add some non-zero values
#         mock_volume[:, :, 2] = 1  # This should become slice index 2 after transpose
#         mock_volume[:, :, 5] = 1  # This should become slice index 5 after transpose
#
#         mock_np_load.return_value = mock_volume
#
#         # Test with transpose to (2, 0, 1) to get (10, 20, 20)
#         indices = get_indices_mask_slices("test.npy", transpose_order=(2, 0, 1))
#
#         # Expected indices where the mask has non-zero values
#         expected_indices = {2, 5}
#         self.assertEqual(indices, expected_indices)


# class TestLoadNIISlices(unittest.TestCase):
#     @patch('numpy.load')
#     def test_load_npy_with_specified_range(self, mock_np_load):
#         # Create mock volume with slices
#         mock_volume = np.zeros((10, 20, 20))
#         for i in range(10):
#             mock_volume[i] = np.ones((20, 20)) * i
#
#         mock_np_load.return_value = mock_volume
#
#         # Test with specified slice range
#         slices, indices = smart_load_slices("test.npy", transpose_order=None,
#                                             image_size=None, min_slices_index=2,
#                                             max_slices_index=5, target_zero_ratio=0.9,
#                                             compute_optimal_slice_range=False)
#
#         # Check results
#         self.assertEqual(len(slices), 4)  # slices 2, 3, 4, 5
#         self.assertEqual(list(indices), [2, 3, 4, 5])
#
#         # Check slice values
#         for i, slice_data in enumerate(slices):
#             self.assertTrue(np.array_equal(slice_data, np.ones((20, 20)) * (i + 2)))
#
#     @patch('nibabel.load')
#     def test_load_nii_with_auto_range(self, mock_nib_load):
#         # Create mock volume
#         mock_volume = np.zeros((10, 20, 20))
#         # Set background to 0 (most common value)
#         # Make some slices have less than target_zero_ratio of background
#         for i in [3, 4, 5]:
#             # For these slices, set 80% of pixels to non-zero (leaving 20% as background)
#             slice_data = np.ones((20, 20))
#             # Keep some zeros as background (less than 90% which is target_zero_ratio)
#             slice_data[:4, :5] = 0  # 20% of pixels remain as background
#             mock_volume[i] = slice_data
#
#         # Setup mock nib.load
#         mock_nib = MagicMock()
#         mock_nib.get_fdata.return_value = mock_volume
#         mock_nib_load.return_value = mock_nib
#
#         # Test with auto slice range (min_slice_index=-1, max_slices_index=-1)
#         slices, indices = smart_load_slices("test.nii", transpose_order=None,
#                                             image_size=None, min_slices_index=-1,
#                                             max_slices_index=-1, target_zero_ratio=0.9)
#
#         # Check results - should only return slices with less than 90% background
#         self.assertEqual(len(slices), 3)  # slices 3, 4, 5
#         self.assertEqual(set(indices), {3, 4, 5})


class TestGetNIIFilepaths(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure for testing
        self.test_dir = tempfile.mkdtemp()
        self.filepaths_from_data_dir = {
            't1': 't1.nii',
            't2': 't2.nii'
        }

        # Create test patient directories
        self.patient_dirs = ['patient1', 'patient2', 'patient3']
        for patient in self.patient_dirs:
            patient_dir = os.path.join(self.test_dir, patient)
            os.makedirs(patient_dir)

            # Create test modality files
            with open(os.path.join(patient_dir, 't1.nii'), 'w') as f:
                f.write("dummy")
            with open(os.path.join(patient_dir, 't2.nii'), 'w') as f:
                f.write("dummy")

    def tearDown(self):
        # Remove temporary directory
        shutil.rmtree(self.test_dir)

    def test_get_all_patients(self):
        # Get filepaths for all patients
        result = get_nii_filepaths(self.test_dir, self.filepaths_from_data_dir)

        # Check results
        self.assertEqual(len(result['t1']), 3)
        self.assertEqual(len(result['t2']), 3)

        # Check that filepaths are correct
        for modality in ['t1', 't2']:
            for filepath in result[modality]:
                self.assertTrue(os.path.exists(filepath))
                self.assertTrue(filepath.endswith(f'{modality}.nii'))

    def test_limit_patients(self):
        # Get filepaths limited to 2 patients
        result = get_nii_filepaths(self.test_dir, self.filepaths_from_data_dir, n_patients=2)

        # Check results
        self.assertEqual(len(result['t1']), 2)
        self.assertEqual(len(result['t2']), 2)

    def setup_dirs_with_missing_modality(self):
        # Create a patient directory with missing modality
        patient_dir = os.path.join(self.test_dir, 'patient4')
        os.makedirs(patient_dir)
        with open(os.path.join(patient_dir, 't1.nii'), 'w') as f:
            f.write("dummy")
        # Note: intentionally not creating t2.nii for this patient

    def test_missing_modality(self):
        self.setup_dirs_with_missing_modality()

        # Get filepaths - should skip patient4 since it doesn't have t2.nii
        result = get_nii_filepaths(self.test_dir, self.filepaths_from_data_dir)

        # Should still have 3 results (patient4 skipped)
        self.assertEqual(len(result['t1']), 3)
        self.assertEqual(len(result['t2']), 3)

    def test_equal_number_of_files_for_all_modalities(self):
        result = get_nii_filepaths(self.test_dir, self.filepaths_from_data_dir)
        # check if all have the same modality
        filepaths_lists_lens = [len(filepaths) for filepaths in result.values()]
        self.assertTrue(all(length == filepaths_lists_lens[0] for length in filepaths_lists_lens), "Not all the filepath lists have the same length")

        # and the second try with a missing modality directory
        self.setup_dirs_with_missing_modality()

        # Get filepaths - should skip patient4 since it doesn't have t2.nii
        result = get_nii_filepaths(self.test_dir, self.filepaths_from_data_dir)

        # check if all have the same modality
        filepaths_lists_lens = [len(filepaths) for filepaths in result.values()]
        self.assertTrue(all(length == filepaths_lists_lens[0] for length in filepaths_lists_lens), "Not all the filepath lists have the same length")


class TestGetYoungestDir(unittest.TestCase):
    def test_get_youngest_dir(self):
        # Test different filepath formats
        self.assertEqual(get_youngest_dir('\\path\\to\\patient1\\file.nii'), 'patient1')
        self.assertEqual(get_youngest_dir('C:\\path\\to\\patient2\\file.nii'), 'patient2')
        self.assertEqual(get_youngest_dir('patient3\\file.nii'), 'patient3')
        self.assertEqual(get_youngest_dir('\\complicated\\path\\with\\patient4\\nested\\file.nii'), 'nested')


class TestTransformVolumesToNumpySlices(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.transformer = TransformVolumesToNumpySlices(
            target_root_dir="test_target",
            origin_data_dir="test_origin",
            transpose_order=(2, 0, 1),
            target_zero_ratio=0.9
        )

        # Sample 3D volume with background (0) and foreground (1)
        # Shape: (5, 10, 10) - 5 slices, each 10x10
        self.sample_volume = np.zeros((5, 10, 10))

        # Slice 0: Empty (all zeros)
        # Slice 1: 15% foreground (85% background)
        self.sample_volume[1, :3, :5] = 1
        # Slice 2: 30% foreground (70% background)
        self.sample_volume[2, :3, :] = 1
        # Slice 3: 5% foreground (95% background)
        self.sample_volume[3, :1, :5] = 1
        # Slice 4: Empty (all zeros)

        # Sample mask volume with same shape
        self.sample_mask = np.zeros((5, 10, 10))
        # Only slices 1 and 2 have mask values
        self.sample_mask[1, 1:3, 1:3] = 1
        self.sample_mask[2, 2:4, 2:4] = 1

    def test_get_optimal_slice_range(self):
        """Test get_optimal_slice_range with different target ratios"""
        # Test with target ratio 0.9 (should return slices with less than 90% background)
        expected_slices = {1, 2}  # Slices 1 and 2 have less than 90% background
        result = self.transformer.get_optimal_slice_range(self.sample_volume, 0.9)
        self.assertEqual(result, expected_slices)

        # Test with target ratio 0.8 (should return only slice 2)
        expected_slices = {2}  # Only slice 2 has less than 80% background
        result = self.transformer.get_optimal_slice_range(self.sample_volume, 0.8)
        self.assertEqual(result, expected_slices)

        # Test with target ratio 1.0 (should return all slices with any foreground)
        expected_slices = {1, 2, 3}  # Slices 1, 2, and 3 have some foreground
        result = self.transformer.get_optimal_slice_range(self.sample_volume, 1.0)
        self.assertEqual(result, expected_slices)

        # Test with empty volume
        empty_volume = np.zeros((3, 10, 10))
        result = self.transformer.get_optimal_slice_range(empty_volume, 0.9)
        self.assertEqual(result, set())  # Should return empty set

    def test_get_indices_mask_slices(self):
        """Test get_indices_mask_slices with different mask volumes"""
        # Test with sample mask
        expected_slices = {1, 2}  # Slices 1 and 2 have mask values
        result = self.transformer.get_indices_mask_slices(self.sample_mask)
        self.assertEqual(result, expected_slices)

        # Test with empty mask
        empty_mask = np.zeros((5, 10, 10))
        result = self.transformer.get_indices_mask_slices(empty_mask)
        self.assertEqual(result, set())  # Should return empty set

        # Test with all slices having mask
        full_mask = np.ones((5, 10, 10))
        expected_slices = {0, 1, 2, 3, 4}  # All slices have mask
        result = self.transformer.get_indices_mask_slices(full_mask)
        self.assertEqual(result, expected_slices)

        # Test with single slice having mask
        single_mask = np.zeros((5, 10, 10))
        single_mask[3, 5, 5] = 1
        expected_slices = {3}  # Only slice 3 has mask
        result = self.transformer.get_indices_mask_slices(single_mask)
        self.assertEqual(result, expected_slices)

    @patch('src.utils.files_operations.TransformVolumesToNumpySlices.load_slice')
    @patch('src.utils.files_operations.TransformVolumesToNumpySlices.get_optimal_slice_range')
    def test_smart_load_slices(self, mock_get_optimal_slice_range, mock_load_slice):
        """Test smart_load_slices with different parameters"""
        # Setup mocks
        mock_load_slice.return_value = self.sample_volume
        mock_get_optimal_slice_range.return_value = {1, 2}

        # Test with default parameters (compute_optimal_slice_range=True)
        slices, indices = self.transformer.smart_load_slices("test_file.npy")

        # Check if load_slice was called
        mock_load_slice.assert_called_once_with("test_file.npy")
        # Check if get_optimal_slice_range was called
        mock_get_optimal_slice_range.assert_called_once()

        # Check results
        self.assertEqual(len(slices), 2)  # Should return 2 slices
        self.assertEqual(indices, {1, 2})  # Should return indices {1, 2}

        # Reset mocks
        mock_load_slice.reset_mock()
        mock_get_optimal_slice_range.reset_mock()

        # Test with min_slices_index and max_slices_index provided
        slices, indices = self.transformer.smart_load_slices("test_file.npy", min_slices_index=1, max_slices_index=3)

        # Check results
        self.assertEqual(len(slices), 3)  # Should return 3 slices (1, 2, 3)
        self.assertEqual(indices, {1, 2, 3})  # Should still return optimal indices {1, 2}

        # Reset mocks
        mock_load_slice.reset_mock()
        mock_get_optimal_slice_range.reset_mock()

        # Test with compute_optimal_slice_range=False
        slices, indices = self.transformer.smart_load_slices(
            "test_file.npy",
            min_slices_index=1,
            max_slices_index=3,
            compute_optimal_slice_range=False
        )

        # Check if get_optimal_slice_range was not called
        mock_get_optimal_slice_range.assert_not_called()

        # Check results
        self.assertEqual(len(slices), 3)  # Should return 3 slices (1, 2, 3)
        self.assertEqual(indices, {1, 2, 3})  # Should return all indices in range

        # Test with invalid parameters (compute_optimal_slice_range=False and no indices)
        with self.assertRaises(ValueError):
            self.transformer.smart_load_slices("test_file.npy", compute_optimal_slice_range=False)

    @patch('logging.log')
    @patch('src.utils.files_operations.TransformVolumesToNumpySlices.load_slice')
    @patch('src.utils.files_operations.TransformVolumesToNumpySlices.get_optimal_slice_range')
    def test_smart_load_slices_with_empty_result(self, mock_get_optimal_slice_range, mock_load_slice, mock_logging):
        """Test smart_load_slices when get_optimal_slice_range returns empty set"""
        # Setup mocks
        mock_load_slice.return_value = self.sample_volume
        mock_get_optimal_slice_range.return_value = set()  # Return empty set

        # Test with default parameters should raise ValueError
        with self.assertRaises(ValueError):
            self.transformer.smart_load_slices("test_file.npy")


if __name__ == '__main__':
    unittest.main()
