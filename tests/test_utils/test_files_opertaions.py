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
    TransformNIIDataToNumpySlices,
    trim_image,
    get_indices_mask_slices,
    load_nii_slices,
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


class TestGetIndicesMaskSlices(unittest.TestCase):
    @patch('numpy.load')
    @patch('nibabel.load')
    def test_npy_file(self, mock_nib_load, mock_np_load):
        # Setup mock data
        mock_volume = np.zeros((10, 20, 20))
        # Add some non-zero values in specific slices
        mock_volume[3] = np.ones((20, 20))
        mock_volume[7] = np.ones((20, 20))

        mock_np_load.return_value = mock_volume

        # Test with .npy file
        indices = get_indices_mask_slices("test.npy", transpose_order=None)

        # Expected indices where the mask has non-zero values
        expected_indices = {3, 7}
        self.assertEqual(indices, expected_indices)
        mock_np_load.assert_called_once()

    @patch('numpy.load')
    @patch('nibabel.load')
    def test_nii_file(self, mock_nib_load, mock_np_load):
        # Setup mock data
        mock_volume = np.zeros((10, 20, 20))
        # Add some non-zero values in specific slices
        mock_volume[3] = np.ones((20, 20))
        mock_volume[7] = np.ones((20, 20))

        # For nib.load, we need to simulate the get_fdata method
        mock_nib = MagicMock()
        mock_nib.get_fdata.return_value = mock_volume
        mock_nib_load.return_value = mock_nib

        # Test with .nii file
        indices = get_indices_mask_slices("test.nii", transpose_order=None)

        # Expected indices where the mask has non-zero values
        expected_indices = {3, 7}
        self.assertEqual(indices, expected_indices)
        mock_nib_load.assert_called_once()

    def test_unsupported_file_extension(self):
        with self.assertRaises(ValueError):
            get_indices_mask_slices("test.txt", transpose_order=None)

    @patch('numpy.load')
    def test_with_transpose(self, mock_np_load):
        # Setup mock data - assume original shape is (20, 20, 10)
        mock_volume = np.zeros((20, 20, 10))
        # Add some non-zero values
        mock_volume[:, :, 2] = 1  # This should become slice index 2 after transpose
        mock_volume[:, :, 5] = 1  # This should become slice index 5 after transpose

        mock_np_load.return_value = mock_volume

        # Test with transpose to (2, 0, 1) to get (10, 20, 20)
        indices = get_indices_mask_slices("test.npy", transpose_order=(2, 0, 1))

        # Expected indices where the mask has non-zero values
        expected_indices = {2, 5}
        self.assertEqual(indices, expected_indices)


class TestLoadNIISlices(unittest.TestCase):
    @patch('numpy.load')
    def test_load_npy_with_specified_range(self, mock_np_load):
        # Create mock volume with slices
        mock_volume = np.zeros((10, 20, 20))
        for i in range(10):
            mock_volume[i] = np.ones((20, 20)) * i

        mock_np_load.return_value = mock_volume

        # Test with specified slice range
        slices, indices = load_nii_slices("test.npy", transpose_order=None,
                                          image_size=None, min_slices_index=2,
                                          max_slices_index=5, target_zero_ratio=0.9,
                                          compute_optimal_slice_range=False)

        # Check results
        self.assertEqual(len(slices), 4)  # slices 2, 3, 4, 5
        self.assertEqual(list(indices), [2, 3, 4, 5])

        # Check slice values
        for i, slice_data in enumerate(slices):
            self.assertTrue(np.array_equal(slice_data, np.ones((20, 20)) * (i + 2)))

    @patch('nibabel.load')
    def test_load_nii_with_auto_range(self, mock_nib_load):
        # Create mock volume
        mock_volume = np.zeros((10, 20, 20))
        # Set background to 0 (most common value)
        # Make some slices have less than target_zero_ratio of background
        for i in [3, 4, 5]:
            # For these slices, set 80% of pixels to non-zero (leaving 20% as background)
            slice_data = np.ones((20, 20))
            # Keep some zeros as background (less than 90% which is target_zero_ratio)
            slice_data[:4, :5] = 0  # 20% of pixels remain as background
            mock_volume[i] = slice_data

        # Setup mock nib.load
        mock_nib = MagicMock()
        mock_nib.get_fdata.return_value = mock_volume
        mock_nib_load.return_value = mock_nib

        # Test with auto slice range (min_slice_index=-1, max_slices_index=-1)
        slices, indices = load_nii_slices("test.nii", transpose_order=None,
                                          image_size=None, min_slices_index=-1,
                                          max_slices_index=-1, target_zero_ratio=0.9)

        # Check results - should only return slices with less than 90% background
        self.assertEqual(len(slices), 3)  # slices 3, 4, 5
        self.assertEqual(set(indices), {3, 4, 5})


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


class TestTransformNIIDataToNumpySlices(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        self.origin_dir = tempfile.mkdtemp()
        self.target_dir = tempfile.mkdtemp()

        # Create necessary subdirectories
        os.makedirs(os.path.join(self.origin_dir, 'patient1'))
        os.makedirs(os.path.join(self.origin_dir, 'patient2'))

        # Setup basic parameters for the transformer
        self.transformer = TransformNIIDataToNumpySlices(
            target_root_dir=self.target_dir,
            origin_data_dir=self.origin_dir,
            transpose_order=(2, 0, 1),
            mask_volume_filename='mask',
            leading_modality='t1',
        )

    def tearDown(self):
        # Clean up temporary directories
        shutil.rmtree(self.origin_dir)
        shutil.rmtree(self.target_dir)

    def test_create_empty_dirs(self):
        # Test directory creation
        test_dir = tempfile.mkdtemp()
        try:
            dir_names = ['dir1', 'dir2', 'dir3']
            TransformNIIDataToNumpySlices.create_empty_dirs(test_dir, dir_names)

            # Check if directories were created
            for dirname in dir_names:
                dir_path = os.path.join(test_dir, dirname)
                self.assertTrue(os.path.exists(dir_path))
                self.assertTrue(os.path.isdir(dir_path))
        finally:
            shutil.rmtree(test_dir)

    @patch('src.utils.files_operations.get_nii_filepaths')
    @patch('src.utils.files_operations.TransformNIIDataToNumpySlices.create_set')
    def test_create_train_val_test_sets(self, mock_create_set, mock_get_filepaths):
        # Mock data
        mock_filepaths = {
            't1': ['path/to/patient1/t1.nii', 'path/to/patient2/t1.nii',
                   'path/to/patient3/t1.nii', 'path/to/patient4/t1.nii'],
            't2': ['path/to/patient1/t2.nii', 'path/to/patient2/t2.nii',
                   'path/to/patient3/t2.nii', 'path/to/patient4/t2.nii']
        }
        mock_get_filepaths.return_value = mock_filepaths

        # Create a fresh target directory
        shutil.rmtree(self.target_dir)

        # Call the method
        self.transformer.create_train_val_test_sets(
            paths_from_local_dirs={'t1': 't1.nii', 't2': 't2.nii'},
            train_size=0.5,
            validation_size=0.25
        )

        # Check that create_set was called for each set type
        self.assertEqual(mock_create_set.call_count, 3)

        # Check that create_set was called with correct arguments
        calls = mock_create_set.call_args_list

        # Check train set (first 2 patients)
        train_filepaths = calls[0][0][0]
        self.assertEqual(len(train_filepaths['t1']), 2)
        self.assertEqual(calls[0][0][1], 'train')

        # Check validation set (1 patient)
        val_filepaths = calls[1][0][0]
        self.assertEqual(len(val_filepaths['t1']), 1)
        self.assertEqual(calls[2][0][1], 'validation')

        # Check test set (1 patient)
        test_filepaths = calls[2][0][0]
        self.assertEqual(len(test_filepaths['t1']), 1)
        self.assertEqual(calls[1][0][1], 'test')

    @patch('numpy.save')
    def test_save_slices(self, mock_np_save):
        # Create mock slices
        slices = [np.ones((10, 10)), np.zeros((10, 10))]

        # Call save_slices
        self.transformer.save_slices(
            slices=slices,
            patient_name='patient1',
            modality='t1',
            main_dir=self.target_dir,
            slice_min_index=50
        )

        # Check directory creation
        for i in range(len(slices)):
            slice_dir = os.path.join(self.target_dir, 'patient1', f'slice{50 + i}')
            self.assertTrue(os.path.exists(slice_dir))

        # Check that numpy.save was called correctly
        self.assertEqual(mock_np_save.call_count, 2)

        # Verify the save paths
        save_paths = [call_args[0][0] for call_args in mock_np_save.call_args_list]
        self.assertIn(os.path.join(self.target_dir, 'patient1', 'slice50', 't1.npy'), save_paths)
        self.assertIn(os.path.join(self.target_dir, 'patient1', 'slice51', 't1.npy'), save_paths)

    @patch('src.utils.files_operations.get_indices_mask_slices')
    @patch('src.utils.files_operations.get_youngest_dir')
    @patch('src.utils.files_operations.load_nii_slices')
    @patch('src.utils.files_operations.TransformNIIDataToNumpySlices.save_slices')
    def test_create_set(self, mock_save_slices, mock_load_slices, mock_get_youngest_dir, mock_get_indices):
        # Setup mocks
        mock_get_youngest_dir.return_value = 'patient1'
        mock_get_indices.return_value = {50, 51, 52}
        mock_load_slices.return_value = ([np.ones((10, 10))], [50])

        # Mock modality paths
        modality_paths = {
            't1': ['path/to/patient1/t1.nii'],
            't2': ['path/to/patient1/t2.nii'],
            'mask': ['path/to/patient1/mask.nii']
        }

        # Call create_set
        self.transformer.create_set(modality_paths, 'train')

        # Check that load_nii_slices was called for each modality
        self.assertEqual(mock_load_slices.call_count, 3)

        # Check that save_slices was called for each modality
        self.assertEqual(mock_save_slices.call_count, 3)


if __name__ == '__main__':
    unittest.main()