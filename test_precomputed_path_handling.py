#!/usr/bin/env python3
"""
Test for precomputed data file vs directory handling.

This test validates the fix for the issue where precomputed data paths 
were incorrectly treated as directories when they were actually files.

The fix ensures that:
- Single .npz files are loaded directly 
- Directories are searched for matching files
- Both PrecomputedDataset and DistributedPrecomputedDataset work with both input types
- Appropriate logging is provided for each case
"""
import os
import sys
import unittest

# Add the main directory to path
sys.path.append('/home/runner/work/PDFParameterInference/PDFParameterInference')

from precomputed_datasets import PrecomputedDataset, DistributedPrecomputedDataset, create_precomputed_dataloader


class TestPrecomputedDataPathHandling(unittest.TestCase):
    """Test file vs directory handling in precomputed datasets."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data paths."""
        cls.data_dir = "/home/runner/work/PDFParameterInference/PDFParameterInference/precomputed_data"
        cls.single_file = f"{cls.data_dir}/mceg_ns3_ne100000_nr1.npz"
        
        # Verify test data exists
        if not os.path.exists(cls.data_dir):
            raise unittest.SkipTest(f"Test data directory not found: {cls.data_dir}")
        if not os.path.exists(cls.single_file):
            raise unittest.SkipTest(f"Test data file not found: {cls.single_file}")
    
    def test_precomputed_dataset_with_directory(self):
        """Test PrecomputedDataset with directory input."""
        dataset = PrecomputedDataset(self.data_dir, "gaussian")
        self.assertGreater(len(dataset), 0)
        
        metadata = dataset.get_metadata()
        self.assertEqual(metadata['problem'], 'gaussian')
        self.assertGreater(metadata['num_samples'], 0)
        self.assertGreater(len(metadata['data_files']), 0)
    
    def test_precomputed_dataset_with_single_file(self):
        """Test PrecomputedDataset with single file input."""
        dataset = PrecomputedDataset(self.single_file, "mceg")
        self.assertGreater(len(dataset), 0)
        
        metadata = dataset.get_metadata()
        self.assertEqual(metadata['problem'], 'mceg')
        self.assertGreater(metadata['num_samples'], 0)
        self.assertEqual(len(metadata['data_files']), 1)
        self.assertEqual(metadata['data_files'][0], self.single_file)
    
    def test_distributed_dataset_with_directory(self):
        """Test DistributedPrecomputedDataset with directory input."""
        dataset = DistributedPrecomputedDataset(self.data_dir, "gaussian", rank=0, world_size=2)
        self.assertGreater(len(dataset), 0)
        
        metadata = dataset.get_metadata()
        self.assertEqual(metadata['problem'], 'gaussian')
        self.assertGreater(metadata['num_samples'], 0)
    
    def test_distributed_dataset_with_single_file(self):
        """Test DistributedPrecomputedDataset with single file input."""
        dataset = DistributedPrecomputedDataset(self.single_file, "mceg", rank=0, world_size=2)
        self.assertGreater(len(dataset), 0)
        
        metadata = dataset.get_metadata()
        self.assertEqual(metadata['problem'], 'mceg')
        self.assertGreater(metadata['num_samples'], 0)
        self.assertEqual(len(metadata['data_files']), 1)
    
    def test_dataloader_with_directory(self):
        """Test create_precomputed_dataloader with directory input."""
        dataloader = create_precomputed_dataloader(self.data_dir, "gaussian", batch_size=2)
        self.assertGreater(len(dataloader.dataset), 0)
        
        # Test getting a batch
        for theta, events in dataloader:
            self.assertEqual(len(theta.shape), 2)  # [batch_size, theta_dim]
            self.assertEqual(len(events.shape), 4)  # [batch_size, n_repeat, num_events, feature_dim]
            break
    
    def test_dataloader_with_single_file(self):
        """Test create_precomputed_dataloader with single file input."""
        dataloader = create_precomputed_dataloader(self.single_file, "mceg", batch_size=2)
        self.assertGreater(len(dataloader.dataset), 0)
        
        # Test getting a batch
        for theta, events in dataloader:
            self.assertEqual(len(theta.shape), 2)  # [batch_size, theta_dim]
            self.assertEqual(len(events.shape), 4)  # [batch_size, n_repeat, num_events, feature_dim]
            break
    
    def test_nonexistent_file_error(self):
        """Test that nonexistent files raise appropriate errors."""
        with self.assertRaises(FileNotFoundError):
            PrecomputedDataset("/nonexistent/file.npz", "test")
    
    def test_nonexistent_directory_error(self):
        """Test that nonexistent directories raise appropriate errors."""
        with self.assertRaises(FileNotFoundError):
            PrecomputedDataset("/nonexistent/directory", "test")
    
    def test_wrong_file_extension_error(self):
        """Test that non-.npz files raise appropriate errors."""
        with self.assertRaises((ValueError, FileNotFoundError)):
            PrecomputedDataset("/tmp/test.txt", "test")
    
    def test_no_matching_files_error(self):
        """Test that directories with no matching files raise appropriate errors."""
        with self.assertRaises(FileNotFoundError):
            PrecomputedDataset(self.data_dir, "nonexistent_problem")


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)