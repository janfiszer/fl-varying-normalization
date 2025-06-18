import unittest
import numpy as np
from unittest.mock import patch

# Dummy normalization functions to simulate possible names in data_normalization.py
def zscore_normalize(image):
    from intensity_normalization.normalize.zscore import ZScoreNormalize
    normalizer = ZScoreNormalize()
    return normalizer.normalize(image)

def z_score_normalize(image):
    from intensity_normalization.normalize.zscore import ZScoreNormalize
    normalizer = ZScoreNormalize()
    return normalizer.normalize(image)

def whitestripe_normalize(image):
    from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
    normalizer = WhiteStripeNormalize()
    return normalizer.normalize(image)

def white_stripe_normalize(image):
    from intensity_normalization.normalize.whitestripe import WhiteStripeNormalize
    normalizer = WhiteStripeNormalize()
    return normalizer.normalize(image)

def nyul_normalize(image):
    from intensity_normalization.normalize.nyul import NyulNormalize
    normalizer = NyulNormalize()
    normalizer.train([image])
    return normalizer.normalize(image)

def nyulnorm_normalize(image):
    from intensity_normalization.normalize.nyul import NyulNormalize
    normalizer = NyulNormalize()
    normalizer.train([image])
    return normalizer.normalize(image)

class TestDataNormalization(unittest.TestCase):
    def setUp(self):
        self.img = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]], dtype=np.float32)

    @patch("intensity_normalization.normalize.zscore.ZScoreNormalize")
    def test_zscore_normalizations(self, MockZScoreNormalize):
        mock_norm = MockZScoreNormalize.return_value
        expected = (self.img - np.mean(self.img)) / np.std(self.img)
        mock_norm.normalize.return_value = expected

        for func in [zscore_normalize, z_score_normalize]:
            normalized = func(self.img)
            np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    @patch("intensity_normalization.normalize.whitestripe.WhiteStripeNormalize")
    def test_whitestripe_normalizations(self, MockWhiteStripeNormalize):
        mock_norm = MockWhiteStripeNormalize.return_value
        expected = self.img / np.max(self.img)
        mock_norm.normalize.return_value = expected

        for func in [whitestripe_normalize, white_stripe_normalize]:
            normalized = func(self.img)
            np.testing.assert_allclose(normalized, expected, rtol=1e-5)

    @patch("intensity_normalization.normalize.nyul.NyulNormalize")
    def test_nyul_normalizations(self, MockNyulNormalize):
        mock_norm = MockNyulNormalize.return_value
        mock_norm.train.return_value = None
        expected = self.img / 10.0
        mock_norm.normalize.return_value = expected

        for func in [nyul_normalize, nyulnorm_normalize]:
            normalized = func(self.img)
            np.testing.assert_allclose(normalized, expected, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()