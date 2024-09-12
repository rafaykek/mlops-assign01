import unittest
import joblib
import numpy as np
import os
from main import saveModel, predictPrice 

class TestHousingModel(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        saveModel()

    def testSavedModel(self):
        self.assertTrue(os.path.exists('linear_regression_model.pkl'), "Model file was not saved")

    def testModelLoad(self):
        try:
            model = joblib.load('linear_regression_model.pkl')
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Loading model failed with error: {e}")

    def testPredicted(self):
        test_input = {
            'area': 3500,
            'bedrooms': 3,
            'bathrooms': 2,
            'stories': 2,
            'parking': 1,
            'mainroad_yes': 1,
            'guestroom_yes': 0,
            'basement_yes': 0,
            'hotwaterheating_yes': 0,
            'airconditioning_yes': 1,
            'prefarea_yes': 1,
            'furnishingstatus_semi_furnished': 0,
            'furnishingstatus_unfurnished': 0,
        }
        predicted_price = predictPrice(**test_input)
        self.assertIsInstance(predicted_price, np.ndarray, "Prediction is not in the correct format")
        self.assertEqual(predicted_price.shape, (1,), "Prediction should be a single value")

    def testInvalid(self):
        with self.assertRaises(ValueError):
            predictPrice(-1000, 3, 2, 2, 1, 1, 0, 0, 0, 1, 1, 0, 0)

if __name__ == '__main__':
    unittest.main()
