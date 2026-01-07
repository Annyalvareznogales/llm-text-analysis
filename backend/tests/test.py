import unittest
from fastapi.testclient import TestClient
from main import app  


class TestFastAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_get_api(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    def test_predict_moral_model(self):
        """Test 'moral_model'"""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "moral_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted", response.json())
        self.assertIn("Probabilities", response.json())
        
        expected_classes = [
            "NO-MORAL", 
            "MORAL"
        ]
        for class_label in expected_classes:
            self.assertIn(class_label, response.json()["Probabilities"])



    def test_predict_moralpolarity_model(self):
        """Test 'moralpolarity_model'"""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "moralpolarity_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 200)  
        self.assertIn("Predicted_Moral_Polarity", response.json())
        self.assertIn("Probabilities", response.json())
        
        expected_classes = [
            "NO-MORAL", 
            "VIRTUE",
            "VICE"
        ]
        for class_label in expected_classes:
            self.assertIn(class_label, response.json()["Probabilities"])




    def test_predict_multimoral_model(self):
        """Test 'multimoral_model'."""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "multimoral_model"
        }
        response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)

        self.assertIn("Predicted_Moral_Trait", response.json())
        self.assertIn("Probabilities", response.json())
        
        expected_classes = [
            "NO-MORAL", 
            "CARE/HARM", 
            "FAIRNESS/CHEATING", 
            "LOYALTY/BETRAYAL", 
            "AUTHORITY/SUBVERSION", 
            "PURITY/DEGRADATION"
        ]
        for class_label in expected_classes:
            self.assertIn(class_label, response.json()["Probabilities"])


    def test_predict_multimoralpolarity_model(self):
        """Test 'multimoralpolarity_model'."""
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "multimoralpolarity_model"
        }
        response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)
        
        self.assertIn("Predicted_Moral", response.json())
        self.assertIn("Probabilities", response.json())
        
        expected_classes = [
            "NO-MORAL", "CARE", "HARM", "FAIRNESS", "CHEATING", 
            "LOYALTY", "BETRAYAL", "AUTHORITY", "SUBVERSION", 
            "PURITY", "DEGRADATION"
        ]
        for class_label in expected_classes:
            self.assertIn(class_label, response.json()["Probabilities"])


    def test_invalid_model(self):
        data = {
            "text": "The government should protect its citizens and maintain law and order.",
            "model_name": "non_existing_model"
        }
        response = self.client.post("/predict", json=data)
        
        self.assertEqual(response.status_code, 400) 
        self.assertIn("Model name not valid", response.json()['detail'])  



if __name__ == "__main__":
    unittest.main()
