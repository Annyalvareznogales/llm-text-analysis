from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from litserve import LitAPI

class ModelLitAPI(LitAPI):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
    
    def setup(self, device):
        """
        Load the tokenizer and model, and move the model to the specified device.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(device)
        self.model.eval()
    
    def decode_request(self, request):
        """
        Preprocess the request data (tokenize).
        """
        inputs = self.tokenizer(
            request["text"], 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        return inputs

    def predict(self, inputs):
        """
        Perform the inference.
        """
        with torch.no_grad():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
        return outputs.logits

    def encode_response(self, logits):
        """
        This method must be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement the method`encode_response`.")



class RobertaMoralPresenceAPI(ModelLitAPI):
    def encode_response(self, logits):
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        probabilities = probabilities.astype(float)
        class_labels = ["NO-MORAL", "MORAL"]
        predicted_class_idx = probabilities.argmax()
        predicted_class = class_labels[predicted_class_idx]

        
        class_probabilities = {label: prob for label, prob in zip(class_labels, probabilities)}

        return {
            "Predicted": predicted_class,
            "Probabilities": class_probabilities
        }

class RobertaMoralPolarityAPI(ModelLitAPI):
    def encode_response(self, logits):
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        probabilities = probabilities.astype(float)

        class_labels = [
            "NO-MORAL", "VIRTUE", "VICE"]

        predicted_class_idx = probabilities.argmax()
        predicted_class = class_labels[predicted_class_idx]

        
        class_probabilities = {label: prob for label, prob in zip(class_labels, probabilities)}

        return {
            "Predicted_Moral_Polarity": predicted_class,
            "Probabilities": class_probabilities
        }


    
class RobertaMultiMoralPresenceAPI(ModelLitAPI):
    def encode_response(self, logits):
        """
        Process the model output and return the probabilities for all classes.
        """
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        probabilities = probabilities.astype(float)

        class_labels = [
            "NO-MORAL", 
            "CARE/HARM", 
            "FAIRNESS/CHEATING", 
            "LOYALTY/BETRAYAL", 
            "AUTHORITY/SUBVERSION", 
            "PURITY/DEGRADATION"
        ]

        predicted_class_idx = probabilities.argmax()
        predicted_class = class_labels[predicted_class_idx]

       
        class_probabilities = {label: prob for label, prob in zip(class_labels, probabilities)}

        return {
            "Predicted_Moral_Trait": predicted_class,
            "Probabilities": class_probabilities
        }


class RobertaMultiMoralPolarityAPI(ModelLitAPI):
    def encode_response(self, logits):
        """
        Process the model output and return the probabilities for all classes.
        """
        probabilities = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
        probabilities = probabilities.astype(float)

        class_labels = [
            "NO-MORAL", "CARE", "HARM", "FAIRNESS", "CHEATING", 
            "LOYALTY", "BETRAYAL", "AUTHORITY", "SUBVERSION", 
            "PURITY", "DEGRADATION"
        ]

        predicted_class_idx = probabilities.argmax()
        predicted_class = class_labels[predicted_class_idx]

        
        class_probabilities = {label: prob for label, prob in zip(class_labels, probabilities)}

        return {
            "Predicted_Moral": predicted_class,
            "Probabilities": class_probabilities
        }
