import os
import torch
from litserve import LitServer
from huggingface_hub import login
from models.models import * 

HF_MODEL_NAME = os.environ["HF_MODEL_NAME"]


if __name__ == "__main__":

    if HF_MODEL_NAME == "gsi-upm/Roberta-MultiMoral-Polarity":
        api = RobertaMultiMoralPolarityAPI(HF_MODEL_NAME)
    elif HF_MODEL_NAME == "gsi-upm/Roberta-MultiMoral-Presence":
        api = RobertaMultiMoralPresenceAPI(HF_MODEL_NAME)
    elif HF_MODEL_NAME == "gsi-upm/Roberta-Moral-Presence":
        api = RobertaMoralPresenceAPI(HF_MODEL_NAME)
    elif HF_MODEL_NAME == "gsi-upm/Roberta-Moral-Porality":
        api = RobertaMoralPolarityAPI(HF_MODEL_NAME)
    else:
        raise ValueError(f"HF_MODEL_NAME env is not set properly. Its value is {HF_MODEL_NAME}")
    
    server = LitServer(api, accelerator='cuda' if torch.cuda.is_available() else 'cpu', devices=1)
    
    server.run(port=8000)

