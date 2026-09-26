import os
import numpy as np
from django.conf import settings
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Carga el token desde openAI.env (un nivel arriba de DjangoProjectBase)
load_dotenv(os.path.join(settings.BASE_DIR, '..', 'openAI.env'))
client = InferenceClient(token=os.environ.get('HF_TOKEN'))

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding(text):
    emb = client.feature_extraction(text, model=MODEL)
    emb = np.array(emb, dtype=np.float32)
    if emb.ndim > 1:
        emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
    return emb

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))