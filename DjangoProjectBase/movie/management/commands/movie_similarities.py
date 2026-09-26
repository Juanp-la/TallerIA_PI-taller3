import os
import numpy as np
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

class Command(BaseCommand):
    help = "Compare two movies and a prompt using embeddings (Hugging Face)"

    def handle(self, *args, **kwargs):
        # ✅ Carga el token desde el .env
        load_dotenv('../openAI.env')
        client = InferenceClient(token=os.environ.get('HF_TOKEN'))

        # 🎬 Películas a comparar (cámbialas para la actividad)
        movie1 = Movie.objects.get(title="Castillo medieval")
        movie2 = Movie.objects.get(title="La captura")

        # 📝 Prompt de búsqueda (cámbialo para la actividad)
        prompt = "película de fantasía con caballeros y hechiceros"

        # ✅ Obtener el embedding de cualquier texto
        def get_embedding(text):
            emb = client.feature_extraction(
                text,
                model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            )
            emb = np.array(emb, dtype=np.float32)
            # Si devuelve un vector por palabra, se promedian
            if emb.ndim > 1:
                emb = emb.reshape(-1, emb.shape[-1]).mean(axis=0)
            return emb

        # ✅ Similitud de coseno
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        # 🎬 Película vs película
        emb1 = get_embedding(movie1.description)
        emb2 = get_embedding(movie2.description)

        similarity = cosine_similarity(emb1, emb2)
        self.stdout.write(f"🎬 {movie1.title} vs {movie2.title}: {similarity:.4f}")

        # 📝 Prompt vs cada película
        prompt_emb = get_embedding(prompt)

        sim_prompt_movie1 = cosine_similarity(prompt_emb, emb1)
        sim_prompt_movie2 = cosine_similarity(prompt_emb, emb2)

        self.stdout.write(f"📝 Prompt: '{prompt}'")
        self.stdout.write(f"📝 Similitud prompt vs '{movie1.title}': {sim_prompt_movie1:.4f}")
        self.stdout.write(f"📝 Similitud prompt vs '{movie2.title}': {sim_prompt_movie2:.4f}")