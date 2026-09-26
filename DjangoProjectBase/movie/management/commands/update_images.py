import os
from openai import OpenAI
from huggingface_hub import InferenceClient
from django.core.management.base import BaseCommand
from movie.models import Movie
from dotenv import load_dotenv

class Command(BaseCommand):
    help = "Generate images with Hugging Face and update movie image field"

    def handle(self, *args, **kwargs):
        # ✅ Load environment variables from the .env file
        load_dotenv('../openAI.env')

        # ✅ Initialize the OpenAI client with the API key
        client = OpenAI(
            api_key=os.environ.get('openai_apikey'),
        )
        # ✅ Folder to save images
        images_folder = 'media/movie/images/'
        os.makedirs(images_folder, exist_ok=True)

        # ✅ Fetch all movies
        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        for movie in movies:
            try:
                # ✅ Call the helper function
                image_relative_path = self.generate_and_download_image(client, movie.title, images_folder)

                # ✅ Update database
                movie.image = image_relative_path
                movie.save()
                self.stdout.write(self.style.SUCCESS(f"Saved and updated image for: {movie.title}"))

            except Exception as e:
                self.stderr.write(f"Failed for {movie.title}: {e}")

            # 🔎 Process just the first movie for demonstration
            break

        self.stdout.write(self.style.SUCCESS("Process finished (only first movie updated)."))

    def generate_and_download_image(self, client, movie_title, save_folder):
        hf = InferenceClient(token=os.environ.get('HF_TOKEN'))
        image = hf.text_to_image(
            f"Movie poster of {movie_title}",
            model="black-forest-labs/FLUX.1-schnell",
        )

        image_filename = f"m_{movie_title}.png"
        image_path_full = os.path.join(save_folder, image_filename)
        image.save(image_path_full)

        return os.path.join('movie/images', image_filename)