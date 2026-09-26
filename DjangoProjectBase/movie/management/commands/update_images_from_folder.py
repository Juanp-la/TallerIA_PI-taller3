import os
from django.conf import settings
from django.core.management.base import BaseCommand
from movie.models import Movie

class Command(BaseCommand):
    help = "Assign images from media/movie/images/ to each movie in the database"

    def handle(self, *args, **kwargs):
        # 📂 Carpeta donde están las imágenes entregadas
        images_folder = os.path.join(settings.MEDIA_ROOT, 'movie', 'images')

        if not os.path.exists(images_folder):
            self.stderr.write(f"Folder '{images_folder}' not found.")
            return

        movies = Movie.objects.all()
        self.stdout.write(f"Found {movies.count()} movies")

        updated_count = 0

        for movie in movies:
            # 🖼️ Nombre esperado: m_NOMBRE_PELICULA.png
            image_filename = f"m_{movie.title}.png"
            image_path_full = os.path.join(images_folder, image_filename)

            if os.path.exists(image_path_full):
                movie.image = os.path.join('movie/images', image_filename)
                movie.save()
                updated_count += 1
                self.stdout.write(self.style.SUCCESS(f"Updated image for: {movie.title}"))
            else:
                self.stderr.write(f"Image not found for: {movie.title}")

        self.stdout.write(self.style.SUCCESS(f"Finished updating {updated_count} movies with images."))