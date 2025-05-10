from PIL import Image
from pathlib import Path

class FileConvert:
    def __init__(self, path):
        self.path = path

    def convert(self):
        # Définir les chemins
        chemin_jpg = Path(self.path)
        chemin_png = chemin_jpg.with_suffix('.png')

        # Ouvrir et convertir l'image
        image = Image.open(chemin_jpg)
        image.save(chemin_png, 'PNG')
        return chemin_png