"""
FileConvert.py - Module pour la conversion d'images.

Ce module fait partie du projet DataMorse.

Il permet de convertir des images de format JPG en format PNG pour une utilisation ultérieure
dans le chiffrement et le déchiffrement des messages.

Auteur : Lucas BATTAGLIA

Version : v1.1
"""
from PIL import Image
from pathlib import Path
import os


class FileConvert:
    """
        Classe pour convertir des images entre différents formats.

        Cette classe permet de convertir des images de format JPG en format PNG.
    """

    def __init__(self, path):
        """
            Initialise un objet FileConvert avec le chemin de l'image.

            Args:
                path (str): Le chemin de l'image à convertir.

            Raises:
                FileNotFoundError: Si le chemin fourni n'existe pas.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError("Le lien vers l'image fournis n'existe pas !")
        self.__path = path

    def convert(self):
        """
            Convertit l'image de JPG à PNG.

            Returns:
                str: Le chemin de l'image convertie au format PNG.
        """
        # Définir les chemins
        chemin_jpg = Path(self.__path)
        chemin_png = chemin_jpg.with_suffix('.png')

        # Ouvrir et convertir l'image
        image = Image.open(chemin_jpg)
        image.save(chemin_png, 'PNG')
        return chemin_png
