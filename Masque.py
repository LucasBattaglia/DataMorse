"""
Masque.py - Module pour l'application et la gestion des masques sur des matrices binaires.

Ce module fait partie du projet DataMorse.

Il permet d'appliquer différents masques à des matrices binaires pour améliorer la robustesse
du chiffrement, ainsi que de retirer ces masques lors du déchiffrement.

Auteur : Lucas BATTAGLIA

Version : v1.0
"""

import numpy as np


class Masque:
    """
        Classe pour appliquer et gérer des masques sur des matrices binaires.

        Cette classe permet d'appliquer différents masques à des matrices binaires pour améliorer
        la robustesse du chiffrement, ainsi que de retirer ces masques lors du déchiffrement.
    """
    def __init__(self, matrix):
        """
            Initialise un objet Masque avec une matrice donnée.

            Args:
                matrix (list): La matrice binaire sur laquelle appliquer le masque.
        """
        self.__matrix = np.array(matrix)
        self.__height, self.__width = self.__matrix.shape

        # Dictionnaire des masques disponibles
        self.__masks = {
            1: lambda i, j: (i % 3 + j % 2) % 2 == 0,
            2: lambda i, j: (i * j) % (i + j + 1) % 2 == 0 if (i + j + 1) != 0 else False,
            3: lambda i, j: (i * j) // (i + j + 1) % 2 == 0 if (i + j + 1) != 0 else False,
            4: lambda i, j: (i ** 2 + 3 * j ** 2) % 5 % 2 == 0,
            5: lambda i, j: (i // 3 + j) % 2 == 0,
            6: lambda i, j: (i * (j + 1)) % 7 % 2 == 0,
        }

    def __apply_mask(self, mask_id):
        """
            Applique un masque à la matrice binaire.

            Args:
                mask_id (int): L'identifiant du masque à appliquer.

            Raises:
                ValueError: Si l'identifiant du masque est invalide.
        """
        if mask_id not in self.__masks:
            raise ValueError("Identifiant de masque invalide : {}".format(mask_id))

        mask_func = self.__masks[mask_id]
        for i in range(self.__height):
            for j in range(self.__width):
                if mask_func(i, j):
                    self.__matrix[i, j] ^= 1  # inversion du bit

    def add_mask(self, mask_id=None):
        """
            Applique un masque à la matrice et retourne la matrice masquée.

            Args:
                mask_id (int, optional): L'identifiant du masque à appliquer. Si None, le meilleur masque est utilisé.

            Returns:
                tuple: La matrice masquée et l'identifiant du masque appliqué.
        """
        if mask_id is None:
            mask_id = self.__best_mask()
        self.__apply_mask(mask_id)
        return self.__matrix.tolist(), mask_id

    def remove_mask(self, mask_id):
        """
            Retire un masque de la matrice.

            Args:
                mask_id (int): L'identifiant du masque à retirer.

            Returns:
                tuple: La matrice restaurée et l'identifiant du masque retiré.
        """
        self.__apply_mask(mask_id)
        return self.__matrix, mask_id

    def __best_mask(self):
        """
            Détermine le meilleur masque à appliquer à la matrice.

            Returns:
                int: L'identifiant du meilleur masque.
        """
        best_id = None
        best_diff = float('inf')
        total_bits = self.__height * self.__width

        for mask_id, mask_func in self.__masks.items():
            temp_matrix = self.__matrix.copy()
            for i in range(self.__height):
                for j in range(self.__width):
                    if mask_func(i, j):
                        temp_matrix[i, j] ^= 1

            ones_count = np.sum(temp_matrix)
            diff = abs(ones_count - total_bits // 2)

            if diff < best_diff:
                best_diff = diff
                best_id = mask_id

        return best_id


if __name__ == "__main__":
    data = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]

    print("Matrice de base :", data)

    masker = Masque(data)
    masked = masker.add_mask()
    print("Matrice masquée :\n", masked[0], "\navec le masque :", masked[1])

    original = masker.remove_mask(masked[1])
    print("Matrice restaurée :", original[0])
