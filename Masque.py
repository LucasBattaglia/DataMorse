import numpy as np

class Masque:
    def __init__(self, matrix):
        """
        Initialise l'objet avec une matrice de données à masquer.
        """
        self.matrix = np.array(matrix)
        self.height, self.width = self.matrix.shape

        # Dictionnaire des masques disponibles
        self.masks = {
            1: lambda i, j: (i % 3 + j % 2) % 2 == 0,
            2: lambda i, j: (i * j) % (i + j + 1) % 2 == 0 if (i + j + 1) != 0 else False,
            3: lambda i, j: (i * j) // (i + j + 1) % 2 == 0 if (i + j + 1) != 0 else False,
            4: lambda i, j: (i**2 + 3 * j**2) % 5 % 2 == 0,
            5: lambda i, j: (i // 3 + j) % 2 == 0,
            6: lambda i, j: (i * (j + 1)) % 7 % 2 == 0,
        }

    def apply_mask(self, mask_id):
        """
        Applique le masque d'identifiant donné à la matrice (XOR des bits).
        """
        if mask_id not in self.masks:
            raise ValueError("Identifiant de masque invalide.")

        mask_func = self.masks[mask_id]
        for i in range(self.height):
            for j in range(self.width):
                if mask_func(i, j):
                    self.matrix[i, j] ^= 1  # inversion du bit

    def add_mask(self, mask_id=None):
        """
            Ajoute le masque.
        """
        if mask_id is None:
            mask_id = self.best_mask()
        self.apply_mask(mask_id)
        return self.matrix.tolist(), mask_id

    def remove_mask(self, mask_id):
        """
        Supprime le masque (même opération XOR, car XOR est involutif).
        """
        self.apply_mask(mask_id)
        return self.matrix, mask_id

    def best_mask(self):
        """
        Détermine le masque qui équilibre au mieux le nombre de 1 et de 0 dans la matrice.
        Retourne un tuple contenant :
            - l'identifiant du meilleur masque
            - la matrice masquée correspondante (sous forme de liste de listes)
        """
        best_id = None
        best_diff = float('inf')
        total_bits = self.height * self.width

        for mask_id, mask_func in self.masks.items():
            temp_matrix = self.matrix.copy()
            for i in range(self.height):
                for j in range(self.width):
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

