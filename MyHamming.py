"""
MyHamming.py - Module de correction d'erreurs par code de Hamming.

Ce module fait partie du projet DataMorse.
Il permet d’encoder et de décoder des matrices binaires avec contrôle de parité,
et de corriger certaines erreurs simples ou doubles via redondance.

Auteur : Lucas BATTAGLIA

Version : v1.2
"""
from collections import Counter


class MyHamming:
    """
        Classe pour encoder et décoder des matrices en utilisant un code de Hamming étendu.

        Cette classe permet l'encodage avec bits de parité, la détection d’erreurs,
        et la tentative de correction simple ou double en comparant deux matrices.
        """
    def __init__(self, matrix1, matrix2=None):
        """
            Initialise un objet MyHamming avec une ou deux matrices.

            Args:
                matrix1 (list[list[int]]): Matrice principale.
                matrix2 (list[list[int]], optional): Matrice secondaire pour comparaison (double détection).
        """
        self.__matrix1 = matrix1
        self.__matrix2 = matrix2

    @staticmethod
    def __add_parity(matrix):
        """
            Ajoute des bits de parité aux lignes et colonnes d'une matrice.

            Args:
                matrix (list[list[int]]): Matrice binaire à encoder.

            Returns:
                list[list[int]]: Matrice avec bits de parité ajoutés.
        """
        for row in matrix:
            parity_bit = sum(row) % 2
            row.insert(0, parity_bit)

        num_columns = len(matrix[0])
        parity_row = []
        for col in range(num_columns):
            col_sum = sum(row[col] for row in matrix)
            parity_bit = col_sum % 2
            parity_row.append(parity_bit)

        matrix.append(parity_row)
        return matrix

    @staticmethod
    def __find_error(matrix):
        """
            Détecte les lignes et colonnes contenant des erreurs de parité.

            Args:
                matrix (list[list[int]]): Matrice encodée avec parité.

            Returns:
                tuple: (liste des lignes erronées, liste des colonnes erronées)
        """
        error_ligne = []
        error_colonne = []

        for index, row in enumerate(matrix):
            if sum(row) % 2 != 0:
                error_ligne.append(index)

        for i in range(len(matrix[0]) - 1):
            col = [matrix[j][i] for j, _ in enumerate(matrix)]
            if sum(col) % 2 != 0:
                error_colonne.append(i)

        return error_ligne, error_colonne

    @staticmethod
    def __merge_matrices(m1, m2):
        """
            Compare deux matrices et produit une matrice fusionnée avec `None` aux positions divergentes.

            Args:
                m1 (list[list[int]]): Première matrice.
                m2 (list[list[int]]): Deuxième matrice.

            Returns:
                list[list[Optional[int]]]: Matrice fusionnée.
        """
        rows = len(m1)
        cols = len(m1[0])
        merged = []
        for r in range(rows):
            row = []
            for c in range(cols):
                if m1[r][c] == m2[r][c]:
                    row.append(m1[r][c])
                else:
                    row.append(None)
            merged.append(row)
        return merged

    def __changement_matrices(self, m1, m2):
        """
            Identifie les positions où deux matrices diffèrent.

            Args:
                m1 (list[list[int]]): Première matrice.
                m2 (list[list[int]]): Deuxième matrice.

            Returns:
                list[tuple[int, int]]: Liste des coordonnées des différences.
        """
        newmatrice = self.__merge_matrices(m1, m2)
        error_positions = []
        for r in range(len(newmatrice)):
            for c in range(len(newmatrice[r])):
                if newmatrice[r][c] is None:
                    error_positions.append((r, c))
        return error_positions

    @staticmethod
    def __common_tuples(list1, list2):
        """
            Retourne les éléments communs à deux listes, en tenant compte de leur fréquence.

            Args:
                list1 (list[tuple]): Première liste.
                list2 (list[tuple]): Deuxième liste.

            Returns:
                list[tuple]: Éléments communs aux deux listes.
        """
        counter1 = Counter(list1)
        counter2 = Counter(list2)
        return list((counter1 & counter2).elements())

    def __decode_double(self):
        """
            Tente de corriger les erreurs en comparant deux matrices.

            Returns:
                list[list[int]]: Matrice corrigée.

            Raises:
                ValueError: Si les erreurs ne peuvent pas être corrigées.
        """
        try:
            return self.__decode_simple(self.__matrix1)
        except ValueError:
            pass

        try:
            return self.__decode_simple(self.__matrix2)
        except ValueError:
            pass

        chang = self.__changement_matrices(self.__matrix1, self.__matrix2)

        erreur = self.__find_error(self.__matrix1)
        possible_position = []
        for e in erreur[0]:
            for a in erreur[1]:
                possible_position.append((e, a))

        list_erreur = self.__common_tuples(possible_position, chang)

        for e in list_erreur:
            self.__matrix1[e[0]][e[1]] ^= 1
            try:
                return self.__decode_simple(self.__matrix1)
            except ValueError:
                continue

        erreur = self.__find_error(self.__matrix2)
        possible_position = []
        for e in erreur[0]:
            for a in erreur[1]:
                possible_position.append((e, a))

        list_erreur = self.__common_tuples(possible_position, chang)

        for e in list_erreur:
            self.__matrix2[e[0]][e[1]] ^= 1
            try:
                return self.__decode_simple(self.__matrix2)
            except ValueError:
                continue

        raise ValueError("Trop d'erreur pour pouvoir etre corriger")

    def __decode_simple(self, matrix):
        """
            Détecte et corrige les erreurs simples dans une seule matrice.

            Args:
                matrix (list[list[int]]): Matrice encodée avec parité.

            Returns:
                list[list[int]]: Matrice décodée (sans parité).

            Raises:
                ValueError: Si les erreurs sont trop nombreuses ou incohérentes.
        """
        erreur = self.__find_error(matrix)
        if len(erreur[0]) == 0 and len(erreur[1]) == 0:
            return [row[1:] for row in matrix[:-1]]
        elif len(erreur[0]) == 1 and len(erreur[1]) == 1:
            r, c = erreur[0][0], erreur[1][0]
            corrected = [row[:] for row in matrix]
            corrected[r][c] ^= 1
            return [row[1:] for row in corrected[:-1]]
        elif len(erreur[0]) == 0 or len(erreur[1]) == 0:
            raise ValueError("Erreur dans la recherche des erreurs")
        else:
            raise ValueError("Trop d'erreurs")

    def encode(self):
        """
            Encode la matrice principale avec des bits de parité ligne/colonne.

            Returns:
                list[list[int]]: Matrice encodée avec contrôle de parité.
        """
        return self.__add_parity(self.__matrix1)

    def decode(self):
        """
            Décode la matrice principale (ou les deux) et corrige les erreurs détectables.

            Returns:
                list[list[int]]: Matrice binaire corrigée.

            Raises:
                ValueError: Si trop d'erreurs sont présentes pour être corrigées.
        """
        if self.__matrix2 is None:
            return self.__decode_simple(self.__matrix1)
        else:
            return self.__decode_double()


if __name__ == "__main__":
    # Matrice de base (non encodée)
    base_matrix = [
        [1, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 1, 0, 1],
    ]

    # Création des objets et encodage
    hm1 = MyHamming([row[:] for row in base_matrix])
    encoded1 = hm1.encode()

    hm2 = MyHamming([row[:] for row in base_matrix])
    encoded2 = hm2.encode()

    print("Matrice encodée originale 1:")
    for r in encoded1:
        print(r)

    print("\nMatrice encodée originale 2:")
    for r in encoded2:
        print(r)

    # Ajout d'erreurs différentes
    import copy

    m1_err = copy.deepcopy(encoded1)
    m2_err = copy.deepcopy(encoded2)

    # Erreurs dans m1
    m1_err[0][2] ^= 1

    # Erreurs dans m2
    m2_err[1][1] ^= 1
    m2_err[2][0] ^= 1
    m2_err[3][4] ^= 1

    print("\nMatrice 1 avec erreurs introduites:")
    for r in m1_err:
        print(r)

    m1 = MyHamming(m1_err).decode()

    print("\nMatrice corrigé:")
    for r in m1:
        print(r)

    m1_err[2][3] ^= 1
    m1_err[2][4] ^= 1
    print("\nMatrice 1 avec erreurs introduites:")
    for r in m1_err:
        print(r)

    print("\nMatrice 2 avec erreurs introduites:")
    for r in m2_err:
        print(r)

    m2 = MyHamming(m1_err, m2_err).decode()

    print("\nMatrice corrigé:")
    for r in m2:
        print(r)
