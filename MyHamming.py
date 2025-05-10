from collections import Counter

class MyHamming:
    def __init__(self, matrix1, matrix2=None):
        """
        Initialisation avec la matrice brute (sans bits de parité).
        """
        self.matrix1 = matrix1
        self.matrix2 = matrix2

    @staticmethod
    def add_parity(matrix):
        """
                Encode the matrix by adding parity bits:
                - Add a parity bit at the start of each row (row parity)
                - Add a parity row at the bottom (column parity)
                """
        # Ajout bit de parité à chaque ligne (au début)
        for row in matrix:
            parity_bit = sum(row) % 2
            row.insert(0, parity_bit)

        # Calcul de la ligne de parité (pour chaque colonne)
        num_columns = len(matrix[0])
        parity_row = []
        for col in range(num_columns):
            col_sum = sum(row[col] for row in matrix)
            parity_bit = col_sum % 2
            parity_row.append(parity_bit)

        # Ajout de la ligne de parité à la fin
        matrix.append(parity_row)
        return matrix


    def find_error(self, matrix):
        """
        Detect all error positions where parity checks fail in the given matrix.
        Returns:
            List of (row_index, column_index) tuples pointing to suspected error bits.
        """
        error_ligne = []
        error_colonne = []

        for index, row in enumerate(matrix):
            if sum(row) % 2 != 0:
                error_ligne.append(index)

        for i in range(len(matrix[0])-1):
            col = [matrix[j][i] for j,_  in enumerate(matrix)]
            if sum(col) % 2 != 0:
                error_colonne.append(i)

        return error_ligne, error_colonne

    @staticmethod
    def merge_matrices(m1, m2):
        """
        Fusionne deux matrices bit à bit en appliquant la règle suivante :
        - Si bits égaux, bit conservé
        - Sinon, bit = None (erreur probable)
        Renvoie une matrice de même taille avec bits 0,1 ou None
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


    def changement_matrices(self, m1, m2):
        newmatrice = self.merge_matrices(m1, m2)
        error_positions = []
        for r in range(len(newmatrice)):
            for c in range(len(newmatrice[r])):
                if newmatrice[r][c] is None:
                    error_positions.append((r,c))
        return error_positions


    def common_tuples(self, list1, list2):
        counter1 = Counter(list1)
        counter2 = Counter(list2)
        return list((counter1 & counter2).elements())

    def decode_double(self):
        try:
            return self.decode_simple(self.matrix1)
        except ValueError:
            pass

        try:
            return self.decode_simple(self.matrix2)
        except ValueError:
            pass

        chang = self.changement_matrices(self.matrix1, self.matrix2)

        erreur = self.find_error(self.matrix1)
        possible_position = []
        for e in erreur[0]:
            for a in erreur[1]:
                possible_position.append((e, a))

        list_erreur = self.common_tuples(possible_position, chang)

        for e in list_erreur:
            self.matrix1[e[0]][e[1]] ^= 1
            try:
                return self.decode_simple(self.matrix1)
            except ValueError:
                continue

        erreur = self.find_error(self.matrix2)
        possible_position = []
        for e in erreur[0]:
            for a in erreur[1]:
                possible_position.append((e, a))

        list_erreur = self.common_tuples(possible_position, chang)

        for e in list_erreur:
            self.matrix2[e[0]][e[1]] ^= 1
            try:
                return self.decode_simple(self.matrix2)
            except ValueError:
                continue

        raise ValueError("Trop d'erreur pour pouvoir etre corriger")


    def decode_simple(self, matrix):
        erreur = self.find_error(matrix)
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
        return self.add_parity(self.matrix1)


    def decode(self):
        if self.matrix2 is None:
            return self.decode_simple(self.matrix1)
        else:
            return self.decode_double()



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

