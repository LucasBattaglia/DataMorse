import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt
import unicodedata

import FileConvert
import ReedSolomon
import MyHamming
import Masque

morse_dict = {
                'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.', 'H': '....',
                'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 'P': '.--.',
                'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
                'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
                '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.', '.': '.-.-.-.',
                ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.-----.', '/': '-..-.', '(' : '-.--.',
                ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
                '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.'
              }


class DataMorseEncoder:
    def __init__(self):
        print("Votre message : ", end="")
        text = input()
        print("\033[36mEncryptage de votre message en DataMorse en cours ... \033[0m")
        self.RS = ReedSolomon.ReedSolomon(self.text_to_morse(self.remplacer_accents(text)))
        encoded_message, enc_params = self.RS.encode()
        binary_data = self.morse_to_matrix(encoded_message)
        me = Masque.Masque(binary_data)
        binary_data, masque = me.add_mask()
        enc_params['masque'] = masque
        donnee = self.entete(enc_params, min(len(binary_data[0])-2, 10))
        datamorseencoder = self.ajouter_lignes_et_colonnes(binary_data, donnee)
        print("\033[36mMessage encrypter !\033[0m")
        print("\033[36mGeneration de l'image ...\033[0m")
        img = self.generate_image(datamorseencoder)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        cv2.imwrite('./img/message.png', img)
        print("Image du message : img/message.png")
        print("Vos paramettres de decodage : ", end="")
        print(enc_params)


    @staticmethod
    def remplacer_accents(texte):
        # Supprimer les accents
        texte_sans_accents = unicodedata.normalize('NFD', texte)
        texte_sans_accents = ''.join(
            c for c in texte_sans_accents
            if unicodedata.category(c) != 'Mn'
        )
        # Convertir en majuscules
        return texte_sans_accents

    def text_to_morse(self, text):
        morse = []
        for word in text.upper().split():
            morse.extend(self.word_to_morse(word))
            morse.extend([0,0,0,0,0,0,0])
        return morse


    @staticmethod
    def word_to_morse(text):
        morse_text = [morse_dict[i] for i in text]
        morse_deux = [[[1, 0] if carac == '.' else [1, 1, 1, 0] for carac in elem] for elem in morse_text]
        morse_trois = []
        for i in range(len(morse_deux)):
            for j in range(len(morse_deux[i])):
                morse_trois.extend(morse_deux[i][j])
            morse_trois.extend([0, 0, 0])
        return morse_trois

    @staticmethod
    def subdiviser_list(list, size_list):
        # Créer les sous-listes avec des tranches
        sublists = [list[i:i + size_list] for i in range(0, len(list), size_list)]

        # Remplir la dernière sous-liste avec des 0 si nécessaire
        if len(sublists) > 0 and len(sublists[-1]) < size_list:
            sublists[-1].extend([0] * (size_list - len(sublists[-1])))

        return sublists


    def morse_to_matrix(self, morse):
        size = int(np.ceil(np.sqrt(len(morse))))
        return self.subdiviser_list(morse, size)


    @staticmethod
    def entier_vers_bits(n, taille=10):
        bits_str = format(n, '0'+str(taille)+'b')
        bits = [int(b) for b in bits_str]
        return bits


    def entete(self, valeur, taille=10):
        masque = valeur['masque']
        n_parity = valeur['n_parity']
        pad_bits = valeur['pad_bits']
        message_len_bytes = valeur['message_len_bytes']

        masque_morse = self.entier_vers_bits(masque, taille)
        n_parity_morse = self.entier_vers_bits(n_parity, taille)
        pad_bits_morse = self.entier_vers_bits(pad_bits, taille)
        message_len_bytes_morse = self.entier_vers_bits(message_len_bytes, taille)

        lignes_sup = []
        lignes_sup.append(masque_morse)
        lignes_sup.append(n_parity_morse)
        lignes_sup.append(pad_bits_morse)
        lignes_sup.append(message_len_bytes_morse)

        hm1 = MyHamming.MyHamming(lignes_sup)
        lignes_sup = hm1.encode()

        return lignes_sup

    @staticmethod
    def ajouter_lignes_et_colonnes(matrice, ajout_lignes):
        x = len(matrice[0])
        y = len(matrice)

        new_matrice = []

        line1 = []
        line1.extend(ajout_lignes[0])
        line1.extend([0 if i % 2 == 0 else 1  for i in range(len(ajout_lignes[0]), x)])
        bisline1 = list(line1)
        bisline1.extend([1, 1, 1, 1, 1])
        new_matrice.append(bisline1)

        line2 = []
        line2.extend(ajout_lignes[1])
        line2.extend([1 if i % 2 == 0 else 0  for i in range(len(ajout_lignes[1]), x)])
        bisline2 = list(line2)
        bisline2.extend([1, 0, 1, 0, 1])
        new_matrice.append(bisline2)

        line3 = []
        line3.extend(ajout_lignes[2])
        line3.extend([0 if i % 2 == 0 else 1 for i in range(len(ajout_lignes[2]), x)])
        bisline3 = list(line3)
        bisline3.extend([1, 1, 1, 1, 1])
        new_matrice.append(bisline3)

        line4 = []
        line4.extend(ajout_lignes[3])
        line4.extend([1 if i % 2 == 0 else 0 for i in range(len(ajout_lignes[3]), x)])
        bisline4 = list(line4)
        bisline4.extend([1, 0, 1, 0, 1])
        new_matrice.append(bisline4)

        line5 = []
        line5.extend(ajout_lignes[4])
        line5.extend([0 if i % 2 == 0 else 1 for i in range(len(ajout_lignes[4]), x)])
        bisline5 = list(line5)
        bisline5.extend([1, 1, 1, 1, 1])
        new_matrice.append(bisline5)


        for i in range(y):
            new_line = matrice[i]
            if i < len(line1):
                new_line.append(line5[i])
                new_line.append(line4[i])
                new_line.append(line3[i])
                new_line.append(line2[i])
                new_line.append(line1[i])
            else:
                new_line.extend([0, 0, 0, 0, 0])
            new_matrice.append(new_line)

        return new_matrice

    @staticmethod
    def draw_triangles(x, y, image, width, height):
        vertices = np.array([[x + width/2, y], [x + width, y + height], [x, y + height]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.fillPoly(image, [pts], color=(0, 0, 255))


    def generate_image(self, matrix):
        size = max(len(matrix), len(matrix[0]))
        width = 20
        height = 20
        image = np.ones(((size+10) * (width+5), (size+10) * (height+5)), dtype=np.uint8) * 255


        # Orientation
        #x, y = size * 24 + 10, 0 * 24 + 10
        #self.draw_triangles(x, y, image, 50, 50)

        # Donnee
        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                x, y = i * (width + 4) + 20, (j+3) * (height + 4) + 20
                if matrix[j][i] == 1:
                    self.draw_triangles(x, y, image, width, height)
        return image

class DataMorseDecoder:
    ###### Attribut  ########
    decodeur = None

    ###### Methode  ########
    def __init__(self, data=None):
        print("\033[36mInitialisation du decodeur ...\033[0m")
        if type(data) is None:
            self.decodeur = self.DataMorseDecoderCamera()
        elif type(data) is str:
            self.decodeur = self.DataMorseDecoderImage(data)
        elif type(data) is list:
            self.decodeur = self.DataMorseDecoderMatrix(data)
        elif type(data) is np.ndarray:
            self.decodeur = self.DataMorseDecoderMatrix(data)
        else:
            raise TypeError("Le type du parametre entrer n'est pas correct: Soit rien (camera), soit un str (path de l'image), soit une liste (matrice de bit)")
        print("\033[36mDecodeur Initialiser\033[0m")


    def run_decodeur(self):
        if self.decodeur is None:
            raise ValueError("Votre decoder n'a pas étais initialisé avant d'être utilisé. Faite DataMorseDecoder(data=None)")
        self.decodeur.decoder()


    ###### Sous Classe  ########
    class DataMorseDecoderCamera:
        def __init__(self):
            self.cap = cv2.VideoCapture(0)  # Ouvre la caméra

            if not self.cap.isOpened():
                raise SystemError("Erreur : Impossible d'ouvrir la caméra")

        def decoder(self):
            print("\033[36mDecodage de l'image\033[0m")
            raise NotImplementedError("La fonction decoder de la classe DataMorseDecoderCamera n'a pas encore été implémentée.")


    class DataMorseDecoderImage:
        def __init__(self, path):
            if path.lower().endswith(('.jpg', '.jpeg')):
                convert = FileConvert.FileConvert(path)
                path = convert.convert()
            self.path = path


        @staticmethod
        def calculate_average_distance(points):
            """Calcule la distance moyenne entre tous les points."""
            distances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                    distances.append(distance)
            return np.mean(distances) if distances else 0

        @staticmethod
        def is_point_isolated(point, points, average_distance, percentage):
            """Vérifie si un point est isolé par rapport à une liste de points."""
            distances = [np.linalg.norm(np.array(point) - np.array(other_point)) for other_point in points if
                         other_point != point]
            if not distances:  # Si le point est le seul dans la liste
                return True
            min_distance = min(distances)
            threshold = average_distance * (1 + percentage / 100)  # Seuil basé sur la moyenne
            return min_distance > threshold

        def filter_isolated_points(self, points, percentage=2):
            """Retourne une liste de points sans les points isolés."""
            if not points:
                return []

            # Calculer la distance moyenne
            average_distance = self.calculate_average_distance(points)

            # Filtrer les points isolés
            filtered_points = [point for point in points if
                               not self.is_point_isolated(point, points, average_distance, percentage)]

            return filtered_points

        @staticmethod
        def draw_matrix_with_points(points):
            """Dessine une grille sur l'image et place les points dans les cases."""
            # Calculer l'espacement entre les points
            if len(points) < 2:
                raise ValueError("Pas assez de points pour déterminer l'espacement.")

            # Calculer les distances entre les points pour déterminer l'espacement
            distances = []
            for i in range(len(points)):
                dist = []
                for j in range(len(points)):
                    if i != j:
                        distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                        dist.append(distance)
                distances.append(np.min(dist))

            # Utiliser la distance moyenne comme espacement
            distances.sort()
            m = int(len(distances) / 2 - 10 / 100 * len(distances))
            n = int(len(distances) / 2 + 10 / 100 * len(distances))
            distances = distances[m:n]
            average_distance = np.mean(distances)
            cell_size = int(average_distance)  # Taille de chaque case de la grille

            # Determiner la taille de la grille
            min_x = int(min(point[0] for point in points) - cell_size / 2)
            max_x = int(max(point[0] for point in points) + cell_size / 2)
            min_y = int(min(point[1] for point in points) - cell_size / 2)
            max_y = int(max(point[1] for point in points) + cell_size / 2)

            # Calculer la largeur et la hauteur de l'image
            width = max_x - min_x  # Ajouter un peu de marge
            height = max_y - min_y  # Ajouter un peu de marge

            # Déterminer le nombre de cases
            num_columns = width // cell_size + 1
            num_rows = height // cell_size + 1

            # Generation de la matrice
            matrix = []
            for j in range(1, num_rows):
                line_matrix = [0 for _ in range(1, num_columns)]
                for i in range(1, num_columns):
                    for k, point in enumerate(points):
                        if (min_x + (i - 1) * cell_size) < point[0] < (min_x + i * cell_size) and (
                                min_y + (j - 1) * cell_size) < point[1] < (min_y + j * cell_size):
                            line_matrix[i - 1] = 1

                matrix.append(line_matrix)

            return np.array(matrix)


        @staticmethod
        def recup_triangle(contours):
            triangles = []
            areas = []

            # list for storing names of shapes
            for contour in contours:
                # Approximer le contour avec un seuil plus élevé pour des lignes plus droites
                approx = cv2.approxPolyDP(contour, 0.19 * cv2.arcLength(contour, True), True)

                # Vérifier si le contour est proche d'un triangle
                num_vertices = len(approx)
                if num_vertices == 3:
                    # Calculer l'aire du triangle
                    area = cv2.contourArea(contour)

                    areas.append(area)
                    triangles.append(approx)
            return triangles, areas

        @staticmethod
        def position(triangles, areas, average_area):
            position = []
            for i, triangle in enumerate(triangles):
                if areas[i] > average_area - 2 / 100 * average_area:
                    M = cv2.moments(triangle)
                    if M['m00'] != 0.0:
                        x = int(M['m10'] / M['m00'])
                        y = int(M['m01'] / M['m00'])
                        position.append((x, y))
            return position

        def decoder(self):
            img = cv2.imread(self.path)

            print("\033[36mLecture de l'image en cours ...\033[0m")

            scale_percent = 2000  # Ajuste si besoin
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            img = cv2.resize(img, (width, height))

            # Traitement de l'image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            adjust = cv2.convertScaleAbs(blurred, alpha=1, beta=0)
            _, threshold = cv2.threshold(adjust, 127, 255, cv2.THRESH_BINARY)

            # On recupere les formes
            contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            triangles, areas = self.recup_triangle(contours)
            if areas:
                average_area = sum(areas) / len(areas)
            else:
                average_area = 0

            # On filtre pour recuperer au maximum les
            position = self.position(triangles, areas, average_area)
            filtered_points = self.filter_isolated_points(position)

            # On cree la matrice
            matrix = self.draw_matrix_with_points(filtered_points)

            # On recreer une instance de notre decodeur pour decoder la matrice
            mere = DataMorseDecoder(matrix)
            mere.run_decodeur()



    class DataMorseDecoderMatrix:
        def __init__(self, matrix):
            self.matrix = matrix


        @staticmethod
        def create_chaine(matrix):
            chaine = []
            for ligne in matrix:
                chaine.extend(ligne)
            return chaine


        @staticmethod
        def split(lst, sep):
            result = []
            i = 0
            buffer = []
            n = len(sep)

            while i < len(lst):
                # Si une séquence correspond à sep à partir de l'indice i
                if list(lst[i:i + n]) == list(sep):
                    if buffer:
                        result.append(buffer)
                        buffer = []
                    i += n  # on saute la séquence trouvée
                else:
                    buffer.append(lst[i])
                    i += 1

            if buffer:
                result.append(buffer)

            return result


        def splitMot(self, chaine):
            morse = self.split(chaine, [0, 0, 0, 0, 0, 0, 0, 0])
            return morse


        def splitLetter(self, morse):
            symbol = self.split(morse, [0, 0, 0, 0])
            return symbol


        def splitSymbol(self, morse):
            symbol = self.split(morse, [0])
            return symbol


        @staticmethod
        def get_key_from_value(value):
            for key, val in morse_dict.items():
                if val == value:
                    return key
            return None  # si la valeur n'est pas trouvée


        def morse_to_letter(self, morse):
            symbolMorse = self.splitSymbol(morse)
            symbolText = ['.' if x == [1] else '-' if x == [1, 1, 1] else 5 for x in symbolMorse]
            if 5 in symbolText:
                raise Exception('Erreur de lecture du Code DataMorse! Essayer avec une image plus nettes ou avec un DataMorse plus zoomer !')
            text = ''.join(symbolText)
            if not symbolText:
                Text = ""
            else:
                Text = self.get_key_from_value(text)
                if Text is None:
                    raise Exception(
                        'Erreur de lecture du Code DataMorse! Essayer avec une image plus nettes ou avec un DataMorse plus zoomer !')
            return Text


        def morse_to_word(self, morse):
            wordMorse = self.splitLetter(morse)
            wordText = ''
            for word in wordMorse:
                wordText += self.morse_to_letter(word)
            return wordText


        def morse_to_text(self, morse):
            mots = self.splitMot(morse)
            text = ''
            for mot in mots:
                text = text + " " + self.morse_to_word(mot)
            return text

        @staticmethod
        def rotate_matrix_anticlockwise(matrix):
            transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
            rotated = [row[::-1] for row in transposed]
            return rotated

        @staticmethod
        def bits_vers_entier(bits):
            """
            Convertit une liste de bits (par exemple 10 bits) en entier décimal.

            Args:
                bits (list of int): Liste contenant des 0 ou 1, de longueur 10.

            Returns:
                int: La valeur décimale correspondante.
            """
            # Convertir la liste de bits en chaîne puis en entier
            bits_str = ''.join(str(b) for b in bits)
            decimal_value = int(bits_str, 2)
            return decimal_value


        def decoder(self):
            print("\033[36mDecodage en cours ...\033[0m")
            print("\033[36mExtraction de l'entete ...\033[0m")

            entete1 = [row[:min(11, len(self.matrix[0])-6)] for row in self.matrix[:5]]
            entete2 = [row[len(self.matrix)-5:] for row in self.matrix[5:min(16, len(self.matrix))]]
            entete2 = self.rotate_matrix_anticlockwise(entete2)

            entete = MyHamming.MyHamming(entete1, entete2).decode()

            print("\033[36mPreparation des donnees ...\033[0m")

            masque = self.bits_vers_entier(entete[0])
            n_parity = self.bits_vers_entier(entete[1])
            pad_bits = self.bits_vers_entier(entete[2])
            message_len_bytes = self.bits_vers_entier(entete[3])
            dict_Solomon = {'n_parity': n_parity, 'pad_bits': pad_bits, 'message_len_bytes': message_len_bytes}

            matrice = copy.deepcopy(self.matrix)
            self.matrix = []
            for i in range(5, len(matrice)):
                self.matrix.append(matrice[i][:len(matrice[i]) - 5])

            me = Masque.Masque(self.matrix)
            self.matrix = me.remove_mask(masque)[0]

            chaine = self.create_chaine(self.matrix)

            print("\033[36mCorrection des erreurs dans les donnees ...\033[0m")
            RS = ReedSolomon.ReedSolomon(chaine)
            encoded_bits = RS.decode(dict_Solomon)

            print("\033[36mConversion en hexadecimal des donnees ...\033[0m")
            texte = self.morse_to_text(encoded_bits)

            print("\033[36mFin du decodage !\033[0m")
            print("Votre message est :")
            print(texte)


if __name__ == "__main__":
    run = int(input("\033[33m### DataMorce ###\033[0m\n\nQue voullez vous faire ?\n\t1 - Encripter\n\t2 - Decripter\n"))
    if run == 1:
        DataMorseEncoder()
    else:
        decoder = DataMorseDecoder("img/message.png")
        decoder.run_decodeur()
