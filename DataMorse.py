"""
DataMorse.py - Module pour le chiffrement et le déchiffrement de messages en utilisant le code Morse.

Ce module est la partie principale du projet DataMorse.

Il permet d'encoder des messages en code Morse, de les transformer en une matrice binaire,
et de générer une image représentant ce message. Il inclut également un décodeur pour
récupérer le message original à partir de l'image.

Auteur : Lucas BATTAGLIA

Version : v2.8
"""
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
    ',': '--..--', '?': '..--..', "'": '.----.', '!': '-.-.-----.', '/': '-..-.', '(': '-.--.',
    ')': '-.--.-', '&': '.-...', ':': '---...', ';': '-.-.-.', '=': '-...-', '+': '.-.-.', '-': '-....-',
    '_': '..--.-', '"': '.-..-.', '$': '...-..-', '@': '.--.-.'
}


class DataMorseEncoder:
    """
        Classe pour encoder des messages en utilisant le code Morse.

        Cette classe permet de prendre un message en entrée, de le convertir en code Morse,
        de le transformer en une matrice binaire, et de générer une image représentant ce message.
    """
    def __init__(self):
        """
            Initialise l'encodeur DataMorse et demande un message à l'utilisateur.

            Cette méthode effectue le chiffrement du message en DataMorse et génère une image.
        """
        print("Votre message : ", end="")
        text = input()
        print("\033[36mChiffrement  de votre message en DataMorse en cours ... \033[0m")
        self.__RS = ReedSolomon.ReedSolomon(self.__text_to_morse(self.__remplacer_accents(text)))
        encoded_message, enc_params = self.__RS.encode()
        binary_data = self.__morse_to_matrix(encoded_message)
        me = Masque.Masque(binary_data)
        binary_data, masque = me.add_mask()
        enc_params['masque'] = masque
        donnee = self.__entete(enc_params, min(len(binary_data[0]) - 2, 10))
        datamorseencoder = self.__ajouter_lignes_et_colonnes(binary_data, donnee)
        print("\033[36mMessage chiffrer !\033[0m")
        print("\033[36mGeneration de l'image ...\033[0m")
        img = self.__generate_image(datamorseencoder)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        cv2.imwrite('./img/message.png', img)
        print("Image du message : img/message.png")
        print("Vos paramettres de decodage : ", end="")
        print(enc_params)

    @staticmethod
    def __remplacer_accents(texte):
        """
            Supprime les accents d'un texte donné.

            Args:
                texte (str): Le texte d'entrée avec des accents.

            Returns:
                str: Le texte sans accents.
        """
        # Supprimer les accents
        texte_sans_accents = unicodedata.normalize('NFD', texte)
        texte_sans_accents = ''.join(
            c for c in texte_sans_accents
            if unicodedata.category(c) != 'Mn'
        )
        # Convertir en majuscules
        return texte_sans_accents

    def __text_to_morse(self, text):
        """
            Convertit un texte en une liste de bits représentant le code Morse.

            Args:
                text (str): Le texte à convertir.

            Returns:
                list: Une liste de bits représentant le code Morse.
        """
        morse = []
        for word in text.upper().split():
            morse.extend(self.__word_to_morse(word))
            morse.extend([0, 0, 0, 0, 0, 0, 0])
        return morse

    @staticmethod
    def __word_to_morse(text):
        """
            Convertit un mot en code Morse sous forme de liste de bits.

            Args:
                text (str): Le mot à convertir.

            Returns:
                list: Une liste de bits représentant le mot en code Morse.
        """
        morse_text = [morse_dict[i] for i in text]
        morse_deux = [[[1, 0] if carac == '.' else [1, 1, 1, 0] for carac in elem] for elem in morse_text]
        morse_trois = []
        for i in range(len(morse_deux)):
            for j in range(len(morse_deux[i])):
                morse_trois.extend(morse_deux[i][j])
            morse_trois.extend([0, 0, 0])
        return morse_trois

    @staticmethod
    def __subdiviser_list(liste, size_list):
        """
            Crée des sous-listes à partir d'une liste donnée.

            Args:
                liste (list): La liste à subdiviser.
                size_list (int): La taille de chaque sous-liste.

            Returns:
                list: Une liste de sous-listes.
        """
        # Créer les sous-listes avec des tranches
        sublists = [liste[i:i + size_list] for i in range(0, len(liste), size_list)]

        # Remplir la dernière sous-liste avec des 0 si nécessaire
        if len(sublists) > 0 and len(sublists[-1]) < size_list:
            sublists[-1].extend([0] * (size_list - len(sublists[-1])))

        return sublists

    def __morse_to_matrix(self, morse):
        """
            Transforme une liste de bits Morse en une matrice.

            Args:
                morse (list): La liste de bits Morse à transformer.

            Returns:
                list: Une matrice représentant les bits Morse.
        """
        size = int(np.ceil(np.sqrt(len(morse))))
        return self.__subdiviser_list(morse, size)

    @staticmethod
    def __entier_vers_bits(n, taille=10):
        """
            Convertit un entier en une liste de bits.

            Args:
                n (int): L'entier à convertir.
                taille (int): La taille de la liste de bits.

            Returns:
                list: Une liste de bits représentant l'entier.
        """
        bits_str = format(n, '0' + str(taille) + 'b')
        bits = [int(b) for b in bits_str]
        return bits

    def __entete(self, valeur, taille=10):
        """
            Crée l'en-tête pour le message encodé.

            Args:
                valeur (dict): Les valeurs nécessaires pour créer l'en-tête.
                taille (int): La taille des bits à utiliser.

            Returns:
                list: Une liste représentant l'en-tête encodé.
        """
        masque = valeur['masque']
        n_parity = valeur['n_parity']
        pad_bits = valeur['pad_bits']
        message_len_bytes = valeur['message_len_bytes']

        masque_morse = self.__entier_vers_bits(masque, taille)
        n_parity_morse = self.__entier_vers_bits(n_parity, taille)
        pad_bits_morse = self.__entier_vers_bits(pad_bits, taille)
        message_len_bytes_morse = self.__entier_vers_bits(message_len_bytes, taille)

        lignes_sup = [masque_morse, n_parity_morse, pad_bits_morse, message_len_bytes_morse]

        hm1 = MyHamming.MyHamming(lignes_sup)
        lignes_sup = hm1.encode()

        return lignes_sup

    @staticmethod
    def __ajouter_lignes_et_colonnes(matrice, ajout_lignes):
        """
            Ajoute des lignes et des colonnes à une matrice donnée.

            Args:
                matrice (list): La matrice à modifier.
                ajout_lignes (list): Les lignes à ajouter.

            Returns:
                list: La matrice modifiée avec les lignes et colonnes ajoutées.
        """
        x = len(matrice[0])
        y = len(matrice)

        new_matrice = []

        line1 = []
        line1.extend(ajout_lignes[0])
        line1.extend([0 if i % 2 == 0 else 1 for i in range(len(ajout_lignes[0]), x)])
        bisline1 = list(line1)
        bisline1.extend([1, 1, 1, 1, 1])
        new_matrice.append(bisline1)

        line2 = []
        line2.extend(ajout_lignes[1])
        line2.extend([1 if i % 2 == 0 else 0 for i in range(len(ajout_lignes[1]), x)])
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
    def __draw_triangles(x, y, image, width, height):
        """
            Dessine des triangles sur une image à des coordonnées données.

            Args:
                x (int): La coordonnée x du triangle.
                y (int): La coordonnée y du triangle.
                image (ndarray): L'image sur laquelle dessiner.
                width (int): La largeur du triangle.
                height (int): La hauteur du triangle.
        """
        vertices = np.array([[x + width / 2, y], [x + width, y + height], [x, y + height]], np.int32)
        pts = vertices.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.fillPoly(image, [pts], color=(0, 0, 255))

    def __generate_image(self, matrix):
        """
            Génère une image à partir d'une matrice binaire.

            Args:
                matrix (list): La matrice binaire à transformer en image.

            Returns:
                ndarray: L'image générée.
        """
        size = max(len(matrix), len(matrix[0]))
        width = 20
        height = 20
        image = np.ones(((size + 10) * (width + 5), (size + 10) * (height + 5)), dtype=np.uint8) * 255

        # Donnee
        for i in range(len(matrix[0])):
            for j in range(len(matrix)):
                x, y = i * (width + 4) + 20, (j + 3) * (height + 4) + 20
                if matrix[j][i] == 1:
                    self.__draw_triangles(x, y, image, width, height)
        return image


class DataMorseDecoder:
    """
        Classe pour décoder des messages en utilisant le code Morse.

        Cette classe permet de décoder des messages à partir d'images, de matrices binaires,
        ou d'autres sources, et de récupérer le message original.
    """
    def __init__(self, data=None):
        """
            Initialise le décodeur DataMorse avec les données fournies.

            Args:
                data (optional): Les données à décoder, qui peuvent être une image, une matrice, etc.
        """
        self.__decodeur = None
        print("\033[36mInitialisation du decodeur ...\033[0m")
        if type(data) is None:
            self.__decodeur = self.DataMorseDecoderCamera()
        elif type(data) is str:
            self.__decodeur = self.DataMorseDecoderImage(data)
        elif type(data) is list:
            self.__decodeur = self.DataMorseDecoderMatrix(data)
        elif type(data) is np.ndarray:
            self.__decodeur = self.DataMorseDecoderMatrix(data)
        else:
            raise TypeError(
                "Le type du parametre entrer n'est pas correct: Soit rien (camera), soit un str (path de l'image), soit une liste (matrice de bit)")
        print("\033[36mDecodeur Initialisé\033[0m")

    def run_decodeur(self):
        """
            Exécute le décodeur pour traiter les données et récupérer le message.

            Raises:
                ValueError: Si le décodeur n'a pas été initialisé correctement.
        """
        if self.__decodeur is None:
            raise ValueError(
                "Votre decodeur n'a pas étais initialisé avant d'être utilisé. Faite DataMorseDecoder(data=None)")
        self.__decodeur.decoder()

    ###### Sous Classe  ########
    class DataMorseDecoderCamera:
        """
            Sous-classe pour le décodeur utilisant une caméra.

            Cette classe permet de décoder des messages à partir d'images capturées par une caméra.
        """
        def __init__(self):
            """
                Initialise le décodeur de caméra.

                Ouvre la caméra pour capturer des images.

                Raises:
                    SystemError: Si la caméra ne peut pas être ouverte.
            """
            self.__cap = cv2.VideoCapture(0)  # Ouvre la caméra

            if not self.__cap.isOpened():
                raise SystemError("Erreur : Impossible d'ouvrir la caméra")

        def decoder(self):
            """
                Décode l'image capturée par la caméra.

                Cette méthode n'est pas encore implémentée.

                Raises:
                    NotImplementedError: Si la méthode n'est pas implémentée.
            """
            print("\033[36mDecodage de l'image\033[0m")
            raise NotImplementedError(
                "La fonction decoder de la classe DataMorseDecoderCamera n'a pas encore été implémentée.")

    class DataMorseDecoderImage:
        """
            Sous-classe pour le décodeur utilisant une image.

            Cette classe permet de décoder des messages à partir d'images fournies.
        """
        def __init__(self, path):
            """
                Initialise le décodeur d'image avec le chemin de l'image.

                Args:
                    path (str): Le chemin de l'image à décoder.
            """
            if path.lower().endswith(('.jpg', '.jpeg')):
                convert = FileConvert.FileConvert(path)
                path = convert.convert()
            self.__path = path

        @staticmethod
        def __calculate_average_distance(points):
            """
                Calcule la distance moyenne entre un ensemble de points.

                Args:
                    points (list): Une liste de points sous forme de tuples (x, y).

                Returns:
                    float: La distance moyenne entre les points.
            """
            distances = []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    distance = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
                    distances.append(distance)
            return np.mean(distances) if distances else 0

        @staticmethod
        def __is_point_isolated(point, points, average_distance, percentage):
            """
                Détermine si un point est isolé par rapport à un ensemble de points.

                Args:
                    point (tuple): Le point à vérifier.
                    points (list): La liste des autres points.
                    average_distance (float): La distance moyenne entre les points.
                    percentage (float): Le pourcentage pour déterminer l'isolement.
                Returns:
                    bool: True si le point est isolé, False sinon.
            """
            distances = [np.linalg.norm(np.array(point) - np.array(other_point)) for other_point in points if
                         other_point != point]
            if not distances:  # Si le point est le seul dans la liste
                return True
            min_distance = min(distances)
            threshold = average_distance * (1 + percentage / 100)  # Seuil basé sur la moyenne
            return min_distance > threshold

        def __filter_isolated_points(self, points, percentage=2):
            """
                Filtre les points isolés d'une liste de points.

                Args:
                    points (list): La liste de points à filtrer.
                    percentage (float): Le pourcentage pour déterminer l'isolement.

                Returns:
                    list: La liste des points filtrés sans points isolés.
            """
            if not points:
                return []

            # Calculer la distance moyenne
            average_distance = self.__calculate_average_distance(points)

            # Filtrer les points isolés
            filtered_points = [point for point in points if
                               not self.__is_point_isolated(point, points, average_distance, percentage)]

            return filtered_points

        @staticmethod
        def __draw_matrix_with_points(points):
            """
                Dessine une matrice à partir d'une liste de points.

                Args:
                    points (list): La liste de points à dessiner.

                Raises:
                    ValueError: Si le nombre de points est insuffisant pour déterminer l'espacement.

                Returns:
                    np.ndarray: La matrice binaire représentant les points.
            """
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
        def __recup_triangle(contours):
            """
                Récupère les triangles à partir des contours détectés.

                Args:
                    contours (list): La liste des contours détectés.
                Returns:
                    tuple: Une liste de triangles et une liste de leurs aires respectives.
            """
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
        def __position(triangles, areas, average_area):
            """
                Détermine la position des triangles en fonction de leur aire.

                Args:
                    triangles (list): La liste des triangles.
                    areas (list): La liste des aires des triangles.
                    average_area (float): L'aire moyenne pour le filtrage.

                Returns:
                    list: Une liste de positions (x, y) des triangles valides.
            """
            position = []
            for i, triangle in enumerate(triangles):
                if areas[i] > average_area - 2 / 100 * average_area:
                    m = cv2.moments(triangle)
                    if m['m00'] != 0.0:
                        x = int(m['m10'] / m['m00'])
                        y = int(m['m01'] / m['m00'])
                        position.append((x, y))
            return position

        def decoder(self):
            """
                Décode l'image fournie et récupère le message.

                Cette méthode traite l'image pour extraire les points et reconstruire le message.
            """
            img = cv2.imread(self.__path)

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
            triangles, areas = self.__recup_triangle(contours)
            if areas:
                average_area = sum(areas) / len(areas)
            else:
                average_area = 0

            # On filtre pour recuperer au maximum les
            position = self.__position(triangles, areas, average_area)
            filtered_points = self.__filter_isolated_points(position)

            # On cree la matrice
            matrix = self.__draw_matrix_with_points(filtered_points)

            # On recreer une instance de notre decodeur pour decoder la matrice
            mere = DataMorseDecoder(matrix)
            mere.run_decodeur()

    class DataMorseDecoderMatrix:
        """
            Sous-classe pour le décodeur utilisant une matrice.

            Cette classe permet de décoder des messages à partir de matrices binaires.
        """
        def __init__(self, matrix):
            """
                Initialise le décodeur de matrice avec la matrice fournie.

                Args:
                    matrix (list): La matrice binaire à décoder.
            """
            self.__matrix = matrix

        @staticmethod
        def __create_chaine(matrix):
            """
                Crée une chaîne à partir d'une matrice binaire.

                Args:
                    matrix (list): La matrice binaire à transformer.

                Returns:
                    list: Une liste représentant la chaîne de bits.
            """
            chaine = []
            for ligne in matrix:
                chaine.extend(ligne)
            return chaine

        @staticmethod
        def __split(lst, sep):
            """
                Divise une liste en sous-listes en fonction d'un séparateur donné.

                Args:
                    lst (list): La liste à diviser.
                    sep (list): La séquence de séparation.

                Returns:
                    list: Une liste de sous-listes.
            """
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

        def __split_mot(self, chaine):
            """
                Divise une chaîne en mots en utilisant un séparateur spécifique.

                Args:
                    chaine (list): La chaîne à diviser.

                Returns:
                    list: Une liste de mots.
            """
            morse = self.__split(chaine, [0, 0, 0, 0, 0, 0, 0, 0])
            return morse

        def __split_letter(self, morse):
            """
                Divise une liste de Morse en lettres.

                Args:
                    morse (list): La liste de Morse à diviser.

                Returns:
                    list: Une liste de lettres.
            """
            symbol = self.__split(morse, [0, 0, 0, 0])
            return symbol

        def __split_symbol(self, morse):
            """
                Divise une liste de Morse en symboles.

                Args:
                    morse (list): La liste de Morse à diviser.
                Returns:
                    list: Une liste de symboles.
            """
            symbol = self.__split(morse, [0])
            return symbol

        @staticmethod
        def __get_key_from_value(value):
            """
                Récupère la clé correspondant à une valeur dans le dictionnaire Morse.

                Args:
                    value (str): La valeur à rechercher.

                Returns:
                    str: La clé correspondante, ou None si non trouvée.
            """
            for key, val in morse_dict.items():
                if val == value:
                    return key
            return None  # si la valeur n'est pas trouvée

        def __morse_to_letter(self, morse):
            """
                Convertit une séquence de Morse en lettre.

                Args:
                    morse (list): La séquence de Morse à convertir.
                Returns:
                    str: La lettre correspondante.
            """
            symbol_morse = self.__split_symbol(morse)
            symbol_text = ['.' if x == [1] else '-' if x == [1, 1, 1] else 5 for x in symbol_morse]
            if 5 in symbol_text:
                raise Exception(
                    'Erreur de lecture du Code DataMorse! Essayer avec une image plus nettes ou avec un DataMorse plus zoomer !')
            text = ''.join(symbol_text)
            if not symbol_text:
                text = ""
            else:
                text = self.__get_key_from_value(text)
                if text is None:
                    raise Exception(
                        'Erreur de lecture du Code DataMorse! Essayer avec une image plus nettes ou avec un DataMorse plus zoomer !')
            return text

        def __morse_to_word(self, morse):
            """
                Convertit une séquence de Morse en mot.

                Args:
                    morse (list): La séquence de Morse à convertir.
                Returns:
                    str: Le mot correspondant.
            """
            word_morse = self.__split_letter(morse)
            word_text = ''
            for word in word_morse:
                word_text += self.__morse_to_letter(word)
            return word_text

        def __morse_to_text(self, morse):
            """
                Convertit une séquence de Morse en texte.

                Args:
                    morse (list): La séquence de Morse à convertir.

                Returns:
                    str: Le texte correspondant.
            """
            mots = self.__split_mot(morse)
            text = ''
            for mot in mots:
                text = text + " " + self.__morse_to_word(mot)
            return text

        @staticmethod
        def __rotate_matrix_anticlockwise(matrix):
            """
                Fait une rotation antihoraire d'une matrice.

                Args:
                    matrix (list): La matrice à faire tourner.

                Returns:
                    list: La matrice tournée.
            """
            transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
            rotated = [row[::-1] for row in transposed]
            return rotated

        @staticmethod
        def __bits_vers_entier(bits):
            """
                Convertit une liste de bits en entier.

                Args:
                    bits (list): La liste de bits à convertir.
                Returns:
                    int: L'entier correspondant.
            """
            # Convertir la liste de bits en chaîne puis en entier
            bits_str = ''.join(str(b) for b in bits)
            decimal_value = int(bits_str, 2)
            return decimal_value

        @staticmethod
        def __compter_differences(m1, m2):
            """
                Compte le nombre de différences entre deux matrices.

                Args:
                    m1 (list): La première matrice.
                    m2 (list): La deuxième matrice.

                Returns:
                    int: Le nombre de différences entre les deux matrices.
            """
            differences = 0
            for i in range(len(m1)):
                for j in range(len(m1[0])):
                    if m1[i][j] != m2[i][j]:
                        differences += 1
            return differences

        def decoder(self):
            """
                Décode la matrice fournie et récupère le message.

                Cette méthode traite la matrice pour extraire le message original.
            """
            print("\033[36mDécodage en cours ...\033[0m")
            print("\033[36mGestion de l'orientation du DataMorse ...\033[0m")
            orientations = []
            for i in range(4):
                orientation = [row[len(row) - 5:] for row in self.__matrix[:5]]
                verif = [[1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]]
                if np.array_equal(verif, orientation):
                    break
                orientations.append(self.__compter_differences(orientation, verif))
                self.__matrix = self.__rotate_matrix_anticlockwise(self.__matrix)

            if len(orientations) == 4:
                indice_min = orientations.index(min(orientations))
                for _ in range(indice_min):
                    self.__matrix = self.__rotate_matrix_anticlockwise(self.__matrix)

            print("\033[36mExtraction de l'entête ...\033[0m")

            entete1 = [row[:min(11, len(row) - 6)] for row in self.__matrix[:5]]
            entete2 = [row[len(row) - 5:] for row in self.__matrix[5:min(16, len(self.__matrix))]]
            entete2 = self.__rotate_matrix_anticlockwise(entete2)

            entete = MyHamming.MyHamming(entete1, entete2).decode()

            print("\033[36mPreparation des donnees ...\033[0m")

            masque = self.__bits_vers_entier(entete[0])
            n_parity = self.__bits_vers_entier(entete[1])
            pad_bits = self.__bits_vers_entier(entete[2])
            message_len_bytes = self.__bits_vers_entier(entete[3])
            dict_solomon = {'n_parity': n_parity, 'pad_bits': pad_bits, 'message_len_bytes': message_len_bytes}

            matrice = copy.deepcopy(self.__matrix)
            self.__matrix = []
            for i in range(5, len(matrice)):
                self.__matrix.append(matrice[i][:len(matrice[i]) - 5])

            me = Masque.Masque(self.__matrix)
            self.__matrix = me.remove_mask(masque)[0]

            chaine = self.__create_chaine(self.__matrix)

            print("\033[36mCorrection des erreurs dans les donnees ...\033[0m")
            rs = ReedSolomon.ReedSolomon(chaine)
            encoded_bits = rs.decode(dict_solomon)

            print("\033[36mConversion en alphanumérique des donnees ...\033[0m")
            texte = self.__morse_to_text(encoded_bits)

            print("\033[36mFin du decodage !\033[0m")
            print("Votre message est :")
            print(texte)


if __name__ == "__main__":
    run = int(input("\033[33m### DataMorce ###\033[0m\n\nQue voulez vous faire ?\n\t1 - Chiffrer\n\t2 - Déchiffrer\n"))
    if run == 1:
        DataMorseEncoder()
    else:
        image = int(input(
            "Que voulez vous déchiffrer ?\n\t1 - Image simple (votre image chiffrer arrive ici par defaut)\n\t2 - Image complexe (Erreur + texte + rotation)\n\t3 - Autre Image\n"))
        path = ""
        if image == 1:
            path = "img/message.png"
        elif image == 2:
            path = "img/img.png"
        elif image == 3:
            path = input("Quel est le lien (path) de votre image ?\n")
        else:
            raise ValueError("Selection non valide (1, 2 ou 3) !")
        decoder = DataMorseDecoder(path)
        decoder.run_decodeur()
