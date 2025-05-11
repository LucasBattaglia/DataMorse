"""
ReedSolomon.py - Module pour l'encodage et le décodage de messages avec le code Reed-Solomon.

Ce module fait partie du projet DataMorse.

Il permet d'encoder des messages en utilisant le code Reed-Solomon pour la correction d'erreurs,
et de décoder les messages reçus tout en corrigeant les erreurs détectables.

Auteur : Lucas BATTAGLIA

Version : v1.5
"""

import reedsolo
import numpy as np


class ReedSolomon:
    """
        Classe pour l'encodage et le décodage de messages avec le code Reed-Solomon.

        Cette classe permet d'encoder des messages pour la correction d'erreurs et de décoder
        les messages reçus tout en corrigeant les erreurs détectables.
    """
    def __init__(self, message):
        """
            Initialise un objet ReedSolomon avec un message donné.

            Args:
                message (list): Le message à encoder.
        """
        self.message = message

    def encode(self):
        """
            Encode le message en utilisant le code Reed-Solomon.

            Returns:
                tuple: Les bits encodés et les paramètres d'encodage.
        """
        encoder = self.ReedSolomonEncoder(0.2)
        return encoder.encode(self.message)

    def decode(self, param):
        """
            Décode un message encodé en utilisant le code Reed-Solomon.

            Args:
                param (dict): Les paramètres d'encodage nécessaires pour le décodage.

            Returns:
                list: Le message décodé.
        """
        decoder = self.ReedSolomonDecoder()
        return decoder.decode(self.message, param)

    def modifier_message(self, message):
        """
            Modifie le message à décoder.

            Args:
                message (list): Le nouveau message à décoder.
        """
        self.message = message

    class ReedSolomonEncoder:
        """
            Classe interne pour l'encodage des messages avec le code Reed-Solomon.

            Cette classe gère l'encodage des messages en bits et la génération de symboles de parité.
        """
        def __init__(self, pourcentage):
            """
                Initialise un encodeur Reed-Solomon avec un pourcentage de symboles de parité.

                Args:
                    pourcentage (float): Le pourcentage de symboles de parité à utiliser.
            """
            self.rs_codec = None
            self.pourcentage = pourcentage

        @staticmethod
        def _bits_to_bytes(bit_list):
            """
                Convertit une liste de bits en un tableau d'octets.

                Args:
                    bit_list (list): La liste de bits à convertir.

                Returns:
                    tuple: Un tableau d'octets et le nombre de bits de remplissage.
            """
            bit_list = list(bit_list)
            pad_bits = (8 - len(bit_list) % 8) % 8
            bit_list += [0] * pad_bits

            byte_array = bytearray()
            for i in range(0, len(bit_list), 8):
                byte_val = 0
                for j in range(8):
                    byte_val = (byte_val << 1) | bit_list[i + j]
                byte_array.append(byte_val)
            return byte_array, pad_bits

        @staticmethod
        def _bytes_to_bits(byte_data):
            """
                Convertit un tableau d'octets en une liste de bits.

                Args:
                    byte_data (bytearray): Le tableau d'octets à convertir.

                Returns:
                    list: La liste de bits résultante.
            """
            bit_list = []
            for b in byte_data:
                for i in range(7, -1, -1):
                    bit_list.append((b >> i) & 1)
            return bit_list

        def encode(self, bit_list):
            """
                Encode une liste de bits en utilisant le code Reed-Solomon.

                Args:
                    bit_list (list): La liste de bits à encoder.

                Returns:
                    tuple: Les bits encodés et les paramètres d'encodage.
            """
            message_bytes, pad_bits = self._bits_to_bytes(bit_list)

            # Calculer le nombre de symboles de parité autorisé
            max_n_parity = len(message_bytes) - 1
            if max_n_parity <= 0:
                raise ValueError("Message trop court pour Reed-Solomon.")

            n_parity = min(int(self.pourcentage * len(message_bytes)), max_n_parity)

            self.rs_codec = reedsolo.RSCodec(n_parity)
            encoded_bytes = self.rs_codec.encode(message_bytes)
            encoded_bits = self._bytes_to_bits(encoded_bytes)

            params = {
                'n_parity': n_parity,
                'pad_bits': pad_bits,
                'message_len_bytes': len(message_bytes)
            }

            return encoded_bits, params

    class ReedSolomonDecoder:
        """
            Classe interne pour le décodage des messages avec le code Reed-Solomon.

            Cette classe gère le décodage des messages et la correction d'erreurs.
        """
        def __init__(self):
            """
                Initialise un décodeur Reed-Solomon.
            """
            pass

        @staticmethod
        def _bits_to_bytes(bit_list):
            """
                Convertit une liste de bits en un tableau d'octets.

                Args:
                    bit_list (list): La liste de bits à convertir.

                Returns:
                    bytearray: Le tableau d'octets résultant.
            """
            bit_list = list(bit_list)
            if len(bit_list) % 8 != 0:
                remove = len(bit_list) - (len(bit_list) % 8)
                bit_list = bit_list[:remove]
            byte_array = bytearray()
            for i in range(0, len(bit_list), 8):
                byte_val = 0
                for j in range(8):
                    byte_val = (byte_val << 1) | bit_list[i + j]
                byte_array.append(byte_val)
            return byte_array

        @staticmethod
        def _bytes_to_bits(byte_data):
            """
                Convertit un tableau d'octets en une liste de bits.

                Args:
                    byte_data (bytearray): Le tableau d'octets à convertir.

                Returns:
                    list: La liste de bits résultante.
            """
            bit_list = []
            for b in byte_data:
                for i in range(7, -1, -1):
                    bit_list.append((b >> i) & 1)
            return bit_list

        def decode(self, received_bits, params):
            """
                Décode les bits reçus en utilisant le code Reed-Solomon.

                Args:
                    received_bits (list): Les bits reçus à décoder.
                    params (dict): Les paramètres d'encodage nécessaires pour le décodage.

                Returns:
                    list: Le message décodé.
            """
            n_parity = params.get('n_parity')
            if n_parity == 0:
                return received_bits
            pad_bits = params.get('pad_bits')
            message_len_bytes = params.get('message_len_bytes')

            rs_codec = reedsolo.RSCodec(n_parity)

            received_bytes = self._bits_to_bytes(received_bits)

            try:
                decoded_bytes = rs_codec.decode(received_bytes)
                # `reedsolo.decode` peut parfois retourner une seule valeur, parfois un tuple selon la version
                if isinstance(decoded_bytes, tuple):
                    decoded_bytes = decoded_bytes[0]
            except reedsolo.ReedSolomonError as e:
                raise ValueError(f"Erreur lors du décodage Reed-Solomon : {e}")

            original_bytes = decoded_bytes[:message_len_bytes]
            original_bits = self._bytes_to_bits(original_bytes)
            if pad_bits:
                original_bits = original_bits[:-pad_bits]

            return np.array(original_bits, dtype=int)


if __name__ == "__main__":
    message_bits = [
        0, 1, 0, 1, 0, 1, 0, 0,  # T
        0, 1, 0, 0, 0, 1, 0, 1,  # E
        0, 1, 0, 1, 0, 0, 1, 1,  # S
        0, 1, 0, 1, 0, 1, 0, 0  # T
    ]

    RS = ReedSolomon(message_bits)

    print("Message original (bits):")
    print(message_bits)

    encoded_bits, enc_params = RS.encode()

    print("\nMessage encodé (bits) :")
    print(encoded_bits)
    print("Paramètres d'encodage :", enc_params)

    received_bits = encoded_bits.copy()
    received_bits[7] ^= 1
    received_bits[1] ^= 1
    received_bits[17] ^= 1
    received_bits[22] ^= 1
    received_bits[37] ^= 1

    RS.modifier_message(received_bits)
    try:
        decoded_bits = RS.decode(enc_params)

        print("\nMessage décodé (bits) :")
        print(decoded_bits.tolist())

        if decoded_bits.tolist() == message_bits:
            print("\n✅ Le décodage a réussi, le message original est récupéré.")
        else:
            print("\n❌ Le décodage a échoué ou le message a été modifié.")
    except ValueError as e:
        print("\n❌ Erreur lors du décodage :", e)
