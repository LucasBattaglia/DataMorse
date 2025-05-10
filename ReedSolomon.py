import reedsolo
import numpy as np

class ReedSolomon:
    def __init__(self, message):
        self.message = message

    def encode(self):
        encoder = self.ReedSolomonEncoder(0.2)
        return encoder.encode(self.message)

    def decode(self, param):
        decoder = self.ReedSolomonDecoder()
        return decoder.decode(self.message, param)

    def modifier_message(self, message):
        self.message = message

    class ReedSolomonEncoder:
        def __init__(self, pourcentage):
            self.pourcentage = pourcentage

        def _bits_to_bytes(self, bit_list):
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

        def _bytes_to_bits(self, byte_data):
            bit_list = []
            for b in byte_data:
                for i in range(7, -1, -1):
                    bit_list.append((b >> i) & 1)
            return bit_list

        def encode(self, bit_list):
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
        def __init__(self):
            pass

        def _bits_to_bytes(self, bit_list):
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

        def _bytes_to_bits(self, byte_data):
            bit_list = []
            for b in byte_data:
                for i in range(7, -1, -1):
                    bit_list.append((b >> i) & 1)
            return bit_list

        def decode(self, received_bits, params):
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
        0, 1, 0, 1, 0, 1, 0, 0   # T
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
