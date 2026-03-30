import array
import math

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Bit-level compression using Elias Gamma Encoding.

    Elias Gamma encodes a positive integer n as:
      1. N = floor(log2(n))
      2. Write N zeros
      3. Write the N+1 bit binary representation of n
    
    Example: 13 -> N = floor(log2(13)) = 3
             Output: 000 1101 = 0001101 (7 bits)

    Since Elias Gamma only works for positive integers (>= 1),
    we offset gap values by +1 during encoding (gaps can be 0)
    and -1 during decoding.

    For TF values (which are already >= 1), no offset is needed.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!
    """

    @staticmethod
    def elias_gamma_encode_number(n):
        """
        Encode a single positive integer n (>= 1) using Elias Gamma coding.

        Parameters
        ----------
        n : int
            Positive integer >= 1 to encode

        Returns
        -------
        List[int]
            List of bits (0s and 1s)
        """
        if n <= 0:
            raise ValueError(f"Elias Gamma requires n >= 1, got {n}")
        
        N = int(math.floor(math.log2(n)))
        # N zeros
        bits = [0] * N
        # binary representation of n in (N+1) bits
        binary_repr = []
        val = n
        for _ in range(N + 1):
            binary_repr.insert(0, val % 2)
            val //= 2
        bits.extend(binary_repr)
        return bits

    @staticmethod
    def elias_gamma_decode_stream(bits):
        """
        Decode a stream of bits back into a list of positive integers.

        Parameters
        ----------
        bits : List[int]
            List of bits

        Returns
        -------
        List[int]
            Decoded positive integers
        """
        numbers = []
        i = 0
        while i < len(bits):
            # Count leading zeros
            N = 0
            while i < len(bits) and bits[i] == 0:
                N += 1
                i += 1
            
            if i >= len(bits):
                break

            # Read N+1 bits as the number
            if i + N + 1 > len(bits):
                break
            
            val = 0
            for j in range(N + 1):
                val = val * 2 + bits[i]
                i += 1
            numbers.append(val)
        
        return numbers

    @staticmethod
    def bits_to_bytes(bits):
        """
        Convert a list of bits to a bytestream.
        Pads the last byte with zeros and prepends a byte
        indicating how many padding bits were added.

        Parameters
        ----------
        bits : List[int]

        Returns
        -------
        bytes
        """
        # Calculate padding needed
        padding = (8 - (len(bits) % 8)) % 8
        padded_bits = bits + [0] * padding

        # First byte stores the number of padding bits
        result = [padding]
        for i in range(0, len(padded_bits), 8):
            byte_val = 0
            for j in range(8):
                byte_val = byte_val * 2 + padded_bits[i + j]
            result.append(byte_val)
        
        return array.array('B', result).tobytes()

    @staticmethod
    def bytes_to_bits(encoded_bytestream):
        """
        Convert a bytestream back to a list of bits,
        removing the padding indicated by the first byte.

        Parameters
        ----------
        encoded_bytestream : bytes

        Returns
        -------
        List[int]
        """
        decoded = array.array('B')
        decoded.frombytes(encoded_bytestream)
        byte_list = decoded.tolist()

        if len(byte_list) == 0:
            return []

        padding = byte_list[0]
        bits = []
        for byte_val in byte_list[1:]:
            for j in range(7, -1, -1):
                bits.append((byte_val >> j) & 1)

        # Remove padding bits from the end
        if padding > 0:
            bits = bits[:-padding]
        
        return bits

    @staticmethod
    def elias_gamma_encode(list_of_numbers):
        """
        Encode a list of positive integers (>= 1) using Elias Gamma.

        Parameters
        ----------
        list_of_numbers : List[int]

        Returns
        -------
        bytes
        """
        all_bits = []
        for n in list_of_numbers:
            all_bits.extend(EliasGammaPostings.elias_gamma_encode_number(n))
        return EliasGammaPostings.bits_to_bytes(all_bits)

    @staticmethod
    def elias_gamma_decode(encoded_bytestream):
        """
        Decode a bytestream of Elias Gamma encoded numbers.

        Parameters
        ----------
        encoded_bytestream : bytes

        Returns
        -------
        List[int]
        """
        bits = EliasGammaPostings.bytes_to_bits(encoded_bytestream)
        return EliasGammaPostings.elias_gamma_decode_stream(bits)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes menggunakan
        Elias Gamma Encoding. Diubah dulu ke gap-based list,
        lalu setiap gap di-offset +1 (karena gap bisa 0, tapi
        Elias Gamma butuh >= 1).

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings), sorted ascending

        Returns
        -------
        bytes
        """
        # Convert to gap-based
        gap_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_list.append(postings_list[i] - postings_list[i - 1])
        
        # Offset by +1 to ensure all values >= 1
        offset_list = [g + 1 for g in gap_list]
        return EliasGammaPostings.elias_gamma_encode(offset_list)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari stream of bytes (Elias Gamma).
        Decode -> offset -1 -> reconstruct from gaps.

        Parameters
        ----------
        encoded_postings_list: bytes

        Returns
        -------
        List[int]
        """
        offset_list = EliasGammaPostings.elias_gamma_decode(encoded_postings_list)
        # Remove offset
        gap_list = [g - 1 for g in offset_list]
        # Reconstruct original postings
        postings = [gap_list[0]]
        for i in range(1, len(gap_list)):
            postings.append(postings[-1] + gap_list[i])
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menggunakan Elias Gamma.
        TF values are already >= 1, so no offset needed.

        Parameters
        ----------
        tf_list: List[int]

        Returns
        -------
        bytes
        """
        return EliasGammaPostings.elias_gamma_encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari stream of bytes.

        Parameters
        ----------
        encoded_tf_list: bytes

        Returns
        -------
        List[int]
        """
        return EliasGammaPostings.elias_gamma_decode(encoded_tf_list)


if __name__ == '__main__':
    
    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded TF list    : ", len(encoded_tf_list), "bytes")
        
        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, \
            f"hasil decoding tidak sama dengan postings original: {decoded_posting_list} != {postings_list}"
        assert decoded_tf_list == tf_list, \
            f"hasil decoding tidak sama dengan TF original: {decoded_tf_list} != {tf_list}"
        print()
