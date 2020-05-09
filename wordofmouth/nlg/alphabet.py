import numpy as np  # type: ignore
import random

from typing import List, Set


class Alphabet:
    """
    Handle the alphabet recognized and served by our model.

    We need to maps letters to numbers and back to letters when
    learning RNN.  This class maintains this mapping.
    """

    def __init__(self, sentences: List[str]):
        """
        :param sentences: List of sentences to build the vocabulary
            for.
        """

        chars: Set[str] = set()
        for sentence in sentences:
            chars.update(sentence)

        self._padding_id = 0
        self._id_to_char = np.array(["<"] + sorted(chars))
        self._char_to_id = {c: i for (i, c) in enumerate(self._id_to_char)}

    def encode(self, text: str, *, textwidth=None):
        """Encode a text into a list of character ids.

        An exception will be thrown if the text contains characters
        that are not in the alphabet.

        :param text: Text to encode.
        :param padded: Whether the text should be padded or not.

        :return: Encoded text.
        """
        if textwidth:
            assert (
                len(text) <= textwidth
            ), f"Text length should be at most {textwidth}, got {len(text)}."

        encoded_text = [self._char_to_id[c] for c in text]

        if textwidth:
            padding_size = textwidth - len(text)
            padding = [self._padding_id] * padding_size
            encoded_text.extend(padding)

        return encoded_text

    def decode(self, encoded_text: List[int]):
        """Decode a list of letter ids to a word.

        :param encoded_text: Vector to decode.

        :return: Decoded text.
        """
        return "".join(
            (self._id_to_char[i] for i in encoded_text if i != self._padding_id)
        )

    def sample(self):
        """Sample one character from the alphabet.

        :return: A random character from the alphabet.
        """
        return random.choice(self._id_to_char[1:])

    @property
    def padding_id(self) -> int:
        """
        :return: The id of the padding character.
        """
        return self._padding_id

    def __len__(self) -> int:
        """
        :return: The size of this alphabet.
        """
        return len(self._id_to_char)
