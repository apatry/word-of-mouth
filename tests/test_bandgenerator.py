import unittest

from wordofmouth.nlg.alphabet import Alphabet


class AlphabetTests(unittest.TestCase):
    def test_empty_alphabet(self):
        alphabet = Alphabet([])

        # the alphabet should only contain the padding character
        self.assertEqual(
            len(alphabet), 1, "Only the padding character should be in the alphabet."
        )
        self.assertEqual(
            len(alphabet.encode("", textwidth=10)),
            10,
            "Padding should be added for the empty string.",
        )
        self.assertEqual(
            alphabet.decode(alphabet.encode("")),
            "",
            "Padding should be ignored when decoding.",
        )

    def test_alphabet(self):
        alphabet = Alphabet(["roses", "are" "red"])

        with self.assertRaises(KeyError) as cm:
            alphabet.encode("unknown letters")

        self.assertEqual(
            alphabet.decode(alphabet.encode("redroses")),
            "redroses",
            "Decoding should revert the word that was encoded.",
        )

        self.assertEqual(
            len(alphabet.encode("roses", textwidth=10)),
            10,
            "Padding should be added at the end.",
        )

        self.assertEqual(
            alphabet.decode(alphabet.encode("roses", textwidth=10)),
            "roses",
            "Padding shouldn't be included when decoding a word.",
        )
