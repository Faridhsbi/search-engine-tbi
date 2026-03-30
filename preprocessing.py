"""
preprocessing.py

Centralized text preprocessing pipeline for the search engine.
Provides tokenization, stopword removal, and Porter stemming.

All retrieval and indexing code should use preprocess() to ensure
consistent text normalization.
"""

import re
import os
import pickle

# --------------------------------------------------------------------------- #
#  Porter Stemmer – implemented from scratch (no external dependency needed)  #
# --------------------------------------------------------------------------- #

class PorterStemmer:
    """
    A compact, self-contained implementation of the Porter Stemming Algorithm
    (M.F. Porter, 1980 – "An algorithm for suffix stripping").

    Reference: https://tartarus.org/martin/PorterStemmer/def.txt
    """

    def __init__(self):
        # vowels
        self.vowels = frozenset("aeiou")

    # helpers #

    def _is_consonant(self, word, i):
        """Return True if word[i] is a consonant."""
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem):
        """
        Return the 'measure' of a stem – the number of VC sequences.
        [C](VC){m}[V]  → m
        """
        cv = ""
        for i in range(len(stem)):
            cv += "C" if self._is_consonant(stem, i) else "V"
        return cv.count("VC")

    def _has_vowel(self, stem):
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _ends_double_consonant(self, word):
        if len(word) >= 2 and word[-1] == word[-2]:
            return self._is_consonant(word, len(word) - 1)
        return False

    def _cvc(self, word):
        """*o – the stem ends cvc, where the second c is not W, X or Y."""
        if len(word) >= 3:
            if (self._is_consonant(word, len(word) - 1)
                    and not self._is_consonant(word, len(word) - 2)
                    and self._is_consonant(word, len(word) - 3)):
                if word[-1] not in ('w', 'x', 'y'):
                    return True
        return False

    def _replace_suffix(self, word, suffix, replacement, m_gt=None):
        """If word ends with suffix and measure > m_gt, replace."""
        if word.endswith(suffix):
            stem = word[:-len(suffix)]
            if m_gt is None or self._measure(stem) > m_gt:
                return stem + replacement
            return word
        return None

    # steps

    def _step1a(self, word):
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        return word

    def _step1b(self, word):
        flag = False
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]
            return word
        elif word.endswith("ed"):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                flag = True
            else:
                return word
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                flag = True
            else:
                return word

        if flag:
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                return word + "e"
            if self._ends_double_consonant(word) and word[-1] not in ('l', 's', 'z'):
                return word[:-1]
            if self._measure(word) == 1 and self._cvc(word):
                return word + "e"
        return word

    def _step1c(self, word):
        if word.endswith("y"):
            stem = word[:-1]
            if self._has_vowel(stem):
                return stem + "i"
        return word

    def _step2(self, word):
        pairs = [
            ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
            ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
            ("alli", "al"), ("entli", "ent"), ("eli", "e"),
            ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
            ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
            ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
            ("iviti", "ive"), ("biliti", "ble"),
        ]
        for suffix, replacement in pairs:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step3(self, word):
        pairs = [
            ("icate", "ic"), ("ative", ""), ("alize", "al"),
            ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", ""),
        ]
        for suffix, replacement in pairs:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step4(self, word):
        suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible",
            "ant", "ement", "ment", "ent", "ion", "ou", "ism",
            "ate", "iti", "ous", "ive", "ize",
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:
                    if suffix == "ion":
                        if len(stem) > 0 and stem[-1] in ('s', 't'):
                            return stem
                    else:
                        return stem
                return word
        return word

    def _step5a(self, word):
        if word.endswith("e"):
            stem = word[:-1]
            if self._measure(stem) > 1:
                return stem
            if self._measure(stem) == 1 and not self._cvc(stem):
                return stem
        return word

    def _step5b(self, word):
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word[-1] == 'l':
            return word[:-1]
        return word

    def stem(self, word):
        """Stem a single word using the Porter algorithm."""
        if len(word) <= 2:
            return word
        word = word.lower()
        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)
        return word


# --------------------------------------------------------------------------- #
#  Stopwords list (standard English IR stopwords)                              #
# --------------------------------------------------------------------------- #

ENGLISH_STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can't",
    "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having",
    "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's",
    "will", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves",
    # extra common IR stopwords
    "also", "just", "like", "well", "back", "even", "still", "way",
    "take", "since", "another", "however", "two", "three",
})

# Singleton stemmer
_stemmer = PorterStemmer()


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #

def tokenize(text):
    """
    Tokenize text into a list of lowercase alphabetic tokens.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str]
    """
    return re.findall(r'[a-zA-Z]+', text.lower())


def remove_stopwords(tokens):
    """
    Remove English stopwords from a token list.

    Parameters
    ----------
    tokens : List[str]

    Returns
    -------
    List[str]
    """
    return [t for t in tokens if t not in ENGLISH_STOP_WORDS]


def stem_tokens(tokens):
    """
    Apply Porter stemming to every token.

    Parameters
    ----------
    tokens : List[str]

    Returns
    -------
    List[str]
    """
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text):
    """
    Full preprocessing pipeline: tokenize → remove stopwords → stem.

    Parameters
    ----------
    text : str

    Returns
    -------
    List[str]
        Preprocessed tokens
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens


if __name__ == "__main__":
    # quick test
    sample = "The crystalline lens in vertebrates, including humans."
    print("Original :", sample)
    print("Tokenized:", tokenize(sample))
    print("No stops :", remove_stopwords(tokenize(sample)))
    print("Stemmed  :", preprocess(sample))
