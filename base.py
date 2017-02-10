"""
    Author: Gaetano Rossiello
    Email: gaetano.rossiello@uniba.it
"""
import string
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine


def similarity(v1, v2):
    score = 0.0
    if np.count_nonzero(v1) != 0 and np.count_nonzero(v2) != 0:
        score = ((1 - cosine(v1, v2)) + 1) / 2
    return score


class BaseSummarizer:

    extra_stopwords = ["''", "``", "'s"]

    def __init__(self,
                 language='english',
                 stopwords_remove=True,
                 debug=False):
        self.language = language
        self.stopwords_remove = stopwords_remove
        self.debug = debug
        return

    def sent_tokenize(self, text):
        return sent_tokenize(text, self.language)

    def preprocess_text(self, text):
        sentences = self.sent_tokenize(text)
        sentences_cleaned = []
        for sent in sentences:
            words = word_tokenize(sent, self.language)
            words = [w for w in words if w not in string.punctuation]
            words = [w for w in words if w not in self.extra_stopwords]
            words = [w.lower() for w in words]
            if self.stopwords_remove:
                stops = set(stopwords.words(self.language))
                words = [w for w in words if w not in stops]
            sentences_cleaned.append(" ".join(words))
        return sentences_cleaned

    def summarize(self, text, limit_type='word', limit=100):
        raise NotImplementedError("Abstract method")
