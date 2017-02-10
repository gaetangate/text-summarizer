"""
     Author: Gaetano Rossiello
     Email: gaetano.rossiello@uniba.it
"""
import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from gensim.models.word2vec import Word2Vec


class CentroidW2VSummarizer(base.BaseSummarizer):
    def __init__(self,
                 word2vec_model_path,
                 language='english',
                 stopwords_remove=True,
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.95,
                 reordering=True):
        super().__init__(language, stopwords_remove, debug)

        self.word2vec = Word2Vec.load_word2vec_format(word2vec_model_path, binary=True)
        self.index2word_set = set(self.word2vec.wv.index2word)
        self.word_vectors = dict()

        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        self.reordering = reordering
        return

    def get_topic_idf(self, sentences):
        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        # print(centroid_vector.max())

        feature_names = vectorizer.get_feature_names()
        word_list = []
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] > self.topic_threshold:
                print(feature_names[i], centroid_vector[i])
                word_list.append(feature_names[i])

        return word_list

    def word_vectors_cache(self, sentences):
        for s in sentences:
            words = s.split()
            for w in words:
                if w not in self.word_vectors and w in self.index2word_set:
                    self.word_vectors[w] = self.word2vec[w]
        return

    # Sentence representation with sum of word vectors
    def compose_vectors(self, vectors):
        composed_vector = np.zeros(self.word2vec.vector_size, dtype="float32")
        count = 0
        word_vectors_keys = set(self.word_vectors.keys())
        for vector in vectors:
            if vector in word_vectors_keys:
                composed_vector = composed_vector + self.word_vectors[vector]
                count += 1

        return composed_vector

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)
        centroid_words = self.get_topic_idf(clean_sentences)

        self.word_vectors_cache(clean_sentences)
        centroid_vector = self.compose_vectors(centroid_words)

        sentences_scores = []
        for i in range(len(clean_sentences)):
            words = clean_sentences[i].split()
            sentence_vector = self.compose_vectors(words)
            score = base.similarity(sentence_vector, centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, sentence_vector))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)
        # print(sentence_scores_sort)

        count = 0
        sentences_summary = []
        for s in sentence_scores_sort:
            if count > limit:
                break
            include_flag = True
            for ps in sentences_summary:
                sim = base.similarity(s[3], ps[3])
                # print(s[0], ps[0], sim)
                if sim > self.sim_threshold:
                    include_flag = False
            if include_flag:
                # print(s[0], s[1])
                sentences_summary.append(s)
                if limit_type == 'word':
                    count += len(s[1].split())
                else:
                    count += len(s[1])

        if self.reordering:
            sentences_summary = sorted(sentences_summary, key=lambda el: el[0], reverse=False)

        summary = "\n".join([s[1] for s in sentences_summary])

        return summary
