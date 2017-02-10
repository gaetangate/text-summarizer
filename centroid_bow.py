"""
     Author: Gaetano Rossiello
     Email: gaetano.rossiello@uniba.it
"""
import base
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class CentroidBOWSummarizer(base.BaseSummarizer):

    def __init__(self,
                 language='english',
                 stopwords_remove=True,
                 debug=False,
                 topic_threshold=0.3,
                 sim_threshold=0.95):
        super().__init__(language, stopwords_remove, debug)
        self.topic_threshold = topic_threshold
        self.sim_threshold = sim_threshold
        return

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        clean_sentences = self.preprocess_text(text)

        vectorizer = CountVectorizer()
        sent_word_matrix = vectorizer.fit_transform(clean_sentences)

        transformer = TfidfTransformer(norm=None, sublinear_tf=False, smooth_idf=False)
        tfidf = transformer.fit_transform(sent_word_matrix)
        tfidf = tfidf.toarray()

        centroid_vector = tfidf.sum(0)
        centroid_vector = np.divide(centroid_vector, centroid_vector.max())
        for i in range(centroid_vector.shape[0]):
            if centroid_vector[i] <= self.topic_threshold:
                centroid_vector[i] = 0

        sentences_scores = []
        for i in range(tfidf.shape[0]):
            score = base.similarity(tfidf[i, :], centroid_vector)
            sentences_scores.append((i, raw_sentences[i], score, tfidf[i, :]))

        sentence_scores_sort = sorted(sentences_scores, key=lambda el: el[2], reverse=True)

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

        summary = "\n".join([s[1] for s in sentences_summary])
        return summary
