"""
     Author: Gaetano Rossiello
     Email: gaetano.rossiello@uniba.it
"""
import base


class LeadSummarizer(base.BaseSummarizer):

    def __init__(self,
                 language='english',
                 stopwords_remove=True,
                 debug=False):
        super().__init__(language, stopwords_remove, debug)
        return

    def summarize(self, text, limit_type='word', limit=100):
        raw_sentences = self.sent_tokenize(text)
        count = 0
        sentences_summary = []
        for s in raw_sentences:
            if count > limit:
                break
            sentences_summary.append(s)
            if limit_type == 'word':
                count += len(s.split())
            else:
                count += len(s)

        summary = "\n".join([s for s in sentences_summary])
        return summary
