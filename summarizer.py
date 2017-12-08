from operator import itemgetter
import wikipedia
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords as sw
from nltk.cluster.util import cosine_distance
from nltk.stem.porter import PorterStemmer
from scraper import Scraper


class TextRankSummarizer(object):
    def __init__(self, language=None):
        self.stopwords = sw.words(language)
        self.stemmer = PorterStemmer()

    def sentence_similarity(self, sentence1, sentence2):
        if self.stopwords is None:
            self.stopwords = []

        sentence1 = [self.stemmer.stem(w.lower()) for w in sentence1]
        sentence2 = [self.stemmer.stem(w.lower()) for w in sentence2]

        all_words = list(set(sentence1 + sentence2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for w in sentence1:
            if w in self.stopwords:
                continue
            vector1[all_words.index(w)] += 1

        for w in sentence2:
            if w in self.stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def build_similarity_matrix(self, sentences):
        similarity_matrix = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                similarity_matrix[idx1][idx2] = self.sentence_similarity(sentences[idx1], sentences[idx2])

        for idx in range(len(similarity_matrix)):
            similarity_matrix[idx] /= similarity_matrix[idx].sum()

        return similarity_matrix

    def page_rank(self, similarity_matrix, eps=0.0001, d=0.85):
        probs = np.ones(len(similarity_matrix)) / len(similarity_matrix)
        while True:
            new_probs = np.ones(len(similarity_matrix)) * (1 - d) / len(
                similarity_matrix) + d * similarity_matrix.T.dot(probs)
            delta = abs((new_probs - probs).sum())
            if delta <= eps:
                return new_probs
            probs = new_probs

    def summarize(self, query, size=1):
        scraper = Scraper()
        text = scraper.get_intro(query)
        sentences = sent_tokenize(text)
        similarity_matrix = self.build_similarity_matrix(sentences)
        sentence_ranks = self.page_rank(similarity_matrix)
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:size])
        print(selected_sentences)
        summary = itemgetter(*selected_sentences)(sentences)

        if isinstance(summary, tuple):
            return ' '.join(summary)

        return summary

class Summarizer():
    def __init__(self, language='english'):
        self.text_rank_summarizer = TextRankSummarizer(language)

    def summarize(self, type, query, size):
        if type == 'text_rank':
            return self.text_rank_summarizer.summarize(query, size)

        return None
