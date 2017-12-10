import numpy as np
import nltk
import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords as sw
from nltk.cluster.util import cosine_distance
from nltk.stem.porter import PorterStemmer
from operator import itemgetter
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from scraper import Scraper

TEXT_RANK = 'text_rank'

INDONESIAN = 'indonesian'
ENGLISH = 'english'


class TextRankSummarizer(object):
    def __init__(self, language):
        dir_path = os.path.dirname(os.path.realpath(__file__)) + '/nltk_data/'
        nltk.data.path = [dir_path]

        self.stopwords = sw.words(language)
        self.scraper = Scraper(language)
        self.language = language
        if language == INDONESIAN:
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
        else:
            self.stemmer = PorterStemmer()

    def sentence_similarity(self, sentence1, sentence2):
        if self.stopwords is None:
            self.stopwords = []

        sentence1 = [self.stemmer.stem(w.lower()) for w in word_tokenize(sentence1)]
        sentence2 = [self.stemmer.stem(w.lower()) for w in word_tokenize(sentence2)]

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
            if similarity_matrix[idx].sum() != 0:
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
        for ch in ['&',':','-','+','.',',']:
            query = query.replace(ch,' ')
        words = word_tokenize(query.lower())
        filtered_words = [word for word in words if word not in self.stopwords and word.isalpha()]
        new_query = " ".join(filtered_words)
        print("new query : " + new_query)
        suggested_query, status, lang = self.scraper.get_query(new_query)
        if status == -1:
            suggested_query, status, lang = self.scraper.get_query(new_query,isInverse=True)
        if status == -1:
            suggested_query, status, lang = self.scraper.get_query(new_query)

        text = self.scraper.get_intro_lang(suggested_query, lang)

        if not text:
            if self.language == INDONESIAN:
                return "mohon maaf {q} tidak ditemukan".format(q=query)
            else:
                return "{q} not found".format(q=query)

        sentences = sent_tokenize(text)
        similarity_matrix = self.build_similarity_matrix(sentences)
        sentence_ranks = self.page_rank(similarity_matrix)
        ranked_sentence_indexes = [item[0] for item in sorted(enumerate(sentence_ranks), key=lambda item: -item[1])]
        selected_sentences = sorted(ranked_sentence_indexes[:size])

        summary = itemgetter(*selected_sentences)(sentences)

        if isinstance(summary, tuple):
            if status == 0:
                return ' '.join(summary)
            elif lang == self.language:
                res = ' '.join(summary)
                if lang == INDONESIAN:
                    return "mungkin maksud anda adalah {sq}\n{s}".format(sq=suggested_query, s=res)
                else:
                    return "maybe this is what you want {sq}\n{s}".format(sq=suggested_query, s=res)
            else:
                return summary

        if status == 0:
            return summary
        elif lang == self.language:
            if lang == INDONESIAN:
                return "mungkin maksud anda adalah {sq}\n{s}".format(sq=suggested_query, s=summary)
            else:
                return "maybe this is what you want {sq}\n{s}".format(sq=suggested_query, s=res)
        else:
            return summary

class Summarizer():
    def __init__(self):
        self.english_text_rank_summarizer = TextRankSummarizer(ENGLISH)
        self.indonesian_text_rank_summarizer = TextRankSummarizer(INDONESIAN)

    def summarize(self, type, language, query, size):
        if type == TEXT_RANK:
            if language == INDONESIAN:
                return self.indonesian_text_rank_summarizer.summarize(query, size)
            else:
                return self.english_text_rank_summarizer.summarize(query, size)
        return None
