from gensim.models import KeyedVectors

from os.path import dirname

MODEL_PATH = dirname(dirname(dirname(__file__))) + "/word2vec_model/trmodel"


class Word2vec:
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
        self.model.init_sims(replace=True)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value
        self._model.init_sims(replace=True)

    def get_most_similar(self, word: str):
        return self.model.most_similar([word])

    def words_closer_than(self, word1: str, word2: str):
        return self.model.words_closer_than(word1, word2)

    def get_doesnt_match(self, word_list: list):
        return self.model.doesnt_match(word_list)

    def distance(self, word1: str, word2: str):
        return self.model.distance(word1, word2)

    def closer_than(self, word1: str, word2: str):
        return self.model.closer_than(word1, word2)

    def get_vocab(self):
        return self.model.vocab

    def distances(self, word: str, list_of_words: list):
        return self.model.distances(word, list_of_words)

    def get_vector(self, word: str):
        return self.model.get_vector(word)

    def get_most_similar_to_given(self, word: str, list_of_words: list):
        return self.model.most_similar_to_given(word, list_of_words)

    def cosine_similarity(self, word: str, list_of_words: list):
        wc_list = []
        wc = self.get_vector(word)
        for w in list_of_words:
            _wc = self.get_vector(w)
            wc_list.append(_wc)

        return self.model.cosine_similarities(wc, wc_list)

    def similar_by_word(self, word: str, top_n: int = 10):
        return self.model.similar_by_word(word, top_n)

    def similarity(self, w1: str, w2: str):
        return self.model.similarity(w1, w2)

    def get_document_similarity(self, doc1, doc2):
        d1 = [w for w in doc1.split()]
        d2 = [w for w in doc2.split()]
        return self.model.wmdistance(d1, d2)
