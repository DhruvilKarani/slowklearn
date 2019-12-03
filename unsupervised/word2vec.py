import numpy as np 
from itertools import chain

class Word2Vec:
    def __init__(self, n_dim = 100):
        self.n_dim = 100
        self.vocab_ = None
        self._word2index = None
        self._index2word = None

    @staticmethod
    def vocab(sentences):
        assert len(sentences)>0, "Empty list"
        return list(set(chain.from_iterable(sentences)))

    @staticmethod
    def word2index(words):
        return {word:i for i, word in enumerate(words)}
    
    @staticmethod
    def onehot(size, idx):
        vec = np.zeros((1, size), dtype=np.float32)
        vec[0, idx] = 1
        return vec
    
    def fit(self, sentences):
        sentences = list(filter(lambda x: len(x)>0, sentences))
        self.vocab_ = self.vocab(sentences)
        self._word2index = self.word2index(self.vocab_)
        self._index2word = {value: key for key, value in self._word2index.items()}
        sent_idx = map(lambda x: [self._word2index[token] for token in x], sentences)
        



if __name__ == '__main__':
    wordvec = Word2Vec()
    sentences = [['how', 'are', 'you'], ['we', 'are', 'great']]
    print(wordvec.fit(sentences))
    print(wordvec._word2index)