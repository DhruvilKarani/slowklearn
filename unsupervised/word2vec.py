import numpy as np 
from itertools import chain
import sys
sys.path.append('../loss')
sys.path.append('../')
from loss.functions import cat_crossentropy

class Word2Vec:
    def __init__(self, n_dim = 100):
        self.n_dim = 100
        self.vocab_ = None
        self._word2index = None
        self._index2word = None

    @staticmethod
    def _vocab(sentences):
        assert len(sentences)>0, "Empty list"
        return list(set(chain.from_iterable(sentences)))

    @staticmethod
    def word2index(words):
        return {word:i for i, word in enumerate(words)}
    
    @staticmethod
    def _onehot(size, idxs):
        vec = np.zeros((1, size), dtype=np.float32)
        for idx in idxs:
            vec[0, idx] = 1
        return vec


    @staticmethod
    def _softmax(vector):
        vector = np.exp(vector.ravel())
        return vector/np.sum(vector)


    def _forward(self, x, h, u):
        x = x.reshape(len(self.vocab_), -1)
        ht = np.matmul(h.T, x)
        return np.matmul(u.T, ht)
    

    def _grad_u(self, l, x, h, u):
        grad = np.zeros_like(u)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                grad[i,j] = (-1 + np.exp(-l))*(np.matmul((h[:,i]).reshape(1,-1), x.reshape(-1,1)))
        return grad

    def _grad_h(self, probs, y, x, h, u):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        probs = probs.reshape(-1,1)
        grad = np.zeros_like(h)
        print(x.shape, y.shape, probs.shape, grad.shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                grad[i,j] = np.sum(np.multiply(np.multiply(probs-y,u[j,:].reshape(-1,1)), x))
        return grad

    def _idxed_sentences(self, sentences):
        self._word2index = self.word2index(self.vocab_)
        # self._index2word = {value: key for key, value in self._word2index.items()}
        return list(map(lambda x: [self._word2index[token] for token in x], sentences))


    def fit_cbow(self, sentences, lr=10e-3, n_iters=10):
        sentences = list(filter(lambda x: len(x)>0, sentences))
        self.vocab_ = self._vocab(sentences)
        sent_idx = self._idxed_sentences(sentences)
        h = np.random.random((len(self.vocab_), self.n_dim))
        u = np.random.random((self.n_dim, len(self.vocab_)))

        for iter in range(n_iters):
            for sentence in sent_idx:
                for idx in range(len(sentence)):
                #complete
                x = self._onehot(len(self.vocab_), sentence[:-1])
                y = self._onehot(len(self.vocab_), sentence[-1:])
                y_pred = self._forward(x, h, u)
                y_probs = self._softmax(y_pred)
                L = cat_crossentropy(y, y_probs)
                grad_u = self._grad_u(L, x, h, u)
                grad_h = self._grad_h(y_probs, y, x, h, u)
                u -= lr*grad_u
                h -= lr*grad_h
                print(L)



         



if __name__ == '__main__':
    wordvec = Word2Vec()
    sentences = [['how', 'are', 'you'], ['we', 'are', 'great']]
    print(wordvec.fit_cbow(sentences))