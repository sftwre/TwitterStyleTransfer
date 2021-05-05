import nltk
import numpy as np
from torchtext.legacy import data
from torchtext.vocab import Vocab
from collections import Counter
from nltk.corpus import stopwords
# nltk.download('stopwords')

class TwitterDataset():

    def __init__(self, emb_dim=50, batch_size=32, tweet_len=280, gpu=False):

        self.gpu = gpu
        self.emb_dim = emb_dim
        self.batch_size = batch_size
        self.TEXT = data.Field(init_token='<start>', eos_token='<eos>', tokenize='spacy', fix_length=tweet_len)
        self.LABEL = data.Field(sequential=False, unk_token=None)

        vocabPath = './data/vocab.txt'
        trainPath = './data/tweets.train.txt'
        trainLabelsPath = './data/tweets.train.labels'

        testPath = './data/tweets.test.txt'
        testLabelsPath = './data/tweets.test.labels'

        # stopwords to remove from tweets
        # self.sw = set(stopwords.words('english'))

        # load vocab
        self.vocab = self._loadVocab(vocabPath)

        # load training data
        X_train, y_train = self._loadData(trainPath, trainLabelsPath)

        self.X_train = X_train
        self.y_train = y_train

        self.TEXT.vocab = self._buildVocab(self.X_train)
        self.LABEL.vocab = self._buildVocab(self.y_train, labels=True)

        # size of vocabulary
        self.vocab_size = len(self.TEXT.vocab.itos)

        # number of accounts dataset
        self.n_accounts = len(self.LABEL.vocab.itos) - 2

        # define training iterator
        self.trainIterator = None

    def _loadData(self, dataFile, labelsFile):
        """
        Loads twitter data and labels and apply filters.
        """
        # first element is twitter data, second are labels
        data = []
        for path in [dataFile, labelsFile]:
            with open(path, 'r', encoding='UTF-8') as file:
                text = file.read().split('\n')
                data.append(text)

        # remove stopwords and non-vocabulary words
        data[0] = np.array(list(map(self._cleanTweet, data[0])))
        data[1] = np.array(data[1])
        return data

    def _loadVocab(self, path):
        """
        Loads vocabulary file
        :param path:
        :return:
        """
        with open(path, 'r', encoding='UTF-8') as file:
            vocab = file.read().split()
            vocab = set(vocab)

        return vocab

    def _cleanTweet(self, tweet):
        # split into sentences
        tokens = tweet.strip().lower().split()

        # filter out words not in vocabulary
        tokens = list(filter(lambda x: x.isalpha() and x in self.vocab, tokens))

        # create new sentences
        return ' '.join(tokens)

    def _buildVocab(self, data, labels=False):

        counter = Counter()

        for x in data:
            words = x.split()
            counter.update(words)

        specials = ['<unk>', '<pad>', '<start>', '<eos>']
        vocab = Vocab(counter, specials=specials) if not labels else Vocab(counter, specials_first=False)

        return vocab

    def _dataIterator(self, text, labels):
        """
        Generator that yields batch_size items from a dataset
        :param data:
        :param batch_size:
        :return:
        """
        try:
            assert len(text) == len(labels)
        except AssertionError:
            print('Length of text data and labels must match',
                  f'text length: {len(text)}, labels length: {len(labels)}', sep='\n')
            exit(-1)

        # randomly shuffle data
        # TODO sort tweets by length, so the LSTM trains on batches of similair length
        idx = np.arange(text.shape[0])
        np.random.shuffle(idx)

        text = text[idx]
        labels = labels[idx]

        for start in range(0, len(text), self.batch_size):
            end = start + self.batch_size
            encodedText = self.TEXT.process(text[start:end])
            encodedLabels = self.LABEL.process(labels[start:end])

            if self.gpu:
                encodedText = encodedText.cuda()
                encodedLabels = encodedLabels.cuda()

            yield encodedText, encodedLabels

    def resetTrainBatches(self):
        self.trainIterator = self._dataIterator(self.X_train, self.y_train)

    def getVocabVectors(self):
        return self.TEXT.vocab.vectors

    def idxs2sentence(self, idxs):
        return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

    def idx2label(self, idx):
        return self.LABEL.vocab.itos[idx]







