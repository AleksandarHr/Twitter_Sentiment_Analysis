import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from abc import ABC


class TextNet(ABC):
    def word2vec(self, word):
        """
        maps a word to its embedding
        @param word: single word as a string
        @return: corresponding embedding of the input word
        """
        if word in self.word_indexer:
            idx = torch.LongTensor([self.word_indexer[word]]).to(self.device)
            return self.embedding(idx)

    def tweet2vec(self, tweet):
        """
        maps a tweet to a tensor of its words embeddings
        @param tweet: single tweet as a string
        @return: tensor of dimension [number of words, embedding dimension]
        """
        tweet = tweet.split(' ')
        tweet = list(filter(lambda word: word in self.word_indexer, tweet))
        tweet = list(map(lambda word: self.word2vec(word), tweet))
        return torch.cat(tweet)

    def tweets2vec(self, tweets):
        """
        maps a batch of tweets to their respective embeddings
        @param tweets: batch of tweets each as a string
        @return: tensor of dimension [number of tweets, largest number of words per tweet, embedding dimension]
        """
        tweets = list(map(lambda tweet: self.tweet2vec(tweet), tweets))
        tweets = pad_sequence(tweets, batch_first=True)
        return tweets

    def forward_rnn(self, tweets):
        """
        maps a batch of tweets to a batch of sentiment predictions
        @param tweets: batch of tweets each as a string
        @return: tensor of predictions of dimension [batch size, number of different sentiment (2)]
        """
        tweets = self.tweets2vec(tweets)
        if self.is_lstm:
            output, (hidden, cell) = self.rnn(tweets)
        else:
            output, hidden = self.rnn(tweets)

        if self.bidirectional:
            x = self.fc1(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            x = self.fc1(hidden[-1])
        x = self.dropout1(x)
        x = self.bn1(F.relu(x))
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x


class TextLeNetCNN(nn.Module, TextNet):
    def __init__(self, embedding_matrix, embedding_dim, n_filters, filters_size, n_conv_layers, device, word_indexer,
                 dropout):
        """
        initialize attributes of the TextLeNetCNN class
        original paper: https://arxiv.org/abs/1408.5882 (Convolutional Neural Networks for Sentence Classification)
        @param embedding_matrix: pretrained embedding matrix of dimension [number of words, embedding dimension]
        @param embedding_dim: embedding dimension
        @param n_filters: number of different filters per convolutional layer
        @param filters_size: number of words a given convolutional layer uses
        @param n_conv_layers: number of different convolutional layers
        @param device: either 'cpu' or 'cuda' depending on hardware availability
        @param word_indexer: dictionary mapping a word to an index in the embedding matrix
        @param dropout: keep probability of the dropout layers
        """
        super().__init__()
        self.word_indexer = word_indexer
        self.device = device
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=False)
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                    out_channels=n_filters,
                                                    kernel_size=(filter_size, embedding_dim))
                                          for filter_size in filters_size])

        # Linear layers
        self.fc1 = nn.Linear(in_features=n_filters * n_conv_layers, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=2)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(256)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Final activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tweets):
        """
        maps a batch of tweets to a batch of sentiment predictions
        @param tweets: batch of tweets each as a string
        @return: tensor of predictions of dimension [batch size, number of different sentiment (2)]
        """
        tweets = self.tweets2vec(tweets).unsqueeze(1)
        f_maps = [F.relu(conv_layer(tweets).squeeze(-1)) for conv_layer in self.conv_layers]
        pooled = [F.max_pool1d(f_map, f_map.shape[2]).squeeze(-1) for f_map in f_maps]

        # Stack along channel dimension
        x = torch.cat(pooled, dim=-1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(F.relu(x))
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.softmax(x)
        return x


class SimpleRNN(nn.Module, TextNet):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, output_dim, device, word_indexer, dropout,
                 bidirectional):
        """
        initialize attributes of the SimpleRNN class
        @param embedding_matrix: pretrained embedding matrix of dimension [number of words, embedding dimension]
        @param embedding_dim: embedding dimension
        @param hidden_dim: hidden dimension of the RNN layer
        @param output_dim: dimension of the output, corresponds to the number of different sentiment (2)
        @param device: either 'cpu' or 'cuda' depending on hardware availability
        @param word_indexer: dictionary mapping a word to an index in the embedding matrix
        @param dropout: keep probability of the dropout layers
        @param bidirectional: boolean indicating if a tweet is processed in forward and reverse mode w.r.t the words
        """
        super().__init__()
        self.word_indexer = word_indexer
        self.device = device
        self.bidirectional = bidirectional
        self.is_lstm = False
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=False)
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)

        # Linear layers
        if bidirectional:
            self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Final activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tweets):
        """
        maps a batch of tweets to a batch of sentiment predictions
        @param tweets: batch of tweets each as a string
        @return: tensor of predictions of dimension [batch size, number of different sentiment (2)]
        """
        return self.forward_rnn(tweets)


class SimpleLSTM(nn.Module, TextNet):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, output_dim, device, word_indexer, dropout,
                 bidirectional):
        """
        initialize attributes of the SimpleLSTM class
        @param embedding_matrix: pretrained embedding matrix of dimension [number of words, embedding dimension]
        @param embedding_dim: embedding dimension
        @param hidden_dim: hidden dimension of the LSTM layer
        @param output_dim: dimension of the output, corresponds to the number of different sentiment (2)
        @param device: either 'cpu' or 'cuda' depending on hardware availability
        @param word_indexer: dictionary mapping a word to an index in the embedding matrix
        @param dropout: keep probability of the dropout layers
        @param bidirectional: boolean indicating if a tweet is processed in forward and reverse mode w.r.t the words
        """
        super().__init__()
        self.word_indexer = word_indexer
        self.device = device
        self.bidirectional = bidirectional
        self.is_lstm = True
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=False)
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_dim,
                           num_layers=1,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=bidirectional)

        # Linear layers
        if bidirectional:
            self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Final activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tweets):
        """
        maps a batch of tweets to a batch of sentiment predictions
        @param tweets: batch of tweets each as a string
        @return: tensor of predictions of dimension [batch size, number of different sentiment (2)]
        """
        return self.forward_rnn(tweets)


class GRUNet(nn.Module, TextNet):
    def __init__(self, embedding_matrix, embedding_dim, hidden_dim, output_dim, device, word_indexer, dropout,
                 bidirectional):
        """
        initialize attributes of the GRUNet class
        @param embedding_matrix: pretrained embedding matrix of dimension [number of words, embedding dimension]
        @param embedding_dim: embedding dimension
        @param hidden_dim: hidden dimension of the LSTM layer
        @param output_dim: dimension of the output, corresponds to the number of different sentiment (2)
        @param device: either 'cpu' or 'cuda' depending on hardware availability
        @param word_indexer: dictionary mapping a word to an index in the embedding matrix
        @param dropout: keep probability of the dropout layers
        @param bidirectional: boolean indicating if a tweet is processed in forward and reverse mode w.r.t the words
        """
        super().__init__()
        self.word_indexer = word_indexer
        self.device = device
        self.bidirectional = bidirectional
        # self.is_lstm = False
        self.embedding = nn.Embedding.from_pretrained(torch.Tensor(embedding_matrix), freeze=False)
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=1,
                          batch_first=True,
                          dropout=dropout,
                          bidirectional=bidirectional)

        # Linear layers
        if bidirectional:
            self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        else:
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Final activation
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tweets):
        """
        maps a batch of tweets to a batch of sentiment predictions
        @param tweets: batch of tweets each as a string
        @return: tensor of predictions of dimension [batch size, number of different sentiment (2)]
        """
        return self.forward_rnn(tweets)


