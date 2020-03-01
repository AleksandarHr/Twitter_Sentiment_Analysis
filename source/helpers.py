import torch
from torch import nn
from tqdm import tqdm
import numpy as np


def train(model, train_generator, validation_generator, epochs, device, optimizer):
    """
    trains a model for a specified number of epochs
    @param model: model currently trained
    @param train_generator: iterator which generates batch of training samples
    @param validation_generator: iterator which generates batch of evaluation samples
    @param epochs: number of epochs to train the model on
    @param device: either 'cpu' or 'cuda' depending on hardware availability
    @param optimizer: optimizer used to updated the model's parameters
    @return:
    """
    # Set the objective function
    loss_function = nn.CrossEntropyLoss()

    # Move model's parameters to device
    model = model.to(device)

    # Set model in train mode for dropout and batch-normalization layers
    model.train(True)
    for e in range(epochs):
        nb_correct_train = 0
        for batch_tweets, batch_labels in tqdm(train_generator):
            # Move batch to device
            batch_labels = torch.LongTensor(batch_labels).to(device)

            # Reset the gradient for each batch
            optimizer.zero_grad()

            # Forward pass)
            outputs = model.forward(batch_tweets)

            # Compute the predictions
            predictions = torch.argmax(outputs.detach(), dim=1)
            nb_correct_train += torch.sum(predictions == batch_labels)

            loss = loss_function(outputs, batch_labels.squeeze())

            # Backward pass
            loss.backward()
            optimizer.step()

        # Compute training accuracy
        train_acc = nb_correct_train.item() / train_generator.dataset.__len__()
        train_message = 'Train accuracy at epoch {} is {}'
        print(train_message.format(e, train_acc))

    # Set model in eval mode for dropout and batch-normalization layers
    model.train(False)
    with torch.no_grad():
        nb_correct_val = 0
        for batch_tweets, batch_labels in validation_generator:
            batch_labels = torch.LongTensor(batch_labels).to(device)
            outputs = model.forward(batch_tweets)
            predictions = torch.argmax(outputs, dim=1)
            nb_correct_val += torch.sum(predictions == batch_labels)
        val_acc = nb_correct_val.item() / validation_generator.dataset.__len__()
        val_message = 'Test accuracy is {}'
        print(val_message.format(val_acc))


def build_embedding_matrix(embedding_dim, glove, vocabulary):
    """
    completes a pre-trained glove embedding with random initialization for missing entries.
    This implementation is strongly inspired by:
        https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76
    @param embedding_dim: dimension of the the embedding space
    @param glove: pretrained glove embedding dictionary
    @param vocabulary: list of words used in the tweets
    @return: completed embedding matrix and word to index dictionary
    """
    embedding_matrix = np.zeros((len(vocabulary), embedding_dim))
    text_indexer_dict = dict()
    for i, word in enumerate(vocabulary):
        try:
            embedding_matrix[i] = glove[word]
        except KeyError:
            embedding_matrix[i] = np.random.normal(loc=0, scale=1., size=embedding_dim)
        text_indexer_dict[word] = i
    return embedding_matrix, text_indexer_dict


def accuracy(model, inputs, labels, batch_size, device):
    """
    computes the accuracy of the model on the given inputs
    @param model: model currently trained
    @param inputs: array of inputs on which to compute the accuracy
    @param labels: array of true labels
    @param batch_size: size of batches
    @param device: cpu or cuda depending on hardware availability
    @return: accuracy of the predictions
    """
    nb_correct = 0
    for b in range(0, inputs.shape[0], batch_size):
        outputs = model.forward(inputs.narrow(0, b, batch_size).to(device))
        predictions = torch.argmax(outputs, dim=1)
        nb_correct += torch.sum(predictions == labels.narrow(0, b, batch_size).squeeze().to(device))
    return nb_correct.item() / inputs.shape[0]


def load_dict_contractions():
    """
    this list of contractions was taken from a previous versions of :
        https://en.wikipedia.org/wiki/Wikipedia%3AList_of_English_contractions
    @return: dictionary from contraction to corresponding correct spelling
    """
    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "'cause": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "e'er": "ever",
        "em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'll": "he will",
        "he's": "he is",
        "he've": "he have",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "I'd": "I would",
        "I'll": "I will",
        "I'm": "I am",
        "I'm'a": "I am about to",
        "I'm'o": "I am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "I've": "I have",
        "kinda": "kind of",
        "let's": "let us",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "ne'er": "never",
        "o'": "of",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "shalln't": "shall not",
        "shan't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "'tis": "it is",
        "'twas": "it was",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why's": "why is",
        "won't": "will not",
        "wouldn't": "would not",
        "would've": "would have",
        "y'all": "you all",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have",
        "Whatcha": "What are you",
        "luv": "love",
        "sux": "sucks"
    }
