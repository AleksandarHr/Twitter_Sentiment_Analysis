import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from torch.utils import data


class CustomDataset(data.Dataset):
    def __init__(self, data):
        """
        initialize attributes of the CustomDataset class
        @param data: data
        """
        self.data = data

    def __getitem__(self, index):
        """
        @param index: index of the (tweet, label) tuple to return
        @return: single (tweet, label) tuple
        """
        tweet = self.data.iloc[index]['tweet']
        label = self.data.iloc[index]['label']
        return tweet, label

    def __len__(self):
        """
        @return: number of samples in the data
        """
        return self.data.shape[0]


class PreProcessing():
    def __init__(self):
        """
        initialize attributes of the PreProcessing class
        """
        self.regex = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
        self.stop_words = stopwords.words("english")

    def clean(self, tweet):
        """
        @param tweet: single tweet as a string
        @return: cleaned tweet as a string
        """
        tweet = self.basic_cleaning_3(tweet)
        tweet = tweet.split(' ')
        tweet = [self.remove_repeated_letters(word) for word in tweet]
        tweet = ' '.join(tweet)
        return tweet

    def basic_cleaning(self, tweet):
        """
        @param tweet: single tweet as a string
        @return: list of words containing only some characters
        """
        tweet = tweet.replace('<user>', '')
        tweet = tweet.replace('<url>', '')
        tweet = ''.join(c if (c.isalpha() or c.isspace()) else '' for c in tweet)
        tweet = ' '.join(tweet.split()).split(' ')
        tweet = list(filter(lambda x: x not in self.stop_words, tweet))
        return tweet

    def basic_cleaning_2(self, tweet):
        """
        @param tweet: single tweet as a sting
        @return: single cleaned tweet as a string
        """
        tweet = ''.join('' if c == "'" else c for c in tweet)
        tweet = ''.join(' <hashtag> ' if c == '#' else c for c in tweet)
        tweet = ''.join(' <number> ' if c.isnumeric() else c for c in tweet)
        tweet = ''.join(c if (c.isalpha() or c.isspace() or c in ['<', '>']) else ' ' + c + ' ' for c in tweet)
        tweet = ' '.join(tweet.split())
        return tweet

    def basic_cleaning_3(self, tweet):
        """
        @param tweet: single tweet as a string
        @return: single cleaned tweet
        """
        tweet = ''.join('' if c == "'" else c for c in tweet)
        tweet = ''.join(c if c.isalpha() else ' ' + c + ' ' for c in tweet)
        tweet = ' '.join(tweet.split())
        return tweet

    def remove_repeated_letters(self, word):
        """
        recursive method to remove repeated letters from a given word based on a dictionary.
        original implementation: https://www.youtube.com/user/RockyDeRaze
        @param word: word that might contains unnecessary repeated letters
        @return: cleaned word
        """
        if wordnet.synsets(word):
            return word
        result = self.regex.sub(self.repl, word)
        if word == result:
            return result
        else:
            return self.remove_repeated_letters(result)

