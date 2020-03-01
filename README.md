# ML-2019

## Team _Learning Machines_
* Arman Ehsasi: arman.ehasi@epfl.ch
* Aleksandar Hrusanov: aleksandar.hrusanov@epfl.ch
* Thomas Stegmüller: thomas.stegmuller@epfl.ch

## Overview
The repository contains our implementation of a Machine Learning model for performing Twitter Data Sentiment Analysis. In particular, we worked with a large data of tweets previously containing either happy or sad smiley emoticon. The goal of the model was, given such a tweet, to predict whether it used to have one or the other. This project was part of the the EPFL's _CS-433: Machine Learning_ course. With our model, we reached final accuracy of 84.3%.

## Dependencies

### Libraries
* Install Python 3 [Anaconda distribution](https://www.anaconda.com/distribution/)

* Install numpy
```
$ conda install -c anaconda numpy
```
* Install [scikit-learn](https://scikit-learn.org/stable/install.html)
```console
$ conda install -c anaconda scikit-learn
```
* Install [nltk](http://www.nltk.org/)
```
$ python
$ >>> import nltk
$ >>> nltk.download()
```
* Install [pytorch](https://pytorch.org/get-started/locally/)
  * Specify system configurations in the provided link to get the corresponding command
  - Note: In order to avoid errors, use pytorch 1.2 or lower

### Files
* All the necessary files (e.g. train data, test data, glove embeddings) can be downloaded from the following [Google drive](https://drive.google.com/drive/folders/17iCunNXyaIzMgZOTDjN7Qbt3Wabemdo0?usp=sharing).
* Different notebooks make use of different data files. The complete expected directory structure is as follows:
```
README.MD
|___source
|   |   preprocessing.ipynb
|   |   classes.py
|   |   helpers.py
|   |   models.py
|   |   postprocessing.ipynb
|   |   glove.ipynb
|   |   tf_idf.ipynb
|   |   run.py
|
|___Data   
|   |   train_full.txt
|   |   train_small.txt
|   |   train_neg.txt
|   |   train_neg_full.txt
|   |   train_pos.txt
|   |   train_pos_full.txt
|   |   test_full.txt
|   |   test_data.txt
|   |   test_online.txt
|   |   glove.twitter.27B.200d.txt
|   |   glove.twitter.27B.25d.txt
|
|___Models   
|   |   GRUNet
```
  * Note: The _Models_ folder contains the trained models. We achieved our best scores with the provided trained GRUNet model. If you want to train your own model, simply save it in the _Models_ folder and change the respective _model_path_ variable in the _run.py_ script.

## Code Base Structure
The code base contains the following files:
* **preprocessing.ipynb** - a notebook containing functionality used for cleaning the raw Twitter data
* **classes.py** - contains a few helper classes used for better code structure and organization
* **helpers.py** - contains various helper functions used within core methods' implementations and the _run.py_ script
* **models.py** - contains the training models' code
  * Abstract class _TextNet_ with functionality common for all models
  * Class _TextLeNetCNN_ used to build a Convolutional Neural Network model
  * Class _SimpleRNN_ used to build a Recurrent Neural Network model
  * Class _SimpleLSTM_ used to build a Long Short Term Memory model
  * Class _GRUNet_ used to build a Grated Recurrent Unit Network
* **postprocessing.ipynb** - a notebook containing functionality used for post-processing (e.g. get a better understanding of the predictions made by the our final model)
* **glove.ipynb** - a notebook containing functionality used to assess performance of the _GloVe_ word representation with a Logistic Regression model
* **tf_idf.ipynb** - a notebook containing functionality used to assess performance of the _TF-IDF_ word representation with a Logistic Regression model
* **bow.ipynb** - a notebook containing functionality used to assess performance of the _Bag of Words_ word representation with a Logistic Regression model
* **run.py** - the script to run and reproduce the final predictions for the test data

## Reproduce
* For easy reproducability of our best scores, one only needs the provided pre-trained model file _GRUNet_ and the _test_online.txt_ file in order to create a submission file for the challenge on [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-text-classification-2019/).
* Model training was done using [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)
