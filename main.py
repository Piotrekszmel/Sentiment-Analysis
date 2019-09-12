from models.baseline_nbow import baseline_nbow
from models.sentiment_model import Sentiment_Analysis
import pickle
import os
import numpy as np
import sys
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM
from utilities.callbacks import MetricsCallback, PlottingCallback
from utilities.data_preparation import get_labels_to_categories_map, get_class_weights2, onehot_to_categories
from sklearn.metrics import f1_score, precision_score
from sklearn.metrics import recall_score
from data.data_loader import DataLoader
from models.nn_models import build_attention_RNN
from utilities.data_loader import get_embeddings, Loader, prepare_dataset
from evaluate import predict_class

#Sentiment_Analysis("datastories.twitter", 300, 50, PERSIST=False, FINAL=True)

"""
embeddings, word_indices = get_embeddings(corpus='datastories.twitter', dim=300)

loader = Loader(word_indices, text_lengths=50)
tweet = 'i am so happy. :)'
tweet, label = prepare_dataset([tweet], -1, loader.pipeline, False, True)



nn_model = build_attention_RNN(embeddings, classes=3, max_length=50,
                                unit=LSTM, layers=2, cells=150,
                                bidirectional=True,
                                attention="simple",
                                noise=0.3,
                                final_layer=False,
                                dropout_final=0.5,
                                dropout_attention=0.5,
                                dropout_words=0.3,
                                dropout_rnn=0.3,
                                dropout_rnn_U=0.3,
                                clipnorm=1, lr=0.001, loss_l2=0.0001)
nn_model.load_weights('./bi_model_weights_1.h5')



prediction = nn_model.predict(tweet)
print(np.argmax(nn_model.predict(tweet)[0]))
print(prediction)                              
"""

tweets, predicted_y, y = predict_class(["I am happy", "I am sad :(", "Poland is a country"], [2,0,1], "datastories.twitter", 300)
for tweet, label in zip(tweets, predicted_y):
    print("text: ",tweet, " ''' predicted: ", predicted_y, " ||| label: ", label)