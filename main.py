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


tweets, predicted_y, label = predict_class(["I am happy", "I am sad :(", "Poland is a country"], [2,0,1], "datastories.twitter", 300)
os.system("clear")
for predict_y, label in zip(predicted_y, label):
    print(float(predict_y), ' | ', label)