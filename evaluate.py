from utilities.data_loader import get_embeddings, Loader, prepare_dataset
from models.nn_models import build_attention_RNN
from keras.layers import LSTM
import numpy as np

def predict_class(X, y, corpus, dim):
    tweets = []
    labels = []
    
    embeddings, word_indices = get_embeddings(corpus=corpus, dim=dim)
    loader = Loader(word_indices, text_lengths=50)
    X, y = prepare_dataset(X, y, loader.pipeline, False, True)

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

    for tweet in X:
        predicted_y = np.argmax(nn_model.predict(tweet)[0])
        tweets.append(tweet)
        labels.append(predicted_y)
    return tweets, labels, y
