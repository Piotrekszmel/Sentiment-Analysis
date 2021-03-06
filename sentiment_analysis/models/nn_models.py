import sys
sys.path.append('..')
from keras.constraints import maxnorm
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dropout, Dense, Bidirectional, LSTM, \
    Embedding, GaussianNoise, Activation, Flatten, \
    RepeatVector, MaxoutDense, GlobalMaxPooling1D, \
    Convolution1D, MaxPooling1D, concatenate, Conv1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from ..utilities.layers import AttentionWithContext, Attention, MeanOverTime
from sklearn import preprocessing


def embeddings_layer(max_length, embeddings, trainable=False, masking=False,
                     scale=False, normalize=False):
    """
    Create embeddings layer
    :param max_length : maximum length of input
    """
    if scale:
        print("Scaling embedding weights...")
        embeddings = preprocessing.scale(embeddings)
    if normalize:
        print("Normalizing embedding weights...")
        embeddings = preprocessing.normalize(embeddings)

    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    _embedding = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_size,
        input_length=max_length if max_length > 0 else None,
        trainable=trainable,
        mask_zero=masking if max_length > 0 else False,
        weights=[embeddings]
    )

    return _embedding


def get_RNN(unit=LSTM, cells=64, bi=False, return_sequences=True, dropout_U=0.,
            consume_less='cpu', l2_reg=0):
    rnn = unit(cells, return_sequences=return_sequences, consume_less=consume_less,
               dropout_U=dropout_U, W_regularizer=l2(l2_reg))
    if bi:
        return Bidirectional(rnn)
    else:
        return rnn


def build_attention_RNN(embeddings, classes, max_length, unit=LSTM, cells=64,
                        layers=1, **kwargs):
    """Builds RNN model with attention based on given parameters"""
    
    # parameters
    bi = kwargs.get("bidirectional", False)
    noise = kwargs.get("noise", 0.)
    dropout_words = kwargs.get("dropout_words", 0)
    dropout_rnn = kwargs.get("dropout_rnn", 0)
    dropout_rnn_U = kwargs.get("dropout_rnn_U", 0)
    dropout_attention = kwargs.get("dropout_attention", 0)
    dropout_final = kwargs.get("dropout_final", 0)
    attention = kwargs.get("attention", None)
    final_layer = kwargs.get("final_layer", False)
    clipnorm = kwargs.get("clipnorm", 1)
    loss_l2 = kwargs.get("loss_l2", 0.)
    lr = kwargs.get("lr", 0.001)

    model = Sequential()
    model.add(embeddings_layer(max_length=max_length, embeddings=embeddings, trainable=False,
                               masking=True, scale=False, normalize=False))

    if noise > 0:
        model.add(GaussianNoise(noise))
    if dropout_words > 0:
        model.add(Dropout(dropout_words))

    for i in range(layers):
        rs = (layers > 1 and i < layers - 1) or attention
        model.add(get_RNN(unit, cells, bi, return_sequences=rs, dropout_U=dropout_rnn_U))

        if dropout_rnn > 0:
            model.add(Dropout(dropout_rnn))
    
    if attention == "memory":
        model.add(AttentionWithContext())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    if attention == "simple":
        model.add(Attention())
        if dropout_attention > 0:
            model.add(Dropout(dropout_attention))
    
    if final_layer:
        model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
        if dropout_final > 0:
            model.add(Dropout(dropout_final))

    model.add(Dense(classes, activity_regularizer=l2(loss_l2)))
    model.add(Activation("softmax"))

    model.compile(optimizer=Adam(clipnorm=clipnorm, lr=lr),
                 loss="categorical_crossentropy")

    return model
    