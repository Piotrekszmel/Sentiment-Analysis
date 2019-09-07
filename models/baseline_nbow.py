import numpy as np
from data.data_loader import DataLoader
from utilities.data_loader import get_embeddings
from utilities.sklearn import eval_clf, nbow_model

np.random.seed(1337)


def baseline_nbow(WV_CORPUS, WV_DIM, years=None, datasets=None):
    WV_CORPUS = WV_CORPUS
    WV_DIM = WV_DIM
    embeddings, word_indices = get_embeddings(corpus=WV_CORPUS, dim=WV_DIM)

    train_set = DataLoader(verbose=False).get_data(years=years,
                                                        datasets=datasets)
    X = [obs[1] for obs in train_set]
    y = [obs[0] for obs in train_set]

    test_data = DataLoader(verbose=False).get_gold()
    X_test = [obs[1] for obs in test_data]
    y_test = [obs[0] for obs in test_data]

    print("-----------------------------")
    print("LinearSVC")
    nbow = nbow_model("clf", embeddings, word_indices)
    nbow.fit(X, y)
    results = eval_clf(nbow.predict(X_test), y_test)
    for res, val in results.items():
        print("{}: {:.3f}".format(res, val))
    print("-----------------------------")