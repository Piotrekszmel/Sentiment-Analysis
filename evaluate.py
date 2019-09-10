import pickle

loaded_model = pickle.load(open("hist.pickle", "rb"))
for key in loaded_model.keys():
    print(key)

