import pickle

word_indices = pickle.load(open("model_word_indices.pickle", "rb"))

first2pairs = {k: word_indices[k] for k in sorted(word_indices.keys())[:2]}
print(first2pairs)

