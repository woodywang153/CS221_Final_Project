
import numpy as np

DEFAULT_FILE_PATH = "glove.6B.50d.txt"

def loadWordVectors(filepath=DEFAULT_FILE_PATH, dimensions=50):
    """Read pretrained GloVe vectors"""
    wordVectors = dict()
    with open(filepath) as file:
        for line in file:
            line = line.strip()
            tokens = line.split()
            token = tokens[0]
            data = [float(x) for x in tokens[1:]]
            wordVectors[token] = np.asarray(data)
    return wordVectors

#loadWordVectors()
