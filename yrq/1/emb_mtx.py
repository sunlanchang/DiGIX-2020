from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
path = 'adv_id'
def get_embedding():
        path = f'adv_idword2vec_kv.kv'
        wv = KeyedVectors.load(path, mmap='r')
        feature_tokens = list(wv.vocab.keys())
        tokenizer = Tokenizer(num_words=1187210)
        tokenizer.fit_on_texts(feature_tokens)
        embedding_dim = 128
        embedding_matrix = np.random.randn(
            len(feature_tokens)+1, embedding_dim)
        for feature in feature_tokens:
            embedding_vector = wv[feature]
            if embedding_vector is not None:
                index = tokenizer.texts_to_sequences([feature])[0][0]
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

if __name__ == '__main__': 
    embedding_matrix = get_embedding()
    np.save(path+'embedding_matrix.npy',embedding_matrix)
    