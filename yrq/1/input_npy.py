from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec, KeyedVectors
import numpy as np
npy_path = 'adv_id'
if __name__ == '__main__':
    path = f'adv_idword2vec_kv.kv'
    wv = KeyedVectors.load(path, mmap='r')
    feature_tokens = list(wv.vocab.keys())
    tokenizer = Tokenizer(num_words=1187210, filters='')
    tokenizer.fit_on_texts(feature_tokens)
    
    feature_seq = []

    with open('adv_id.txt') as f:
        for text in f:
            feature_seq.append(text.strip())

    sequences = tokenizer.texts_to_sequences(feature_seq[:41907133//1])
    x_train = pad_sequences(
        sequences, maxlen=40, padding='post')
    print(x_train.shape)
    np.save(npy_path+'x_train.npy', x_train)

    sequences = tokenizer.texts_to_sequences(feature_seq[41907133:])
    x_test = pad_sequences(
        sequences, maxlen=40, padding='post') 
    print(x_test.shape)
    np.save(npy_path+'x_test.npy', x_test)
    
    

    
    