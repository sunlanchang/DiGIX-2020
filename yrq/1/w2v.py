from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec

path = 'adv_id2'

# class bsz_loss(CallbackAny2Vec):
#     def __init__(self):
#         self.bsz = 1

#     def on_batch_end(self, model):
#         loss = model.get_latest_training_loss()
#         print("bsz:%d/loss:%f" % (self.bsz, loss))
#         self.bsz = self.bsz+1

if __name__ == '__main__': 
    print('LineSentence start')
    sentences = word2vec.LineSentence('adv_id2.txt') 
    print('Word2Vec start')
    # model = word2vec.Word2Vec(sentences , size=128, window=35, sg=0, hs=1, min_count=1, iter=5, callbacks=[bsz_loss])
    model = word2vec.Word2Vec(sentences , size=128, window=64, sg=0, hs=1, min_count=1, iter=10, workers=-1)
    print('save start')
    model.save(path+'word2vec.model')
    model.wv.save(path+'word2vec_kv.kv')
    print('Done')

