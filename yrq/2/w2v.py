from gensim.models import word2vec
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm
path = 'adv_id2'

# class bsz_loss(CallbackAny2Vec):
#     def __init__(self):
#         self.bsz = 1

#     def on_batch_end(self, model):
#         loss = model.get_latest_training_loss()
#         print("bsz:%d/loss:%f" % (self.bsz, loss))
#         self.bsz = self.bsz+1

def f1_f2_w2v(f1_f2):
    print('LineSentence start')
    sentences = word2vec.LineSentence(f1_f2+'.txt') 
    print('Word2Vec start')
    # model = word2vec.Word2Vec(sentences , size=128, window=35, sg=0, hs=1, min_count=1, iter=5, callbacks=[bsz_loss])
    model = word2vec.Word2Vec(sentences , size=64, window=10000, sg=0, hs=1, min_count=1, iter=10, workers=-1)
    print('save start')
    model.save(f1_f2+'_word2vec.model')
    model.wv.save(f1_f2+'_word2vec_kv.kv')
    print('Done')
if __name__ == '__main__': 
    f1_f2_list = [['task_id','age'],['task_id','city'],['task_id','city_rank'],['task_id','device_name'],['task_id','career'],['task_id','gender'],['task_id','residence'],['adv_id','age'],['adv_id','city'],['adv_id','city_rank'],['adv_id','device_name'],['adv_id','career'],['adv_id','gender'],['adv_id','residence'],['creat_type_cd','age'],['creat_type_cd','city'],['creat_type_cd','city_rank'],['creat_type_cd','device_name'],['creat_type_cd','career'],['creat_type_cd','gender'],['creat_type_cd','residence'],['indu_name','age'],['indu_name','city'],['indu_name','city_rank'],['indu_name','device_name'],['indu_name','career'],['indu_name','gender'],['indu_name','residence'],['adv_prim_id','age'],['adv_prim_id','city'],['adv_prim_id','city_rank'],['adv_prim_id','device_name'],['adv_prim_id','career'],['adv_prim_id','gender'],['adv_prim_id','residence']]
    for i in f1_f2_list:
        f1_f2_w2v(str(i[0])+'_'+str(i[1]))

