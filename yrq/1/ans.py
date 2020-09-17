import numpy as np
import pandas as pd 
if __name__ == '__main__':
    test_id = pd.read_csv(r'test_data_A.csv', sep='|', dtype=str)['id'].astype(int).values.flatten()
    # test_pred = np.load('ans_mtx.npy',allow_pickle=True)[:,1].flatten()
    test_pred = np.load('ans_mtx.npy',allow_pickle=True).flatten()
    pd.DataFrame({'id':test_id,'probability':test_pred}).to_csv('submission.csv', sep=',', index=0)



    