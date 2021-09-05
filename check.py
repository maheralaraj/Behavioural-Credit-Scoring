import pickle
import numpy as np

with open("pickle/df_cleaned.pickle", "rb") as fp:
    res = pickle.load(fp)
res = res.sort_values(['Key','TRANSACTION DATE'])
first_pos = res.Key.value_counts().sort_index().cumsum().values
first_pos = np.roll(first_pos,shift=1)
first_pos[0] = 0

date_shift = res['TRANSACTION DATE'].shift()
difference = (res['TRANSACTION DATE']-date_shift).dt.days.astype(int, errors='ignore')
difference.iloc[first_pos] = 0
res.insert(2, 'DIFFERENCE', difference)
res = res.head(1000)
res.to_csv("111.csv")
'''
with open("pickle/test.pickle", "rb") as fp:
    test_dict = pickle.load(fp)

aaa=[i.shape[0] for i in test_dict['data']]
print(sum(aaa))

with open("pickle/train.pickle", "rb") as fp:
    train_dict = pickle.load(fp)

aaa=[i.shape[0] for i in train_dict['data']]
print(sum(aaa))
'''
