import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
df = pd.read_csv('UCI_credit_card.csv')
arr_list = []
for i in range(5, 23, 6):
    np_slice = df[df.columns[i:(i+6)]].to_numpy()
    arr_list.append(preprocessing.MinMaxScaler().fit_transform(np_slice))

tensors = np.dstack(arr_list)
tensors = np.split(tensors, 5)
additional = df[df.columns[0:5]]
edu = additional.loc[:, 'EDUCATION'].replace([0,5,6],None)
edu = pd.get_dummies(edu)
additional = pd.concat([additional, edu], axis=1).drop('EDUCATION',axis=1).to_numpy()
additional = preprocessing.MinMaxScaler().fit_transform(additional)
additional = np.split(additional, 5)
labels = np.split(df['default.payment.next.month'].to_numpy(), 5)
with open("processed.pickle", "wb") as fp:
    pickle.dump({'temporal': tensors, 'labels': labels, 'additional': additional}, fp)
