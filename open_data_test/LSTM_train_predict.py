import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense
import pickle
from sklearn.metrics import accuracy_score

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], enable=True)


class TransactionSequence(tf.keras.utils.Sequence):
    def __init__(self, data, drop_index):
        self.data = data['temporal'].copy()
        self.additional = data['additional'].copy()
        self.labels = data['labels'].copy()
        self.data.pop(drop_index)
        self.additional.pop(drop_index)
        self.labels.pop(drop_index)
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        return [self.data[i], self.additional[i]], self.labels[i]


with open("data/processed.pickle", "rb") as fp:
    train_dict = pickle.load(fp)


transactions_input = tf.keras.Input(shape=(None, 3), name='transactions')
additional_input = tf.keras.Input(shape=(8,), name='info')
x = LSTM(4, return_sequences=True)(transactions_input)
x = LSTM(4, return_sequences=False)(x)
x = Dense(4, activation='sigmoid')(x)
x = tf.concat([x, additional_input], axis=1)
out = Dense(1, activation='sigmoid')(x)
model = tf.keras.Model([transactions_input, additional_input], out)
opt=tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy', 'AUC'])
random_weights = model.get_weights()
actual_labels = np.concatenate(train_dict['labels']).flatten()
N = len(actual_labels)
res_df = pd.DataFrame({'actual': actual_labels, 'scores': np.zeros(N)})
step = int(N/5)
for i in range(5):
    tr = TransactionSequence(train_dict, i)
    #model.set_weights(random_weights)
    model.fit(x = tr, epochs=1000)
    scores = model.predict([train_dict['temporal'][i], train_dict['additional'][i]])
    res_df.loc[i*step:(i+1)*step-1, 'scores'] = scores

pred_decision = np.clip(res_df.scores, a_min=0, a_max=1)
pred_decision = np.round(pred_decision)
print(accuracy_score(actual_labels, pred_decision))

#with open("LSTM_results.pickle", "wb") as fp:
#    pickle.dump(res_df, fp)
