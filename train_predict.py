import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
import pickle
import datetime
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], enable=True)


class TransactionSequence(tf.keras.utils.Sequence):
    def __init__(self, data, suffix='latefee', is_perm=True):
        self.is_perm = is_perm
        self.labels = []
        self.additional=[]
        self.len = len(data['data'])
        self.perm = np.random.permutation(self.len)
        self.get_additional_data(data)
        self.data = data['data']
        self.clients = [df.Key for df in data['clients']]
        if suffix == 'latefee':
            self.get_labels_605(data)
        elif suffix == 'purchase':
            self.get_labels_sum_purchase(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.is_perm:
            i = self.perm[idx]
        else:
            i = idx
        return [self.data[i], self.additional[i]], self.labels[i]

    def on_epoch_end(self):
        self.perm = np.random.permutation(self.len)

    def get_additional_data(self, data):
        for df in data['clients']:
            tar = np.array(df.drop(['Key'], axis=1), dtype=np.float32)
            self.additional.append(tar)

    def get_labels_605(self, data):
        for ind, df in enumerate(data['label_source']):
            lab = np.zeros((len(data['clients'][ind].Key), 1))
            tmp = df[df['TRANSACTION CODE'] == 605]
            positive_keys = tmp.Key.unique()
            _, ir, _ = np.intersect1d(data['clients'][ind].Key, positive_keys, return_indices=True)
            lab[ir] = 1
            self.labels.append(lab)

    def get_labels_sum_purchase(self, data):
        for ind, df in enumerate(data['label_source']):
            lab = np.zeros((len(data['clients'][ind].Key), 1))
            tmp = df[df['TRANSACTION CODE'] == 1]
            merga = tmp.groupby('Key').agg({'TRANS AMT IN AED-Signed': np.sum})
            _, ir, _ = np.intersect1d(data['clients'][ind].Key, merga.index.values, return_indices=True)
            lab[ir, 0] = np.log(merga['TRANS AMT IN AED-Signed']+1)
            self.labels.append(lab)


def get_predictions(model, ts):
    actual_labels = np.concatenate(ts.labels).flatten()
    actual_clients = np.concatenate(ts.clients).flatten()
    results = model.evaluate(ts)
    print('loss: {}, test {}: {}, test {}: {}'.format(results[0], metrics[0], results[1], metrics[1], results[2]))
    scores = model.predict(ts)
    df = pd.DataFrame({"Key": actual_clients, "actual": actual_labels, "scores": scores.flatten()})
    return df


with open("pickle/train.pickle", "rb") as fp:
    train_dict = pickle.load(fp)


sss = input("Predict (1 or Enter): late fees; (2): purchase amount").lower()
if sss == '1' or sss == '':
    print("We predict late fees!")
    suffix = 'latefee'
    metrics = ['binary_accuracy', 'AUC']
    loss = 'binary_crossentropy'
    nl = [16, 16, 16]
    last_activation = 'sigmoid'
elif sss == '2':
    print("We predict sum of purchases!")
    suffix = 'purchase'
    loss = 'mae'
    metrics = ['mse', 'mae']
    nl = [32, 32, 32, 32, 32]
    last_activation = None
else:
    print("Wrong choice! Exit")
    exit(0)

tr = TransactionSequence(train_dict, suffix=suffix)

sss = input("%s (Y/n) " % "Load existing model?").lower() != 'n'

if sss:
    model = tf.keras.models.load_model("pickle/model_{}.h5".format(suffix))
else:
    print("Loading model from scratch!")
    transactions_input = tf.keras.Input(shape=(None, 27), name='transactions')
    additional_input = tf.keras.Input(shape=(41,), name='info')
    x=transactions_input
    for ind, cnt in enumerate(nl):
        if ind == len(nl)-1:
            rs = False
        else:
            rs = True
        x = LSTM(cnt, return_sequences=rs)(x)
    x = Dense(20, activation='sigmoid')(x)
    x = tf.concat([x, additional_input], axis=1)
    x = Dense(10, activation='sigmoid')(x)
    out = Dense(1, activation=last_activation)(x)
    model = tf.keras.Model([transactions_input, additional_input], out)

model.compile(loss=loss, optimizer='adam', metrics=metrics)
TensorBoard_callback=[tf.keras.callbacks.ModelCheckpoint("pickle/model_{}.h5".format(suffix))]
if input("%s (y/N)" % "Is train model?").lower() == 'y':
    log_dir = "jupyter/log/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    TensorBoard_callback.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))
    model.fit(x=tr, epochs=100, callbacks=TensorBoard_callback)

with open("pickle/test.pickle", "rb") as fp:
    test_dict = pickle.load(fp)
ts = TransactionSequence(test_dict, suffix=suffix, is_perm=False)

prediction_dict = {"train": get_predictions(model, tr), "test": get_predictions(model, ts)}
with open("pickle/results_{}.pickle".format(suffix), "wb") as fp:
    pickle.dump(prediction_dict, fp)
