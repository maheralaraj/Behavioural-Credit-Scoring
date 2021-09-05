import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import math


def split(df):
    df.insert(1, 'months_since_BC', df['YEAR']*12+df['MONTH'])
    grouped_single = df.groupby('Key').agg({'months_since_BC': 'max'})
    res = df.merge(grouped_single, on='Key', how='left')
    first = res[res['months_since_BC_x'] < res['months_since_BC_y']]
    second = res[res['months_since_BC_x'] == res['months_since_BC_y']]
    keys_leave = first['Key'].unique()
    second = second[second['Key'].isin(keys_leave)]
    df = df.drop(['months_since_BC'], 1)
    return df, first.drop(['months_since_BC_x', 'months_since_BC_y'], 1), second.drop(['months_since_BC_x', 'months_since_BC_y'], 1)


def get_quantile(interval_number, transaction):
    res = np.zeros(interval_number+1, dtype=int)
    for i in range(1,interval_number+1):
        res[i] = transaction.quantile(math.pow(i/interval_number, 0.2))
    return np.unique(res)


def get_tensor(client_dataframe, client_key, codes):
    MAX_TRANSACTION_SIZE = max(client_dataframe.groupby('Key').size())
    tensor = np.zeros((len(client_key), MAX_TRANSACTION_SIZE, 15+len(codes)))
    for ind, key in enumerate(tqdm(client_key)):
        current_client = client_dataframe[client_dataframe['Key'] == key]
        dif = (current_client['TRANSACTION DATE'] - current_client['TRANSACTION DATE'].values[0]).dt.days.astype(int)
        difference = np.diff(dif)
        difference = np.insert(difference, 0, 0., axis=0)
        current_client.insert(2, 'DIFFERENCE', difference)
        current_client = current_client.drop(['Key', 'TRANSACTION DATE'], 1)
        result_array = current_client.to_numpy()
        tensor[ind][MAX_TRANSACTION_SIZE - len(result_array):] = result_array
    return tensor


def convert_to_tensors(res, dflabels, K, additional):
    codes = res['TRANSACTION CODE'].unique()
    TC_dummies = pd.get_dummies(res['TRANSACTION CODE'])
    res = pd.concat([res, TC_dummies], axis=1)
    MC_dummies = pd.get_dummies(res['MCG CODE'])
    res = pd.concat([res, MC_dummies], axis=1)
    res = res.drop(['TRANSACTION CODE', 'MCG CODE'], 1)
    KEY_GROUPS = res.groupby('Key').size()
    KEY_GROUPS = KEY_GROUPS.sort_values()
    CLIENT_TRANSACTIONS = KEY_GROUPS.to_frame()
    quantiles = get_quantile(K, CLIENT_TRANSACTIONS)
    client_groups = []
    for i in range(K):
        tmp = CLIENT_TRANSACTIONS[CLIENT_TRANSACTIONS[0].between(quantiles[i], quantiles[i + 1]+1, inclusive=False)]
        tmp = pd.DataFrame(data={'Key': tmp.index})
        tmp = tmp.merge(additional, on=['Key'], how='left')
        tmp.fillna(value=0, inplace=True)
        client_groups.append(tmp)
    label_source = [dflabels[dflabels['Key'].isin(i.Key)] for i in client_groups]

    tensors = []
    for i in client_groups:
        clients = res[res['Key'].isin(i.Key)]
        tensors.append(get_tensor(clients, i.Key, codes))
    return {'data': tensors, 'label_source': label_source, 'clients': client_groups}


def main():
    K = 40
    with open("pickle/df_cleaned.pickle", "rb") as fp:
        res = pickle.load(fp)
    with open("pickle/df_additional.pickle", "rb") as fp:
        additional = pickle.load(fp)
    my = res['YEAR']*12+res['MONTH']
    my = (my == max(my))
    res_m1 = res.loc[~my, :]
    ytest = res.loc[my, :]
    res_m1, res_m2, ytrain = split(res_m1)
    res_tr = convert_to_tensors(res_m2, ytrain, K, additional)
    with open("pickle/train.pickle", "wb") as fp:
       pickle.dump(res_tr, fp)
    clients_test = ytest['Key'].unique()
    res_m1 = res_m1[res_m1['Key'].isin(clients_test)]
    res_ts = convert_to_tensors(res_m1, ytest, K, additional)
    with open("pickle/test.pickle", "wb") as fp:
        pickle.dump(res_ts, fp)

if __name__ == "__main__":
    main()

