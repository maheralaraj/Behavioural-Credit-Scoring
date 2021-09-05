import pandas as pd
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

train_data_path = 'RScripts/RData/merged_key_train_allclients.csv'
test_data_path = 'RScripts/RData/merged_key_test_allclients.csv'

def prepare_data(file_path):
    date_cols = ['total.first.date', 'total.last.date', 'purchase.first.date', 'purchase.last.date', 'activity.first.date',
                 'activity.last.date', 'latefee.first.date', 'latefee.last.date']
    df = pd.read_csv(file_path, parse_dates=date_cols, low_memory=False)
    df = df.drop(['Unnamed: 0', 'latefee.first.after.activity'], 1)

    df['total.first.date'] = (df['total.last.date']-df['total.first.date']).dt.days
    df['purchase.first.date'] = (df['purchase.last.date'] - df['purchase.first.date']).dt.days
    df['activity.first.date'] = (df['activity.last.date'] - df['activity.first.date']).dt.days
    df['latefee.first.date'] = (df['latefee.last.date'] - df['latefee.first.date']).dt.days

    df = df.drop(['total.last.date','purchase.last.date', 'activity.last.date', 'latefee.last.date'], 1)
    df['bank.account.number'] = df['bank.account.number'].astype(int)
    return df

def add(column_name, model, dictionary, dt_train, dt_test, x_train, x_test):
    train_prediction = model.predict_proba(x_train)
    test_prediction = model.predict_proba(x_test)
    dictionary[column_name] = dict(train=dt_train.assign(scores=train_prediction[:, 1]),
                                   test=dt_test.assign(scores=test_prediction[:, 1]))
    return dictionary

def main():
    train_df = prepare_data(train_data_path)
    test_df = prepare_data(test_data_path)

    scaler = StandardScaler()
    scaler.fit(train_df.drop(['label', 'Key'], axis=1))

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    x_train = train_df.drop(['label', 'Key'], axis=1)
    y_train = train_df['label']
    x_test = test_df.drop(['label', 'Key'], axis=1)
    y_test = test_df['label']

    res = {}
    dt_train = pd.DataFrame({'Key': train_df['Key'], 'actual': y_train})
    dt_test = pd.DataFrame({'Key': test_df['Key'], 'actual': y_test})

    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    model1 = MLPClassifier(random_state = 1, max_iter= 500).fit(x_train, y_train)
    res = add("Neural Network", model1, res, dt_train, dt_test, x_train, x_test)

    model2 = svm.SVC(probability=True).fit(x_train, y_train)
    res = add("SVM", model2, res, dt_train, dt_test, x_train, x_test)

    model3 = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
    res = add("Random Forest", model3, res, dt_train, dt_test, x_train, x_test)

    model4 = LogisticRegression(random_state=0, max_iter=500).fit(x_train, y_train)
    res = add("Logistic Regression", model4, res, dt_train, dt_test, x_train, x_test)

    with open("pickle/classifiers_allclients.pickle", "wb") as fp:
       pickle.dump(res, fp)


if __name__ == "__main__":
    main()