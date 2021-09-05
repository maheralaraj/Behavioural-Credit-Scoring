import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn import preprocessing
from sklearn.model_selection import KFold
df = pd.read_csv('data/UCI_credit_card.csv')
labels = df['default.payment.next.month']
df.drop(['default.payment.next.month'],axis=1,inplace=True)
min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(df.to_numpy())

models = {'Gradient Boosting': GradientBoostingClassifier(),
          'Bagging NN': BaggingClassifier(base_estimator=MLPClassifier(random_state=1, max_iter=500)),
          'SVM': svm.SVC(probability=True),
          'Random Forest': RandomForestClassifier(max_depth=2, random_state=0),
          'Logistic Regression': LogisticRegression(random_state=0, max_iter=500)}

res = {}

for key in models:
    print(key)
    scores = cross_val_predict(models[key], data, labels, cv=KFold(n_splits=5), method="predict_proba", n_jobs=8,verbose=1)
    scores = scores[:, 1]
    res[key] = {'scores':scores, 'actual': labels.to_numpy()}

with open("classifiers_results_enhanced.pickle", "wb") as fp:
    pickle.dump(res, fp)
