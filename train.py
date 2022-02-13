from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline
from data_handler import pipe, data
from sklearn.model_selection import cross_validate
import numpy as np


def training():
    df = data('./train.csv')
    preprocessor = pipe()

    X_train,Y_train = df.drop(['Survived'], axis=1), df['Survived']

    tree_classifiers = {
  "Decision Tree":        DecisionTreeClassifier(),
  "Extra Trees":          ExtraTreesClassifier(n_estimators=100),
  "Random Forest":        RandomForestClassifier(n_estimators=100),
  "AdaBoost":             AdaBoostClassifier(n_estimators=100),
  "GBClassifier":         GradientBoostingClassifier(n_estimators=100),
  "XGBoost":              XGBClassifier(n_estimators=100),
  "LightGBM":             LGBMClassifier(n_estimators=100),
  "CatBoost":             CatBoostClassifier(n_estimators=100),
  'KNeighborsClassifier': KNeighborsClassifier(),
  'SVC':                  SVC()
 }
    results = {}
  
    for name, cl in tree_classifiers.items():
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", cl)]
        )
        
        cv = cross_validate(clf, X_train, Y_train, cv=5)

        results[name] = cv['test_score'].mean()
   
    return results, tree_classifiers

def predict(x):

    df = data('./train.csv')
    results, tree_classifiers = training()
    preprocessor = pipe()

    X_train,Y_train = df.drop(['Survived'], axis=1), df['Survived']

    answer = []

    for name, cl in tree_classifiers.items():
        clf = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", cl)]
        )
        clf.fit(X_train, Y_train)
        
        answer.append(clf.predict(x))
        
    return sum(answer) / len(answer)