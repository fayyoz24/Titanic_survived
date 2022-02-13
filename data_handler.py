
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def data(path):

    df = pd.read_csv(path)

    df1 = df.copy()

    df1 = df1.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1)

    df1['Family_size'] = df1['SibSp']+df1['Parch']+1
    df1['Family_size']
    df1 = df1.drop(['SibSp'], axis=1)
   
    df1['Age'].fillna(value=28, inplace=True)

    df1['Embarked'].fillna(value='C', inplace=True)

    return df1

def pipe():

    numeric_features = ["Age", "Fare", 'Family_size', 'Parch']
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_features = ["Embarked", "Sex", "Pclass"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor