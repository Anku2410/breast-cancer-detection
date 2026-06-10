import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data.csv')
df = df.dropna(axis=1)

le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'].values)

X = df.iloc[:, 2:32].values
Y = df['diagnosis'].values.astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
forest.fit(X_train, Y_train)

joblib.dump(forest, 'breast_cancer_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model retrained and saved!")