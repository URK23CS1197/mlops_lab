import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data=pd.read_csv('Diabetes.csv')

X=data[['Glucose', 'Insulin', 'BMI']]
y=data[['Outcome']]

X_train, X_test, y_train, y_test=train_test_split (X,y,test_size=0.25,random_state=42)

rf=RandomForestClassifier()
model=rf.fit(X_train, y_train)
y_pred=model.predict(X_test)

acc=accuracy_score(y_test, y_pred)

print(f"Accuracy:{acc:.2f}")

joblib.dump(model, "random_forest.pkl")

print("model saved sucessfully")