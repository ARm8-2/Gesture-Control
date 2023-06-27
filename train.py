import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split    
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  
import joblib   

data = pd.read_csv('./logs/data.csv')

y = data['gesture']
x = data.drop('gesture', axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, max_depth=5)

model.fit(x_train.values, y_train.values)
y_pred = model.predict(x_test.values)

print("Accuracy: ", accuracy_score(y_test, y_pred))

joblib.dump(model, 'model/model.pkl')
