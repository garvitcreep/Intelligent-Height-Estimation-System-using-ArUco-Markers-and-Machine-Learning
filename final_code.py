from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd

df = pd.read_csv("/content/output_dataset.csv")

df.head()

df = df.drop('Name',axis=1)
df.head()

x = df.iloc[:,:-1]
y = df.iloc[:,-1]
y.head()

model = LinearRegression()

# Fit the model to the data
model.fit(x, y)

X_new = np.array([[26, 95], [28, 110]])  # New input features for prediction
predictions = model.predict(X_new)

print(predictions)

coefficients = model.coef_
intercept = model.intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)
