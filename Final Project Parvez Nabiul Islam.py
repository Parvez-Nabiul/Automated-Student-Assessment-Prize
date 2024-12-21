import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('E:/ASAP2_train.csv')


essays = df['full_text']
scores = df['score']


tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(essays).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, scores, test_size=0.2, random_state=42)


regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted vs True')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.title('True Scores vs Predicted Scores')
plt.xlabel('True Scores')
plt.ylabel('Predicted Scores')
plt.legend()
plt.grid(True)
plt.show()

