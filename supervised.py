
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
# Sample data
import numpy as np
X = np.array([[1200], [1500], [1700], [2000], [2500]])  # Square footage
y = np.array([200000, 250000, 270000, 320000, 400000])  # Price
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Model training
model = LinearRegression()
model.fit(X_train, y_train)
# Predictions
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)
# Evaluate
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Sample data
X = np.array([1200, 1500, 1700, 2000, 2500]).reshape(-1, 1)  # Square footage
y = np.array([200000, 250000, 270000, 320000, 400000])  # House prices
# Train a linear regression model
model = LinearRegression()
model.fit(X, y)
# Predictions
y_pred = model.predict(X)
# Plot the data
plt.scatter(X, y, color='blue', label="Data Points")
plt.plot(X, y_pred, color='red', label="Regression Line")
plt.title("House Price Prediction")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.legend()
plt.show()
'''