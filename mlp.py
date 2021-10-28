import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from warnings import filterwarnings
filterwarnings('ignore')

# Define dataset
X = np.linspace(-1, 2.5, 500)
y = np.sin(X)

# Define training and validation parts
train_part = 0.80
dataset_indices = np.array(range(len(X)))

train_indices = np.random.choice(dataset_indices, int(len(dataset_indices)*train_part), replace=False)
valid_indices = dataset_indices[~np.isin(dataset_indices, train_indices)]

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[valid_indices], y[valid_indices]

X_train = X_train.reshape(-1, 1)
X_val = X_val.reshape(-1, 1)
print("Lengths: X_train %s, y_train %s, X_val %s, y_val %s" % (len(X_train), len(y_train), len(X_val), len(y_val)))


# Define model
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(max_iter=1, warm_start=True, activation='logistic', solver = "lbfgs",
                     hidden_layer_sizes=1, learning_rate_init=0.001)

# Train and evaluate on validation set
total_epochs = 10
for epoch in range(0, total_epochs):
    # Fit model
    model.fit(X_train, y_train)

    # Get predictions
    train_preds = model.predict(X_train)
    train_loss = mean_squared_error(train_preds, y_train)

    val_preds = model.predict(X_val)
    val_loss = mean_squared_error(val_preds, y_val)
    #print(train_loss, val_loss)


# Plot performance
x, y = X_val[:, 0], y_val
sns.scatterplot(x=x, y=y, color="gray")

x, y = X_val[:, 0], val_preds
sns.scatterplot(x=x, y=y, color="red")

plt.legend(["True scores", "Predicted scores"])
plt.xlabel("Input")
plt.ylabel("Output")
