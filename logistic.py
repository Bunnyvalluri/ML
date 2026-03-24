#1
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# Step 1: Loading a dataset
data = {"X": [1, 2, 3, 4, 5], "Y": [0, 0, 0, 1, 1]}

print(data)

df = pd.DataFrame(data)
print(df)


#2
# Selecting feature column (as DataFrame → 2D)
x = df[["X"]]
print(x.head())       # shows first few rows
print(type(x))        # should be DataFrame

print()

# Selecting target column (as Series → 1D)
y = df["Y"]
print(y.head())
print(type(y))        # should be Series

#3
print(x.info())
print(df["Y"].value_counts())


#4
# Build the model
model = LogisticRegression()

# Train the model (x = DataFrame, y = Series)
model.fit(x, y)

# Get regression coefficients (w1, w2, ...) and intercept (w0)
w0 = model.intercept_
w = model.coef_

print(w0)
print(w) 


#5
 y_pred = model.predict(x)
 print(y_pred)

#6
# Predict probabilities (uses sigmoid/softmax internally)
y_prob = model.predict_proba(x)

print(np.round(y_prob, 2))

#7
# Predict class labels
y_pred = model.predict(x)

# Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

#8
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

#9
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

cm = confusion_matrix(y, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:\n", cm)
