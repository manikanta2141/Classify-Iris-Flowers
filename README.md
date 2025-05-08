# Classify-Iris-Flowers
Build a classification model using the Iris dataset to classify iris flowers into three species (Setosa, Versicolor, Virginica) based on their sepal and petal dimensions.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()
import pandas as pd

# Replace 'IRIS.csv' with the exact filename you uploaded
df = pd.read_csv('IRIS.csv')
df.head()

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode species names into numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode species names into numeric labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC()
svm_model.fit(X_train, y_train)

def evaluate(model, name):
    y_pred = model.predict(X_test)
    print(f"🔹 {name} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\n" + "="*50 + "\n")

evaluate(log_model, "Logistic Regression")
evaluate(tree_model, "Decision Tree")
evaluate(svm_model, "Support Vector Machine")

from sklearn.metrics import confusion_matrix

best_model = svm_model  # change if needed
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

