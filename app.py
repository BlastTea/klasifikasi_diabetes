import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

df1 = pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")
df2 = pd.read_csv("diabetes_binary_health_indicators_BRFSS2021.csv")

data = pd.concat([df1, df2])
data = data.drop(['Sex', 'GenHlth', 'Income'], axis=1)
data = data.drop_duplicates().reset_index()
data = data.drop(['index'], axis=1)

X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# svm_model = SVR()
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [1, 0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'poly', 'linear', 'sigmoid', 'precomputed']
# }

# grid = GridSearchCV(svm_model, param_grid, refit=True, verbose=2)
# grid.fit(X_train, y_train)
# y_pred = grid.predict(X_test)

# svm_model = SVC(kernel='rbf')
svm_model = LinearSVC(dual=False)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)

# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(cr)
print("\nAccuracy:", accuracy)

plt.figure(figsize=(10, 7))
sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# print("Best parameters found: ", grid.best_params_)